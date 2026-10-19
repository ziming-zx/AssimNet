"""Build the three components and assemble optional known features for the proxy."""

from copy import deepcopy
from pathlib import Path
import math

import torch
from torch import nn

from .networks import GaussianEncoder, Decoder, ProxyNetwork, positive_int
from .training import TrainableModel


def check_shape(x, tail, name):
    if not isinstance(x, torch.Tensor) or tuple(x.shape[1:]) != tuple(tail):
        shape = tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, tail))}); got {shape}.")


def check_weight(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return float(value)


def kl_standard_normal(mean, logvar):
    return -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(dim=-1).mean()


def kl_between_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    """KL(q || p), summed over latent dimensions and averaged over samples."""
    return 0.5 * (logvar_p - logvar_q - 1 + (logvar_q - logvar_p).exp()
                  + (mu_q - mu_p).square() * (-logvar_p).exp()).sum(dim=-1).mean()


class Proxy(TrainableModel):
    prediction_key = "y_pred"

    def __init__(self, **config):
        super().__init__()
        self.network = ProxyNetwork(**config)
        self.input_dim = config["input_dim"]
        self.sequence_length = config["sequence_length"]
        self.output_dim = config.get("output_dim", 1)
        self.input_mode = config.get("input_mode", "static")

    def forward(self, x):
        return self.network(x)

    def _validate_x(self, x):
        tail = (self.input_dim,) if self.input_mode == "static" else (self.sequence_length, self.input_dim)
        check_shape(x, tail, "Proxy input")

    def _validate_y(self, x, y):
        check_shape(y, (self.sequence_length, self.output_dim), "Proxy target")

    def _losses(self, x, y, sample):
        return {"loss": self._reconstruction_loss(self(x), y)}

    def _predict_batch(self, x, sample, return_details):
        return {"y_pred": self(x)}


class VAE(TrainableModel):
    metric_names = ("loss", "recons_loss", "kl_loss")
    prediction_key = "theta_pred"
    supports_sampling = True

    def __init__(self, input_dim, latent_dim=32, encoder_hidden_dims=(64, 64),
                 decoder_hidden_dims=(64, 64), activation="relu", dropout=0.2,
                 projection_dim=64, projection_activation="tanh", output_activation="linear"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = GaussianEncoder(input_dim, latent_dim, encoder_hidden_dims, activation, dropout)
        self.decoder = Decoder(latent_dim, input_dim, decoder_hidden_dims, activation, dropout,
                               projection_dim, projection_activation, output_activation)
        self.gamma = 5000.0

    def compile(self, optimizer="adam", lr=1e-4, loss="mse", gamma=5000, **optimizer_kwargs):
        self.gamma = check_weight(gamma, "gamma")
        return super().compile(optimizer, lr, loss, **optimizer_kwargs)

    def forward(self, x, sample=None):
        mean, logvar, z = self.encoder(x, self.training if sample is None else sample)
        return {"theta_pred": self.decoder(z), "z_mean": mean, "z_logvar": logvar}

    def _validate_x(self, x):
        check_shape(x, (self.input_dim,), "VAE input")

    def _validate_y(self, x, y):
        if y is not None:
            raise ValueError("VAE.fit takes parameters as x and reconstructs them; omit y.")

    def _losses(self, x, y, sample):
        result = self(x, sample=sample)
        reconstruction = self._reconstruction_loss(result["theta_pred"], x)
        kl = kl_standard_normal(result["z_mean"], result["z_logvar"])
        return {"loss": self.gamma * reconstruction + kl, "recons_loss": reconstruction, "kl_loss": kl}

    def _predict_batch(self, x, sample, return_details):
        result = self(x, sample=sample)
        return result if return_details else {"theta_pred": result["theta_pred"]}


class InputAssembler(nn.Module):
    """Broadcast static features and reorder columns, entirely inside autograd."""

    def __init__(self, theta_dim, proxy, theta_features=None, known_features=None, proxy_features=None):
        super().__init__()
        theta_features = list(theta_features) if theta_features is not None else [f"theta_{i}" for i in range(theta_dim)]
        known_features = list(known_features) if known_features is not None else []
        combined = theta_features + known_features
        proxy_features = list(proxy_features) if proxy_features is not None else combined.copy()
        if any(not isinstance(name, str) or not name for name in combined + proxy_features):
            raise ValueError("Feature names must be nonempty strings.")
        if len(theta_features) != theta_dim:
            raise ValueError("theta_features must match the VAE input/output dimension.")
        if len(set(combined)) != len(combined):
            raise ValueError("theta_features and known_features must be unique and disjoint.")
        if len(proxy_features) != len(combined) or set(proxy_features) != set(combined):
            raise ValueError("proxy_features must be a permutation of theta_features + known_features.")
        if len(proxy_features) != proxy.input_dim:
            raise ValueError("The assembled feature count must match proxy_config['input_dim'].")
        self.known_dim = len(known_features)
        self.input_mode = proxy.input_mode
        self.sequence_length = proxy.sequence_length
        self.register_buffer("column_order", torch.tensor([combined.index(name) for name in proxy_features]))

    def validate_known(self, x2, count):
        if not self.known_dim:
            if x2 is not None:
                raise ValueError("X2 was supplied, but input_config has no known_features.")
            return
        if x2 is None:
            raise ValueError("X2 is required because input_config specifies known_features.")
        valid = tuple(x2.shape) == (count, self.known_dim)
        if self.input_mode == "sequence":
            valid = valid or tuple(x2.shape) == (count, self.sequence_length, self.known_dim)
        if not valid:
            raise ValueError(f"X2 must be (N, {self.known_dim})"
                             + (f" or (N, {self.sequence_length}, {self.known_dim})." if self.input_mode == "sequence" else "."))

    def forward(self, theta, x2=None):
        if self.input_mode == "sequence":
            theta = theta.unsqueeze(1).expand(-1, self.sequence_length, -1)
            if x2 is not None and x2.ndim == 2:
                x2 = x2.unsqueeze(1).expand(-1, self.sequence_length, -1)
        combined = theta if x2 is None else torch.cat((theta, x2), dim=-1)
        return combined.index_select(-1, self.column_order)


class Assimilation(TrainableModel):
    metric_names = ("loss", "recons_loss", "kl_loss")
    prediction_key = "y_pred"
    supports_sampling = True

    def __init__(self, proxy, vae, assembler, input_dim, hidden_dims=(128, 128),
                 activation="relu", dropout=0.2):
        super().__init__()
        self.input_dim = positive_int(input_dim, "assim input_dim")
        self.encoder = GaussianEncoder(input_dim, vae.latent_dim, hidden_dims, activation, dropout)
        self.vae_encoder = vae.encoder
        self.decoder = vae.decoder
        self.proxy = proxy
        self.assembler = assembler
        self.kl_weight = 0.001

    def training_parameters(self):
        return self.encoder.parameters()

    def _prepare_training(self):
        super()._prepare_training()
        for module in (self.vae_encoder, self.decoder, self.proxy):
            module.requires_grad_(False)
            module.zero_grad(set_to_none=True)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        # Frozen Dropout must stay off even when the assimilation encoder trains.
        for module in (self.vae_encoder, self.decoder, self.proxy):
            module.eval()
        return self

    def compile(self, optimizer="adam", lr=1e-4, loss="mse", kl_weight=0.001, **optimizer_kwargs):
        self.kl_weight = check_weight(kl_weight, "kl_weight")
        return super().compile(optimizer, lr, loss, **optimizer_kwargs)

    def forward(self, x, sample=None):
        mean, logvar, z = self.encoder(x["x1"], self.training if sample is None else sample)
        theta = self.decoder(z)
        proxy_input = self.assembler(theta, x.get("x2"))
        return {"y_pred": self.proxy(proxy_input), "theta_pred": theta,
                "z_mean": mean, "z_logvar": logvar}

    def _validate_x(self, x):
        if not isinstance(x, dict) or "x1" not in x or set(x) - {"x1", "x2"}:
            raise ValueError("Assimilation input must be {'x1': array, 'x2': optional_array}.")
        check_shape(x["x1"], (self.input_dim,), "X1")
        self.assembler.validate_known(x.get("x2"), len(x["x1"]))

    def _validate_y(self, x, y):
        check_shape(y, (self.proxy.sequence_length, self.proxy.output_dim), "Assimilation target")

    def _losses(self, x, y, sample):
        result = self(x, sample=sample)
        # Do not use no_grad here: both frozen paths must differentiate w.r.t. theta.
        mu_p, logvar_p, _ = self.vae_encoder(result["theta_pred"], sample=False)
        kl = kl_between_gaussians(result["z_mean"], result["z_logvar"], mu_p, logvar_p)
        reconstruction = self._reconstruction_loss(result["y_pred"], y)
        return {"loss": reconstruction + self.kl_weight * kl,
                "recons_loss": reconstruction, "kl_loss": kl}

    def _predict_batch(self, x, sample, return_details):
        result = self(x, sample=sample)
        return result if return_details else {"y_pred": result["y_pred"]}


class AssimNet(nn.Module):
    """Shared components; train proxy, then VAE, then assimilation."""

    def __init__(self, proxy_config, vae_config, assim_config, input_config=None):
        super().__init__()
        self.config = deepcopy({"proxy_config": proxy_config, "vae_config": vae_config,
                                "assim_config": assim_config, "input_config": input_config or {}})
        self.proxy = Proxy(**proxy_config)
        self.vae = VAE(**vae_config)
        assembler = InputAssembler(self.vae.input_dim, self.proxy, **(input_config or {}))
        self.assim = Assimilation(self.proxy, self.vae, assembler, **assim_config)

    def forward(self, x, sample=None):
        return self.assim(x, sample=sample)

    def save(self, path):
        """Save an inference checkpoint. Optimizer states/history are not included."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu_state = lambda module: {key: value.detach().cpu() for key, value in module.state_dict().items()}
        torch.save({"format_version": 1, "config": self.config,
                    "proxy": cpu_state(self.proxy), "vae": cpu_state(self.vae),
                    "assim_encoder": cpu_state(self.assim.encoder),
                    "gamma": self.vae.gamma, "kl_weight": self.assim.kl_weight}, path)


def build_assimnet(*, proxy_config, vae_config, assim_config, input_config=None, device="auto"):
    """Construct networks from dictionaries; see README for all configuration keys."""
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    return AssimNet(proxy_config, vae_config, assim_config, input_config).to(device)


def load_assimnet(path, device="auto"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported AssimNet checkpoint version.")
    model = build_assimnet(**checkpoint["config"], device=device)
    model.proxy.load_state_dict(checkpoint["proxy"])
    model.vae.load_state_dict(checkpoint["vae"])
    model.assim.encoder.load_state_dict(checkpoint["assim_encoder"])
    model.vae.gamma = checkpoint["gamma"]
    model.assim.kl_weight = checkpoint["kl_weight"]
    model.eval()
    return model
