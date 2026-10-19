"""Small, configurable neural networks used by the three AssimNet components."""

from contextlib import nullcontext
import math

import torch
from torch import nn


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def get_activation(name):
    choices = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU,
        "silu": nn.SiLU, "linear": nn.Identity,
    }
    if name not in choices:
        raise ValueError(f"activation must be one of {tuple(choices)}.")
    return choices[name]()


def mlp(input_dim, hidden_dims, activation_name="relu", dropout=0.2):
    positive_int(input_dim, "input_dim")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    layers = []
    width = input_dim
    for next_width in hidden_dims:
        positive_int(next_width, "hidden dimension")
        layers.extend([nn.Linear(width, next_width), get_activation(activation_name)])
        if dropout:
            layers.append(nn.Dropout(dropout))
        width = next_width
    return nn.Sequential(*layers), width


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=(128, 128),
                 activation="relu", dropout=0.2):
        super().__init__()
        positive_int(latent_dim, "latent_dim")
        self.body, width = mlp(input_dim, hidden_dims, activation, dropout)
        self.mean = nn.Linear(width, latent_dim)
        self.logvar = nn.Linear(width, latent_dim)

    def forward(self, x, sample=True):
        h = self.body(x)
        mean, logvar = self.mean(h), self.logvar(h)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample else mean
        return mean, logvar, z


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=(128, 128),
                 activation="relu", dropout=0.2, projection_dim=128,
                 projection_activation="tanh", output_activation="linear"):
        super().__init__()
        positive_int(output_dim, "output_dim")
        self.body, width = mlp(latent_dim, hidden_dims, activation, dropout)
        layers = []
        if projection_dim is not None:
            positive_int(projection_dim, "projection_dim")
            layers.extend([nn.Linear(width, projection_dim),
                           get_activation(projection_activation)])
            width = projection_dim
        layers.extend([nn.Linear(width, output_dim), get_activation(output_activation)])
        self.output = nn.Sequential(*layers)

    def forward(self, z):
        return self.output(self.body(z))


class TemporalAttention(nn.Module):
    """A scalar score per timestep, broadcast across all hidden features (CaseI)."""

    def __init__(self, width):
        super().__init__()
        self.score = nn.Linear(width, 1)

    def forward(self, x):
        weights = torch.softmax(torch.tanh(self.score(x)), dim=1)
        return x * weights


class SelfAttention(nn.Module):
    """Single-head scaled dot-product attention, matching the CaseIII structure."""

    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.scale = math.sqrt(width)

    def forward(self, x):
        weights = torch.softmax(self.query(x) @ self.key(x).transpose(-1, -2) / self.scale, dim=-1)
        return weights @ self.value(x)


class ProxyNetwork(nn.Module):
    def __init__(self, input_dim, sequence_length, output_dim=1, input_mode="static",
                 gru_hidden_dims=(128, 128), dense_hidden_dims=(128,),
                 dropout=0.2, attention="temporal", activation="relu"):
        super().__init__()
        positive_int(input_dim, "input_dim")
        positive_int(sequence_length, "sequence_length")
        positive_int(output_dim, "output_dim")
        if input_mode not in ("static", "sequence"):
            raise ValueError("input_mode must be 'static' or 'sequence'.")
        if not gru_hidden_dims:
            raise ValueError("gru_hidden_dims must contain at least one layer.")
        if attention not in (None, "temporal", "self"):
            raise ValueError("attention must be None, 'temporal', or 'self'.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.input_mode, self.sequence_length = input_mode, sequence_length
        self.grus = nn.ModuleList()
        width = input_dim
        for hidden in gru_hidden_dims:
            positive_int(hidden, "GRU hidden dimension")
            self.grus.append(nn.GRU(width, hidden, batch_first=True))
            width = hidden
        first_width = gru_hidden_dims[0]
        self.attention = (TemporalAttention(first_width) if attention == "temporal" else
                          SelfAttention(first_width) if attention == "self" else nn.Identity())
        self.dropout = nn.Dropout(dropout)
        self.head, width = mlp(width, dense_hidden_dims, activation, dropout=0)
        self.output = nn.Linear(width, output_dim)

    def forward(self, x):
        if self.input_mode == "static":
            x = x.unsqueeze(1).expand(-1, self.sequence_length, -1)
        # cuDNN's eval-mode RNN forward does not retain the reserve space needed
        # for backward. A frozen proxy still needs gradients w.r.t. its input.
        # Use native GRUs only on that CUDA path; normal training/predict keep cuDNN.
        native_backward = x.is_cuda and not self.training and torch.is_grad_enabled()
        context = torch.backends.cudnn.flags(enabled=False) if native_backward else nullcontext()
        with context:
            for index, gru in enumerate(self.grus):
                x, _ = gru(x)
                x = self.dropout(x)
                if index == 0:
                    x = self.attention(x)
        return self.output(self.head(x))
