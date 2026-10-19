"""Keras-like training convenience methods; networks remain ordinary nn.Modules."""

from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn

from .networks import positive_int


@dataclass
class History:
    history: dict = field(default_factory=dict)
    best_epoch: int = 0
    stopped_epoch: int = 0
    epochs_ran: int = 0
    monitor: str = ""
    best_value: float = math.inf


def as_tensor(value):
    result = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if result.ndim < 2 or len(result) == 0:
        raise ValueError("Inputs and targets must be nonempty arrays with a batch dimension (N, ...).")
    if not torch.isfinite(result).all():
        raise ValueError("Inputs and targets must contain only finite values.")
    return result


def prepare_x(x):
    if isinstance(x, dict):
        if not x:
            raise ValueError("Input dictionary cannot be empty.")
        result = {key: as_tensor(value) for key, value in x.items()}
        if len({len(value) for value in result.values()}) != 1:
            raise ValueError("All input arrays must have the same sample count.")
        return result
    return as_tensor(x)


def sample_count(x):
    return len(next(iter(x.values()))) if isinstance(x, dict) else len(x)


def subset(x, indices, device="cpu"):
    if isinstance(x, dict):
        return {key: value[indices].to(device) for key, value in x.items()}
    return x[indices].to(device)


class TrainableModel(nn.Module):
    """Subclasses implement validation, loss calculation, and batch prediction."""

    metric_names = ("loss",)
    supports_sampling = False

    @property
    def device(self):
        return next(self.parameters()).device

    def training_parameters(self):
        return self.parameters()

    def _prepare_training(self):
        for parameter in self.training_parameters():
            parameter.requires_grad_(True)

    def compile(self, optimizer="adam", lr=1e-3, loss="mse", **optimizer_kwargs):
        """Configure training; this method does not invoke torch.compile."""
        self._prepare_training()
        parameters = list(self.training_parameters())
        if isinstance(optimizer, torch.optim.Optimizer):
            expected = {id(p) for p in parameters}
            actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if expected != actual:
                raise ValueError("The optimizer must contain exactly this component's training parameters.")
            if optimizer_kwargs:
                raise ValueError("Configure optimizer options on the supplied optimizer instance.")
            self.optimizer = optimizer
        else:
            choices = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
            if optimizer not in choices:
                raise ValueError("optimizer must be 'adam', 'adamw', 'sgd', or a torch optimizer instance.")
            if not math.isfinite(lr) or lr < 0:
                raise ValueError("lr must be finite and nonnegative.")
            self.optimizer = choices[optimizer](parameters, lr=lr, **optimizer_kwargs)
        losses = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.HuberLoss}
        if isinstance(loss, str):
            if loss not in losses:
                raise ValueError("loss must be 'mse', 'mae', 'huber', or a scalar loss callable.")
            loss = losses[loss]()
        if not callable(loss):
            raise ValueError("loss must be callable.")
        # Keep the criterion outside the model's checkpoint and parameter tree.
        self.__dict__["loss_fn"] = loss.to(self.device) if isinstance(loss, nn.Module) else loss
        return self

    def _reconstruction_loss(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}.")
        result = self.loss_fn(prediction, target)
        if not isinstance(result, torch.Tensor) or result.ndim != 0:
            raise ValueError("The loss callable must return a scalar tensor (use mean reduction).")
        return result

    def _prepare_data(self, x, y):
        x = prepare_x(x)
        y = None if y is None else as_tensor(y)
        if y is not None and sample_count(x) != len(y):
            raise ValueError("Inputs and targets must have the same sample count.")
        self._validate_x(x)
        self._validate_y(x, y)
        return x, y

    def fit(self, x, y=None, *, validation_data=None, validation_split=0.0,
            epochs=100, batch_size=256, shuffle=True, seed=None,
            early_stopping=False, monitor=None, patience=50, min_delta=0.0,
            restore_best_weights=True, verbose_every=10, checkpoint_path=None):
        """Train one component. Epoch numbers in History are one-based.

        validation_data is (x_val, y_val), or x_val alone for a VAE.
        validation_split selects a seeded random subset before epoch shuffling.
        All monitored losses are minimized. Best means improvement > min_delta.
        """
        if not hasattr(self, "optimizer"):
            raise RuntimeError("Call compile() before fit().")
        positive_int(epochs, "epochs")
        positive_int(batch_size, "batch_size")
        positive_int(patience, "patience")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative.")
        if not isinstance(verbose_every, int) or verbose_every < 0:
            raise ValueError("verbose_every must be a nonnegative integer; 0 disables logging.")
        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be in [0, 1).")
        if validation_data is not None and validation_split:
            raise ValueError("Use validation_data or validation_split, not both.")
        if seed is not None:
            torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        x, y = self._prepare_data(x, y)
        val = None
        if validation_data is not None:
            if isinstance(validation_data, tuple) and len(validation_data) == 2:
                val = self._prepare_data(*validation_data)
            else:
                val = self._prepare_data(validation_data, None)
        elif validation_split:
            count = sample_count(x)
            val_count = int(count * validation_split)
            if not 0 < val_count < count:
                raise ValueError("validation_split must leave at least one training and one validation sample.")
            order = torch.randperm(count, generator=generator)
            train_indices, val_indices = order[val_count:], order[:val_count]
            val = (subset(x, val_indices), None if y is None else y[val_indices])
            x, y = subset(x, train_indices), None if y is None else y[train_indices]
        available = list(self.metric_names)
        if val is not None:
            available += [f"val_{name}" for name in self.metric_names]
        monitor = monitor or ("val_loss" if val is not None else "loss")
        if monitor not in available:
            raise ValueError(f"Unknown monitor {monitor!r}. Available metrics: {available}")
        self._prepare_training()
        self.zero_grad(set_to_none=True)
        history = History(monitor=monitor)
        best_state, bad_epochs = None, 0
        for epoch in range(1, epochs + 1):
            self.train()
            logs = self._run_epoch(x, y, batch_size, True, shuffle, generator)
            if val is not None:
                self.eval()
                with torch.no_grad():
                    val_logs = self._run_epoch(*val, batch_size, False, False, generator)
                logs.update({f"val_{key}": value for key, value in val_logs.items()})
            for key, value in logs.items():
                history.history.setdefault(key, []).append(value)
            history.epochs_ran = epoch
            current = logs[monitor]
            if current < history.best_value - min_delta:
                history.best_value, history.best_epoch, bad_epochs = current, epoch, 0
                if restore_best_weights:
                    best_state = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}
                if checkpoint_path is not None:
                    self.save_weights(checkpoint_path)
            else:
                bad_epochs += 1
            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                values = " | ".join(f"{key}={value:.6g}" for key, value in logs.items())
                print(f"{type(self).__name__} epoch {epoch}/{epochs} | {values}", flush=True)
            if early_stopping and bad_epochs >= patience:
                history.stopped_epoch = epoch
                if verbose_every:
                    print(f"Early stopping at epoch {epoch}; best {monitor} at epoch {history.best_epoch}.", flush=True)
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return history

    def _run_epoch(self, x, y, batch_size, training, shuffle, generator):
        count = sample_count(x)
        order = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
        totals = {name: 0.0 for name in self.metric_names}
        for start in range(0, count, batch_size):
            indices = order[start:start + batch_size]
            xb = subset(x, indices, self.device)
            yb = None if y is None else y[indices].to(self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            losses = self._losses(xb, yb, sample=training)
            if any(not torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError("Nonfinite loss. Check input scaling, learning rate, and latent variance.")
            if training:
                losses["loss"].backward()
                for parameter in self.training_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Nonfinite gradient. Check input scaling and learning rate.")
                self.optimizer.step()
            for key, value in losses.items():
                totals[key] += value.detach().item() * len(indices)
        return {key: value / count for key, value in totals.items()}

    def predict(self, x, *, batch_size=256, return_details=False, sample=False, n_samples=1):
        """Return NumPy arrays. Multiple draws add a leading (S, N, ...) axis."""
        positive_int(batch_size, "batch_size")
        positive_int(n_samples, "n_samples")
        if (sample or n_samples != 1) and not self.supports_sampling:
            raise ValueError("This component does not support latent sampling.")
        if not sample and n_samples != 1:
            raise ValueError("Use sample=True when n_samples > 1.")
        x = prepare_x(x)
        self._validate_x(x)
        modes = {module: module.training for module in self.modules()}
        self.eval()
        draws = []
        try:
            with torch.inference_mode():
                for _ in range(n_samples):
                    batches = []
                    for start in range(0, sample_count(x), batch_size):
                        xb = subset(x, slice(start, start + batch_size), self.device)
                        output = self._predict_batch(xb, sample, return_details)
                        batches.append({key: value.cpu().numpy() for key, value in output.items()})
                    draws.append({key: np.concatenate([batch[key] for batch in batches], axis=0)
                                  for key in batches[0]})
        finally:
            for module, mode in modes.items():
                module.training = mode
        result = draws[0] if n_samples == 1 else {
            key: np.stack([draw[key] for draw in draws], axis=0) for key in draws[0]
        }
        return result if return_details else result[self.prediction_key]

    def save_weights(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.detach().cpu() for key, value in self.state_dict().items()}, path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return self
