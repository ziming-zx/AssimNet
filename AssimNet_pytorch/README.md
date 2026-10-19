# AssimNet PyTorch

A small, reusable PyTorch implementation of the three-stage AssimNet workflow
from the TensorFlow CaseI and CaseIII examples. It does not import TensorFlow or
modify the existing `AssimNet_tensorflow` directory.

## Environment and GPU setup

Create a local Python 3.12 virtual environment in `.venv/` as described below.
Runtime dependencies are pinned in `requirements.txt`:

| Package | Version |
| --- | --- |
| PyTorch | `2.10.0+cu128` (CUDA 12.8 build) |
| NumPy | `2.2.6` |
| Matplotlib | `3.8.4` |

The Case I quick run checks the three training stages, checkpoint reload, and
prediction on a small synthetic dataset. It does not reproduce manuscript accuracy.

The requirements target Python 3.12 on Windows x64 with an NVIDIA GPU. The CUDA wheel
comes from the [official PyTorch CUDA 12.8 index](https://download.pytorch.org/whl/cu128);
see the [official version-specific installation commands](https://pytorch.org/get-started/previous-versions/).
The direct wheel URL specifies `2.10.0+cu128` and verifies its official SHA-256,
preventing an accidental CPU-only installation and avoiding the download-r2 mirror.
For a different Python version or operating system, select the corresponding
CUDA wheel from the official instructions instead of this Windows-specific URL.
Neither torchvision nor torchaudio is needed by this project.

### Use a created environment (Windows PowerShell)

Open a terminal in `AssimNet_pytorch` and run:

```powershell
.\.venv\Scripts\Activate.ps1
python caseI_main.py --quick --device cuda
```

Alternatively, call the environment's Python directly. This works without
activation and without changing PowerShell's execution policy:

```powershell
.\.venv\Scripts\python.exe caseI_main.py --quick --device cuda
.\.venv\Scripts\python.exe caseI_main.py --device cuda
```

For an editor such as VS Code, select `.venv\Scripts\python.exe` as the Python
interpreter. Installing into this environment does not replace system Python
packages or packages used by the TensorFlow examples.

### Recreate the environment

In `AssimNet_pytorch`, run the following with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The GPU build is a large download. A compatible NVIDIA driver is required.
Prebuilt PyTorch wheels provide the CUDA runtime libraries needed for these
models; this project does not compile CUDA extensions. The CUDA version shown by
`nvidia-smi` describes driver capability and need not equal `torch.version.cuda`.

Verify that this exact environment can see and use the GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0)); x = torch.randn(256, 256, device='cuda', requires_grad=True); (x @ x).square().mean().backward(); torch.cuda.synchronize(); print('CUDA forward/backward OK')"
```

The commands below assume the virtual environment is active. If it is not, replace
`python` with `.\.venv\Scripts\python.exe` on Windows.

## Run CaseI

```powershell
python caseI_main.py --device cuda
```

The script also runs from other working directories when given its path. Without
`--device`, it selects CUDA if available and otherwise CPU. Specifying
`--device cuda` requires a working GPU rather than silently falling back to CPU.

Additional examples:

```powershell
python caseI_main.py --quick --device cuda
python caseI_main.py --device cpu --no-early-stopping
python caseI_main.py --device cuda --output-dir outputs/my_experiment
```

`--quick` uses small networks, small datasets, and 3 / 3 / 5 maximum epochs. It
checks execution, not convergence or research accuracy. A normal run generates
5,000 simulation samples and 200 field samples. The first 4,900 / 100 samples form
the training and validation pools; 10% of each pool is randomly reserved for
validation. The remaining samples are used only for final testing. Maximum epoch
counts are 500 / 1,000 / 2,000 for Proxy / VAE / Assimilation.

Early stopping is enabled by default in this example. Disabling it runs every
configured epoch but still restores the best weights unless that option is also
changed. Dataset sizes, network settings, learning rates, loss weights, and fit
options are collected in four dictionaries at the top of `caseI_main.py`.
`--threads` sets the number of PyTorch CPU threads; its default is 4.

CaseI uses the ground-truth function `d * (cos(t + e) + sin(t + e))` and simulator
`a * cos(b*t + c)`. Inputs are used in their original units without scaling.
Besides predicting through the proxy, the example feeds inferred `[a, b, c]`
back into the simulator to compare the two responses.

Results are saved under `outputs/caseI/`, or `outputs/caseI_quick/` in quick mode:

- `assimnet.pt`: complete model configuration and weights for standalone loading.
- `proxy_best.pt`, `vae_best.pt`, `assim_best.pt`: best weights from each stage.
- `config.json`, `*_history.json`, `metrics.json`: settings, training histories,
  and test metrics.
- `data_and_predictions.npz`: generated data, split indices, predictions, and
  inferred parameters.
- `training_curves.png`, `test_predictions.png`: loss curves and response plots.

Running again in the same output directory overwrites files with the same names.
Use `--output-dir` to retain separate experiments. The virtual environment,
generated outputs, model weights, and Python caches are excluded by `.gitignore`.

## Project structure

```text
AssimNet_pytorch/
  .venv/           Local virtual environment; not committed
  source/
    __init__.py    Public imports
    networks.py    GRU proxy, Gaussian encoder, decoder, attention
    assimnet.py    Component assembly, feature mapping, stage-specific losses
    training.py    compile / fit / predict, early stopping, weight persistence
  caseI_main.py    Single runnable example
  requirements.txt
  .gitignore
  README.md
  outputs/         Generated experiment results; not committed
```

There is no external configuration framework or separate Trainer object. Each
component is an ordinary `torch.nn.Module`. The provided `compile()` configures
training; it is separate from PyTorch's `torch.compile()` graph compiler.

## Build, compile, fit, and predict

```python
from source import build_assimnet, load_assimnet

model = build_assimnet(
    proxy_config={"input_dim": 3, "sequence_length": 100},
    vae_config={"input_dim": 3, "latent_dim": 32},
    assim_config={"input_dim": 2},
    device="auto",
)

# sim_x / theta: (N, 3); sim_y / field_y: (N, 100, 1)
# Stages may use different sample counts. theta contains the simulator parameters.
model.proxy.compile(optimizer="adam", lr=1e-3, loss="mse")
proxy_history = model.proxy.fit(
    sim_x, sim_y, validation_data=(sim_x_val, sim_y_val),
    epochs=500, early_stopping=True, monitor="val_loss",
)

model.vae.compile(optimizer="adam", lr=1e-4, loss="mse", gamma=5000)
vae_history = model.vae.fit(
    theta, validation_data=theta_val, epochs=1000,
    early_stopping=True, monitor="val_recons_loss",
)

model.assim.compile(optimizer="adam", lr=1e-4, loss="mse", kl_weight=0.001)
assim_history = model.assim.fit(
    {"x1": field_x1}, field_y,
    validation_data=({"x1": field_x1_val}, field_y_val),
    epochs=2000, batch_size=128,
    early_stopping=True, monitor="val_recons_loss", patience=100,
)

y_pred = model.assim.predict({"x1": field_x1_test})
model.save("outputs/my_model.pt")
restored = load_assimnet("outputs/my_model.pt", device="cuda")
y_pred = restored.assim.predict({"x1": field_x1_test})
```

The data variables above represent your NumPy arrays or tensors; see
`caseI_main.py` for a complete executable example. Inputs are converted to float32,
batched internally, and sent to the model's device. Set the device at build time.
If you manually move a compiled model to another device, compile it again before
training.

Train sequentially: Proxy, then VAE, then Assimilation. Components share their
network objects, so saving and reloading between stages is unnecessary.
Assimilation trains only its own encoder. The VAE encoder, decoder, and proxy
have frozen parameters and disabled dropout, while gradients through these
networks still reach the assimilation encoder.

On CUDA, backward through the frozen evaluation-mode GRU uses native PyTorch GRU
operations to avoid cuDNN's evaluation-mode backward restriction. Ordinary proxy
training and prediction without gradients can still use cuDNN.

Fitting Proxy or VAE again automatically unfreezes that stage's parameters. This
also changes mappings used by Assimilation, so retraining Assimilation is
normally necessary. Run the stages sequentially, not concurrently.

## Network configuration

Configurations are ordinary dictionaries. Unknown keys raise an error.

| Configuration | Parameters and defaults |
| --- | --- |
| `proxy_config` | Required: `input_dim`, `sequence_length`. Optional: `output_dim=1`, `input_mode="static"`, `gru_hidden_dims=[128,128]`, `dense_hidden_dims=[128]`, `dropout=0.2`, `attention="temporal"`, `activation="relu"`. |
| `vae_config` | Required: `input_dim`. Optional: `latent_dim=32`, `encoder_hidden_dims=[64,64]`, `decoder_hidden_dims=[64,64]`, `activation="relu"`, `dropout=0.2`, `projection_dim=64`, `projection_activation="tanh"`, `output_activation="linear"`. |
| `assim_config` | Required: `input_dim`, the number of X1 features. Optional: `hidden_dims=[128,128]`, `activation="relu"`, `dropout=0.2`. The latent dimension is inherited from the VAE. |
| `input_config` | Optional: `theta_features`, `known_features`, `proxy_features`; see below. |

`gru_hidden_dims` controls the number and width of GRU layers. Attention is placed
after the first GRU and dropout. Supported attention choices are `"temporal"`
(CaseI), `"self"` (CaseIII single-head Q/K/V), and `None`.

Static proxy inputs are repeated to `sequence_length` inside the network;
sequence inputs go directly into the GRU. By default, the decoder adds a tanh
projection after its hidden layers; set `projection_dim=None` to remove it.
Supported activations are `relu`, `tanh`, `gelu`, `silu`, and `linear`.
Automatic parameter clipping is not implemented, matching the active CaseIII
forward computation.

## CaseIII-style X1 and X2 inputs

```text
X1 -> Assimilation Encoder -> z -> Decoder -> theta --+
                                                   +-> Assemble/reorder -> Proxy -> y_pred
                                            X2 ----+

theta -> VAE Encoder -> p(z | theta)
Compute KL(q || p), where q(z | X1) comes from the assimilation encoder.
```

X1 enters the assimilation encoder. X2 supplies known conditions directly to the
proxy input assembly; it does not enter the assimilation encoder. The VAE learns
only the parameters theta that must be reconstructed, rather than all proxy
input features indiscriminately.

For example, the decoder infers `a, b`, X2 provides `depth, time`, and the proxy
expects `a, depth, b, time`:

```python
model = build_assimnet(
    proxy_config={
        "input_dim": 4, "sequence_length": 36,
        "input_mode": "sequence", "attention": "self",
    },
    vae_config={"input_dim": 2, "latent_dim": 16},
    assim_config={"input_dim": 5},
    input_config={
        "theta_features": ["a", "b"],
        "known_features": ["depth", "time"],
        "proxy_features": ["a", "depth", "b", "time"],
    },
)

# Train Proxy and VAE first, as shown above, then train Assimilation.
model.assim.compile(lr=1e-4, kl_weight=0.001)
history = model.assim.fit(
    {"x1": x1, "x2": x2}, y,
    validation_split=0.1, seed=66,
    early_stopping=True, monitor="val_recons_loss",
)
result = model.assim.predict({"x1": x1_test, "x2": x2_test}, return_details=True)
```

| Data | Shape |
| --- | --- |
| X1 | `(N, D1)` |
| theta / VAE training input | `(N, Dtheta)` |
| X2 | `(N, D2)`; a sequence proxy also accepts `(N, T, D2)` |
| Static proxy training input | `(N, Dproxy)` |
| Sequence proxy training input | `(N, T, Dproxy)` |
| Proxy / Assimilation target Y | `(N, T, Dy)`; retain the last axis even for one output |

Static theta and X2 features are broadcast along time when needed. Sequence X2
must match the configured T. Feature names must be unique; `proxy_features` must
be a complete permutation of theta and known feature names. If `known_features`
is configured, both fit and predict require X2; otherwise X2 is rejected.

Assembly handles only broadcasting and column order. Preprocessing belongs in
the case script, with scalers fitted only on training data. **theta, X2, and the
corresponding proxy input columns must use the same scales**, including during
standalone VAE training. Model checkpoints do not include external scalers; save
and reuse them separately for new cases.

## Losses, histories, and early stopping

| Component | `loss` | Additional metrics |
| --- | --- | --- |
| Proxy | Response prediction error | None |
| VAE | `gamma * recons_loss + kl_loss` | `recons_loss`, `kl_loss` |
| Assimilation | `recons_loss + kl_weight * kl_loss` | `recons_loss`, `kl_loss` |

VAE KL is relative to a standard normal; Assimilation uses `KL(q || p)`. KL is
summed over latent dimensions and averaged over samples. The default
reconstruction loss is averaged over all elements.

`compile(loss=...)` accepts `"mse"`, `"mae"`, `"huber"`, or a callable returning a
scalar tensor. It replaces only the reconstruction term. Validation computes all
losses with the `val_` prefix. Training samples latent variables; validation uses
their means with dropout disabled, so training and validation losses need not
have identical values even on the same data.

Optimizers can be `"adam"`, `"adamw"`, `"sgd"`, or an existing PyTorch optimizer.
Extra compile arguments such as `weight_decay` are passed to the optimizer.
An existing optimizer uses its own learning rate and must contain exactly the
parameters returned by `component.training_parameters()`.

Common `fit` options:

| Option | Default / behavior |
| --- | --- |
| `epochs`, `batch_size` | `100`, `256` |
| `shuffle`, `seed` | `True`, `None`. Seed controls splitting, batching, and Torch randomness; seed before build separately for reproducible initialization. |
| `validation_data` | `(x_val, y_val)` for Proxy / Assimilation; `theta_val` or `(theta_val, None)` for VAE. |
| `validation_split` | `0.0`. Split once at random, keeping X1, X2, and Y aligned. Mutually exclusive with validation_data. |
| `early_stopping` | `False`; the CaseI script explicitly enables it. |
| `monitor` | `val_loss` when validation exists, otherwise `loss`. Any recorded loss for the current component can be selected. |
| `patience`, `min_delta` | `50`, `0.0`. Improvement must exceed min_delta; stop after patience consecutive epochs without improvement. |
| `restore_best_weights` | `True`, independently of early_stopping. Uses the same monitor and min_delta criterion. |
| `verbose_every` | `10`; use `0` to disable logging. |
| `checkpoint_path` | `None`; when set, save the component's weights after each qualifying improvement. |

All monitored losses are minimized. An unavailable `val_*` metric, misspelled
monitor, or incompatible input shape raises an error. Patience must be at least
1, and training/validation sets must be nonempty. Set
`restore_best_weights=False` to retain the last epoch's weights.

`fit` returns a `History` object. `history.history` maps metric names to lists;
additional fields are `best_epoch`, `best_value`, `epochs_ran`, `stopped_epoch`,
and `monitor`. Epoch numbers start at 1; stopped_epoch is 0 if early stopping did
not trigger. Training metrics average pre-update batch losses weighted by sample
count; validation metrics use weights at the end of the epoch.

## Prediction and checkpoints

- `proxy.predict(x)` returns `y_pred`.
- `vae.predict(theta)` returns reconstructed `theta_pred`.
- `assim.predict({"x1": ..., "x2": ...})` returns `y_pred`; omit X2 for CaseI.
- `return_details=True` returns a dictionary. VAE includes
  `theta_pred`, `z_mean`, `z_logvar`; Assimilation also includes `y_pred`.

By default, `sample=False` uses z_mean and returns NumPy arrays. Prediction batch
size defaults to 256. VAE and Assimilation accept `sample=True, n_samples=100`,
which adds a leading sampling axis `(100, N, ...)` to every returned array. A
single draw retains shape `(N, ...)`.

The default response is the decoder/proxy output at the latent mean, not a Monte
Carlo predictive mean. Samples reflect latent randomness, not calibrated
confidence intervals:

```python
samples = model.assim.predict({"x1": x1_test}, sample=True, n_samples=100)
y_mean = samples.mean(axis=0)
y_std = samples.std(axis=0)
```

`model.save()` and `load_assimnet()` persist and reconstruct the complete model;
compile is unnecessary for prediction. Checkpoints exclude optimizer states and
external preprocessing. After loading, compile and fit can continue training
with a new optimizer, but do not exactly resume an interrupted training run.

Individual components offer `save_weights()` / `load_weights()` for loading into
an object with the same architecture. Assimilation weight files also include
the frozen networks on which it depends.

## Quick execution check

```powershell
python -m pip check
python caseI_main.py --quick --device cuda --output-dir outputs/caseI_gpu_quick
```

The script trains all three stages by default when starting from an empty output
directory. Use `--no-train-proxy`, `--no-train-vae`, or `--no-train-assim` only
when the corresponding checkpoint already exists in that output directory.

The implementation follows the original network structures but does not import
TensorFlow weights. Framework-specific GRU implementations and random generators
differ, so exact numerical equality with the notebooks is not expected.
