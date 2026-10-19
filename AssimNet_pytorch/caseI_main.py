"""One-command synthetic CaseI: python caseI_main.py (or --quick for a smoke run)."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from source import History, build_assimnet, load_assimnet


# Edit these dictionaries to run your own experiment. No notebook is required.
SEED = 66
DATA_CONFIG = {
    "n_sim": 5000, "n_field": 200, "sequence_length": 100,
    "sim_train_pool": 4900, "field_train_pool": 100, "validation_fraction": 0.1,
}
MODEL_CONFIG = {
    "proxy_config": {
        "input_dim": 3, "sequence_length": 100, "output_dim": 1,
        "input_mode": "static", "gru_hidden_dims": [128, 128],
        "dense_hidden_dims": [128], "attention": "temporal", "dropout": 0.2,
    },
    "vae_config": {
        "input_dim": 3, "latent_dim": 32,
        "encoder_hidden_dims": [64, 64], "decoder_hidden_dims": [64, 64],
        "projection_dim": 64, "dropout": 0.2,
    },
    "assim_config": {"input_dim": 2, "hidden_dims": [128, 128], "dropout": 0.2},
    "input_config": {"theta_features": ["a", "b", "c"]},
}
TRAIN_CONFIG = {
    "proxy": {"epochs": 500, "batch_size": 512, "lr": 1e-3, "monitor": "val_loss"},
    "vae": {"epochs": 1000, "batch_size": 512, "lr": 1e-4,
            "gamma": 1000, "monitor": "val_recons_loss"},
    "assim": {"epochs": 5000, "batch_size": 128, "lr": 5e-4,
              "kl_weight": 0.0001, "monitor": "val_loss"},
}
# Set a stage to False to reuse its best stored weights instead of retraining it.
# The command-line --train-*/--no-train-* options can override these defaults.
TRAIN_SWITCHES = {"proxy": True, "vae": True, "assim": True}
FIT_CONFIG = {
    "early_stopping": True, "patience": 1000, "min_delta": 1e-6,
    "restore_best_weights": True, "verbose_every": 10,
}


def ground_truth(t, features):
    d, e = features[:, 0:1], features[:, 1:2]
    return (d * (np.cos(t + e) + np.sin(t + e)))[..., None].astype(np.float32)


def simulator(t, theta):
    a, b, c = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3]
    return (a * np.cos(b * t + c))[..., None].astype(np.float32)


def generate_data(config, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, config["sequence_length"], dtype=np.float32)
    field_x = rng.uniform([0, 0], [1, 2], size=(config["n_field"], 2)).astype(np.float32)
    sim_x = rng.uniform([0, 0, -np.pi], [1.5, 2, np.pi], size=(config["n_sim"], 3)).astype(np.float32)
    data = {"t": t, "sim_x": sim_x, "sim_y": simulator(t, sim_x),
            "field_x": field_x, "field_y": ground_truth(t, field_x)}
    for prefix in ("sim", "field"):
        pool = config[f"{prefix}_train_pool"]
        count = config[f"n_{prefix}"]
        validation_count = int(pool * config["validation_fraction"])
        if not 0 < validation_count < pool < count:
            raise ValueError("Data configuration must leave nonempty training, validation, and test sets.")
        order = rng.permutation(pool)
        data[f"{prefix}_train_idx"] = order[validation_count:]
        data[f"{prefix}_val_idx"] = order[:validation_count]
        data[f"{prefix}_test_idx"] = np.arange(pool, count)
    return data


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def load_history(path):
    """Load an existing history for plots; weights do not depend on it."""
    if not path.is_file():
        print(f"Notice: no stored training history at {path}; that curve will be omitted.", flush=True)
        return None
    try:
        return History(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Notice: could not read stored training history at {path}: {error}", flush=True)
        return None


def load_stored_component(model, name, output_dir):
    """Restore one stage without replacing the other two stages' weights."""
    stage_path = output_dir / f"{name}_best.pt"
    combined_path = output_dir / "assimnet.pt"
    if stage_path.is_file():
        checkpoint_path = stage_path
        state = torch.load(checkpoint_path, map_location=model.proxy.device, weights_only=True)
        if name == "assim":
            prefix = "encoder."
            state = {key.removeprefix(prefix): value for key, value in state.items()
                     if key.startswith(prefix)}
            if not state:
                raise ValueError(f"Stored AssimNet weights at {checkpoint_path} contain no encoder weights.")
            model.assim.encoder.load_state_dict(state)
        else:
            getattr(model, name).load_state_dict(state)
    elif combined_path.is_file():
        checkpoint_path = combined_path
        state = torch.load(checkpoint_path, map_location=model.proxy.device, weights_only=True)
        key = {"proxy": "proxy", "vae": "vae", "assim": "assim_encoder"}[name]
        if state.get("format_version") != 1 or key not in state:
            raise ValueError(f"Stored model at {checkpoint_path} is not a supported AssimNet checkpoint.")
        target = model.assim.encoder if name == "assim" else getattr(model, name)
        target.load_state_dict(state[key])
    else:
        message = (f"{name} training is disabled, but no stored model was found. "
                   f"Expected {stage_path} or {combined_path}. Enable training or provide a checkpoint.")
        print(f"ERROR: {message}", flush=True)
        raise FileNotFoundError(message)
    print(f"{name}: training disabled; loaded stored weights from {checkpoint_path}", flush=True)


def plot_results(output_dir, histories, t, y_true, y_pred, y_sim):
    if histories:
        fig, axes = plt.subplots(1, len(histories), figsize=(4.4 * len(histories), 3.6),
                                 constrained_layout=True, squeeze=False)
        for ax, (name, history) in zip(axes.flat, histories.items()):
            for key in ("loss", "val_loss"):
                values = history.history.get(key)
                if values:
                    ax.plot(np.arange(1, len(values) + 1), values, label=key)
            ax.set(title=name, xlabel="Epoch", ylabel="Loss", yscale="log")
            ax.legend()
            ax.grid(alpha=0.2)
        fig.savefig(output_dir / "training_curves.png", dpi=160)
        plt.close(fig)
    else:
        print("Notice: no training histories are available; training_curves.png was not updated.", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        index %= len(y_true)
        ax.plot(t, y_true[index, :, 0], label="Ground truth", color="black")
        ax.plot(t, y_pred[index, :, 0], label="AssimNet proxy", linestyle="--")
        ax.plot(t, y_sim[index, :, 0], label="Simulator with inferred parameters", linestyle=":")
        ax.set(title=f"Test sample {index + 1}", xlabel="t", ylabel="Response")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output_dir / "test_predictions.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small networks/data and 3/3/5 epochs; verifies execution, not accuracy.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--output-dir", type=Path, help="Defaults to outputs/caseI (outputs/caseI_quick for --quick).")
    parser.add_argument("--no-early-stopping", action="store_true", help="Train every configured epoch; best weights are still restored.")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads; default 4.")
    for name, label in (("proxy", "proxy"), ("vae", "VAE"), ("assim", "AssimNet")):
        parser.add_argument(
            f"--train-{name}", action=argparse.BooleanOptionalAction, default=None,
            help=f"Train or reuse the stored {label} model (overrides TRAIN_SWITCHES).",
        )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    data_config, model_config = deepcopy(DATA_CONFIG), deepcopy(MODEL_CONFIG)
    train_config, fit_config = deepcopy(TRAIN_CONFIG), deepcopy(FIT_CONFIG)
    train_switches = deepcopy(TRAIN_SWITCHES)
    for name in train_switches:
        override = getattr(args, f"train_{name}")
        if override is not None:
            train_switches[name] = override
    if args.quick:
        data_config.update(n_sim=160, n_field=40, sim_train_pool=128, field_train_pool=24)
        model_config["proxy_config"].update(gru_hidden_dims=[16, 16], dense_hidden_dims=[16])
        model_config["vae_config"].update(latent_dim=4, encoder_hidden_dims=[16], decoder_hidden_dims=[16], projection_dim=16)
        model_config["assim_config"].update(hidden_dims=[16])
        for name, epochs in (("proxy", 3), ("vae", 3), ("assim", 5)):
            train_config[name].update(epochs=epochs, batch_size=32)
        fit_config.update(patience=3, verbose_every=1)
    if args.no_early_stopping:
        fit_config["early_stopping"] = False
    model_config["proxy_config"]["sequence_length"] = data_config["sequence_length"]
    output_dir = args.output_dir or Path(__file__).resolve().parent / "outputs" / ("caseI_quick" if args.quick else "caseI")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = generate_data(data_config, SEED)
    model = build_assimnet(**model_config, device=args.device)
    print(f"Device: {model.proxy.device} | Output: {output_dir}", flush=True)
    if args.quick:
        print("Quick mode is an execution check; its prediction accuracy is not a research result.", flush=True)
    save_json(output_dir / "config.json", {
        "seed": SEED, "quick": args.quick, "device": str(model.proxy.device),
        "data": data_config, "model": model_config, "training": train_config, "fit": fit_config,
        "train_switches": train_switches,
        "preprocessing": "Raw CaseI values; no scaling.",
    })
    histories = {}
    sim_train, sim_val, sim_test = (data[f"sim_{split}_idx"] for split in ("train", "val", "test"))
    field_train, field_val, field_test = (data[f"field_{split}_idx"] for split in ("train", "val", "test"))
    for name in ("proxy", "vae", "assim"):
        history_path = output_dir / f"{name}_history.json"
        if not train_switches[name]:
            load_stored_component(model, name, output_dir)
            history = load_history(history_path)
            if history is not None:
                histories[name] = history
            continue
        options = train_config[name].copy()
        compile_options = {key: options.pop(key) for key in ("lr", "gamma", "kl_weight") if key in options}
        component = getattr(model, name)
        component.compile(optimizer="adam", loss="mse", **compile_options)
        if name == "proxy":
            x, y = data["sim_x"][sim_train], data["sim_y"][sim_train]
            validation = (data["sim_x"][sim_val], data["sim_y"][sim_val])
        elif name == "vae":
            x, y = data["sim_x"][sim_train], None
            validation = data["sim_x"][sim_val]
        else:
            x, y = {"x1": data["field_x"][field_train]}, data["field_y"][field_train]
            validation = ({"x1": data["field_x"][field_val]}, data["field_y"][field_val])
        histories[name] = component.fit(
            x, y, validation_data=validation, seed=SEED,
            checkpoint_path=output_dir / f"{name}_best.pt", **options, **fit_config,
        )
        save_json(history_path, asdict(histories[name]))
    result = model.assim.predict({"x1": data["field_x"][field_test]}, return_details=True)
    y_true, y_pred = data["field_y"][field_test], result["y_pred"]
    y_sim = simulator(data["t"], result["theta_pred"])
    proxy_test = model.proxy.predict(data["sim_x"][sim_test])
    vae_test = model.vae.predict(data["sim_x"][sim_test])
    metrics = {
        "proxy_test_mse": float(np.mean((proxy_test - data["sim_y"][sim_test]) ** 2)),
        "vae_test_recons_mse": float(np.mean((vae_test - data["sim_x"][sim_test]) ** 2)),
        "assim_test_mse": float(np.mean((y_pred - y_true) ** 2)),
        "assim_test_mae": float(np.mean(np.abs(y_pred - y_true))),
        "simulator_test_mse": float(np.mean((y_sim - y_true) ** 2)),
    }
    model.save(output_dir / "assimnet.pt")
    restored = load_assimnet(output_dir / "assimnet.pt", device=args.device)
    restored_predictions = restored.assim.predict({"x1": data["field_x"][field_test]})
    np.testing.assert_allclose(restored_predictions, y_pred, rtol=1e-6, atol=1e-7)
    metrics["checkpoint_reload_max_abs_diff"] = float(np.max(np.abs(restored_predictions - y_pred)))
    save_json(output_dir / "metrics.json", metrics)
    np.savez_compressed(output_dir / "data_and_predictions.npz", **data, **result,
                        y_true=y_true, simulator_y_pred=y_sim)
    plot_results(output_dir, histories, data["t"], y_true, y_pred, y_sim)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Finished. Model, histories, predictions, and figures saved to {output_dir}", flush=True)


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed_time = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
