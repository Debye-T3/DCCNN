import argparse
import gc
import glob
import random
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from modules.datasets.dataset import build_dataset_from_config
from modules.models.ccnn import CCNN
from train.trainer import train_model


def load_config(path: str):
    """Load YAML config with UTF-8 to avoid locale/GBK decoding issues."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_all_configs(config_dir: str, pattern: str = "baseline_v3.yaml"):
    # Default to v3; override pattern/CLI for batch experiments.
    configs = sorted(glob.glob(f"{config_dir}/{pattern}"))
    if not configs:
        raise FileNotFoundError(f"No config files matched pattern '{pattern}' in {config_dir}")

    for cfg_path in configs:
        cfg = load_config(cfg_path)

        # Convert numeric strings to proper types (handles YAML scientific notation as str)
        if "training" in cfg:
            training = cfg["training"]
            training["learning_rate"] = float(training["learning_rate"])
            training["alpha"] = float(training["alpha"])
            training["epochs"] = int(training["epochs"])
            training["batch_size"] = int(training["batch_size"])
            training["early_stopping_patience"] = int(training["early_stopping_patience"])
            if "val_split" in training:
                training["val_split"] = float(training["val_split"])

        if "model" in cfg:
            model = cfg["model"]
            model["kernel_size"] = int(model["kernel_size"])
            model["num_layers"] = int(model["num_layers"])

        model_cfg = cfg["model"]
        train_cfg = cfg["training"]
        path_cfg = cfg["paths"]
        output_dir = Path(path_cfg["output_dir"])
        csv_dir = output_dir / path_cfg.get("csv_subdir")
        model_dir = output_dir / "models"

        output_dir.mkdir(exist_ok=True)
        csv_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask = (
            f"{timestamp}_layers{model_cfg['num_layers']}_batch{train_cfg['batch_size']}_"
            f"kernel{model_cfg['kernel_size']}_alpha{train_cfg['alpha']}"
        )
        csv_path = csv_dir / f"{mask}.csv"
        model_path = model_dir / f"{mask}.pt"

        device = torch.device("cuda" if train_cfg["use_gpu"] and torch.cuda.is_available() else "cpu")
        dataset = build_dataset_from_config(path_cfg, cfg.get("data"))

        total_len = len(dataset)
        val_ratio = train_cfg.get("val_split", 0.2)
        val_size = int(max(1, total_len * val_ratio)) if total_len > 1 else 0
        if val_size >= total_len:
            val_size = max(1, total_len // 5)
        indices = list(range(total_len))
        random.Random(train_cfg.get("split_seed", 42)).shuffle(indices)
        val_indices = indices[:val_size] if val_size > 0 else indices[:1]
        train_indices = indices[val_size:] if val_size > 0 else indices
        if not train_indices:
            train_indices = indices

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            drop_last=len(train_dataset) > train_cfg["batch_size"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(train_cfg["batch_size"], len(val_dataset)),
            shuffle=False,
            drop_last=False,
        )

        model = CCNN(model_cfg["kernel_size"], model_cfg["num_layers"])

        # Train one config; trainer handles warmup/early-stop/cosine
        model, metrics = train_model(
            model,
            train_loader,
            val_loader,
            train_cfg["epochs"],
            train_cfg["learning_rate"],
            train_cfg["alpha"],
            device,
            train_cfg["early_stopping_patience"],
        )

        metrics.to_csv(csv_path, index=False)
        torch.save(model.state_dict(), model_path)

        del model, train_loader, val_loader, dataset
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DCCNN training over one or multiple YAML configs.")
    parser.add_argument(
        "--config",
        default="config/baseline_v3.yaml",
        help="Single config file to run (default: baseline_v3).",
    )
    parser.add_argument(
        "--config-glob",
        default=None,
        help="Optional glob pattern inside config/ (e.g., 'baseline_*.yaml'); overrides --config.",
    )
    args = parser.parse_args()

    if args.config_glob:
        run_all_configs("config", pattern=args.config_glob)
    else:
        run_all_configs(Path(args.config).parent, pattern=Path(args.config).name)
