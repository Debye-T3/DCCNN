"""Strict training configuration tests."""

from pathlib import Path

import pytest
import yaml

from dccnn_arpes.training.config import TrainConfig, load_train_config


def _valid_config(tmp_path: Path) -> dict[str, object]:
    return {
        "paths": {
            "manifest": str(tmp_path / "records.csv"),
            "pairs": str(tmp_path / "pairs.csv"),
            "splits": str(tmp_path / "splits"),
            "output": str(tmp_path / "runs"),
        },
        "seed": 20260727,
        "model": {"name": "residual_denoiser_2d", "channels": 4, "blocks": 1},
        "data": {
            "crop_size": [192, 192],
            "samples_per_epoch": 2,
            "sampling": {"A": 0.0, "B": 0.0, "C": 1.0},
            "identity_probability": 0.1,
            "noise": {
                "poisson_peak_counts": [50.0, 5000.0],
                "background_fraction": [0.0, 0.08],
                "stripe_probability": 0.3,
                "stripe_fraction": [0.0, 0.05],
            },
        },
        "training": {
            "batch_size": 1,
            "epochs": 2,
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-4,
            "workers": 0,
            "device": "cpu",
            "amp": False,
        },
        "loss": {"charbonnier": 0.8, "ms_ssim": 0.15, "gradient": 0.05},
    }


def _write_config(tmp_path: Path, values: dict[str, object]) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_load_train_config_rejects_unknown_nested_key(tmp_path):
    """Silently accepting misspelled options must not alter an experiment."""
    values = _valid_config(tmp_path)
    values["training"]["epochz"] = 2

    with pytest.raises(ValueError, match=r"training.*epochz"):
        load_train_config(_write_config(tmp_path, values))


def test_load_train_config_rejects_missing_required_key(tmp_path):
    """Defaulting an omitted required option must not hide an incomplete experiment."""
    values = _valid_config(tmp_path)
    del values["paths"]["manifest"]

    with pytest.raises(ValueError, match=r"paths.*manifest"):
        load_train_config(_write_config(tmp_path, values))


def test_smoke_overrides_are_in_memory_and_limited(tmp_path):
    """Smoke mode must not mutate YAML or change scientific sampling settings."""
    values = _valid_config(tmp_path)
    values["model"] = {"name": "residual_denoiser_2d", "channels": 64, "blocks": 8}
    values["data"]["samples_per_epoch"] = 10_000
    values["training"]["epochs"] = 100
    values["training"]["device"] = "cuda"
    values["training"]["amp"] = True
    path = _write_config(tmp_path, values)
    original = path.read_bytes()
    config = load_train_config(path)

    smoke = config.for_smoke_test(device="cpu")

    assert isinstance(config, TrainConfig)
    assert smoke.training.epochs == 2
    assert smoke.data.samples_per_epoch == 2
    assert smoke.model.channels == 4
    assert smoke.model.blocks == 1
    assert smoke.paths.output == config.paths.output / "smoke"
    assert smoke.data.sampling == config.data.sampling
    assert smoke.paths.manifest == config.paths.manifest
    assert smoke.paths.pairs == config.paths.pairs
    assert smoke.paths.splits == config.paths.splits
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "seed", True),
        ("model", "channels", 4.9),
        ("data", "samples_per_epoch", "2"),
        ("training", "learning_rate", float("nan")),
        ("training", "weight_decay", float("inf")),
    ],
)
def test_load_train_config_rejects_wrong_types_and_nonfinite_numbers(tmp_path, section, key, value):
    """Coercion or NaN acceptance must not start a different experiment than YAML declares."""
    values = _valid_config(tmp_path)
    target = values if section is None else values[section]
    target[key] = value

    with pytest.raises((TypeError, ValueError), match=key):
        load_train_config(_write_config(tmp_path, values))
