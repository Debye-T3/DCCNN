"""Checkpoint round-trip and schema tests."""

from copy import deepcopy

import pytest
import torch

from dccnn_arpes.models import ResidualDenoiser2D
from dccnn_arpes.training.checkpoints import load_checkpoint, save_checkpoint
from dccnn_arpes.training.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    NoiseConfig,
    PathConfig,
    SamplingConfig,
    TrainConfig,
    TrainingConfig,
)


def _config(tmp_path):
    return TrainConfig(
        paths=PathConfig(
            manifest=tmp_path / "records.csv",
            pairs=tmp_path / "pairs.csv",
            splits=tmp_path / "splits",
            output=tmp_path / "runs",
        ),
        seed=20260727,
        model=ModelConfig(name="residual_denoiser_2d", channels=4, blocks=1),
        data=DataConfig(
            crop_size=(192, 192),
            samples_per_epoch=2,
            sampling=SamplingConfig(A=0.0, B=0.0, C=1.0),
            identity_probability=0.0,
            noise=NoiseConfig(
                poisson_peak_counts=(50.0, 5000.0),
                background_fraction=(0.0, 0.08),
                stripe_probability=0.3,
                stripe_fraction=(0.0, 0.05),
            ),
        ),
        training=TrainingConfig(
            batch_size=1,
            epochs=2,
            learning_rate=1.0e-4,
            weight_decay=1.0e-4,
            workers=0,
            device="cpu",
            amp=False,
        ),
        loss=LossConfig(charbonnier=0.8, ms_ssim=0.15, gradient=0.05),
    )


def test_checkpoint_restores_epoch_optimizer_and_identical_output(tmp_path):
    """Omitting model or optimizer state must prevent an exact training continuation."""
    torch.manual_seed(11)
    model = ResidualDenoiser2D(channels=4, blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    inputs = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8)
    expected = model(inputs)[0].detach().clone()
    path = tmp_path / "last.pt"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=2,
        best_metric=0.125,
        config=_config(tmp_path),
        hashes={"manifest_sha256": "a" * 64, "split_sha256": "b" * 64},
        versions={
            "git_commit": "deadbeef",
            "python": "3.12.0",
            "pytorch": torch.__version__,
            "cuda": None,
            "device": "cpu",
        },
    )
    for parameter in model.parameters():
        parameter.data.add_(1)

    restored = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        map_location="cpu",
    )

    assert restored.epoch == 2
    assert restored.best_metric == 0.125
    assert restored.hashes["manifest_sha256"] == "a" * 64
    assert restored.versions["git_commit"] == "deadbeef"
    assert restored.smoke_test is False
    assert restored.scientific_use is True
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == 2
    torch.testing.assert_close(model(inputs)[0], expected, rtol=0, atol=0)


def _saved_checkpoint(tmp_path):
    model = ResidualDenoiser2D(channels=4, blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=1,
        best_metric=0.25,
        config=_config(tmp_path),
        hashes={"manifest_sha256": "a" * 64},
        versions={"device": "cpu"},
    )
    return path, model


def _rewrite_payload(path, **updates):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.update(updates)
    torch.save(payload, path)


@pytest.mark.parametrize("field", ["smoke_test", "scientific_use"])
def test_checkpoint_v2_rejects_string_intent_flags(tmp_path, field):
    path, model = _saved_checkpoint(tmp_path)
    _rewrite_payload(path, **{field: "false"})

    with pytest.raises(TypeError, match=f"{field} must be a boolean"):
        load_checkpoint(path, model=model)


@pytest.mark.parametrize(
    ("smoke_test", "scientific_use"),
    [(False, False), (True, True)],
)
def test_checkpoint_v2_rejects_non_complementary_intent_flags(tmp_path, smoke_test, scientific_use):
    path, model = _saved_checkpoint(tmp_path)
    _rewrite_payload(path, smoke_test=smoke_test, scientific_use=scientific_use)

    with pytest.raises(ValueError, match="intent flags must be complementary"):
        load_checkpoint(path, model=model)


def test_checkpoint_v2_rejects_intent_mismatch_with_embedded_config(tmp_path):
    path, model = _saved_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mismatched_config = deepcopy(payload["config"])
    mismatched_config["smoke_test"] = True
    mismatched_config["scientific_use"] = False
    _rewrite_payload(path, config=mismatched_config)

    with pytest.raises(ValueError, match="intent flags do not match embedded config"):
        load_checkpoint(path, model=model)


def test_checkpoint_v1_is_rejected_with_distinct_legacy_diagnostic(tmp_path):
    path, model = _saved_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema_version"] = 1
    payload.pop("smoke_test")
    payload.pop("scientific_use")
    torch.save(payload, path)

    with pytest.raises(ValueError, match="legacy checkpoint schema 1"):
        load_checkpoint(path, model=model)
