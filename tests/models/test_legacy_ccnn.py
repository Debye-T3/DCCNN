"""Compatibility tests for the original CCNN checkpoint format."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from dccnn_arpes.models.legacy_ccnn import LegacyCCNN, load_legacy_checkpoint


def _original_ccnn_type():
    path = Path(__file__).parents[2] / "modules" / "models" / "ccnn.py"
    spec = importlib.util.spec_from_file_location("original_ccnn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CCNN


CCNN = _original_ccnn_type()


def test_legacy_ccnn_uses_original_state_keys_and_matches_original_output():
    """Changing a legacy layer registration must break old checkpoint equivalence."""
    torch.manual_seed(17)
    old_model = CCNN(kernel_size=3, num_layers=7)
    model = LegacyCCNN(kernel_size=3, num_layers=7)

    assert tuple(model.state_dict()) == tuple(old_model.state_dict())
    model.load_state_dict(old_model.state_dict())
    inputs = torch.randn(2, 1, 17, 19)

    output = model(inputs)

    assert output.shape == inputs.shape
    torch.testing.assert_close(output, old_model(inputs))


@pytest.mark.parametrize("container_key", [None, "state_dict", "model_state_dict"])
def test_load_legacy_checkpoint_accepts_supported_checkpoint_containers(
    tmp_path, container_key: str | None
):
    """Dropping a supported legacy checkpoint envelope must prevent restoration."""
    torch.manual_seed(23)
    source = CCNN(kernel_size=3, num_layers=7)
    state_dict = {f"module.{key}": value for key, value in source.state_dict().items()}
    checkpoint = state_dict if container_key is None else {container_key: state_dict}
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    restored = LegacyCCNN(kernel_size=3, num_layers=7)

    load_legacy_checkpoint(restored, path)

    inputs = torch.randn(1, 1, 13, 15)
    torch.testing.assert_close(restored(inputs), source(inputs))


def test_load_legacy_checkpoint_names_missing_and_unexpected_keys(tmp_path):
    """Silently accepting a mismatched checkpoint must leave no diagnostic."""
    checkpoint = CCNN(kernel_size=3, num_layers=7).state_dict()
    checkpoint.pop("layers.0.weight")
    checkpoint["obsolete.weight"] = torch.zeros(1)
    path = tmp_path / "mismatched.pt"
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError, match=r"Missing keys: .*layers\.0\.weight.*Unexpected keys: .*obsolete\.weight"):
        load_legacy_checkpoint(LegacyCCNN(kernel_size=3, num_layers=7), path)
