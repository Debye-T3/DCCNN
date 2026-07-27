"""Overlap-tiled inference behavior tests."""

import pytest
import torch
from torch import nn

from dccnn_arpes.inference import tiled_predict


@pytest.mark.parametrize(
    ("shape", "tile_size", "overlap"),
    [
        ((1, 1, 9, 11), 16, 4),
        ((1, 1, 17, 23), 8, 3),
        ((2, 1, 31, 19), 12, 5),
        ((1, 1, 5, 7), 2, 1),
        ((1, 1, 17, 23), 8, 0),
    ],
)
def test_identity_tiled_prediction_matches_full_image_for_small_and_odd_shapes(
    shape, tile_size, overlap
):
    """Incorrect tile coverage or blending must not alter an identity prediction."""
    tensor = torch.linspace(0.0, 1.0, torch.tensor(shape).prod().item()).reshape(shape)
    model = nn.Identity()

    expected = model(tensor)
    actual = tiled_predict(model, tensor, tile_size=tile_size, overlap=overlap)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)


def test_tiled_prediction_rejects_overlap_at_least_tile_size():
    """A zero or negative stride must be rejected rather than hanging or skipping pixels."""
    with pytest.raises(ValueError, match="overlap must be smaller than tile_size"):
        tiled_predict(nn.Identity(), torch.ones(1, 1, 8, 8), tile_size=8, overlap=8)
