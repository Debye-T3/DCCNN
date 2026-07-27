"""Overlap-tiled model inference."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dccnn_arpes.models import denoise_forward


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _hann_weight(
    height: int,
    width: int,
    *,
    top_boundary: bool,
    bottom_boundary: bool,
    left_boundary: bool,
    right_boundary: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    minimum_weight = torch.finfo(dtype).eps
    vertical = torch.hann_window(height, periodic=False, device=device, dtype=dtype).clamp_min(
        minimum_weight
    )
    horizontal = torch.hann_window(width, periodic=False, device=device, dtype=dtype).clamp_min(
        minimum_weight
    )
    if top_boundary:
        vertical[0] = 1
    if bottom_boundary:
        vertical[-1] = 1
    if left_boundary:
        horizontal[0] = 1
    if right_boundary:
        horizontal[-1] = 1
    return vertical[:, None] * horizontal[None, :]


def tiled_predict(model: nn.Module, tensor: Tensor, tile_size: int, overlap: int) -> Tensor:
    """Predict a BCHW tensor in overlapping tiles blended by separable Hann weights."""
    if tensor.ndim != 4:
        raise ValueError("tensor must have BCHW dimensions")
    if not tensor.is_floating_point():
        raise TypeError("tensor must have a floating-point dtype")
    if type(tile_size) is not int or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if type(overlap) is not int or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")

    height, width = tensor.shape[-2:]
    stride = tile_size - overlap
    row_starts = _tile_starts(height, tile_size, stride)
    column_starts = _tile_starts(width, tile_size, stride)
    output = torch.zeros_like(tensor)
    accumulated_weight = torch.zeros_like(tensor)

    with torch.inference_mode():
        for row_start in row_starts:
            row_stop = min(row_start + tile_size, height)
            for column_start in column_starts:
                column_stop = min(column_start + tile_size, width)
                tile = tensor[..., row_start:row_stop, column_start:column_stop]
                prediction, _ = denoise_forward(model, tile)
                if prediction.shape != tile.shape:
                    raise ValueError("model prediction shape must match its input tile")

                weight = _hann_weight(
                    row_stop - row_start,
                    column_stop - column_start,
                    top_boundary=row_start == 0,
                    bottom_boundary=row_stop == height,
                    left_boundary=column_start == 0,
                    right_boundary=column_stop == width,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                weight = weight[None, None, :, :]
                output[..., row_start:row_stop, column_start:column_stop] += prediction * weight
                accumulated_weight[..., row_start:row_stop, column_start:column_stop] += weight

    if torch.any(accumulated_weight == 0):
        raise RuntimeError("tile blending left pixels without prediction coverage")
    return output / accumulated_weight
