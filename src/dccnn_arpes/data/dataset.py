"""Reproducible mixed A/B/C training samples for two-dimensional ARPES cuts."""

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dccnn_arpes.io.xarray_h5 import load_cut

from .noise import NoiseParameters, synthesize_noisy
from .pairing import PairRecord
from .schema import ManifestRecord
from .transforms import IntensityTransform

_PAIR_TYPES = ("A", "B", "C")
_MANUAL_APPROVALS = {"approved", "manually_approved"}


def _stable_rng(*parts: object) -> np.random.Generator:
    material = "\0".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(seed)


def _acquisition_scale(record: ManifestRecord) -> float | None:
    if record.acquisition_time_s is not None:
        if record.acquisition_time_s <= 0:
            raise ValueError(f"record {record.record_id} has invalid acquisition_time_s")
        return float(record.acquisition_time_s)
    if record.sweep_count is not None:
        if record.sweep_count <= 0:
            raise ValueError(f"record {record.record_id} has invalid sweep_count")
        return float(record.sweep_count)
    return None


def _shared_exposures(
    left: ManifestRecord, right: ManifestRecord
) -> tuple[float, float] | None:
    if left.acquisition_time_s is not None and right.acquisition_time_s is not None:
        times = float(left.acquisition_time_s), float(right.acquisition_time_s)
        if times[0] != times[1]:
            return times
    if left.sweep_count is not None and right.sweep_count is not None:
        sweeps = float(left.sweep_count), float(right.sweep_count)
        if sweeps[0] != sweeps[1]:
            return sweeps
    return None


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


class ArpesCutDataset(Dataset):
    """Sample reviewed real pairs and dynamically corrupted measured cuts."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        pairs: Sequence[PairRecord] = (),
        *,
        crop_size: tuple[int, int] = (256, 256),
        samples_per_epoch: int = 10_000,
        sampling: Mapping[str, float] | None = None,
        identity_probability: float = 0.10,
        base_seed: int = 20260727,
        noise_parameters: NoiseParameters | None = None,
    ) -> None:
        super().__init__()
        if len(crop_size) != 2 or any(int(size) <= 0 for size in crop_size):
            raise ValueError("crop_size must contain two positive dimensions")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if not 0 <= identity_probability <= 1:
            raise ValueError("identity_probability must be between zero and one")

        self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        self.samples_per_epoch = int(samples_per_epoch)
        self.identity_probability = float(identity_probability)
        self.base_seed = int(base_seed)
        self.noise_parameters = noise_parameters or NoiseParameters()
        self.epoch = 0
        self._arrays: dict[str, np.ndarray] = {}
        self._records = {
            record.record_id: record
            for record in records
            if record.converted_path and not record.exclusion_reason
        }
        if len(self._records) != len(
            [record for record in records if record.converted_path and not record.exclusion_reason]
        ):
            raise ValueError("record IDs must be unique")

        self._a_pairs, invalid_a = self._build_a_pairs(pairs)
        self._b_groups = self._build_b_groups(pairs)
        paired_ids = {
            record_id
            for pair in pairs
            for record_id in (pair.left_record_id, pair.right_record_id)
            if record_id in self._records
        }
        self._c_records = sorted(set(self._records).difference(paired_ids)) or sorted(self._records)

        requested = sampling or {"A": 0.50, "B": 0.30, "C": 0.20}
        if set(requested) != set(_PAIR_TYPES):
            raise ValueError("sampling must define exactly A, B, and C")
        weights = np.asarray([float(requested[name]) for name in _PAIR_TYPES], dtype=np.float64)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("sampling weights must be finite, non-negative, and have a positive sum")
        pools = {"A": self._a_pairs, "B": self._b_groups, "C": self._c_records}
        for pair_type, weight in zip(_PAIR_TYPES, weights, strict=True):
            if weight > 0 and not pools[pair_type]:
                if pair_type == "A" and invalid_a:
                    raise ValueError("A pair has no usable acquisition scale and is not manually approved")
                raise ValueError(f"no eligible {pair_type} samples")
        self._probabilities = weights / weights.sum()
        self._transform = IntensityTransform()

    def _build_a_pairs(
        self, pairs: Sequence[PairRecord]
    ) -> tuple[list[tuple[str, str, str]], bool]:
        accepted: list[tuple[str, str, str]] = []
        invalid = False
        for pair in pairs:
            if pair.pair_type != "A":
                continue
            if pair.left_record_id not in self._records or pair.right_record_id not in self._records:
                continue
            left = self._records[pair.left_record_id]
            right = self._records[pair.right_record_id]
            _acquisition_scale(left)
            _acquisition_scale(right)
            exposures = _shared_exposures(left, right)
            if exposures is None:
                if pair.review_status not in _MANUAL_APPROVALS:
                    invalid = True
                    continue
                input_id, target_id = pair.left_record_id, pair.right_record_id
            elif exposures[0] <= exposures[1]:
                input_id, target_id = pair.left_record_id, pair.right_record_id
            else:
                input_id, target_id = pair.right_record_id, pair.left_record_id
            accepted.append((input_id, target_id, pair.pair_id))
        return sorted(accepted), invalid

    def _build_b_groups(self, pairs: Sequence[PairRecord]) -> list[tuple[str, ...]]:
        links = [
            pair
            for pair in pairs
            if pair.pair_type == "B"
            and pair.left_record_id in self._records
            and pair.right_record_id in self._records
        ]
        ids = sorted(
            {record_id for pair in links for record_id in (pair.left_record_id, pair.right_record_id)}
        )
        groups = _UnionFind(ids)
        for pair in links:
            groups.union(pair.left_record_id, pair.right_record_id)
        connected: dict[str, list[str]] = defaultdict(list)
        for record_id in ids:
            connected[groups.find(record_id)].append(record_id)
        return sorted(tuple(sorted(group)) for group in connected.values() if len(group) >= 2)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible new crop/noise stream for an epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _load(self, record_id: str) -> np.ndarray:
        if record_id not in self._arrays:
            record = self._records[record_id]
            self._arrays[record_id] = np.asarray(
                load_cut(Path(record.converted_path)).values, dtype=np.float64
            )
        return self._arrays[record_id]

    def _rate(self, record_id: str) -> np.ndarray:
        values = self._load(record_id)
        scale = _acquisition_scale(self._records[record_id])
        return values if scale is None else values / scale

    def _crop_pair(
        self, input_values: np.ndarray, target_values: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        if input_values.shape != target_values.shape:
            raise ValueError("paired cuts must have identical shapes")
        height, width = input_values.shape
        crop_height, crop_width = self.crop_size
        if crop_height > height or crop_width > width:
            raise ValueError("crop_size exceeds an eligible cut")
        row = int(rng.integers(0, height - crop_height + 1))
        column = int(rng.integers(0, width - crop_width + 1))
        region = np.s_[row : row + crop_height, column : column + crop_width]
        return input_values[region], target_values[region], (row, column)

    def _select_source(
        self, pair_type: str, rng: np.random.Generator
    ) -> tuple[str, np.ndarray, np.ndarray, str]:
        if pair_type == "A":
            input_id, target_id, pair_id = self._a_pairs[int(rng.integers(len(self._a_pairs)))]
            return input_id, self._rate(input_id), self._rate(target_id), pair_id
        if pair_type == "B":
            group = self._b_groups[int(rng.integers(len(self._b_groups)))]
            input_id = group[int(rng.integers(len(group)))]
            target_ids = [record_id for record_id in group if record_id != input_id]
            target = np.mean([self._rate(record_id) for record_id in target_ids], axis=0)
            return input_id, self._rate(input_id), target, "|".join(group)
        record_id = self._c_records[int(rng.integers(len(self._c_records)))]
        target = self._rate(record_id)
        return record_id, target, target, ""

    def __getitem__(self, dataset_index: int):
        if not 0 <= dataset_index < len(self):
            raise IndexError(dataset_index)
        selection_rng = _stable_rng(self.base_seed, self.epoch, dataset_index, "selection")
        pair_type = str(selection_rng.choice(_PAIR_TYPES, p=self._probabilities))
        record_id, input_values, target_values, pair_id = self._select_source(
            pair_type, selection_rng
        )
        rng = _stable_rng(self.base_seed, self.epoch, dataset_index, record_id)
        input_crop, target_crop, origin = self._crop_pair(input_values, target_values, rng)

        if pair_type == "C":
            input_crop = synthesize_noisy(target_crop, self.noise_parameters, rng)
        identity = bool(rng.random() < self.identity_probability)
        if identity:
            input_crop = target_crop.copy()

        stats = self._transform.fit(input_crop)
        input_tensor = torch.from_numpy(
            self._transform.forward(input_crop, stats).astype(np.float32, copy=False)
        ).unsqueeze(0)
        target_tensor = torch.from_numpy(
            self._transform.forward(target_crop, stats).astype(np.float32, copy=False)
        ).unsqueeze(0)
        metadata = {
            "record_id": record_id,
            "pair_id": pair_id,
            "pair_type": pair_type,
            "crop_eV": origin[0],
            "crop_alpha": origin[1],
            "crop_origin": origin,
            "transform_stats": {"lower": stats.lower, "scale": stats.scale},
            "identity_constraint": identity,
        }
        return input_tensor, target_tensor, metadata
