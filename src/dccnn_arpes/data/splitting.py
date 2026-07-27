"""Deterministic group-level splits that prevent ARPES data leakage."""

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from .discovery import write_json, write_manifest_csv
from .schema import ManifestRecord

_SPLITS = ("train", "val", "test")
_MAX_EXACT_COMPONENTS = 10


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _key(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned or None


def _components(records: Sequence[ManifestRecord], pair_links: Iterable[tuple[str, str]] = ()) -> list[list[int]]:
    groups = _UnionFind(len(records))
    seen: dict[tuple[str, str], int] = {}
    ids: dict[str, int] = {}
    for index, record in enumerate(records):
        ids[record.record_id] = index
        for field in ("sample_id", "acquisition_group", "pair_id", "source_path"):
            value = _key(getattr(record, field))
            if value is None:
                continue
            token = (field, value)
            if token in seen:
                groups.union(index, seen[token])
            else:
                seen[token] = index
    for left_id, right_id in pair_links:
        if left_id in ids and right_id in ids:
            groups.union(ids[left_id], ids[right_id])
    result: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        result[groups.find(index)].append(index)
    return sorted(result.values(), key=lambda component: tuple(records[index].record_id for index in component))


def _stable_rank(seed: int, component: Sequence[int], records: Sequence[ManifestRecord]) -> str:
    material = f"{seed}\0" + "\0".join(sorted(records[index].record_id for index in component))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _absolute_error(counts: dict[str, int], targets: dict[str, float]) -> float:
    return sum(abs(counts[split] - targets[split]) for split in _SPLITS)


def _allocation_method(component_count: int) -> str:
    """Use an exact solver only within the documented bounded search space."""
    return "exact_bounded" if component_count <= _MAX_EXACT_COMPONENTS else "greedy_local_improvement"


def _exact_assignments(
    components: Sequence[Sequence[int]],
    records: Sequence[ManifestRecord],
    *,
    seed: int,
    counts: dict[str, int],
    targets: dict[str, float],
) -> dict[int, str]:
    """Exhaustively minimize final absolute error for at most ten components."""
    ordered = sorted(enumerate(components), key=lambda item: _stable_rank(seed, item[1], records))
    best: tuple[float, tuple[int, ...], dict[int, str]] | None = None

    def search(position: int, current: dict[str, int], choices: tuple[str, ...], assigned: dict[int, str]) -> None:
        nonlocal best
        if position == len(ordered):
            candidate = (
                _absolute_error(current, targets),
                tuple(_SPLITS.index(split) for split in choices),
                assigned.copy(),
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            return
        component_index, component = ordered[position]
        for split in _SPLITS:
            current[split] += len(component)
            assigned[component_index] = split
            search(position + 1, current, choices + (split,), assigned)
            current[split] -= len(component)
            del assigned[component_index]

    search(0, counts.copy(), (), {})
    assert best is not None
    return best[2]


def _greedy_local_assignments(
    components: Sequence[Sequence[int]],
    records: Sequence[ManifestRecord],
    *,
    seed: int,
    counts: dict[str, int],
    targets: dict[str, float],
) -> dict[int, str]:
    """Scalable deterministic fallback: greedy allocation plus improving single moves."""
    ordered = sorted(enumerate(components), key=lambda item: _stable_rank(seed, item[1], records))
    assigned: dict[int, str] = {}
    for component_index, component in ordered:
        size = len(component)
        split = min(
            _SPLITS,
            key=lambda name: (abs(counts[name] + size - targets[name]) - abs(counts[name] - targets[name]), name),
        )
        assigned[component_index] = split
        counts[split] += size

    while True:
        current_error = _absolute_error(counts, targets)
        candidates: list[tuple[float, str, str, int]] = []
        for component_index, component in ordered:
            old_split = assigned[component_index]
            for new_split in _SPLITS:
                if new_split == old_split:
                    continue
                size = len(component)
                counts[old_split] -= size
                counts[new_split] += size
                candidates.append((_absolute_error(counts, targets), new_split, old_split, component_index))
                counts[new_split] -= size
                counts[old_split] += size
        if not candidates:
            return assigned
        error, new_split, old_split, component_index = min(candidates)
        if error >= current_error:
            return assigned
        size = len(components[component_index])
        counts[old_split] -= size
        counts[new_split] += size
        assigned[component_index] = new_split


def _allocate_components(
    components: Sequence[Sequence[int]],
    records: Sequence[ManifestRecord],
    *,
    seed: int,
    counts: dict[str, int],
    targets: dict[str, float],
    method: str,
) -> dict[int, str]:
    if method == "exact_bounded":
        return _exact_assignments(components, records, seed=seed, counts=counts, targets=targets)
    return _greedy_local_assignments(components, records, seed=seed, counts=counts, targets=targets)


def assign_group_splits(
    records: Iterable[ManifestRecord],
    *,
    seed: int = 20260727,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    pair_links: Iterable[tuple[str, str]] = (),
) -> list[ManifestRecord]:
    """Assign whole connected components to deterministic train/val/test splits."""
    record_list = list(records)
    if len(ratios) != 3 or any(value < 0 for value in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must be three non-negative values summing to 1")
    components = _components(record_list, pair_links)
    method = _allocation_method(len(components))
    counts = {"train": 0, "val": 0, "test": 0}
    targets = dict(zip(_SPLITS, (len(record_list) * value for value in ratios), strict=True))
    assignments: dict[int, str] = {}

    sample_ids = {_key(record.sample_id) for record in record_list} - {None}
    remaining = list(components)
    if len(sample_ids) >= 3:
        candidates = [component for component in remaining if any(_key(record_list[index].sample_id) for index in component)]
        reserved = min(
            candidates,
            key=lambda component: (abs(len(component) - targets["test"]), _stable_rank(seed, component, record_list)),
        )
        for index in reserved:
            assignments[index] = "test"
        counts["test"] += len(reserved)
        remaining.remove(reserved)

    component_splits = _allocate_components(
        remaining, record_list, seed=seed, counts=counts, targets=targets, method=method
    )
    for component_index, component in enumerate(remaining):
        split = component_splits[component_index]
        for index in component:
            assignments[index] = split
    return [replace(record, split=assignments[index]) for index, record in enumerate(record_list)]


def leakage_audit(
    records: Sequence[ManifestRecord],
    pair_links: Iterable[tuple[str, str]] = (),
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, object]:
    """Describe component membership and fail if any component crosses a split."""
    components = _components(records, pair_links)
    component_rows = []
    for component in components:
        splits = sorted({records[index].split for index in component})
        component_rows.append(
            {"record_ids": sorted(records[index].record_id for index in component), "splits": splits}
        )
    leaking = [component for component in component_rows if len(component["splits"]) != 1]
    counts = {name: sum(record.split == name for record in records) for name in _SPLITS}
    targets = dict(zip(_SPLITS, (len(records) * value for value in ratios), strict=True))
    return {
        "component_count": len(component_rows),
        "record_count": len(records),
        "counts": counts,
        "target_counts": targets,
        "allocation_method": _allocation_method(len(components)),
        "absolute_error": _absolute_error(counts, targets),
        "components": component_rows,
        "leakage_free": not leaking,
        "leaking_components": leaking,
    }


def write_split_csvs(
    records: Sequence[ManifestRecord],
    output: Path,
    *,
    pair_links: Iterable[tuple[str, str]] = (),
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, object]:
    """Write split CSVs and an audit, rejecting any detected component leakage."""
    audit = leakage_audit(records, pair_links, ratios=ratios)
    if not audit["leakage_free"]:
        raise ValueError("a connected component appears in multiple splits")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        write_manifest_csv(
            sorted((record for record in records if record.split == split), key=lambda record: record.record_id),
            output / f"{split}.csv",
        )
    write_json(audit, output / "split_audit.json")
    return audit
