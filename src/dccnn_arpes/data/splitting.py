"""Deterministic group-level splits that prevent ARPES data leakage."""

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from .discovery import write_json, write_manifest_csv
from .schema import ManifestRecord


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
    counts = {"train": 0, "val": 0, "test": 0}
    targets = dict(zip(("train", "val", "test"), (len(record_list) * value for value in ratios), strict=True))
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

    for component in sorted(remaining, key=lambda item: _stable_rank(seed, item, record_list)):
        size = len(component)
        split = min(
            ("train", "val", "test"),
            key=lambda name: (abs(counts[name] + size - targets[name]) - abs(counts[name] - targets[name]), name),
        )
        for index in component:
            assignments[index] = split
        counts[split] += size
    return [replace(record, split=assignments[index]) for index, record in enumerate(record_list)]


def leakage_audit(records: Sequence[ManifestRecord], pair_links: Iterable[tuple[str, str]] = ()) -> dict[str, object]:
    """Describe component membership and fail if any component crosses a split."""
    components = _components(records, pair_links)
    component_rows = []
    for component in components:
        splits = sorted({records[index].split for index in component})
        component_rows.append(
            {"record_ids": sorted(records[index].record_id for index in component), "splits": splits}
        )
    leaking = [component for component in component_rows if len(component["splits"]) != 1]
    return {
        "component_count": len(component_rows),
        "record_count": len(records),
        "counts": {name: sum(record.split == name for record in records) for name in ("train", "val", "test")},
        "components": component_rows,
        "leakage_free": not leaking,
        "leaking_components": leaking,
    }


def write_split_csvs(
    records: Sequence[ManifestRecord], output: Path, *, pair_links: Iterable[tuple[str, str]] = ()
) -> dict[str, object]:
    """Write split CSVs and an audit, rejecting any detected component leakage."""
    audit = leakage_audit(records, pair_links)
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
