"""Tests for deterministic, leakage-safe group-level splitting."""

import pytest

from dccnn_arpes.data.schema import ManifestRecord
from dccnn_arpes.data.splitting import assign_group_splits, leakage_audit, write_split_csvs


def _record(record_id: str, **changes) -> ManifestRecord:
    values = {"record_id": record_id, "source_path": f"D:/source/{record_id}.pxt", "sample_id": record_id}
    values.update(changes)
    return ManifestRecord(**values)


def test_connected_relationships_are_never_split_across_partitions():
    """Removing any union relationship must make this test fail."""
    records = [
        _record("one", sample_id="sample-a"),
        _record("two", sample_id="sample-a", acquisition_group="acq-a"),
        _record("three", sample_id="sample-b", acquisition_group="acq-a", pair_id="pair-a"),
        _record("four", sample_id="sample-c", pair_id="pair-a", source_path="D:/source/shared.pxt"),
        _record("five", sample_id="sample-d", source_path="D:/source/shared.pxt"),
        _record("six", sample_id="sample-e"),
        _record("seven", sample_id="sample-f"),
    ]

    assigned = assign_group_splits(records)
    by_id = {record.record_id: record.split for record in assigned}

    assert len({by_id[record_id] for record_id in ("one", "two", "three", "four", "five")}) == 1


def test_same_seed_writes_byte_identical_split_csvs(tmp_path):
    """Nondeterministic ordering or seeding must make this test fail."""
    records = [_record(f"record-{index}", sample_id=f"sample-{index}") for index in range(6)]

    first = assign_group_splits(records, seed=17)
    second = assign_group_splits(records, seed=17)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    write_split_csvs(first, first_output)
    write_split_csvs(second, second_output)

    for name in ("train.csv", "val.csv", "test.csv"):
        assert (first_output / name).read_bytes() == (second_output / name).read_bytes()


def test_three_or_more_samples_reserve_a_whole_sample_for_test():
    """Allowing every sample into train/val must make this test fail."""
    assigned = assign_group_splits([_record(f"record-{index}") for index in range(3)], seed=3)

    assert any(record.split == "test" for record in assigned)


def test_exact_allocator_minimizes_final_record_count_error():
    """Using a greedy-only allocation must make this counterexample fail."""
    records = [
        _record("one", sample_id=""),
        _record("two", sample_id=""),
        _record("three", sample_id="", acquisition_group="paired"),
        _record("four", sample_id="", acquisition_group="paired"),
    ]

    assigned = assign_group_splits(records, seed=4)
    audit = leakage_audit(assigned)

    assert {split: sum(record.split == split for record in assigned) for split in ("train", "val", "test")} == {
        "train": 3,
        "val": 1,
        "test": 0,
    }
    assert audit["allocation_method"] == "exact_bounded"
    assert audit["absolute_error"] == pytest.approx(1.2)


def test_exact_allocator_is_deterministic_for_equally_optimal_assignments():
    """Unstable solver tie-breaking must make this test fail."""
    records = [_record(f"record-{index}", sample_id="") for index in range(4)]

    first = assign_group_splits(records, seed=17)
    second = assign_group_splits(records, seed=17)

    assert [(record.record_id, record.split) for record in first] == [
        (record.record_id, record.split) for record in second
    ]
