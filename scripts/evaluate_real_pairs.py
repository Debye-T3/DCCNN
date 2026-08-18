"""Run and quantify a checkpoint on reviewed real A-type ARPES pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dccnn_arpes.cli.data import _read_manifest_csv
from dccnn_arpes.data.pairing import read_pairs_csv
from dccnn_arpes.evaluation.real_pairs import (
    build_pair_row,
    compare_pair,
    count_rate_normalize,
    effective_exposure,
    orient_pair,
)
from dccnn_arpes.inference import denoise_file
from dccnn_arpes.io import load_cut

_OUTPUT_ROOT = Path(r"D:\Projects\dccnn\outputs").resolve()
_REVIEWED = {"reviewed", "approved", "manually_approved"}


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mean(values: list[object]) -> float | None:
    numeric = [float(value) for value in values if value not in (None, "") and np.isfinite(float(value))]
    return float(np.mean(numeric)) if numeric else None


def _validate_output(path: Path) -> Path:
    destination = path.expanduser().resolve(strict=False)
    if not destination.is_relative_to(_OUTPUT_ROOT):
        raise ValueError(f"output must be under {_OUTPUT_ROOT}")
    return destination


def evaluate_real_pairs(
    *, manifest_path: Path, pairs_path: Path, checkpoint_path: Path, output_dir: Path
) -> dict[str, object]:
    """Run inference and write pair-level count-rate-normalized metrics."""
    output_dir = _validate_output(output_dir)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    records = _read_manifest_csv(Path(manifest_path).expanduser().resolve(strict=True))
    record_by_id = {record.record_id: record for record in records}
    split_by_id = {record.record_id: record.split for record in records}
    pairs = [pair for pair in read_pairs_csv(Path(pairs_path).expanduser().resolve(strict=True)) if pair.pair_type == "A"]
    if not pairs:
        raise ValueError("pairs CSV contains no A-type pairs")

    denoised_dir = output_dir / "denoised"
    denoised_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for pair in pairs:
        if pair.review_status.casefold() not in _REVIEWED:
            raise ValueError(f"pair {pair.pair_id!r} is not reviewed")
        try:
            left = record_by_id[pair.left_record_id]
            right = record_by_id[pair.right_record_id]
        except KeyError as error:
            raise ValueError(f"pair {pair.pair_id!r} has an unknown endpoint") from error
        input_record, reference_record = orient_pair(left, right)
        input_scale = effective_exposure(input_record)
        reference_scale = effective_exposure(reference_record)
        if not input_record.converted_path or not reference_record.converted_path:
            raise ValueError(f"pair {pair.pair_id!r} has a missing converted path")
        input_path = Path(input_record.converted_path).resolve(strict=True)
        reference_path = Path(reference_record.converted_path).resolve(strict=True)
        denoised_path = denoise_file(input_path, checkpoint_path, denoised_dir)

        input_data = load_cut(input_path)
        output_data = load_cut(denoised_path)
        reference_data = load_cut(reference_path)
        normalized_input = count_rate_normalize(input_data, input_scale)
        normalized_output = count_rate_normalize(output_data, input_scale)
        normalized_reference = count_rate_normalize(reference_data, reference_scale)
        metrics = compare_pair(normalized_input, normalized_output, normalized_reference)
        row = build_pair_row(
            pair_id=pair.pair_id,
            split=split_by_id.get(input_record.record_id, ""),
            input_record=input_record,
            reference_record=reference_record,
            metrics=metrics,
        )
        row.update(
            {
                "input_path": str(input_path),
                "reference_path": str(reference_path),
                "denoised_path": str(denoised_path.resolve()),
                "raw_fit_status": metrics["raw"].get("fit_status"),
                "denoised_fit_status": metrics["denoised"].get("fit_status"),
                "raw_fit_failure_reason": metrics["raw"].get("fit_failure_reason", ""),
                "denoised_fit_failure_reason": metrics["denoised"].get("fit_failure_reason", ""),
            }
        )
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    metrics_path = output_dir / "pair_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "pair_count": len(rows),
        "split_counts": dict(Counter(str(row["split"]) for row in rows)),
        "metrics_path": str(metrics_path.resolve()),
        "denoised_directory": str(denoised_dir.resolve()),
        "metrics": {
            name: {
                "raw_mean": _mean([row.get(f"raw_{name}") for row in rows]),
                "denoised_mean": _mean([row.get(f"denoised_{name}") for row in rows]),
                "improvement_mean": _mean([row.get(f"{name}_improvement") for row in rows]),
            }
            for name in ("mae", "nrmse")
        },
        "pairs": [
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "input_file_id": row["input_file_id"],
                "reference_file_id": row["reference_file_id"],
                "raw_nrmse": row["raw_nrmse"],
                "denoised_nrmse": row["denoised_nrmse"],
                "nrmse_improvement": row["nrmse_improvement"],
            }
            for row in rows
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**summary, "summary_path": str(summary_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on reviewed real A-type ARPES pairs.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_real_pairs(
        manifest_path=args.manifest,
        pairs_path=args.pairs,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
    )
    print(f"metrics_path={result['metrics_path']}", flush=True)
    print(f"summary_path={result['summary_path']}", flush=True)
    print(f"pair_count={result['pair_count']}", flush=True)


if __name__ == "__main__":
    main()
