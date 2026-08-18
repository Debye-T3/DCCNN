"""Batch-denoise the locked 19-sample C test set and build qualitative evidence.

The C test set has no paired high-exposure reference.  This script therefore
records only input-to-output diagnostics (noise-score change, profile
correlation, peak shift, and relative change) and creates consistent-scale
contact sheets for manual review.  It deliberately does not label these
diagnostics as scientific acceptance metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from dccnn_arpes.inference import denoise_file
from dccnn_arpes.io import load_cut

OUTPUT_ROOT = Path(r"D:\Projects\dccnn\outputs").resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_or_none(value: str) -> float | None:
    text = (value or "").strip()
    return float(text) if text else None


def _load_rows(manifest_path: Path, acceptance_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row.get("record_id", "").strip(): row for row in rows}
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    ids = acceptance["scientific_gate"]["ineligible_record_ids"]
    if len(ids) != 19:
        raise ValueError(f"expected 19 locked C test IDs, found {len(ids)}")
    selected: list[dict[str, str]] = []
    for record_id in ids:
        if record_id not in by_id:
            raise KeyError(f"record_id {record_id!r} is absent from manifest")
        row = by_id[record_id]
        path = Path(row.get("converted_path", ""))
        if not path.is_file():
            raise FileNotFoundError(f"missing converted C-test input: {path}")
        if row.get("split", "").strip().casefold() != "test":
            raise ValueError(f"locked C-test row {record_id} is not in split=test")
        selected.append(row)
    return selected


def _scale(row: dict[str, str]) -> float:
    acquisition = _float_or_none(row.get("acquisition_time_s", ""))
    if acquisition is not None and acquisition > 0:
        return acquisition
    sweeps = _float_or_none(row.get("sweep_count", ""))
    if sweeps is not None and sweeps > 0:
        return sweeps
    return 1.0


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _noise_score(image: np.ndarray) -> float:
    values = np.asarray(image, dtype=np.float64)
    centered = values - float(np.mean(values))
    base_std = float(np.std(centered))
    if base_std <= 1e-12:
        return 0.0
    dx = float(np.mean(np.abs(np.diff(values, axis=1))))
    dy = float(np.mean(np.abs(np.diff(values, axis=0))))
    return 0.5 * (dx + dy) / base_std


def _peak(values: np.ndarray, coordinate: np.ndarray) -> tuple[int, float]:
    profile = np.asarray(values, dtype=np.float64)
    index = int(np.nanargmax(profile))
    return index, float(coordinate[index])


def _metrics(raw: np.ndarray, output: np.ndarray, energy: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    raw_edc = np.mean(raw, axis=1)
    output_edc = np.mean(output, axis=1)
    raw_mdc = np.mean(raw, axis=0)
    output_mdc = np.mean(output, axis=0)
    raw_e_idx, raw_e = _peak(raw_edc, energy)
    out_e_idx, out_e = _peak(output_edc, energy)
    raw_a_idx, raw_a = _peak(raw_mdc, alpha)
    out_a_idx, out_a = _peak(output_mdc, alpha)
    raw_noise = _noise_score(raw)
    output_noise = _noise_score(output)
    noise_reduction = 1.0 - output_noise / max(raw_noise, 1e-12)
    relative_change = float(np.mean(np.abs(output - raw)) / max(np.mean(np.abs(raw)), 1e-12))
    energy_step = float(np.median(np.abs(np.diff(energy)))) if energy.size > 1 else 0.0
    alpha_step = float(np.median(np.abs(np.diff(alpha)))) if alpha.size > 1 else 0.0
    flags: list[str] = []
    if noise_reduction >= 0.15:
        flags.append("明显降噪")
    elif noise_reduction >= 0.03:
        flags.append("轻微降噪")
    elif noise_reduction < -0.05:
        flags.append("噪声增加")
    else:
        flags.append("变化很小")
    if abs(out_e_idx - raw_e_idx) > 3 or (energy_step and abs(out_e - raw_e) > 3 * energy_step):
        flags.append("EDC峰位需复核")
    if abs(out_a_idx - raw_a_idx) > 3 or (alpha_step and abs(out_a - raw_a) > 3 * alpha_step):
        flags.append("MDC峰位需复核")
    if _corr(raw_edc, output_edc) < 0.97 or _corr(raw_mdc, output_mdc) < 0.97:
        flags.append("剖面结构需复核")
    if relative_change > 0.25:
        flags.append("整体变化较大")
    return {
        "raw_noise_score": raw_noise,
        "output_noise_score": output_noise,
        "noise_reduction": noise_reduction,
        "relative_change": relative_change,
        "edc_correlation": _corr(raw_edc, output_edc),
        "mdc_correlation": _corr(raw_mdc, output_mdc),
        "raw_peak_eV": raw_e,
        "output_peak_eV": out_e,
        "peak_shift_eV": out_e - raw_e,
        "raw_peak_alpha": raw_a,
        "output_peak_alpha": out_a,
        "peak_shift_alpha": out_a - raw_a,
        "output_negative_fraction": float(np.mean(output < 0)),
        "qualitative_flag": "; ".join(flags),
    }


def _display_norm(*arrays: np.ndarray) -> LogNorm:
    positive = np.concatenate([np.asarray(array).ravel() for array in arrays])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        return LogNorm(vmin=1e-6, vmax=1.0)
    vmin = max(float(np.percentile(positive, 1.0)), 1e-6)
    vmax = float(np.percentile(positive, 99.5))
    if vmax <= vmin:
        vmax = max(float(positive.max()), vmin * 1.01)
    return LogNorm(vmin=vmin, vmax=vmax)


def _save_image_pages(
    rows: list[dict[str, Any]], output_dir: Path, model_labels: list[str]
) -> list[str]:
    paths: list[str] = []
    for page_index, start in enumerate(range(0, len(rows), 5), start=1):
        batch = rows[start : start + 5]
        fig, axes = plt.subplots(len(batch), 4, figsize=(16, 3.1 * len(batch)), squeeze=False)
        for row_index, item in enumerate(batch):
            raw = item["raw"]
            first = item["outputs"][model_labels[0]]
            second = item["outputs"][model_labels[1]]
            norm = _display_norm(raw, first, second)
            images = [raw, first, second, np.abs(first - second)]
            cmaps = ["magma", "magma", "magma", "viridis"]
            for col, (axis, image, cmap) in enumerate(zip(axes[row_index], images, cmaps, strict=True)):
                if col < 3:
                    axis.imshow(image, origin="lower", aspect="auto", cmap=cmap, norm=norm)
                else:
                    diff_norm = _display_norm(image)
                    axis.imshow(image, origin="lower", aspect="auto", cmap=cmap, norm=diff_norm)
                axis.set_xticks([])
                axis.set_yticks([])
            axes[row_index, 0].set_ylabel(item["file_id"], rotation=0, ha="right", va="center", fontsize=9)
        titles = ["raw input", model_labels[0], model_labels[1], "absolute model difference"]
        for axis, title in zip(axes[0], titles, strict=True):
            axis.set_title(title, fontsize=10)
        fig.suptitle("C test qualitative map review (shared per-sample scale)", y=0.995, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        path = output_dir / f"image_page_{page_index:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.resolve()))
    return paths


def _save_profile_pages(
    rows: list[dict[str, Any]], output_dir: Path, model_labels: list[str]
) -> list[str]:
    paths: list[str] = []
    for page_index, start in enumerate(range(0, len(rows), 5), start=1):
        batch = rows[start : start + 5]
        fig, axes = plt.subplots(len(batch), 2, figsize=(14, 2.6 * len(batch)), squeeze=False)
        for row_index, item in enumerate(batch):
            raw = item["raw"]
            first = item["outputs"][model_labels[0]]
            second = item["outputs"][model_labels[1]]
            energy = item["energy"]
            alpha = item["alpha"]
            profiles = [
                (energy, np.mean(raw, axis=1), np.mean(first, axis=1), np.mean(second, axis=1), "eV"),
                (alpha, np.mean(raw, axis=0), np.mean(first, axis=0), np.mean(second, axis=0), "alpha"),
            ]
            for axis, (coordinate, raw_profile, first_profile, second_profile, label) in zip(
                axes[row_index], profiles, strict=True
            ):
                axis.plot(coordinate, raw_profile, color="#555555", lw=0.8, label="raw")
                axis.plot(coordinate, first_profile, color="#1769aa", lw=0.9, label=model_labels[0])
                axis.plot(coordinate, second_profile, color="#c66a00", lw=0.9, label=model_labels[1])
                axis.set_ylabel(item["file_id"], rotation=0, ha="right", va="center", fontsize=8)
                axis.set_xlabel(label)
                axis.grid(alpha=0.18)
                if row_index == 0:
                    axis.legend(fontsize=7, ncol=3, loc="best")
        fig.suptitle("C test profile review (mean EDC and MDC)", y=0.995, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        path = output_dir / f"profile_page_{page_index:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.resolve()))
    return paths


def _save_overview(rows: list[dict[str, Any]], output_dir: Path, model_labels: list[str]) -> str:
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, max(16, 1.35 * len(rows))), squeeze=False)
    for row_index, item in enumerate(rows):
        raw = item["raw"]
        first = item["outputs"][model_labels[0]]
        second = item["outputs"][model_labels[1]]
        norm = _display_norm(raw, first, second)
        for col, image in enumerate((raw, first, second)):
            axes[row_index, col].imshow(image, origin="lower", aspect="auto", cmap="magma", norm=norm)
            axes[row_index, col].set_xticks([])
            axes[row_index, col].set_yticks([])
        difference = np.abs(first - second)
        axes[row_index, 3].imshow(difference, origin="lower", aspect="auto", cmap="viridis", norm=_display_norm(difference))
        axes[row_index, 3].set_xticks([])
        axes[row_index, 3].set_yticks([])
        axes[row_index, 0].set_ylabel(item["file_id"], rotation=0, ha="right", va="center", fontsize=8)
    titles = ["raw", model_labels[0], model_labels[1], "|model1-model2|"]
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title)
    fig.suptitle("19 locked C test samples: batch inference overview", y=0.998, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.994))
    path = output_dir / "c_test_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.resolve())


def run(args: argparse.Namespace) -> None:
    output_dir = args.output.expanduser().resolve()
    if not output_dir.is_relative_to(OUTPUT_ROOT):
        raise ValueError(f"output must be under {OUTPUT_ROOT}")
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    selected = _load_rows(args.manifest.resolve(strict=True), args.acceptance.resolve(strict=True))

    model_specs: list[tuple[str, Path]] = []
    for label, checkpoint in args.model:
        path = Path(checkpoint).expanduser().resolve(strict=True)
        model_specs.append((label, path))
    if len(model_specs) != 2:
        raise ValueError("provide exactly two --model LABEL CHECKPOINT arguments")
    model_labels = [label for label, _ in model_specs]

    with (output_dir / "manifest_used.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["record_id", "file_id", "converted_path", "source_path", "acquisition_time_s", "sweep_count"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in selected)

    output_paths: dict[str, dict[str, Path]] = {label: {} for label in model_labels}
    for label, checkpoint in model_specs:
        model_dir = output_dir / label / "denoised"
        model_dir.mkdir(parents=True, exist_ok=True)
        for row in selected:
            source = Path(row["converted_path"])
            destination = model_dir / f"{source.stem}_denoised.h5"
            if not destination.exists():
                denoise_file(source, checkpoint, model_dir)
            output_paths[label][row["record_id"]] = destination.resolve()

    items: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for row in selected:
        record_id = row["record_id"]
        source = Path(row["converted_path"])
        raw_cut = load_cut(source)
        raw = np.asarray(raw_cut.values, dtype=np.float64) / _scale(row)
        energy = np.asarray(raw_cut.coords["eV"].values, dtype=np.float64)
        alpha = np.asarray(raw_cut.coords["alpha"].values, dtype=np.float64)
        outputs: dict[str, np.ndarray] = {}
        for label in model_labels:
            output_cut = load_cut(output_paths[label][record_id])
            if output_cut.shape != raw_cut.shape:
                raise ValueError(f"shape mismatch for {row['file_id']} and model {label}")
            outputs[label] = np.asarray(output_cut.values, dtype=np.float64) / _scale(row)
            metrics = _metrics(raw, outputs[label], energy, alpha)
            metric_rows.append({"record_id": record_id, "file_id": row["file_id"], "model": label, "scale": _scale(row), **metrics})
        items.append({"record_id": record_id, "file_id": row["file_id"], "raw": raw, "outputs": outputs, "energy": energy, "alpha": alpha})

    metrics_path = output_dir / "qualitative_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        fields = sorted({key for metric in metric_rows for key in metric})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    image_pages = _save_image_pages(items, visual_dir, model_labels)
    profile_pages = _save_profile_pages(items, visual_dir, model_labels)
    overview = _save_overview(items, visual_dir, model_labels)

    summary = {
        "sample_count": len(selected),
        "record_ids": [row["record_id"] for row in selected],
        "file_ids": [row["file_id"] for row in selected],
        "models": [
            {"label": label, "checkpoint": str(path), "checkpoint_sha256": _sha256(path)}
            for label, path in model_specs
        ],
        "method": {
            "selection": "locked IDs from population-only-20260728 acceptance scientific_gate.ineligible_record_ids",
            "rate_normalization": "acquisition_time_s, then sweep_count, otherwise unity",
            "qualitative_note": "C has no paired high-exposure reference; metrics are input-to-output diagnostics only",
        },
        "artifacts": {
            "manifest_used": str((output_dir / "manifest_used.csv").resolve()),
            "qualitative_metrics": str(metrics_path.resolve()),
            "overview": overview,
            "image_pages": image_pages,
            "profile_pages": profile_pages,
        },
    }
    (output_dir / "qualitative_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"sample_count={len(selected)}")
    print(f"output_dir={output_dir}")
    print(f"metrics_path={metrics_path.resolve()}")
    print(f"overview={overview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", nargs=2, action="append", metavar=("LABEL", "CHECKPOINT"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
