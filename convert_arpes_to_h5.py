import argparse
import glob
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


@dataclass
class ConversionStats:
    converted: int = 0
    skipped: int = 0
    failed: int = 0


def parse_shape(text: str) -> Tuple[int, ...]:
    parts = [p for chunk in text.lower().replace("x", ",").split(",") for p in [chunk.strip()] if p]
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(f"Expected shape with 2 or 3 dimensions, got '{text}'.")
    try:
        shape = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Shape must contain integers: {text}") from exc
    if any(v <= 0 for v in shape):
        raise argparse.ArgumentTypeError(f"Shape entries must be >0, got {shape}.")
    return shape  # type: ignore[return-value]


def gather_inputs(
    direct_inputs: Sequence[str],
    input_root: Optional[Path],
    extensions: Sequence[str],
    interactive: bool,
) -> List[Path]:
    resolved: List[Path] = []

    def normalize_extension(exts: Sequence[str]) -> List[str]:
        return [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts]

    extensions = normalize_extension(extensions)

    for item in direct_inputs:
        matches = glob.glob(item, recursive=True)
        if matches:
            resolved.extend(Path(match).resolve() for match in matches)
            continue
        candidate = Path(item).expanduser()
        if candidate.exists():
            resolved.append(candidate.resolve())
        else:
            print(f"[WARN] No match for input '{item}'.")

    if input_root:
        if not input_root.exists():
            raise FileNotFoundError(f"Input root '{input_root}' does not exist.")
        candidates = sorted(
            path.resolve()
            for ext in extensions
            for path in input_root.rglob(f"*{ext}")
        )
        if not candidates:
            print(f"[WARN] No files with extensions {extensions} under '{input_root}'.")
        elif interactive:
            print("Discovered files:")
            for idx, candidate in enumerate(candidates, start=1):
                print(f"  [{idx:02d}] {candidate}")
            selection = input(
                "Enter indexes to convert (e.g. '1 3 5', empty for all, 'q' to cancel): "
            ).strip()
            if selection.lower().startswith("q"):
                return []
            if selection:
                tokens = selection.replace(",", " ").split()
                picked: List[Path] = []
                for token in tokens:
                    try:
                        index = int(token)
                    except ValueError as exc:
                        raise ValueError(f"Invalid selection token '{token}'. Expected integer index.") from exc
                    if 1 <= index <= len(candidates):
                        picked.append(candidates[index - 1])
                    else:
                        raise ValueError(f"Index {index} out of range 1..{len(candidates)}.")
                resolved.extend(picked)
            else:
                resolved.extend(candidates)
        elif not direct_inputs:
            resolved.extend(candidates)

    if not resolved:
        default_bin = Path("data/Spectrum_MAP.bin").resolve()
        if default_bin.exists():
            print("[INFO] No inputs provided; falling back to data/Spectrum_MAP.bin.")
            resolved.append(default_bin)
        else:
            raise FileNotFoundError(
                "No input files provided and sample data/Spectrum_MAP.bin not found."
            )

    seen = set()
    unique: List[Path] = []
    for path in resolved:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def load_bin(path: Path, shape: Tuple[int, ...], dtype: str) -> np.ndarray:
    dtype_obj = np.dtype(dtype)
    expected_bytes = math.prod(shape) * dtype_obj.itemsize
    actual_bytes = path.stat().st_size
    if expected_bytes != actual_bytes:
        raise ValueError(
            f"{path}: size mismatch. Expected {expected_bytes} bytes for shape {shape}, dtype {dtype}, "
            f"found {actual_bytes}."
        )
    data = np.fromfile(path, dtype=dtype_obj)
    return data.reshape(shape)


def load_pxt(
    path: Path,
    *,
    energy_offset_override: Optional[float] = None,
    energy_step_override: Optional[float] = None,
    angle_offset_override: Optional[float] = None,
    angle_step_override: Optional[float] = None,
    channel: int = 0,
    subtract_dark: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any], Dict[str, np.ndarray]]:
    raw = path.read_bytes()
    if len(raw) < 256:
        raise ValueError(f"{path}: file too small to be a valid PXT container.")

    def read_uint(index: int) -> int:
        return struct.unpack_from("<I", raw, index * 4)[0]

    def read_double(index: int) -> float:
        return struct.unpack_from("<d", raw, index * 4)[0]

    total_points = read_uint(21)
    channel_count = max(1, read_uint(22))
    frame_type_bytes = raw[25 * 4 : 27 * 4]
    frame_type = frame_type_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore") or "unknown"

    width = read_uint(35)
    height = read_uint(36)
    if width == 0 or height == 0:
        raise ValueError(f"{path}: reported shape {width}x{height} is invalid.")
    if total_points not in {0, width * height}:
        print(
            f"[WARN] {path}: total_points ({total_points}) does not match width*height ({width * height})."
        )

    energy_step_raw = read_double(39)
    angle_step_raw = read_double(41)
    energy_offset_raw = read_double(47)
    angle_offset_raw = read_double(49)

    energy_step = energy_step_override if energy_step_override is not None else energy_step_raw
    angle_step = angle_step_override if angle_step_override is not None else angle_step_raw
    energy_offset = energy_offset_override if energy_offset_override is not None else energy_offset_raw
    angle_offset = angle_offset_override if angle_offset_override is not None else angle_offset_raw

    itemsize = np.dtype("<i2").itemsize
    data_bytes = width * height * channel_count * itemsize
    if data_bytes <= 0 or data_bytes > len(raw):
        raise ValueError(f"{path}: payload size is inconsistent with header values.")
    header_bytes = len(raw) - data_bytes
    if header_bytes < 0:
        raise ValueError(f"{path}: negative header size computed.")

    payload = np.frombuffer(raw, dtype="<i2", count=width * height * channel_count, offset=header_bytes)
    payload = payload.reshape(height, width, channel_count)

    if not 0 <= channel < channel_count:
        raise ValueError(f"{path}: channel index {channel} out of range for {channel_count} available channels.")

    signal = payload[..., channel].astype(np.float32)
    subtracted_from: Optional[int] = None
    if subtract_dark and channel_count > 1:
        dark_idx = 1 if channel == 0 else (channel - 1 if channel > 0 else None)
        if dark_idx is not None and 0 <= dark_idx < channel_count:
            signal = signal - payload[..., dark_idx].astype(np.float32)
            subtracted_from = dark_idx
    signal = np.clip(signal, a_min=0.0, a_max=None)
    spectrum = signal.T.copy()

    axes: Dict[str, np.ndarray] = {}
    if energy_offset is not None and energy_step is not None:
        axes["energy"] = (np.arange(width, dtype=np.float32) * energy_step + energy_offset).astype(np.float32)
    else:
        axes["energy_index"] = np.arange(width, dtype=np.float32)
    if angle_offset is not None and angle_step is not None:
        axes["thetax"] = (np.arange(height, dtype=np.float32) * angle_step + angle_offset).astype(np.float32)
    else:
        axes["angle_index"] = np.arange(height, dtype=np.float32)

    attrs: Dict[str, Any] = {
        "frame_type": frame_type,
        "channels_total": int(channel_count),
        "channel_used": int(channel),
        "raw_energy_offset": float(energy_offset_raw),
        "raw_energy_step": float(energy_step_raw),
        "raw_angle_offset": float(angle_offset_raw),
        "raw_angle_step": float(angle_step_raw),
        "unit_energy": "eV",
        "unit_angle": "deg",
    }
    if subtracted_from is not None:
        attrs["subtracted_channel"] = int(subtracted_from)

    extras: Dict[str, np.ndarray] = {
        "raw_channels": payload.transpose(2, 0, 1).copy(),
    }
    return spectrum.astype(np.float32, copy=False), axes, attrs, extras


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_bin_axes(
    shape: Tuple[int, int, int],
    energy_offset: float,
    energy_step: float,
    thetax_offset: float,
    thetax_step: float,
    thetay_offset: float,
    thetay_step: float,
) -> Dict[str, np.ndarray]:
    width, height, depth = shape
    energy_axis = np.arange(width, dtype=np.float32) * energy_step + energy_offset
    thetax_axis = np.arange(height, dtype=np.float32) * thetax_step + thetax_offset
    thetay_axis = np.arange(depth, dtype=np.float32) * thetay_step + thetay_offset
    return {
        "energy": energy_axis,
        "thetax": thetax_axis,
        "thetay": thetay_axis,
    }


def write_h5(
    spectrum: np.ndarray,
    axes: Dict[str, np.ndarray],
    destination: Path,
    attrs: Dict[str, Any],
    overwrite: bool,
    extra_datasets: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists. Use --force to overwrite.")
    ensure_directory(destination.parent)
    with h5py.File(destination, "w") as handle:
        handle.create_dataset("spectrum", data=spectrum)
        for name, values in axes.items():
            handle.create_dataset(name, data=values)
        if extra_datasets:
            for name, values in extra_datasets.items():
                handle.create_dataset(name, data=values)
        for key, value in attrs.items():
            handle.attrs[key] = value


def save_preview(
    spectrum: np.ndarray,
    axes: Dict[str, np.ndarray],
    destination: Path,
    slice_index: int,
    thetax_column: int,
) -> None:
    ensure_directory(destination.parent)

    if spectrum.ndim == 3:
        width, height, depth = spectrum.shape
        slice_idx = int(np.clip(slice_index, 0, depth - 1))
        column_idx = int(np.clip(thetax_column, 0, height - 1))

        spectrum_slice = np.clip(spectrum[:, :, slice_idx], a_min=0.0, a_max=None)
        positive = spectrum_slice[spectrum_slice > 0]
        if positive.size:
            vmin = max(np.percentile(positive, 5), 1e-6)
            vmax = np.percentile(positive, 99)
            if vmax <= vmin:
                vmax = positive.max()
                vmin = max(vmax * 1e-3, 1e-6)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

        fig, (ax_slice, ax_energy) = plt.subplots(1, 2, figsize=(12, 5))
        heatmap_kwargs = {
            "origin": "lower",
            "aspect": "auto",
            "cmap": "inferno",
        }
        if norm is not None:
            heatmap_kwargs["norm"] = norm
        heatmap = ax_slice.imshow(np.rot90(spectrum_slice), **heatmap_kwargs)
        fig.colorbar(heatmap, ax=ax_slice)
        ax_slice.set_title(f"Slice {slice_idx} (log scale)" if norm is not None else f"Slice {slice_idx}")
        ax_slice.set_xlabel("Thetax bin")
        ax_slice.set_ylabel("Energy bin")

        energy_axis = axes.get("energy")
        if energy_axis is not None and energy_axis.size == width:
            ax_energy.plot(
                energy_axis,
                spectrum[:, column_idx, slice_idx],
                "b-",
                linewidth=1,
            )
            ax_energy.set_xlabel("Energy [eV]")
        else:
            ax_energy.plot(
                spectrum[:, column_idx, slice_idx],
                "b-",
                linewidth=1,
            )
            ax_energy.set_xlabel("Energy index")
        ax_energy.set_title(f"Energy spectrum @ column {column_idx}")
        ax_energy.set_ylabel("Intensity")
        ax_energy.grid(True, alpha=0.3)
    elif spectrum.ndim == 2:
        data = np.clip(spectrum, a_min=0.0, a_max=None)
        energy_axis = axes.get("energy")
        angle_axis = axes.get("thetax")
        if angle_axis is None:
            angle_axis = axes.get("angle")

        extent = None
        if (
            energy_axis is not None
            and angle_axis is not None
            and energy_axis.size == data.shape[0]
            and angle_axis.size == data.shape[1]
        ):
            extent = [
                float(angle_axis[0]),
                float(angle_axis[-1]),
                float(energy_axis[0]),
                float(energy_axis[-1]),
            ]

        positive = data[data > 0]
        if positive.size:
            vmin = max(np.percentile(positive, 5), 1e-6)
            vmax = np.percentile(positive, 99)
            if vmax <= vmin:
                vmax = positive.max()
                vmin = max(vmax * 1e-3, 1e-6)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

        fig, ax = plt.subplots(figsize=(7, 5))
        heatmap_kwargs = {
            "origin": "lower",
            "aspect": "auto",
            "cmap": "inferno",
        }
        if extent is not None:
            heatmap_kwargs["extent"] = extent
        if norm is not None:
            heatmap_kwargs["norm"] = norm
        heatmap = ax.imshow(data, **heatmap_kwargs)
        fig.colorbar(heatmap, ax=ax)
        ax.set_title("Cut (log scale)" if norm is not None else "Cut")
        ax.set_xlabel("Angle [deg]" if angle_axis is not None else "Angle index")
        ax.set_ylabel("Energy [eV]" if energy_axis is not None else "Energy index")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(spectrum.size), spectrum.ravel(), "b-")
        ax.set_title("Spectrum (1D)")
        ax.set_xlabel("Point index")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def format_relative(path: Path, root: Optional[Path]) -> Path:
    if root:
        try:
            return path.relative_to(root)
        except ValueError:
            return Path(path.name)
    return Path(path.name)


def convert_file(
    path: Path,
    args: argparse.Namespace,
    stats: ConversionStats,
    root: Optional[Path],
) -> None:
    suffix = path.suffix.lower()
    output_dir: Path = args.output_dir
    relative = format_relative(path, root)
    output_path = output_dir / relative.with_suffix(".h5")
    preview_path = args.preview_dir / relative.with_suffix(".png") if args.preview else None

    try:
        extras: Optional[Dict[str, np.ndarray]] = None
        if suffix == ".bin":
            shape = args.bin_shape
            spectrum = load_bin(path, shape, args.bin_dtype)
            axes = build_bin_axes(
                shape,
                args.energy_offset,
                args.energy_step,
                args.thetax_offset,
                args.thetax_step,
                args.thetay_offset,
                args.thetay_step,
            )
            attrs = {
                "source_format": "bin",
                "source_path": str(path),
                "unit_energy": "eV",
                "unit_angle": "deg",
            }
        elif suffix == ".pxt":
            spectrum, axes, pxt_attrs, extras = load_pxt(
                path,
                energy_offset_override=args.pxt_energy_offset,
                energy_step_override=args.pxt_energy_step,
                angle_offset_override=args.pxt_angle_offset,
                angle_step_override=args.pxt_angle_step,
                channel=args.pxt_channel,
                subtract_dark=args.pxt_subtract_dark,
            )
            attrs = {
                "source_format": "pxt",
                "source_path": str(path),
                **pxt_attrs,
            }
        else:
            stats.skipped += 1
            print(f"[SKIP] {path}: unsupported extension.")
            return

        attrs["shape"] = str(tuple(int(dim) for dim in spectrum.shape))

        if args.dry_run:
            print(f"[DRY-RUN] Would convert '{path}' -> '{output_path}'.")
            return

        write_h5(spectrum, axes, output_path, attrs, overwrite=args.force, extra_datasets=extras)
        print(f"[OK] {path} -> {output_path}")
        if preview_path:
            save_preview(spectrum, axes, preview_path, args.slice_index, args.thetax_column)
            print(f"     Preview -> {preview_path}")
        stats.converted += 1
    except Exception as exc:  # pylint: disable=broad-except
        stats.failed += 1
        print(f"[FAIL] {path}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ARPES datasets (*.bin, *.pxt) into HDF5 for ML workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Explicit file paths or glob patterns to convert.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Directory to scan recursively for candidate files.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".bin", ".pxt"],
        help="Extensions considered when scanning input-root.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt to pick files discovered under input-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/converted_h5"),
        help="Directory where converted HDF5 files are written.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate PNG previews alongside HDF5 outputs.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("data/previews"),
        help="Directory where preview PNGs are written when --preview is set.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=25,
        help="Slice index used for 3D previews.",
    )
    parser.add_argument(
        "--thetax-column",
        type=int,
        default=285,
        help="Column index used for the representative energy spectrum.",
    )
    parser.add_argument(
        "--bin-shape",
        type=parse_shape,
        default=(365, 571, 51),
        help="Shape of binary cubes (width x height x depth).",
    )
    parser.add_argument(
        "--bin-dtype",
        default="float32",
        help="NumPy dtype used when loading binary cubes.",
    )
    parser.add_argument("--energy-offset", type=float, default=14.58)
    parser.add_argument("--energy-step", type=float, default=0.01)
    parser.add_argument("--thetax-offset", type=float, default=-8.946134)
    parser.add_argument("--thetax-step", type=float, default=0.031226)
    parser.add_argument("--thetay-offset", type=float, default=-5.0)
    parser.add_argument("--thetay-step", type=float, default=0.2)
    parser.add_argument(
        "--pxt-energy-offset",
        type=float,
        help="Override the energy offset parsed from PXT headers.",
    )
    parser.add_argument(
        "--pxt-energy-step",
        type=float,
        help="Override the energy step parsed from PXT headers.",
    )
    parser.add_argument(
        "--pxt-angle-offset",
        type=float,
        help="Override the angle offset parsed from PXT headers.",
    )
    parser.add_argument(
        "--pxt-angle-step",
        type=float,
        help="Override the angle step parsed from PXT headers.",
    )
    parser.add_argument(
        "--pxt-channel",
        type=int,
        default=0,
        help="Channel index to export from multi-channel PXT files (0-based).",
    )
    parser.add_argument(
        "--pxt-subtract-dark",
        action="store_true",
        help="Subtract the adjacent channel (typically channel 1) when exporting PXT intensity.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without writing files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        inputs = gather_inputs(args.inputs, args.input_root, args.extensions, args.interactive)
    except Exception as exc:  # pylint: disable=broad-except
        parser.error(str(exc))
        return

    stats = ConversionStats()
    root = args.input_root.resolve() if args.input_root else None

    for file_path in inputs:
        convert_file(Path(file_path), args, stats, root)

    summary = (
        f"Converted: {stats.converted}, "
        f"Skipped: {stats.skipped}, "
        f"Failed: {stats.failed}"
    )
    if stats.failed:
        print(f"[DONE] {summary} (check logs above for failures).")
    else:
        print(f"[DONE] {summary}.")


if __name__ == "__main__":
    main()
