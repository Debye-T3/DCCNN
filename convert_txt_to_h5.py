import argparse
import re
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


def parse_axes(lines) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Parse dimension sizes and axes from txt header."""
    def find_line(prefix: str) -> str:
        for line in lines:
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip()
        raise ValueError(f"Missing '{prefix}' in txt header.")

    n_energy = int(find_line("Dimension 1 size"))
    n_angle = int(find_line("Dimension 2 size"))
    energy_axis = np.fromstring(find_line("Dimension 1 scale"), sep=" ")
    angle_axis = np.fromstring(find_line("Dimension 2 scale"), sep=" ")
    if energy_axis.size != n_energy:
        raise ValueError(f"Energy axis length {energy_axis.size} != {n_energy}")
    if angle_axis.size != n_angle:
        raise ValueError(f"Angle axis length {angle_axis.size} != {n_angle}")
    return energy_axis.astype(np.float32), angle_axis.astype(np.float32), n_energy, n_angle


def parse_data(lines, start_idx: int, n_energy: int, n_angle: int) -> np.ndarray:
    """
    Parse data rows starting at start_idx. Each row should contain either:
    - n_angle values (pure intensity), or
    - n_angle + 1 values (first column = energy, rest = intensity).
    """
    data_rows = []
    for line in lines[start_idx:]:
        if not line.strip():
            continue
        nums = np.fromstring(line, sep=" ")
        if nums.size == 0:
            continue
        if nums.size == n_angle + 1:
            nums = nums[1:]
        elif nums.size > n_angle + 1:
            nums = nums[-n_angle:]  # take last n_angle values
        elif nums.size < n_angle:
            # pad short lines with zeros to expected length
            nums = np.pad(nums, (0, n_angle - nums.size), mode="constant", constant_values=0)
        data_rows.append(nums)
        if len(data_rows) >= n_energy:
            break
    if len(data_rows) != n_energy:
        # pad missing rows with zeros
        missing = n_energy - len(data_rows)
        if missing > 0:
            pad_row = np.zeros((missing, n_angle), dtype=np.float32)
            data_arr = np.vstack([data_rows, pad_row])
        else:
            data_arr = np.stack(data_rows[:n_energy])
    else:
        data_arr = np.stack(data_rows)
    return data_arr.astype(np.float32)


def convert_txt(txt_path: Path, out_path: Path) -> None:
    txt_str = txt_path.read_text(encoding="utf-8", errors="ignore")
    lines = txt_str.splitlines()
    # Find first numeric row (many numbers) to start data
    numeric_re = re.compile(r"^[0-9eE+\-.\s]+$")
    start_idx = None
    for i, l in enumerate(lines):
        if numeric_re.match(l.strip()) and len(l.split()) > 5:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find numeric data block in txt file.")
    energy_axis, angle_axis, n_energy, n_angle = parse_axes(lines)
    spectrum = parse_data(lines, start_idx + 1, n_energy, n_angle)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("spectrum", data=spectrum)
        f.create_dataset("energy", data=energy_axis)
        f.create_dataset("thetax", data=angle_axis)
        # also store as raw_channels[1,H,W] for compatibility
        f.create_dataset("raw_channels", data=spectrum[None, ...])
        f.attrs["source_format"] = "txt"
        f.attrs["source_path"] = str(txt_path)
        f.attrs["shape"] = str(spectrum.shape)
    print(f"[OK] {txt_path} -> {out_path}")
    return spectrum, energy_axis, angle_axis


def compute_contrast(data: np.ndarray, pmin: float, pmax: float) -> dict:
    positive = data[data > 0]
    if positive.size == 0:
        return {"vmin": 1e-6, "vmax": 1.0}
    vmin = max(np.percentile(positive, pmin), 1e-6)
    vmax = np.percentile(positive, pmax)
    if vmax <= vmin:
        vmax = positive.max()
        vmin = max(vmax * 1e-3, 1e-6)
    return {"vmin": float(vmin), "vmax": float(vmax)}


def save_preview(spectrum: np.ndarray, energy: np.ndarray, angle: np.ndarray, out_path: Path, cmap: str, pmin: float, pmax: float, use_log: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = compute_contrast(spectrum, pmin, pmax)
    norm = LogNorm(vmin=stats["vmin"], vmax=stats["vmax"]) if use_log else None
    extent = None
    if energy.size == spectrum.shape[0] and angle.size == spectrum.shape[1]:
        extent = [float(angle[0]), float(angle[-1]), float(energy[0]), float(energy[-1])]
    fig, ax = plt.subplots(figsize=(7, 5))
    kwargs = {"origin": "lower", "aspect": "auto", "cmap": cmap}
    if extent is not None:
        kwargs["extent"] = extent
    if norm is not None:
        kwargs["norm"] = norm
    im = ax.imshow(spectrum, **kwargs)
    fig.colorbar(im, ax=ax)
    ax.set_title("Spectrum (log scale)" if norm is not None else "Spectrum")
    ax.set_xlabel("Angle [deg]" if extent else "Angle index")
    ax.set_ylabel("Energy [eV]" if extent else "Energy index")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Convert ARPES txt export to HDF5.")
    parser.add_argument("inputs", nargs="+", help="Txt files to convert (glob or paths).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/converted_h5"), help="Output directory for H5.")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix before .h5 (e.g., '_txt').")
    parser.add_argument("--preview", action="store_true", help="Generate preview PNGs.")
    parser.add_argument("--preview-dir", type=Path, default=Path("data/previews"), help="Directory for preview PNGs.")
    parser.add_argument("--preview-cmap", type=str, default="inferno", help="Colormap for preview.")
    parser.add_argument("--preview-pmin", type=float, default=1.0, help="Lower percentile for contrast.")
    parser.add_argument("--preview-pmax", type=float, default=99.5, help="Upper percentile for contrast.")
    parser.add_argument("--preview-log", action="store_true", help="Use log color scale for preview.")
    args = parser.parse_args()

    for pat in args.inputs:
        for path in Path().glob(pat):
            if not path.is_file():
                continue
            out_name = path.with_suffix("").name + (args.suffix or "") + ".h5"
            out_path = args.output_dir / out_name
            spectrum, energy_axis, angle_axis = convert_txt(path, out_path)
            if args.preview:
                prev_path = args.preview_dir / (path.with_suffix("").name + (args.suffix or "") + ".png")
                save_preview(spectrum, energy_axis, angle_axis, prev_path, args.preview_cmap, args.preview_pmin, args.preview_pmax, args.preview_log)
                print(f"     Preview -> {prev_path}")


if __name__ == "__main__":
    main()
