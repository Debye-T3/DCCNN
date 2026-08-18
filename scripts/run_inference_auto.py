import argparse
import csv
import glob
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_inference import run_inference


def _load_input_2d(path: Path, input_key: str, input_channel: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if input_key not in handle:
            raise KeyError(f"{path}: input '{input_key}' not found.")
        data = np.asarray(handle[input_key])
    if data.ndim == 3:
        return np.asarray(data[input_channel], dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def _noise_score(img: np.ndarray) -> float:
    # High-frequency ratio: larger => noisier
    x = img.astype(np.float32)
    x = x - float(x.mean())
    base_std = float(x.std() + 1e-8)
    if base_std <= 1e-8:
        return 0.0
    dx = np.abs(np.diff(x, axis=1)).mean()
    dy = np.abs(np.diff(x, axis=0)).mean()
    hf = float((dx + dy) * 0.5)
    return hf / base_std


def _route_files(
    files: list[Path],
    input_key: str,
    input_channel: int,
    threshold: float,
) -> tuple[list[Path], list[Path], list[dict[str, object]]]:
    clean_like: list[Path] = []
    noisy_like: list[Path] = []
    rows: list[dict[str, object]] = []

    for p in files:
        arr = _load_input_2d(p, input_key=input_key, input_channel=input_channel)
        score = _noise_score(arr)
        use_noisy_model = score >= threshold
        if use_noisy_model:
            noisy_like.append(p)
            route = "v5"
        else:
            clean_like.append(p)
            route = "v6"
        rows.append(
            {
                "file": str(p),
                "noise_score": score,
                "threshold": threshold,
                "selected_model": route,
            }
        )
    return clean_like, noisy_like, rows


def _write_routing_csv(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "auto_routing.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "noise_score", "threshold", "selected_model"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[AUTO] routing saved to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto route inference between v6(clean) and v5(noisy).")
    parser.add_argument("--input-glob", type=str, required=True, help="Input HDF5 glob pattern.")
    parser.add_argument("--target-key", type=str, default="spectrum", help="Target dataset key.")
    parser.add_argument("--input-key", type=str, default="raw_channels", help="Input dataset key.")
    parser.add_argument("--input-channel", type=int, default=0, help="Input channel for 3D raw tensor.")
    parser.add_argument("--noise-threshold", type=float, default=0.62, help="Noise score threshold.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_auto"),
        help="Auto inference output directory.",
    )
    parser.add_argument(
        "--config-v6",
        type=Path,
        default=Path("config/config_nbvt_train_v6_stable.yaml"),
        help="Config used with v6 model.",
    )
    parser.add_argument(
        "--model-v6",
        type=Path,
        default=Path("results_nbvt_v6_stable/models/20260429_221939_layers7_batch6_kernel3_alpha0.1.pt"),
        help="Clean-like model path (v6).",
    )
    parser.add_argument(
        "--config-v5",
        type=Path,
        default=Path("config/config_nbvt_train_v5_balanced_plus.yaml"),
        help="Config used with v5 model.",
    )
    parser.add_argument(
        "--model-v5",
        type=Path,
        default=Path("results_nbvt_v5_balanced_plus/models/20260427_113404_layers7_batch6_kernel3_alpha0.1.pt"),
        help="Noisy-like model path (v5).",
    )
    args = parser.parse_args()

    files = sorted(Path(p) for p in glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No HDF5 files matched pattern '{args.input_glob}'.")

    out_dir = args.output_dir.resolve()
    clean_files, noisy_files, rows = _route_files(
        files,
        input_key=args.input_key,
        input_channel=args.input_channel,
        threshold=float(args.noise_threshold),
    )
    _write_routing_csv(rows, out_dir)

    print(f"[AUTO] total={len(files)}, v6(clean-like)={len(clean_files)}, v5(noisy-like)={len(noisy_files)}")

    if clean_files:
        run_inference(
            config_path=args.config_v6.resolve(),
            model_path=args.model_v6.resolve(),
            input_glob="",
            output_dir=(out_dir / "v6_cleanlike"),
            target_key_override=args.target_key,
            input_files=[str(p) for p in clean_files],
            preview_noise=False,
            inference_noise=False,
            inference_noise_std=None,
        )

    if noisy_files:
        run_inference(
            config_path=args.config_v5.resolve(),
            model_path=args.model_v5.resolve(),
            input_glob="",
            output_dir=(out_dir / "v5_noisylike"),
            target_key_override=args.target_key,
            input_files=[str(p) for p in noisy_files],
            preview_noise=False,
            inference_noise=False,
            inference_noise_std=None,
        )


if __name__ == "__main__":
    main()
