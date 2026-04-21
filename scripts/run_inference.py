import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import LogNorm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim  # 新增: PSNR/SSIM

from modules.datasets.dataset import ArpesH5Dataset
from modules.models.ccnn import CCNN


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_latest_model(models_dir: Path) -> Path:
    candidates = sorted(models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No model checkpoints found in {models_dir}")
    return candidates[0]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_extent(energy: np.ndarray, angle: np.ndarray) -> List[float]:
    if energy.size and angle.size:
        return [float(angle[0]), float(angle[-1]), float(energy[0]), float(energy[-1])]
    return None


def log_scale(data: np.ndarray) -> Dict[str, float]:
    positive = data[data > 0]
    if positive.size == 0:
        return {"vmin": 1e-6, "vmax": 1.0}
    vmin = max(np.percentile(positive, 1.0), 1e-6)
    vmax = np.percentile(positive, 99.8)
    if vmax <= vmin:
        vmax = positive.max()
        vmin = max(vmax * 1e-3, 1e-6)
    return {"vmin": vmin, "vmax": vmax}


def save_preview(noisy: np.ndarray, denoised: np.ndarray, extent, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    titles = ["Noisy input", "Denoised"]
    datasets = [noisy, denoised]

    for ax, title, data in zip(axes, titles, datasets):
        stats = log_scale(data)
        kwargs = {"origin": "lower", "aspect": "auto", "cmap": "inferno", "norm": LogNorm(**stats)}
        if extent is not None:
            kwargs["extent"] = extent
        im = ax.imshow(data, **kwargs)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Angle [deg]" if extent else "Angle index")
        ax.set_ylabel("Energy [eV]" if extent else "Energy index")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def compute_energy_mask_info(
    spectrum: np.ndarray,
    energy: np.ndarray,
    threshold_ratio: float,
    min_run: int,
    method: str,
    percentile: float,
    smooth_window: int,
) -> Tuple[np.ndarray, Optional[int], Optional[float], int]:
    if energy.ndim != 1 or energy.size == 0:
        return np.ones_like(spectrum, dtype=np.float32), None, None, 0

    axis = 0
    if energy.size == spectrum.shape[1]:
        axis = 1

    profile = spectrum.mean(axis=1 - axis)
    if smooth_window and smooth_window > 1:
        win = max(3, int(smooth_window))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float32) / float(win)
        profile = np.convolve(profile, kernel, mode="same")
    max_val = float(profile.max())
    if max_val <= 0.0:
        return np.ones_like(spectrum, dtype=np.float32), None, None, axis

    if method == "percentile":
        threshold = float(np.percentile(profile, percentile))
    else:
        threshold = max_val * threshold_ratio
    below = profile < threshold
    run_len = max(1, int(min_run))

    high_at_start = energy[0] > energy[-1]
    mask_1d = np.ones_like(profile, dtype=np.float32)
    run = 0
    cutoff = None
    if high_at_start:
        for idx in range(profile.size):
            if below[idx]:
                run += 1
                if run >= run_len:
                    cutoff = idx
                    break
            else:
                run = 0
        if cutoff is not None:
            mask_1d[: cutoff + 1] = 0.0
    else:
        for idx in range(profile.size - 1, -1, -1):
            if below[idx]:
                run += 1
                if run >= run_len:
                    cutoff = idx
                    break
            else:
                run = 0
        if cutoff is not None:
            mask_1d[cutoff:] = 0.0

    cutoff_energy = None
    if cutoff is not None and energy.size == profile.size:
        cutoff_energy = float(energy[cutoff])

    if axis == 0:
        mask = np.repeat(mask_1d[:, None], spectrum.shape[1], axis=1).astype(np.float32)
    else:
        mask = np.repeat(mask_1d[None, :], spectrum.shape[0], axis=0).astype(np.float32)
    return mask, cutoff, cutoff_energy, axis


def apply_preview_noise(noisy: np.ndarray, noise_std: float, mask: Optional[np.ndarray]) -> np.ndarray:
    if noise_std <= 0.0:
        return noisy
    sigma = float(noisy.std() + 1e-6) * noise_std
    if sigma <= 0.0:
        return noisy
    rng = np.random.default_rng(seed=0)
    noise = rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32)
    if mask is not None:
        noise = noise * mask
    return noisy + noise


def save_mask(mask: np.ndarray, extent, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    kwargs = {"origin": "lower", "aspect": "auto", "cmap": "gray", "vmin": 0.0, "vmax": 1.0}
    if extent is not None:
        kwargs["extent"] = extent
    im = ax.imshow(mask, **kwargs)
    ax.set_title("Noise mask")
    ax.set_xlabel("Angle [deg]" if extent else "Angle index")
    ax.set_ylabel("Energy [eV]" if extent else "Energy index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_inference(
    config_path: Path,
    model_path: Path,
    input_glob: str,
    output_dir: Path,
    target_key_override: str = "",
    input_files: Optional[List[str]] = None,
    preview_noise: bool = False,
) -> None:
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    data_cfg = cfg.get("data", {})
    path_cfg = cfg["paths"]

    files: List[str] = []
    if input_files:
        files = [str(p) for p in input_files]
    else:
        if not input_glob:
            input_glob = path_cfg.get("h5_glob", "")
        if not input_glob:
            raise ValueError("No input glob specified. Provide --input-glob or set paths.h5_glob in config.")
        files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No HDF5 files matched pattern '{input_glob}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCNN(model_cfg["kernel_size"], model_cfg["num_layers"])  # 与训练配置对齐
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)  # strict=True 默认，避免权重错配被忽略
    model.to(device)
    model.eval()
    model_tag = model_path.stem

    summaries = []
    preview_dir = output_dir / "previews" / model_tag
    ensure_dir(preview_dir)

    input_key = data_cfg.get("input_key", "raw_channels")
    target_key = target_key_override or data_cfg.get("target_key", "spectrum")
    input_channel = data_cfg.get("input_channel", 0)

    normalize = bool(data_cfg.get("normalize", False))
    noise_std = float(data_cfg.get("noise_std", 0.0))
    noise_energy_threshold = float(data_cfg.get("noise_energy_threshold", 0.0))
    noise_min_run = int(data_cfg.get("noise_min_run", 0))
    noise_energy_method = str(data_cfg.get("noise_energy_method", "max_ratio"))
    noise_energy_percentile = float(data_cfg.get("noise_energy_percentile", 5.0))
    noise_energy_smooth = int(data_cfg.get("noise_energy_smooth", 0))

    for file_path in files:
        path = Path(file_path)
        with h5py.File(path, "r") as handle:
            if target_key not in handle:
                if "spectrum" in handle:
                    print(f"[WARN] {path}: target '{target_key}' missing, falling back to 'spectrum'.")
                    clean = np.asarray(handle["spectrum"])
                else:
                    raise KeyError(f"{path}: target '{target_key}' and fallback 'spectrum' not found.")
            else:
                clean = np.asarray(handle[target_key])
            if input_key not in handle:
                raise KeyError(f"{path}: input '{input_key}' not found.")
            noisy_data = np.asarray(handle[input_key])
            energy = handle["energy"][:] if "energy" in handle else np.arange(clean.shape[0])
            angle = handle["thetax"][:] if "thetax" in handle else np.arange(clean.shape[1])

        if noisy_data.ndim == 3:
            noisy = noisy_data[input_channel]
        else:
            noisy = noisy_data

        if noisy.shape != clean.shape:
            if noisy.T.shape == clean.shape:
                noisy = noisy.T
            else:
                raise ValueError(f"{path}: shape mismatch between input {noisy.shape} and target {clean.shape}")

        if normalize:
            # Norm 用 noisy 范围 (训时 noisy/target 同 norm)
            noisy_norm, noisy_mean, noisy_std = ArpesH5Dataset._normalize(noisy, return_stats=True)
        else:
            noisy_norm = noisy.astype(np.float32)
            noisy_mean, noisy_std = 0.0, 1.0

        tensor_in = torch.from_numpy(noisy_norm).unsqueeze(0).unsqueeze(0).to(device).float()  # 确保 float
        with torch.no_grad():
            clean_pred = model(tensor_in)  # 直接预测干净谱图，shape: [1, 1, H, W]

        clean_pred_np = clean_pred.squeeze(0).squeeze(0).cpu().numpy()  # 从 [1,1,H,W] -> [H,W]

        denoised_norm = clean_pred_np  # 模型输出即干净谱图的标准化结果
        if normalize:
            denoised = ArpesH5Dataset._denormalize(denoised_norm, noisy_mean, noisy_std)  # 用 noisy std 反转
        else:
            denoised = denoised_norm
        
        mae = float(np.mean(np.abs(denoised - clean)))
        mse = float(np.mean((denoised - clean) ** 2))

        # 新增: PSNR/SSIM 计算
        psnr_val = psnr(clean, denoised, data_range=clean.max() - clean.min())
        ssim_val = ssim(clean, denoised, data_range=clean.max() - clean.min())

        dataset_name = f"denoised_{model_tag}"
        with h5py.File(path, "a") as handle:
            if dataset_name in handle:
                del handle[dataset_name]
            dset = handle.create_dataset(dataset_name, data=denoised.astype(np.float32))
            dset.attrs["source_model"] = str(model_path)
            dset.attrs["mae_vs_spectrum"] = mae
            dset.attrs["mse_vs_spectrum"] = mse

        extent = prepare_extent(energy, angle)
        preview_path = preview_dir / f"{path.stem}_{model_tag}_comparison.png"
        preview_mask = None
        cutoff_idx = None
        cutoff_energy = None
        cutoff_axis = None
        if preview_noise and energy.size and (
            noise_energy_method == "percentile" or noise_energy_threshold > 0.0
        ):
            preview_mask, cutoff_idx, cutoff_energy, cutoff_axis = compute_energy_mask_info(
                clean,
                energy,
                noise_energy_threshold,
                noise_min_run,
                noise_energy_method,
                noise_energy_percentile,
                noise_energy_smooth,
            )
        preview_noisy = apply_preview_noise(noisy, noise_std, preview_mask) if preview_noise else noisy
        save_preview(preview_noisy, denoised, extent, preview_path)
        if preview_mask is not None:
            mask_path = preview_dir / f"{path.stem}_{model_tag}_mask.png"
            save_mask(preview_mask, extent, mask_path)

        summaries.append(
            {
                "file": str(path),
                "model_tag": model_tag,
                "dataset": dataset_name,
                "mae": mae,
                "mse": mse,
                "psnr": psnr_val,  # 新增
                "ssim": ssim_val,  # 新增
                "preview": str(preview_path),
                "cutoff_idx": cutoff_idx if cutoff_idx is not None else "",
                "cutoff_energy": cutoff_energy if cutoff_energy is not None else "",
                "cutoff_axis": cutoff_axis if cutoff_axis is not None else "",
            }
        )
        if cutoff_idx is not None:
            print(f"[MASK] {path.name}: cutoff_idx={cutoff_idx}, cutoff_energy={cutoff_energy}, axis={cutoff_axis}")
        print(f"[OK] {path.name}: MAE={mae:.4f}, MSE={mse:.4f}, PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")

    if summaries:
        import csv

        metrics_dir = output_dir / "metrics" / model_tag
        ensure_dir(metrics_dir)
        csv_path = metrics_dir / "inference_metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file",
                    "model_tag",
                    "dataset",
                    "mae",
                    "mse",
                    "psnr",
                    "ssim",
                    "preview",
                    "cutoff_idx",
                    "cutoff_energy",
                    "cutoff_axis",
                ],
            )
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Inference metrics saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run denoising inference on ARPES HDF5 files.")
    parser.add_argument("--config", type=Path, default=Path("config/config_baseline.yaml"), help="Path to YAML config.")
    parser.add_argument("--model", type=Path, default=None, help="Path to trained .pt checkpoint.")
    parser.add_argument("--input-glob", type=str, default="", help="Override HDF5 glob pattern.")
    parser.add_argument(
        "--target-key",
        type=str,
        default="",
        help="Override target dataset name (default uses config; falls back to 'spectrum' if missing).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/inference"), help="Directory for previews/metrics.")
    parser.add_argument(
        "--preview-noise",
        action="store_true",
        help="Apply synthetic noise to preview-only noisy input (uses data.noise_std).",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    model_path = args.model
    if model_path is None:
        model_dir = Path("results/models")
        model_path = find_latest_model(model_dir)
    else:
        model_path = model_path.resolve()

    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    run_inference(
        config_path,
        model_path,
        args.input_glob,
        output_dir,
        target_key_override=args.target_key,
        preview_noise=args.preview_noise,
    )


if __name__ == "__main__":
    main()
