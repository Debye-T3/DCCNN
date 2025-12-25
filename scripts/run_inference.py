import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List

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


def save_preview(noisy: np.ndarray, denoised: np.ndarray, clean: np.ndarray, extent, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Noisy input", "Denoised", "Reference spectrum"]
    datasets = [noisy, denoised, clean]

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


def run_inference(
    config_path: Path,
    model_path: Path,
    input_glob: str,
    output_dir: Path,
    target_key_override: str = "",
) -> None:
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    data_cfg = cfg.get("data", {})
    path_cfg = cfg["paths"]

    if not input_glob:
        input_glob = path_cfg.get("h5_glob", "")
    if not input_glob:
        raise ValueError("No input glob specified. Provide --input-glob or set paths.h5_glob in config.")

    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No HDF5 files matched pattern '{input_glob}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = CCNN(model_cfg["kernel_size"], model_cfg["num_layers"])
    # 匹配 v2 .pt 实际层数 (从 debug max=13 +1=14)
    num_layers = 14
    model = CCNN(model_cfg["kernel_size"], num_layers)
    state = torch.load(model_path, map_location=device)
    #model.load_state_dict(state)
    model.load_state_dict(state, strict=False)  # 忽略 missing/unexpected 键
    print("Loaded with strict=False; mismatched keys ignored.")
    model.to(device)
    model.eval()
    model_tag = model_path.stem

    summaries = []
    preview_dir = output_dir / "previews"
    ensure_dir(preview_dir)

    input_key = data_cfg.get("input_key", "raw_channels")
    target_key = target_key_override or data_cfg.get("target_key", "spectrum")
    input_channel = data_cfg.get("input_channel", 0)

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

                       # Norm 用 noisy 范围 (训时 noisy/target 同 norm)
        noisy_norm, noisy_mean, noisy_std = ArpesH5Dataset._normalize(noisy, return_stats=True)

        tensor_in = torch.from_numpy(noisy_norm).unsqueeze(0).unsqueeze(0).to(device).float()  # 确保 float
        with torch.no_grad():
            noise_pred = model(tensor_in)  # 预测噪声，shape: [1, 1, H, W]

        # === 修复开始：将 noise_pred 转为 numpy 并 squeeze ===
        noise_pred_np = noise_pred.squeeze(0).squeeze(0).cpu().numpy()  # 从 [1,1,H,W] -> [H,W]
        # === 修复结束 ===

        denoised_norm = noisy_norm - noise_pred_np  # 现在都是 numpy.ndarray，可以相减
        denoised = ArpesH5Dataset._denormalize(denoised_norm, noisy_mean, noisy_std)  # 用 noisy std 反转
        
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
        save_preview(noisy, denoised, clean, extent, preview_path)

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
            }
        )
        print(f"[OK] {path.name}: MAE={mae:.4f}, MSE={mse:.4f}, PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")

    if summaries:
        import csv

        csv_path = output_dir / "inference_metrics.csv"
        ensure_dir(output_dir)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["file", "model_tag", "dataset", "mae", "mse", "psnr", "ssim", "preview"]  # 新增 psnr/ssim
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

    run_inference(config_path, model_path, args.input_glob, output_dir, target_key_override=args.target_key)


if __name__ == "__main__":
    main()