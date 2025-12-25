import torch
import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cfg = {'kernel_size': 3, 'num_layers': 7}
crop_size = (256, 256)

# 加载模型（替换为你的实际 .pt 全名）
from modules.models.ccnn import CCNN
model = CCNN(**model_cfg).to(device)
model.load_state_dict(torch.load('results/models/20251216_164813_layers7_batch6_kernel3_alpha0.1.pt', map_location=device))  # 替换为你的实际 .pt 全名
model.eval()
print("Model loaded!")

# 加载测试数据
h5_path = 'data/converted_h5/MS30013.h5'
with h5py.File(h5_path, 'r') as f:
    print("HDF5 keys:", list(f.keys()))
    
    # 加载 raw_channels (3D) — 取 channel=0 作为 2D noisy
    if 'raw_channels' in f:
        raw_3d = np.array(f['raw_channels'][:])
        noisy_full = raw_3d[0]  # 第一个 channel
        print("Raw 3D shape:", raw_3d.shape, "→ Noisy 2D:", noisy_full.shape)
        print("Noisy min/max:", noisy_full.min(), noisy_full.max())  # debug 范围
    else:
        print("[WARN] No 'raw_channels'; simulating")
        spectrum = np.array(f['spectrum'][:])
        noisy_full = spectrum + np.random.poisson(spectrum * 0.1).astype(np.float32)

    # 加载 spectrum (2D) — 转置匹配 noisy
    target_full = np.array(f['spectrum'][:])
    if target_full.shape != noisy_full.shape:
        target_full = np.transpose(target_full)
        print("Transposed target to match noisy:", target_full.shape)
    print("Target shape:", target_full.shape)
    print("Target min/max:", target_full.min(), target_full.max())  # debug

# 安全 Crop/Pad
h, w = target_full.shape
print(f"Full dimensions: H={h}, W={w}")

if min(h, w) < 256:
    print("[INFO] Padding smaller dim")
    if h < w:
        pad_h = (256 - h) // 2
        noisy = np.pad(noisy_full, ((pad_h, 256 - h - pad_h), (0, 0)), mode='constant', constant_values=0)
        target = np.pad(target_full, ((pad_h, 256 - h - pad_h), (0, 0)), mode='constant', constant_values=0)
    else:
        pad_w = (256 - w) // 2
        noisy = np.pad(noisy_full, ((0, 0), (pad_w, 256 - w - pad_w)), mode='constant', constant_values=0)
        target = np.pad(target_full, ((0, 0), (pad_w, 256 - w - pad_w)), mode='constant', constant_values=0)
else:
    ch_start = max(0, (h - 256) // 2)
    ch_end = min(h, ch_start + 256)
    cw_start = max(0, (w - 256) // 2)
    cw_end = min(w, cw_start + 256)
    noisy = noisy_full[ch_start:ch_end, cw_start:cw_end]
    target = target_full[ch_start:ch_end, cw_start:cw_end]
    print(f"Cropped: H={noisy.shape[0]}, W={noisy.shape[1]}")

if noisy.shape[0] == 0 or noisy.shape[1] == 0:
    raise ValueError(f"Crop failed: shape {noisy.shape}")

noisy = noisy.astype(np.float32)
target = target.astype(np.float32)
print(f"Final shapes: Noisy {noisy.shape}, Target {target.shape}")

# Norm to [0,1] (MinMax, 基于数据范围)
noisy_min, noisy_max = noisy.min(), noisy.max()
target_min, target_max = target.min(), target.max()
range_noisy = noisy_max - noisy_min if noisy_max > noisy_min else 1
range_target = target_max - target_min if target_max > target_min else 1
noisy_norm = (noisy - noisy_min) / range_noisy
target_norm = (target - target_min) / range_target
print(f"Noisy range: [{noisy_min:.1f}, {noisy_max:.1f}] → Norm [0,1]")

# 转 tensor
noisy_tensor = torch.from_numpy(noisy_norm).unsqueeze(0).unsqueeze(0).to(device)
target_tensor = torch.from_numpy(target_norm).unsqueeze(0).unsqueeze(0).to(device)
print("Tensor shape:", noisy_tensor.shape)

# 推理
with torch.no_grad():
    noise_pred = model(noisy_tensor)  # 预测噪声 (DCCNN 常见)
    denoised_norm = noisy_tensor - noise_pred  # 残差: input - noise
    denoised = denoised_norm.squeeze().cpu().numpy() * range_target + target_min  # 反 norm 到原范围

# 指标 (原范围)
psnr_val = psnr(target, denoised, data_range=range_target)
ssim_val = ssim(target, denoised, data_range=range_target)
print(f"PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

# 可视化 (原范围)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im0 = axs[0].imshow(noisy, cmap='hot', aspect='auto'); axs[0].set_title('Noisy Input')
im1 = axs[1].imshow(denoised, cmap='hot', aspect='auto'); axs[1].set_title('Denoised (residual)')
im2 = axs[2].imshow(target, cmap='hot', aspect='auto'); axs[2].set_title('Reference')
plt.colorbar(im0, ax=axs[0]); plt.colorbar(im1, ax=axs[1]); plt.colorbar(im2, ax=axs[2])
plt.suptitle(f'Comparison (residual) | PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}')
plt.tight_layout()
plt.savefig('results/comparison_residual.png', dpi=150)
plt.show()
print("Saved 'results/comparison_residual.png'")