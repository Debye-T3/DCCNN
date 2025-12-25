import torch
import h5py  # HDF5 加载
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim  # 指标
import matplotlib.pyplot as plt

# 配置（从你的 yaml 复制）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cfg = {'kernel_size': 3, 'num_layers': 7}  # CCNN 参数
crop_size = (256, 256)  # 从 yaml

# 加载模型
from modules.models.ccnn import CCNN  # 导入你的模型类
model = CCNN(**model_cfg).to(device)
model.load_state_dict(torch.load('results/models/20251216_164813_layers7_batch6_kernel3_alpha0.1.pt', map_location=device))  # 替换 your_model.pt
model.eval()  # 推理模式
print("Model loaded!")

# 加载测试数据（用 HDF5 直接取一个 slice，避免全载）
h5_path = 'data/converted_h5/MS30013.h5'  # 你的 H5
with h5py.File(h5_path, 'r') as f:
    # 取一个 3D slice（e.g., Thetay=25），crop 到 256x256
    if 'spectrum' in f:  # 参考（干净）
        target_full = f['spectrum'][:, :, 25]  # 中间 slice，形状 e.g., (365,571)
    else:
        target_full = f['raw_channels'][:, :, 25]  # fallback
    noisy_full = f['raw_channels'][:, :, 25] if 'raw_channels' in f else target_full + np.random.poisson(target_full * 0.1)  # 噪声输入

# Crop 到 256x256（中心）
h, w = target_full.shape
ch = (h - 256) // 2
cw = (w - 256) // 2
target = target_full[ch:ch+256, cw:cw+256].astype(np.float32)
noisy = noisy_full[ch:ch+256, cw:cw+256].astype(np.float32)

# 转 tensor (1,1,H,W)
noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device) / 255.0  # 规范 [0,1] 如果需
target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(device) / 255.0

# 推理
with torch.no_grad():
    denoised_tensor = model(noisy_tensor)
    denoised = denoised_tensor.squeeze().cpu().numpy() * 255.0  # 反规范

# 量化指标
psnr_val = psnr(target, denoised, data_range=target.max() - target.min())
ssim_val = ssim(target, denoised, data_range=target.max() - target.min())
print(f"PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

# 可视化
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im0 = axs[0].imshow(noisy, cmap='hot', aspect='auto'); axs[0].set_title('Noisy Input')
im1 = axs[1].imshow(denoised, cmap='hot', aspect='auto'); axs[1].set_title('Denoised')
im2 = axs[2].imshow(target, cmap='hot', aspect='auto'); axs[2].set_title('Reference')
plt.colorbar(im0, ax=axs[0]); plt.colorbar(im1, ax=axs[1]); plt.colorbar(im2, ax=axs[2])
plt.tight_layout()
plt.savefig('results/comparison_alpha0.1.png', dpi=150)
plt.show()
print("Comparison saved as 'results/comparison_alpha0.1.png'")