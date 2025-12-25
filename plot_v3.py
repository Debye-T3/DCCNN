import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# v3 CSV 路径 (替换为你的实际)
csv_file = 'results_v3/csv/20251217_110239_layers7_batch6_kernel3_alpha0.1.csv'  # e.g., '20251217_110239_layers7_batch6_kernel3_alpha0.1.csv'

# 提取目录 (results_v3)
csv_dir = Path(csv_file).parent.parent  # csv/ 的父 = results_v3
plot_dir = csv_dir / 'loss_plots'  # 新子文件夹 results_v3/loss_plots/
plot_dir.mkdir(exist_ok=True)

# 加载 CSV
df = pd.read_csv(csv_file)
print(df.head())  # 检查列

# 画 Loss & MS-SSIM 曲线
plt.figure(figsize=(12, 4))

# 子图1: Loss
plt.subplot(1, 2, 1)
plt.plot(df['Epoch'], df['Loss'], label='Train Loss', marker='o', linewidth=2)
plt.plot(df['Epoch'], df['Val-Loss'], label='Val Loss', marker='s', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('v3 Training & Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: MS-SSIM
if 'Msssim' in df.columns and 'Val-Msssim' in df.columns:
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Msssim'], label='Train MS-SSIM', marker='o', linewidth=2)
    plt.plot(df['Epoch'], df['Val-Msssim'], label='Val MS-SSIM', marker='s', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MS-SSIM')
    plt.title('v3 Training & Validation MS-SSIM')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()

# 保存到 results_v3/loss_plots/
plot_path = plot_dir / 'loss_analysis_v3.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"v3 Loss plot saved to {plot_path}")