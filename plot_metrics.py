import pandas as pd
import matplotlib.pyplot as plt

# 替换为你的实际 CSV 文件路径
csv_file = 'results/csv/20251216_164813_layers7_batch6_kernel3_alpha0.1.csv'  # e.g., 'results/csv/20251216_174032_layers7_batch6_kernel3_alpha0.1.csv'

# 加载 CSV
df = pd.read_csv(csv_file)
print(df.head())  # 先打印前几行，检查数据（Epoch, Loss, Val-Loss 等）

# 绘制 Loss 曲线
plt.figure(figsize=(12, 4))

# 子图1: Loss
plt.subplot(1, 2, 1)
plt.plot(df['Epoch'], df['Loss'], label='Train Loss', marker='o', linewidth=2)
plt.plot(df['Epoch'], df['Val-Loss'], label='Val Loss', marker='s', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: MS-SSIM（如果列存在；否则注释掉）
if 'Msssim' in df.columns and 'Val-Msssim' in df.columns:
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Msssim'], label='Train MS-SSIM', marker='o', linewidth=2)
    plt.plot(df['Epoch'], df['Val-Msssim'], label='Val MS-SSIM', marker='s', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MS-SSIM')
    plt.title('Training & Validation MS-SSIM')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/loss_analysis.png', dpi=150, bbox_inches='tight')  # 保存高清 PNG
plt.show()  # 显示图（如果终端支持）
print("Plot saved as 'results/loss_analysis.png'")