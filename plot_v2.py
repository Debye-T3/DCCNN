import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# v2 CSV 路径
csv_file = 'results_v2/csv/20251216_190449_layers8_batch6_kernel3_alpha0.05.csv'

# 加载 CSV
df = pd.read_csv(csv_file)
print(df.head())  # 检查列

# 创建 lr_finder 目录
lr_dir = Path('results_v2/lr_finder')
lr_dir.mkdir(parents=True, exist_ok=True)

# 画 LR 曲线 (log scale)
plt.figure(figsize=(10, 4))
plt.plot(df['Epoch'], df['Learning-Rate'], marker='o', linewidth=2, color='blue')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')  # log scale
plt.title('v2 Learning Rate Plot (alpha=0.05, layers=8)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存到 results_v2/lr_finder/
plot_path = lr_dir / 'lr_plot_v2.png'
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"v2 LR plot saved to {plot_path}")