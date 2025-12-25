import torch
from modules.models.ccnn import CCNN

# v2 .pt 路径
pt_path = 'results_v2/models/20251216_190449_layers8_batch6_kernel3_alpha0.05.pt'

# 加载 state_dict
state_dict = torch.load(pt_path, map_location='cpu')
print("State dict keys (first 20):", list(state_dict.keys())[:20])
print("Total keys:", len(state_dict))

# 找 layers 键
layers_keys = [k for k in state_dict if 'layers.' in k]
print("Layers keys sample:", layers_keys[:10])

# 修: 提取 index (split('.')[1])
layer_indices = [int(k.split('.')[1]) for k in layers_keys]
print("Layer indices:", sorted(set(layer_indices)))
print("Max layer index:", max(layer_indices))