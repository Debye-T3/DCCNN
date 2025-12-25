import h5py
import numpy as np

with h5py.File('data/sample_arpes.h5', 'r') as f:
    spec = f['spectrum'][:]
    print('Shape:', spec.shape)
    print('Min/Max/Mean/Std:', np.min(spec), np.max(spec), np.mean(spec), np.std(spec))
    print('Slice 25 (中Thetay) Max/Mean:', np.max(spec[:, :, 25]), np.mean(spec[:, :, 25]))
    print('Sample peaks (top 5):', np.sort(spec.flatten())[-5:])  # 最高5个值