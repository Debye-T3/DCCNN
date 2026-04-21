import os
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir):

        self.high_res_dir = high_res_dir    # Folder of high_res_images
        self.low_res_dir = low_res_dir      # Folder of low_res_images

        self.high_res_image_list = [f for f in os.listdir(high_res_dir) if f.endswith(".npy")]   # List of high_res_images

    def __len__(self):
        return len(self.high_res_image_list) * 50 # There exist 50 low_res_images for every high_res_image. TODO: This only works for a hardcoded fix number. Needs Generalization

    def __getitem__(self, i):
        high_res_image_i = i // 50   # Index for finding the right high_res_image 
        low_res_image_i = i % 50    # Index for the low_res_image of the high_res_image

        high_res_path = os.path.join(self.high_res_dir, self.high_res_image_list[high_res_image_i])
        low_res_path = os.path.join(self.low_res_dir, self.high_res_image_list[high_res_image_i].replace(".npy", "_gen.npy")) # low_res_images have a _gen at the end of the name. TODO: This only works for this name. Needs Generalization

        high_res_image_np = np.load(high_res_path)

        low_res_image_stack_np = np.load(low_res_path)  # The low_res_images are in a stack. Thats why you have to pick one. [x, y, image_index]
        low_res_image_np = low_res_image_stack_np[:, :, low_res_image_i]

        return (
            torch.from_numpy(low_res_image_np).float().unsqueeze(0),
            torch.from_numpy(high_res_image_np).float().unsqueeze(0),
        ) # Ready to use for PyTorch!


class ArpesH5Dataset(Dataset):
    """
    Dataset that reads converted ARPES HDF5 files (see convert_arpes_to_h5.py) and
    generates random crops for denoising training.
    """

    def __init__(
        self,
        file_paths: Sequence[Path],
        samples_per_file: int = 128,
        crop_size: Optional[Tuple[int, int]] = (256, 256),
        input_key: str = "raw_channels",
        target_key: str = "spectrum",
        input_channel: int = 0,
        cache_in_memory: bool = True,
        strict: bool = True,
        normalize: bool = False,  # 新增: normalize 标志 (log1p + z-score)
        add_noise: bool = False,
        noise_std: float = 0.0,
        stripe_noise_prob: float = 0.0,
        stripe_strength: float = 0.0,
        blur_prob: float = 0.0,
        blur_sigma: float = 0.0,
        noise_energy_threshold: float = 0.0,
        noise_min_run: int = 0,
        noise_energy_method: str = "max_ratio",
        noise_energy_percentile: float = 5.0,
        noise_energy_smooth: int = 0,
    ) -> None:
        candidate_paths = [Path(p) for p in file_paths]
        if not candidate_paths:
            raise ValueError("ArpesH5Dataset: no input files found.")

        validated_paths: List[Path] = []
        missing: List[Path] = []
        for path in candidate_paths:
            try:
                with h5py.File(path, "r") as handle:
                    if target_key not in handle or input_key not in handle:
                        missing.append(path)
                    else:
                        validated_paths.append(path)
            except OSError:
                missing.append(path)

        if strict and missing:
            raise KeyError(
                "ArpesH5Dataset: required datasets missing in files: "
                + ", ".join(str(p) for p in missing)
            )
        if not validated_paths:
            raise ValueError(
                "ArpesH5Dataset: no files contain both "
                f"'{input_key}' and '{target_key}'."
            )
        if missing:
            print(
                "[WARN] Skipping files without required datasets: "
                + ", ".join(str(p) for p in missing)
            )
        self.file_paths = validated_paths

        self.samples_per_file = max(1, int(samples_per_file))
        self.crop_size = crop_size
        self.input_key = input_key
        self.target_key = target_key
        self.input_channel = input_channel
        self.cache_in_memory = cache_in_memory
        self.normalize = normalize  # 新增: normalize 标志
        self.add_noise = add_noise
        self.noise_std = max(0.0, float(noise_std))
        self.stripe_noise_prob = max(0.0, float(stripe_noise_prob))
        self.stripe_strength = max(0.0, float(stripe_strength))
        self.blur_prob = max(0.0, float(blur_prob))
        self.blur_sigma = max(0.0, float(blur_sigma))
        self.noise_energy_threshold = max(0.0, float(noise_energy_threshold))
        self.noise_min_run = max(0, int(noise_min_run))
        self.noise_energy_method = str(noise_energy_method)
        self.noise_energy_percentile = float(noise_energy_percentile)
        self.noise_energy_smooth = max(0, int(noise_energy_smooth))
        self._cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
        self._energy_cache: Dict[Path, np.ndarray] = {}
        self._energy_mask_cache: Dict[Path, np.ndarray] = {}
        self._energy_mask_axis: Dict[Path, int] = {}
        self._length = self.samples_per_file * len(self.file_paths)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        file_idx = index // self.samples_per_file
        file_path = self.file_paths[file_idx]

        spectrum, noisy = self._load_file(file_path)

        rng = np.random.default_rng(seed=index)
        noisy_patch, origin = self._sample_patch(noisy, rng)
        clean_patch, _ = self._sample_patch(spectrum, rng, origin=origin)

        mask = self._get_energy_mask(file_path, spectrum, origin, noisy_patch.shape)

        if self.add_noise and (self.noise_std > 0.0 or self.stripe_strength > 0.0 or self.blur_sigma > 0.0):
            noisy_patch = self._add_noise(
                noisy_patch,
                rng,
                self.noise_std,
                self.stripe_noise_prob,
                self.stripe_strength,
                self.blur_prob,
                self.blur_sigma,
                mask=mask,
            )

        # 新增: 应用 normalize (训时 [0,1]，一致性)
        if self.normalize:
            noisy_patch = self._normalize(noisy_patch)
            clean_patch = self._normalize(clean_patch)

        noisy_tensor = torch.from_numpy(noisy_patch).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_patch).unsqueeze(0)
        return noisy_tensor, clean_tensor

    def _load_file(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_in_memory and path in self._cache:
            return self._cache[path]

        with h5py.File(path, "r") as handle:
            spectrum = np.asarray(handle[self.target_key], dtype=np.float32)
            if self.input_key not in handle:
                raise KeyError(f"{path}: dataset '{self.input_key}' not found.")
            noisy_data = np.asarray(handle[self.input_key], dtype=np.float32)
            energy = None
            if "energy" in handle:
                energy = np.asarray(handle["energy"], dtype=np.float32)

        if noisy_data.ndim == 3:
            if not (0 <= self.input_channel < noisy_data.shape[0]):
                raise IndexError(
                    f"{path}: input_channel {self.input_channel} out of range for "
                    f"dataset '{self.input_key}' with shape {noisy_data.shape}."
                )
            noisy_data = noisy_data[self.input_channel]

        if noisy_data.shape != spectrum.shape:
            if noisy_data.T.shape == spectrum.shape:
                noisy_data = noisy_data.T
            else:
                raise ValueError(
                    f"{path}: input ({noisy_data.shape}) and target ({spectrum.shape}) shapes mismatch."
                )

        # 移到 __getitem__ (per-patch norm 更好，避免全文件范围)
        if energy is not None and path not in self._energy_cache:
            self._energy_cache[path] = energy
        if self.cache_in_memory:
            self._cache[path] = (spectrum, noisy_data)
        return spectrum, noisy_data

    def _sample_patch(
        self,
        array: np.ndarray,
        rng: np.random.Generator,
        origin: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        height, width = array.shape
        if not self.crop_size:
            return array, (0, 0)

        crop_h = min(self.crop_size[0], height)
        crop_w = min(self.crop_size[1], width)
        if origin is None:
            top = 0 if height == crop_h else int(rng.integers(0, height - crop_h + 1))
            left = 0 if width == crop_w else int(rng.integers(0, width - crop_w + 1))
        else:
            top, left = origin
        patch = array[top : top + crop_h, left : left + crop_w]
        return patch, (top, left)

    @staticmethod
    def _normalize(array: np.ndarray, return_stats: bool = False):
        """Log1p + z-score normalization."""
        arr = np.array(array, dtype=np.float32, copy=True)
        arr = np.log1p(np.clip(arr, a_min=0.0, a_max=None))
        mean = float(arr.mean())
        std = float(arr.std() + 1e-6)
        arr = (arr - mean) / std
        if return_stats:
            return arr, mean, std
        return arr

    @staticmethod
    def _denormalize(array: np.ndarray, mean: float, std: float) -> np.ndarray:
        return np.expm1(array * std + mean)

    @staticmethod
    def _add_noise(
        array: np.ndarray,
        rng: np.random.Generator,
        noise_std: float,
        stripe_noise_prob: float,
        stripe_strength: float,
        blur_prob: float,
        blur_sigma: float,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        patch = np.array(array, dtype=np.float32, copy=True)
        sigma = float(patch.std() + 1e-6) * noise_std

        if sigma > 0.0:
            noise = rng.normal(0.0, sigma, size=patch.shape).astype(np.float32)
            if mask is not None:
                noise = noise * mask
            patch = patch + noise

        if stripe_noise_prob > 0.0 and stripe_strength > 0.0:
            if float(rng.random()) < stripe_noise_prob:
                patch = ArpesH5Dataset._add_stripe_noise(patch, rng, stripe_strength, mask=mask)

        if blur_prob > 0.0 and blur_sigma > 0.0:
            if float(rng.random()) < blur_prob:
                patch = ArpesH5Dataset._apply_blur(patch, blur_sigma, mask=mask)

        return patch

    @staticmethod
    def _add_stripe_noise(
        array: np.ndarray,
        rng: np.random.Generator,
        strength: float,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        patch = np.array(array, dtype=np.float32, copy=True)
        sigma = float(patch.std() + 1e-6) * strength
        if sigma <= 0.0:
            return patch
        height, width = patch.shape
        if float(rng.random()) < 0.5:
            stripe = rng.normal(0.0, sigma, size=(height, 1)).astype(np.float32)
        else:
            stripe = rng.normal(0.0, sigma, size=(1, width)).astype(np.float32)
        if mask is not None:
            stripe = stripe * mask
        return patch + stripe

    @staticmethod
    def _apply_blur(array: np.ndarray, sigma: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
        patch = np.array(array, dtype=np.float32, copy=True)
        if sigma <= 0.0:
            return patch
        radius = max(1, int(round(sigma * 2)))
        size = radius * 2 + 1
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel_1d = np.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d /= float(kernel_1d.sum())
        kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
        pad = radius
        padded = np.pad(patch, pad, mode="edge")
        out = np.zeros_like(patch)
        for i in range(size):
            for j in range(size):
                out += kernel[i, j] * padded[i : i + patch.shape[0], j : j + patch.shape[1]]
        if mask is not None:
            out = patch * (1.0 - mask) + out * mask
        return out

    def _get_energy_mask(
        self,
        path: Path,
        spectrum: np.ndarray,
        origin: Tuple[int, int],
        patch_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if self.noise_energy_threshold <= 0.0 and self.noise_energy_method != "percentile":
            return None

        mask_1d = self._energy_mask_cache.get(path)
        axis = self._energy_mask_axis.get(path, 0)
        if mask_1d is None:
            energy = self._energy_cache.get(path)
            mask_1d, axis = self._compute_energy_mask(
                spectrum=spectrum,
                energy=energy,
                threshold_ratio=self.noise_energy_threshold,
                min_run=self.noise_min_run,
                method=self.noise_energy_method,
                percentile=self.noise_energy_percentile,
                smooth_window=self.noise_energy_smooth,
            )
            self._energy_mask_cache[path] = mask_1d
            self._energy_mask_axis[path] = axis

        if axis == 0:
            top = origin[0]
            rows = mask_1d[top : top + patch_shape[0]]
            return np.repeat(rows[:, None], patch_shape[1], axis=1).astype(np.float32)
        left = origin[1]
        cols = mask_1d[left : left + patch_shape[1]]
        return np.repeat(cols[None, :], patch_shape[0], axis=0).astype(np.float32)

    @staticmethod
    def _compute_energy_mask(
        spectrum: np.ndarray,
        energy: Optional[np.ndarray],
        threshold_ratio: float,
        min_run: int,
        method: str,
        percentile: float,
        smooth_window: int,
    ) -> Tuple[np.ndarray, int]:
        axis = 0
        if energy is not None and energy.ndim == 1:
            if energy.size == spectrum.shape[1]:
                axis = 1

        if threshold_ratio <= 0.0 and method != "percentile":
            length = spectrum.shape[axis]
            return np.ones(length, dtype=np.float32), axis

        profile = spectrum.mean(axis=1 - axis)
        if smooth_window and smooth_window > 1:
            win = max(3, int(smooth_window))
            if win % 2 == 0:
                win += 1
            kernel = np.ones(win, dtype=np.float32) / float(win)
            profile = np.convolve(profile, kernel, mode="same")
        max_val = float(profile.max())
        if max_val <= 0.0:
            return np.ones_like(profile, dtype=np.float32), axis

        if method == "percentile":
            threshold = float(np.percentile(profile, percentile))
        else:
            threshold = max_val * threshold_ratio
        below = profile < threshold
        run_len = max(1, int(min_run))

        high_at_start = False
        if energy is not None and energy.ndim == 1 and energy.size == profile.size:
            high_at_start = energy[0] > energy[-1]

        mask = np.ones_like(profile, dtype=np.float32)
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
                mask[: cutoff + 1] = 0.0
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
                mask[cutoff:] = 0.0

        return mask, axis


def build_dataset_from_config(
    path_cfg: Dict[str, Any],
    data_cfg: Optional[Dict[str, Any]] = None,
) -> Dataset:
    data_cfg = data_cfg or {}
    crop_size = data_cfg.pop("crop_size", None)
    if crop_size is None:
        crop_h = data_cfg.pop("crop_height", None)
        crop_w = data_cfg.pop("crop_width", None)
        if crop_h is not None and crop_w is not None:
            crop_size = (int(crop_h), int(crop_w))
    elif isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_size = (int(crop_size[0]), int(crop_size[1]))
    else:
        crop_size = None

    if "h5_glob" in path_cfg or "h5_files" in path_cfg:
        if "h5_files" in path_cfg:
            file_paths: List[str] = [str(p) for p in path_cfg["h5_files"]]
        else:
            file_paths = sorted(glob.glob(path_cfg["h5_glob"]))
        if not file_paths:
            raise FileNotFoundError(
                f"No HDF5 files found for pattern '{path_cfg.get('h5_glob', path_cfg.get('h5_files'))}'"
            )
        return ArpesH5Dataset(
            file_paths=file_paths,
            crop_size=crop_size,
            samples_per_file=data_cfg.get("samples_per_file", 128),
            input_key=data_cfg.get("input_key", "raw_channels"),
            target_key=data_cfg.get("target_key", "spectrum"),
            input_channel=data_cfg.get("input_channel", 0),
            cache_in_memory=data_cfg.get("cache_in_memory", True),
            strict=data_cfg.get("strict", False),
            normalize=data_cfg.get("normalize", False),  # 新增: 传 normalize 标志
            add_noise=data_cfg.get("add_noise", False),
            noise_std=data_cfg.get("noise_std", 0.0),
            stripe_noise_prob=data_cfg.get("stripe_noise_prob", 0.0),
            stripe_strength=data_cfg.get("stripe_strength", 0.0),
            blur_prob=data_cfg.get("blur_prob", 0.0),
            blur_sigma=data_cfg.get("blur_sigma", 0.0),
            noise_energy_threshold=data_cfg.get("noise_energy_threshold", 0.0),
            noise_min_run=data_cfg.get("noise_min_run", 0),
            noise_energy_method=data_cfg.get("noise_energy_method", "max_ratio"),
            noise_energy_percentile=data_cfg.get("noise_energy_percentile", 5.0),
            noise_energy_smooth=data_cfg.get("noise_energy_smooth", 0),
        )

    # Fallback to legacy .npy dataset
    return CustomDataset(
        path_cfg["high_res_dir"],
        path_cfg["low_res_dir"],
    )
