# ARPES Data Converter GUI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PySide6 GUI application that converts ARPES raw data (.txt, .pxt, .bin) to HDF5 with experimental parameter entry, SES header auto-extraction, and optional k-space preview.

**Architecture:** Two packages — `converter/` (backend: readers, writer, preview, engine) and `gui/` (frontend: 4 tabs in a QTabWidget). Backend has zero UI dependency; GUI calls ConversionEngine via a QThread worker for non-blocking conversion.

**Tech Stack:** PySide6, h5py, numpy, matplotlib (all already in project deps except PySide6)

---

## File Structure

```
converter_app.py              # Create — entry point
run.bat                       # Create — double-click launcher
converter/
├── __init__.py               # Create
├── engine.py                 # Create — ConversionEngine + ConvertWorker(QThread)
├── readers/
│   ├── __init__.py           # Create
│   ├── txt_reader.py         # Create — refactored from convert_txt_to_h5.py
│   └── pxt_reader.py         # Create — refactored from convert_arpes_to_h5.py
├── writer.py                 # Create — HDF5 writer with metadata attrs
└── preview.py                # Create — preview generator + k-space conversion
gui/
├── __init__.py               # Create
├── main_window.py            # Create — QMainWindow + QTabWidget shell
├── file_tab.py               # Create — Tab 1: file selection
├── params_tab.py             # Create — Tab 2: batch defaults + override table
├── preview_tab.py            # Create — Tab 3: matplotlib preview + controls
└── convert_tab.py            # Create — Tab 4: progress, log, output dir
```

No existing files are modified — the old scripts (`convert_txt_to_h5.py`, `convert_arpes_to_h5.py`) are left in place. The new readers re-implement their parsing logic cleanly.

---

### Task 1: Create directory structure and install PySide6

**Files:**
- Create: `converter/__init__.py`, `converter/readers/__init__.py`, `gui/__init__.py`

- [ ] **Step 1: Create all directories**

```bash
mkdir -p converter/readers gui
```

- [ ] **Step 2: Create package init files**

`converter/__init__.py`:
```python
"""ARPES data converter backend — format readers, HDF5 writer, preview generator."""
```

`converter/readers/__init__.py`:
```python
"""Format-specific parsers for ARPES raw data files."""
```

`gui/__init__.py`:
```python
"""PySide6 GUI for the ARPES data converter."""
```

- [ ] **Step 3: Install PySide6**

```bash
pip install PySide6
```
Expected: PySide6 installed successfully

- [ ] **Step 4: Commit**

```bash
git add converter/ gui/ && git commit -m "chore: create converter and gui package structure"
```

---

### Task 2: Implement txt_reader.py

**Files:**
- Create: `converter/readers/txt_reader.py`

- [ ] **Step 1: Write the module**

```python
"""Scienta DA30L .txt export parser."""

import re
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_axes(lines: list) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Parse dimension sizes and axes from DA30L txt header."""

    def _find(prefix: str) -> str:
        for line in lines:
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip()
        raise ValueError(f"Missing '{prefix}' in txt header.")

    n_energy = int(_find("Dimension 1 size"))
    n_angle = int(_find("Dimension 2 size"))
    energy_axis = np.fromstring(_find("Dimension 1 scale"), sep=" ")
    angle_axis = np.fromstring(_find("Dimension 2 scale"), sep=" ")
    if energy_axis.size != n_energy:
        raise ValueError(f"Energy axis length {energy_axis.size} != {n_energy}")
    if angle_axis.size != n_angle:
        raise ValueError(f"Angle axis length {angle_axis.size} != {n_angle}")
    return energy_axis.astype(np.float32), angle_axis.astype(np.float32), n_energy, n_angle


def parse_data(lines: list, start_idx: int, n_energy: int, n_angle: int) -> np.ndarray:
    """Parse numeric data rows starting at start_idx."""
    data_rows = []
    for line in lines[start_idx:]:
        if not line.strip():
            continue
        nums = np.fromstring(line, sep=" ")
        if nums.size == 0:
            continue
        if nums.size == n_angle + 1:
            nums = nums[1:]
        elif nums.size > n_angle + 1:
            nums = nums[-n_angle:]
        elif nums.size < n_angle:
            nums = np.pad(nums, (0, n_angle - nums.size), mode="constant", constant_values=0)
        data_rows.append(nums)
        if len(data_rows) >= n_energy:
            break
    if len(data_rows) != n_energy:
        missing = n_energy - len(data_rows)
        if missing > 0:
            pad_row = np.zeros((missing, n_angle), dtype=np.float32)
            data_arr = np.vstack([data_rows, pad_row])
        else:
            data_arr = np.stack(data_rows[:n_energy])
    else:
        data_arr = np.stack(data_rows)
    return data_arr.astype(np.float32)


def extract_ses_params(lines: list) -> dict:
    """Extract all SES/DA30L parameters from the header lines into a dict."""
    params = {}
    for line in lines:
        if "=" in line and not line.strip()[0].isdigit():
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and value:
                params[key] = value
    return params


def read_txt(path: Path) -> dict:
    """Read a DA30L .txt file and return spectrum + axes + SES params.

    Returns dict with keys: spectrum (2D float32), energy (1D float32),
    thetax (1D float32), ses_params (dict of str).
    """
    txt_str = path.read_text(encoding="utf-8", errors="ignore")
    lines = txt_str.splitlines()

    numeric_re = re.compile(r"^[0-9eE+\-.\s]+$")
    start_idx = None
    for i, line in enumerate(lines):
        if numeric_re.match(line.strip()) and len(line.split()) > 5:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Could not find numeric data block in {path}")

    energy_axis, angle_axis, n_energy, n_angle = parse_axes(lines)
    spectrum = parse_data(lines, start_idx + 1, n_energy, n_angle)
    ses_params = extract_ses_params(lines[:start_idx])

    return {
        "spectrum": spectrum,
        "energy": energy_axis,
        "thetax": angle_axis,
        "ses_params": ses_params,
    }
```

- [ ] **Step 2: Verify with a quick smoke test**

```bash
python -c "
from pathlib import Path
from converter.readers.txt_reader import read_txt
result = read_txt(Path('data/txtdata/MS30013.txt'))
assert result['spectrum'].ndim == 2
assert result['spectrum'].dtype.name == 'float32'
assert result['energy'].ndim == 1
assert result['thetax'].ndim == 1
assert len(result['ses_params']) > 5
print('OK - shape:', result['spectrum'].shape, 'energy:', result['energy'].shape, 'SES keys:', len(result['ses_params']))
"
```
Expected: `OK - shape: (1221, 463) energy: (1221,) SES keys: ...`

- [ ] **Step 3: Commit**

```bash
git add converter/readers/txt_reader.py && git commit -m "feat: add DA30L .txt reader with SES header extraction"
```

---

### Task 3: Implement pxt_reader.py

**Files:**
- Create: `converter/readers/pxt_reader.py`

- [ ] **Step 1: Write the module**

```python
"""Scienta PXT binary and raw .bin cube parser."""

import math
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_bin(path: Path, shape: Tuple[int, ...], dtype: str) -> np.ndarray:
    """Load a raw binary cube with given shape and dtype."""
    dtype_obj = np.dtype(dtype)
    expected_bytes = math.prod(shape) * dtype_obj.itemsize
    actual_bytes = path.stat().st_size
    if expected_bytes != actual_bytes:
        raise ValueError(
            f"{path}: size mismatch. Expected {expected_bytes} bytes, got {actual_bytes}."
        )
    data = np.fromfile(path, dtype=dtype_obj)
    return data.reshape(shape)


def read_pxt(
    path: Path,
    *,
    energy_offset_override: Optional[float] = None,
    energy_step_override: Optional[float] = None,
    angle_offset_override: Optional[float] = None,
    angle_step_override: Optional[float] = None,
    channel: int = 0,
    subtract_dark: bool = False,
) -> dict:
    """Read a Scienta PXT binary file.

    Returns dict with keys: spectrum (2D float32), energy (1D float32),
    thetax (1D float32), ses_params (dict), raw_channels (3D float32 array
    [channels, H, W]).
    """
    raw = path.read_bytes()
    if len(raw) < 256:
        raise ValueError(f"{path}: file too small to be a valid PXT container.")

    def _uint(idx: int) -> int:
        return struct.unpack_from("<I", raw, idx * 4)[0]

    def _double(idx: int) -> float:
        return struct.unpack_from("<d", raw, idx * 4)[0]

    total_points = _uint(21)
    channel_count = max(1, _uint(22))
    frame_type_bytes = raw[25 * 4: 27 * 4]
    frame_type = frame_type_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore") or "unknown"

    width = _uint(35)
    height = _uint(36)
    if width == 0 or height == 0:
        raise ValueError(f"{path}: reported shape {width}x{height} is invalid.")

    energy_step_raw = _double(39)
    angle_step_raw = _double(41)
    energy_offset_raw = _double(47)
    angle_offset_raw = _double(49)

    energy_step = energy_step_override if energy_step_override is not None else energy_step_raw
    angle_step = angle_step_override if angle_step_override is not None else angle_step_raw
    energy_offset = energy_offset_override if energy_offset_override is not None else energy_offset_raw
    angle_offset = angle_offset_override if angle_offset_override is not None else angle_offset_raw

    itemsize = np.dtype("<i2").itemsize
    data_bytes = width * height * channel_count * itemsize
    header_bytes = len(raw) - data_bytes
    if header_bytes < 0:
        raise ValueError(f"{path}: negative header size computed.")

    payload = np.frombuffer(
        raw, dtype="<i2", count=width * height * channel_count, offset=header_bytes
    )
    payload = payload.reshape(height, width, channel_count)

    chosen_channel = channel
    if channel < 0:
        pos_means = []
        for ch in range(channel_count):
            ch_data = payload[..., ch].astype(np.float32)
            pos_means.append(float(np.mean(np.clip(ch_data, a_min=0.0, a_max=None))))
        chosen_channel = int(np.argmax(pos_means))

    if not 0 <= chosen_channel < channel_count:
        raise ValueError(
            f"{path}: channel {chosen_channel} out of range ({channel_count} channels)."
        )

    signal = payload[..., chosen_channel].astype(np.float32)
    subtracted_from = None
    if subtract_dark and channel_count > 1:
        dark_idx = 1 if chosen_channel == 0 else (chosen_channel - 1)
        if 0 <= dark_idx < channel_count:
            signal = signal - payload[..., dark_idx].astype(np.float32)
            subtracted_from = dark_idx

    signal = np.clip(signal, a_min=0.0, a_max=None)
    spectrum = signal.T.copy()

    energy_axis = (np.arange(width, dtype=np.float32) * energy_step + energy_offset).astype(np.float32)
    angle_axis = (np.arange(height, dtype=np.float32) * angle_step + angle_offset).astype(np.float32)

    ses_params = {
        "frame_type": frame_type,
        "channels_total": int(channel_count),
        "channel_used": int(chosen_channel),
        "energy_offset_eV": float(energy_offset_raw),
        "energy_step_eV": float(energy_step_raw),
        "angle_offset_deg": float(angle_offset_raw),
        "angle_step_deg": float(angle_step_raw),
        "total_points": int(total_points),
        "width": int(width),
        "height": int(height),
    }
    if subtracted_from is not None:
        ses_params["subtracted_channel"] = int(subtracted_from)

    raw_channels = payload.transpose(2, 0, 1).copy()

    return {
        "spectrum": spectrum.astype(np.float32, copy=False),
        "energy": energy_axis,
        "thetax": angle_axis,
        "ses_params": ses_params,
        "raw_channels": raw_channels,
    }
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
from pathlib import Path
from converter.readers.pxt_reader import read_pxt
result = read_pxt(Path('data/pxtdata/MS30013.pxt'))
assert result['spectrum'].ndim == 2
assert result['energy'].ndim == 1
assert result['thetax'].ndim == 1
assert result['raw_channels'].ndim == 3
assert 'frame_type' in result['ses_params']
print('OK - shape:', result['spectrum'].shape, 'channels:', result['raw_channels'].shape[0])
"
```
Expected: `OK - shape: (365, 571) channels: 4`

- [ ] **Step 3: Commit**

```bash
git add converter/readers/pxt_reader.py && git commit -m "feat: add PXT binary reader"
```

---

### Task 4: Implement writer.py

**Files:**
- Create: `converter/writer.py`

- [ ] **Step 1: Write the module**

```python
"""HDF5 writer for converted ARPES data with full metadata attributes."""

from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np


def write_h5(
    spectrum: np.ndarray,
    energy: np.ndarray,
    thetax: np.ndarray,
    destination: Path,
    *,
    source_format: str,
    source_path: str,
    manual_params: Dict[str, Any],
    ses_params: Dict[str, Any],
    raw_channels: Optional[np.ndarray] = None,
    overwrite: bool = False,
) -> None:
    """Write spectrum + axes + metadata to HDF5.

    Args:
        spectrum: 2D float32 array [energy, angle]
        energy: 1D float32 energy axis
        thetax: 1D float32 angle axis
        destination: output .h5 path
        source_format: "txt", "pxt", or "bin"
        source_path: original file path string
        manual_params: user-entered parameters dict
        ses_params: auto-extracted SES header parameters dict
        raw_channels: optional [C, H, W] array for multi-channel PXT data
        overwrite: if True, overwrite existing file
    """
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists. Use overwrite=True.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(destination, "w") as f:
        f.create_dataset("spectrum", data=spectrum)
        f.create_dataset("energy", data=energy)
        f.create_dataset("thetax", data=thetax)
        f.create_dataset("raw_channels", data=spectrum[None, ...].astype(np.float32))

        if raw_channels is not None:
            f.create_dataset("raw_channels_all", data=raw_channels)

        f.attrs["source_format"] = source_format
        f.attrs["source_path"] = source_path
        f.attrs["shape"] = str(tuple(int(d) for d in spectrum.shape))

        for key, value in manual_params.items():
            if value is not None and value != "":
                try:
                    f.attrs[key] = value
                except TypeError:
                    f.attrs[key] = str(value)

        for key, value in ses_params.items():
            if isinstance(value, bool):
                f.attrs[key] = int(value)
            elif isinstance(value, (int, float, str, np.integer, np.floating)):
                f.attrs[key] = value
            else:
                try:
                    f.attrs[key] = str(value)
                except Exception:
                    pass
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
import tempfile, os
import numpy as np
from pathlib import Path
from converter.writer import write_h5
import h5py

spec = np.ones((100, 50), dtype=np.float32)
energy = np.arange(100, dtype=np.float32)
angle = np.arange(50, dtype=np.float32)

tmp = Path(tempfile.mktemp(suffix='.h5'))
try:
    write_h5(spec, energy, angle, tmp,
             source_format='txt', source_path='/fake/test.txt',
             manual_params={'sample_name': 'Test', 'temperature_K': 77.0, 'work_function_eV': 4.2},
             ses_params={'pass_energy_eV': 20, 'lens_mode': 'WideAngle'})
    with h5py.File(tmp, 'r') as f:
        assert 'spectrum' in f
        assert 'raw_channels' in f
        assert f.attrs['sample_name'] == 'Test'
        assert f.attrs['temperature_K'] == 77.0
        assert f.attrs['pass_energy_eV'] == 20
    print('OK')
finally:
    os.unlink(tmp)
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add converter/writer.py && git commit -m "feat: add HDF5 writer with metadata attributes"
```

---

### Task 5: Implement preview.py

**Files:**
- Create: `converter/preview.py`

- [ ] **Step 1: Write the module**

```python
"""Preview image generator with k-space conversion support."""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


K_CONSTANT = 0.5123  # sqrt(2m_e) / hbar in eV^{-1/2} * A^{-1}


def compute_contrast(data: np.ndarray, pmin: float, pmax: float) -> Tuple[float, float]:
    """Percentile-based contrast limits."""
    if data.size == 0:
        return 1e-6, 1.0
    positive = data[data > 0]
    if positive.size == 0:
        positive = np.abs(data.ravel())
    vmin = float(np.percentile(positive, pmin)) if positive.size else 0.0
    vmax = float(np.percentile(positive, pmax)) if positive.size else 1.0
    if vmax <= vmin:
        vmax = float(positive.max()) if positive.size else 1.0
        vmin = max(vmax * 1e-3, 1e-6)
    return vmin, vmax


def to_kspace(
    energy_axis: np.ndarray,
    angle_axis: np.ndarray,
    hv: float,
    work_function: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert angle axis to k-parallel momentum axis.

    k_parallel [A^{-1}] = K_CONSTANT * sqrt(E_kin) * sin(theta)
    where E_kin = hv - work_function - E_binding
    and E_binding is referenced such that the Fermi level is at E=0.

    For raw ARPES data where the energy axis may be kinetic energy
    (not binding), we compute E_kin directly from the axis values:
    if values are decreasing (typical for kinetic energy scale),
    E_kin = energy_axis (already in eV). If values appear to be
    binding energy, E_kin = hv - work_function - energy_axis.
    """
    if hv is None or hv <= 0:
        raise ValueError("Photon energy (hv) is required for k-space conversion.")

    # Detect energy axis type: if values increase with index (typical binding
    # energy: low to high), treat as binding energy; if decrease (kinetic:
    # high to low), treat as kinetic.
    if energy_axis.size > 1 and energy_axis[-1] < energy_axis[0]:
        e_kin = energy_axis.astype(np.float64)
    else:
        e_kin = hv - work_function - energy_axis.astype(np.float64)

    e_kin = np.clip(e_kin, 0.01, None)
    theta_rad = np.radians(angle_axis.astype(np.float64))
    k_parallel = K_CONSTANT * np.sqrt(e_kin) * np.sin(theta_rad)

    return k_parallel.astype(np.float32), energy_axis


def generate_preview(
    spectrum: np.ndarray,
    energy_axis: np.ndarray,
    angle_axis: np.ndarray,
    destination: Path,
    *,
    cmap: str = "inferno",
    pmin: float = 1.0,
    pmax: float = 99.5,
    use_log: bool = True,
    use_kspace: bool = False,
    hv: Optional[float] = None,
    work_function: float = 4.2,
) -> None:
    """Generate a preview PNG of the ARPES spectrum.

    Args:
        spectrum: 2D float32 array [energy, angle]
        energy_axis: 1D energy axis
        angle_axis: 1D angle axis
        destination: output PNG path
        cmap: matplotlib colormap name
        pmin: lower percentile for contrast (0-100)
        pmax: upper percentile for contrast (0-100)
        use_log: apply LogNorm if True
        use_kspace: convert angle to k-parallel if True
        hv: photon energy in eV (required for k-space)
        work_function: work function in eV (default 4.2)
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(spectrum, a_min=0.0, a_max=None)

    # Determine axes for display
    x_axis = angle_axis.copy()
    x_label = "Angle [deg]"

    if use_kspace:
        if hv is None or hv <= 0:
            raise ValueError("Photon energy (hv) is required for k-space preview.")
        k_axis, e_axis = to_kspace(energy_axis, angle_axis, hv, work_function)
        x_axis = k_axis
        x_label = r"$k_{\parallel}$ [$\AA^{-1}$]"
    else:
        e_axis = energy_axis

    # Compute extent
    extent = [
        float(x_axis[0]), float(x_axis[-1]),
        float(e_axis[0]), float(e_axis[-1]),
    ]

    # Contrast
    norm = None
    if use_log:
        vmin, vmax = compute_contrast(data, pmin, pmax)
        norm = LogNorm(vmin=vmin, vmax=vmax)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    kwargs = {"origin": "lower", "aspect": "auto", "cmap": cmap, "extent": extent}
    if norm is not None:
        kwargs["norm"] = norm
    im = ax.imshow(data, **kwargs)
    fig.colorbar(im, ax=ax)
    title = "ARPES Spectrum"
    if use_log:
        title += " (log scale)"
    if use_kspace:
        title += " — k-space"
    ax.set_title(title)
    ax.set_xlabel(x_label)
    y_label = "Energy [eV]"
    ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
import tempfile, os
import numpy as np
from pathlib import Path
from converter.preview import generate_preview, to_kspace

spec = np.random.rand(200, 100).astype(np.float32) * 100
energy = np.linspace(40, 35, 200, dtype=np.float32)
angle = np.linspace(-15, 15, 100, dtype=np.float32)

tmp = Path(tempfile.mktemp(suffix='.png'))
try:
    generate_preview(spec, energy, angle, tmp, use_kspace=True, hv=40.8, work_function=4.2)
    assert tmp.stat().st_size > 1000
    print('OK - kspace preview size:', tmp.stat().st_size)
finally:
    os.unlink(tmp)
"
```
Expected: `OK - kspace preview size: ...`

- [ ] **Step 3: Commit**

```bash
git add converter/preview.py && git commit -m "feat: add preview generator with k-space conversion"
```

---

### Task 6: Implement engine.py

**Files:**
- Create: `converter/engine.py`

- [ ] **Step 1: Write the module**

```python
"""Conversion engine — orchestrates read + write. ConvertWorker runs in a QThread."""

from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, Signal

import numpy as np

from converter.readers.txt_reader import read_txt
from converter.readers.pxt_reader import read_pxt, load_bin
from converter.writer import write_h5
from converter.preview import generate_preview


MANUAL_PARAM_KEYS = [
    "sample_name", "sample_id",
    "position_x", "position_y", "position_z",
    "position_polar", "position_tilt", "position_azimuth",
    "temperature_K", "photon_energy_eV", "polarization", "slit",
    "work_function_eV",
]


def detect_format(path: Path) -> str:
    """Detect file format from extension. Returns 'txt', 'pxt', or 'bin'."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return "txt"
    elif suffix == ".pxt":
        return "pxt"
    elif suffix == ".bin":
        return "bin"
    raise ValueError(f"Unsupported file extension: {suffix}")


def merge_params(batch_defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge batch defaults with per-file overrides. Override wins if non-empty."""
    merged = dict(batch_defaults)
    for key, value in overrides.items():
        if value is not None and value != "":
            merged[key] = value
    return merged


class ConvertWorker(QThread):
    """Worker thread that runs conversion without blocking the GUI."""

    progress = Signal(int, str)  # file_index, message
    file_done = Signal(str, bool, str)  # output_path, success, message
    all_done = Signal(int, int)  # success_count, fail_count

    def __init__(self, file_paths, batch_params, output_dir, preview_enabled, preview_settings, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.batch_params = batch_params
        self.output_dir = Path(output_dir)
        self.preview_enabled = preview_enabled
        self.preview_settings = preview_settings  # dict with cmap, pmin, pmax, use_log, use_kspace

    def run(self):
        success = 0
        fail = 0
        for i, file_path in enumerate(self.file_paths):
            path = Path(file_path)
            self.progress.emit(i, f"Converting {path.name}...")
            try:
                fmt = detect_format(path)
                file_overrides = self.batch_params.get("_overrides", {}).get(str(path), {})
                params = merge_params(self.batch_params, file_overrides)

                # Read
                if fmt == "txt":
                    result = read_txt(path)
                elif fmt == "pxt":
                    result = read_pxt(path)
                elif fmt == "bin":
                    spectrum = load_bin(path, (365, 571, 51), "float32")
                    result = {
                        "spectrum": spectrum[:, :, 25],
                        "energy": np.arange(spectrum.shape[0], dtype=np.float32),
                        "thetax": np.arange(spectrum.shape[1], dtype=np.float32),
                        "ses_params": {},
                    }
                else:
                    raise ValueError(f"Unknown format: {fmt}")

                spectrum = result["spectrum"]
                energy = result["energy"]
                thetax = result["thetax"]
                ses_params = result.get("ses_params", {})
                raw_channels = result.get("raw_channels")

                # Build HDF5 path
                out_name = path.stem + ".h5"
                out_path = self.output_dir / out_name

                # Write
                write_h5(
                    spectrum, energy, thetax, out_path,
                    source_format=fmt,
                    source_path=str(path),
                    manual_params=params,
                    ses_params=ses_params,
                    raw_channels=raw_channels,
                    overwrite=True,
                )

                self.file_done.emit(str(out_path), True, f"[OK] {path.name} -> {out_path}")

                # Preview
                if self.preview_enabled:
                    try:
                        p_settings = self.preview_settings or {}
                        prev_path = self.output_dir.parent / "previews" / (path.stem + ".png")
                        generate_preview(
                            spectrum, energy, thetax, prev_path,
                            cmap=p_settings.get("cmap", "inferno"),
                            pmin=p_settings.get("pmin", 1.0),
                            pmax=p_settings.get("pmax", 99.5),
                            use_log=p_settings.get("use_log", True),
                            use_kspace=p_settings.get("use_kspace", False),
                            hv=params.get("photon_energy_eV"),
                            work_function=float(params.get("work_function_eV", 4.2)),
                        )
                        self.file_done.emit(str(prev_path), True, f"  Preview saved -> {prev_path}")
                    except Exception as exc:
                        self.file_done.emit("", False, f"  Preview failed: {exc}")

                success += 1

            except Exception as exc:
                fail += 1
                self.file_done.emit(str(path), False, f"[FAIL] {path.name}: {exc}")

        self.all_done.emit(success, fail)
```

- [ ] **Step 2: Smoke test engine (no GUI, just the read+write pipeline)**

```bash
python -c "
import tempfile, os
from pathlib import Path
from converter.engine import detect_format, merge_params, ConvertWorker

assert detect_format(Path('test.txt')) == 'txt'
assert detect_format(Path('test.pxt')) == 'pxt'
assert detect_format(Path('test.bin')) == 'bin'

merged = merge_params({'sample_name': 'A', 'temperature_K': 300}, {'temperature_K': 77})
assert merged['sample_name'] == 'A'
assert merged['temperature_K'] == 77
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add converter/engine.py && git commit -m "feat: add ConversionEngine and ConvertWorker thread"
```

---

### Task 7: Implement main_window.py

**Files:**
- Create: `gui/main_window.py`

- [ ] **Step 1: Write the module**

```python
"""Main window with QTabWidget containing all four converter tabs."""

from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from gui.file_tab import FileTab
from gui.params_tab import ParamsTab
from gui.preview_tab import PreviewTab
from gui.convert_tab import ConvertTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARPES Data Converter")
        self.resize(960, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.file_tab = FileTab()
        self.params_tab = ParamsTab()
        self.preview_tab = PreviewTab()
        self.convert_tab = ConvertTab()

        self.tabs.addTab(self.file_tab, "1. Select Files")
        self.tabs.addTab(self.params_tab, "2. Parameters")
        self.tabs.addTab(self.preview_tab, "3. Preview")
        self.tabs.addTab(self.convert_tab, "4. Convert")

        # Wire navigation signals
        self.file_tab.files_changed.connect(self._on_files_changed)
        self.convert_tab.request_params.connect(self._on_params_requested)
        self.convert_tab.request_preview_settings.connect(self._on_preview_settings_requested)

    def _on_files_changed(self, file_paths):
        self.params_tab.set_files(file_paths)
        self.preview_tab.set_files(file_paths)

    def _on_params_requested(self):
        return self.params_tab.get_all_params()

    def _on_preview_settings_requested(self):
        return self.preview_tab.get_settings()
```

- [ ] **Step 2: Commit**

```bash
git add gui/main_window.py && git commit -m "feat: add MainWindow with tabbed layout"
```

---

### Task 8: Implement file_tab.py

**Files:**
- Create: `gui/file_tab.py`

- [ ] **Step 1: Write the module**

```python
"""Tab 1: File selection with drag-drop, browse, and file list."""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Signal, Qt


VALID_EXTENSIONS = {".txt", ".pxt", ".bin"}


class FileTab(QWidget):
    files_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._file_paths = []

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Select ARPES Data Files")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Drop zone
        self.drop_label = QLabel("Drag & drop .txt, .pxt, or .bin files here\n— or —")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet(
            "border: 2px dashed #888; border-radius: 8px; padding: 30px; color: #888;"
        )
        layout.addWidget(self.drop_label)

        # Browse button
        btn_row = QHBoxLayout()
        browse_btn = QPushButton("Browse Files")
        browse_btn.clicked.connect(self._browse)
        btn_row.addStretch()
        btn_row.addWidget(browse_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # File list
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        # Bottom buttons
        bottom = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        self.next_btn = QPushButton("Next: Parameters →")
        self.next_btn.clicked.connect(self._go_next)
        bottom.addWidget(remove_btn)
        bottom.addWidget(clear_btn)
        bottom.addStretch()
        bottom.addWidget(self.next_btn)
        layout.addLayout(bottom)

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select ARPES Data Files", "",
            "ARPES Files (*.txt *.pxt *.bin);;All Files (*)"
        )
        for p in paths:
            self._add_file(Path(p))

    def _add_file(self, path: Path):
        path = path.resolve()
        if str(path) in self._file_paths:
            return
        suffix = path.suffix.lower()
        if suffix not in VALID_EXTENSIONS:
            return
        self._file_paths.append(str(path))
        item = QListWidgetItem(f"{path.name}  —  {suffix[1:].upper()}")
        item.setToolTip(str(path))
        self.file_list.addItem(item)
        self.files_changed.emit(self._file_paths)

    def _remove_selected(self):
        for item in self.file_list.selectedItems():
            idx = self.file_list.row(item)
            self.file_list.takeItem(idx)
            if idx < len(self._file_paths):
                del self._file_paths[idx]
        self.files_changed.emit(self._file_paths)

    def _clear_all(self):
        self.file_list.clear()
        self._file_paths.clear()
        self.files_changed.emit(self._file_paths)

    def _go_next(self):
        if not self._file_paths:
            QMessageBox.warning(self, "No Files", "Please add at least one file.")
            return
        w = self.window()
        if w and hasattr(w, "tabs"):
            w.tabs.setCurrentIndex(1)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file():
                self._add_file(path)

    def get_files(self):
        return list(self._file_paths)
```

- [ ] **Step 2: Commit**

```bash
git add gui/file_tab.py && git commit -m "feat: add FileTab with drag-drop and browse"
```

---

### Task 9: Implement params_tab.py

**Files:**
- Create: `gui/params_tab.py`

- [ ] **Step 1: Write the module**

```python
"""Tab 2: Batch parameter defaults + per-file override table."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QPushButton, QSplitter,
)
from PySide6.QtCore import Signal, Qt


FIELD_KEYS = [
    "sample_name", "sample_id",
    "position_x", "position_y", "position_z",
    "position_polar", "position_tilt", "position_azimuth",
    "temperature_K", "photon_energy_eV", "polarization", "slit",
    "work_function_eV",
]

FIELD_LABELS = [
    "Sample Name", "Sample ID",
    "Position X", "Position Y", "Position Z",
    "Polar", "Tilt", "Azimuth",
    "Temperature (K)", "Photon Energy hv (eV)", "Polarization", "Slit",
    "Work Function Φ (eV)",
]

FIELD_DEFAULTS = {
    "work_function_eV": "4.2",
}


class ParamsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_paths = []
        self._overrides = {}  # {filepath: {key: value}}

        layout = QVBoxLayout(self)

        title = QLabel("Experiment Parameters")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)

        # Left: batch defaults form
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        form_group = QGroupBox("Batch Defaults")
        form_layout = QFormLayout(form_group)
        self._fields = {}
        for key, label in zip(FIELD_KEYS, FIELD_LABELS):
            edit = QLineEdit()
            edit.setPlaceholderText(label)
            if key in FIELD_DEFAULTS:
                edit.setText(FIELD_DEFAULTS[key])
            form_layout.addRow(label, edit)
            self._fields[key] = edit

        left_layout.addWidget(form_group)

        tip = QLabel("These values apply to all files. Edit individual\nfiles in the table on the right.")
        tip.setStyleSheet("color: #888; font-size: 10pt; padding: 8px;")
        left_layout.addWidget(tip)
        left_layout.addStretch()

        splitter.addWidget(left)

        # Right: per-file override table
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        table_group = QGroupBox("Per-File Overrides")
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget()
        self.table.setColumnCount(len(FIELD_KEYS) + 1)
        self.table.setHorizontalHeaderLabels(["File"] + FIELD_LABELS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.cellChanged.connect(self._on_cell_changed)
        self._rebuilding = False
        table_layout.addWidget(self.table)

        table_tip = QLabel("Click any cell to override the batch default for that file. Empty = use default.")
        table_tip.setStyleSheet("color: #888; font-size: 10pt; padding: 4px;")
        table_layout.addWidget(table_tip)

        right_layout.addWidget(table_group)
        splitter.addWidget(right)

        splitter.setSizes([360, 540])
        layout.addWidget(splitter)

        # Navigation
        nav = QHBoxLayout()
        back_btn = QPushButton("← Back: Select Files")
        back_btn.clicked.connect(self._go_back)
        next_btn = QPushButton("Next: Preview →")
        next_btn.clicked.connect(self._go_next)
        nav.addWidget(back_btn)
        nav.addStretch()
        nav.addWidget(next_btn)
        layout.addLayout(nav)

    def set_files(self, file_paths):
        self._file_paths = file_paths
        self._rebuild_table()

    def _rebuild_table(self):
        self._rebuilding = True
        self.table.setRowCount(len(self._file_paths))
        for i, fp in enumerate(self._file_paths):
            name_item = QTableWidgetItem(Path(fp).name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, name_item)
            for j, key in enumerate(FIELD_KEYS):
                val = self._overrides.get(fp, {}).get(key, "")
                self.table.setItem(i, j + 1, QTableWidgetItem(str(val) if val else ""))
        self._rebuilding = False

    def _on_cell_changed(self, row, col):
        if self._rebuilding or row >= len(self._file_paths):
            return
        fp = self._file_paths[row]
        key = FIELD_KEYS[col - 1]
        item = self.table.item(row, col)
        text = item.text().strip() if item else ""
        if fp not in self._overrides:
            self._overrides[fp] = {}
        if text:
            self._overrides[fp][key] = text
        else:
            self._overrides[fp].pop(key, None)
            if not self._overrides[fp]:
                del self._overrides[fp]

    def get_all_params(self):
        """Return {batch_params, _overrides} for the engine."""
        batch = {}
        for key, edit in self._fields.items():
            text = edit.text().strip()
            if text:
                try:
                    batch[key] = float(text)
                except ValueError:
                    batch[key] = text
        batch["_overrides"] = dict(self._overrides)
        return batch

    def _go_back(self):
        w = self.window()
        if w and hasattr(w, "tabs"):
            w.tabs.setCurrentIndex(0)

    def _go_next(self):
        w = self.window()
        if w and hasattr(w, "tabs"):
            w.tabs.setCurrentIndex(2)


from pathlib import Path
```

- [ ] **Step 2: Commit**

```bash
git add gui/params_tab.py && git commit -m "feat: add ParamsTab with batch form and override table"
```

---

### Task 10: Implement preview_tab.py

**Files:**
- Create: `gui/preview_tab.py`

- [ ] **Step 1: Write the module**

```python
"""Tab 3: Preview with k-space toggle, colormap picker, contrast sliders."""

from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QCheckBox, QSlider, QGroupBox, QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from converter.readers.txt_reader import read_txt
from converter.readers.pxt_reader import read_pxt
from converter.preview import compute_contrast, to_kspace
from converter.engine import detect_format


COLORMAPS = ["inferno", "viridis", "plasma", "gray", "jet", "turbo"]


class PreviewTab(QWidget):
    preview_enabled_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_paths = []
        self._current_data = None
        self._current_energy = None
        self._current_angle = None

        layout = QVBoxLayout(self)

        title = QLabel("Preview")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Controls row
        ctrl_row = QHBoxLayout()

        ctrl_row.addWidget(QLabel("File:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self._on_file_selected)
        ctrl_row.addWidget(self.file_combo)

        ctrl_row.addSpacing(16)

        self.kspace_cb = QCheckBox("k-space")
        self.kspace_cb.stateChanged.connect(lambda: self._refresh_preview())
        ctrl_row.addWidget(self.kspace_cb)

        ctrl_row.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(lambda: self._refresh_preview())
        ctrl_row.addWidget(self.cmap_combo)

        self.log_cb = QCheckBox("Log scale")
        self.log_cb.setChecked(True)
        self.log_cb.stateChanged.connect(lambda: self._refresh_preview())
        ctrl_row.addWidget(self.log_cb)

        self.save_preview_cb = QCheckBox("Save with conversion")
        self.save_preview_cb.setChecked(True)
        self.save_preview_cb.stateChanged.connect(
            lambda s: self.preview_enabled_changed.emit(s == Qt.Checked.value)
        )
        ctrl_row.addWidget(self.save_preview_cb)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Main area: canvas + contrast panel
        main_row = QHBoxLayout()

        self.canvas = FigureCanvas(Figure(figsize=(7, 5)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.canvas.figure.tight_layout()
        main_row.addWidget(self.canvas, 1)

        # Contrast controls
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QVBoxLayout(contrast_group)

        contrast_layout.addWidget(QLabel("vmin percentile:"))
        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, 10)
        self.vmin_slider.setValue(1)
        self.vmin_label = QLabel("1%")
        self.vmin_slider.valueChanged.connect(
            lambda v: self.vmin_label.setText(f"{v}%")
        )
        contrast_layout.addWidget(self.vmin_slider)
        contrast_layout.addWidget(self.vmin_label)

        contrast_layout.addWidget(QLabel("vmax percentile:"))
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmax_slider.setRange(90, 100)
        self.vmax_slider.setValue(99)
        self.vmax_label = QLabel("99%")
        self.vmax_slider.valueChanged.connect(
            lambda v: self.vmax_label.setText(f"{v}%")
        )
        contrast_layout.addWidget(self.vmax_slider)
        contrast_layout.addWidget(self.vmax_label)

        refresh_btn = QPushButton("Refresh Preview")
        refresh_btn.clicked.connect(self._refresh_preview)
        contrast_layout.addWidget(refresh_btn)

        contrast_layout.addStretch()
        main_row.addWidget(contrast_group)

        layout.addLayout(main_row)

        # Navigation
        nav = QHBoxLayout()
        back_btn = QPushButton("← Back: Parameters")
        back_btn.clicked.connect(lambda: self._nav_to(1))
        next_btn = QPushButton("Next: Convert →")
        next_btn.clicked.connect(lambda: self._nav_to(3))
        nav.addWidget(back_btn)
        nav.addStretch()
        nav.addWidget(next_btn)
        layout.addLayout(nav)

    def set_files(self, file_paths):
        self._file_paths = file_paths
        self.file_combo.clear()
        for fp in file_paths:
            self.file_combo.addItem(Path(fp).name, fp)
        if file_paths:
            self._on_file_selected(0)

    def _on_file_selected(self, index):
        if index < 0 or not self._file_paths:
            return
        fp = self._file_paths[index]
        try:
            fmt = detect_format(Path(fp))
            if fmt == "txt":
                data = read_txt(Path(fp))
            elif fmt == "pxt":
                data = read_pxt(Path(fp))
            else:
                return
            self._current_data = data["spectrum"]
            self._current_energy = data["energy"]
            self._current_angle = data["thetax"]
            self._refresh_preview()
        except Exception as exc:
            QMessageBox.warning(self, "Preview Error", f"Could not read file:\n{exc}")

    def _refresh_preview(self):
        if self._current_data is None:
            return
        self.ax.clear()
        data = np.clip(self._current_data, a_min=0.0, a_max=None)
        use_kspace = self.kspace_cb.isChecked()
        use_log = self.log_cb.isChecked()
        cmap = self.cmap_combo.currentText()
        pmin = self.vmin_slider.value()
        pmax = self.vmax_slider.value()

        # k-space conversion
        if use_kspace:
            params_tab = self._get_params_tab()
            hv = None
            work_function = 4.2
            if params_tab:
                batch = params_tab.get_all_params()
                hv = batch.get("photon_energy_eV")
                wf = batch.get("work_function_eV", 4.2)
                if wf:
                    try:
                        work_function = float(wf)
                    except (ValueError, TypeError):
                        pass
            if hv is None or hv == "":
                QMessageBox.warning(
                    self, "Missing hv",
                    "Please enter Photon Energy (hv) in the Parameters tab for k-space conversion."
                )
                self.kspace_cb.setChecked(False)
                use_kspace = False

        if use_kspace:
            try:
                hv_val = float(hv)
                k_axis, e_axis = to_kspace(self._current_energy, self._current_angle, hv_val, work_function)
                x_label = r"$k_{\parallel}$ [$\AA^{-1}$]"
            except Exception:
                use_kspace = False
                e_axis = self._current_energy
                k_axis = self._current_angle
                x_label = "Angle [deg]"
        else:
            e_axis = self._current_energy
            k_axis = self._current_angle
            x_label = "Angle [deg]"

        extent = [
            float(k_axis[0]), float(k_axis[-1]),
            float(e_axis[0]), float(e_axis[-1]),
        ]

        if use_log:
            vmin, vmax = compute_contrast(data, pmin, pmax)
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

        kwargs = {"origin": "lower", "aspect": "auto", "cmap": cmap, "extent": extent}
        if norm is not None:
            kwargs["norm"] = norm
        im = self.ax.imshow(data, **kwargs)
        self.canvas.figure.colorbar(im, ax=self.ax)
        title = "ARPES Spectrum"
        if use_log:
            title += " (log)"
        if use_kspace:
            title += " — k-space"
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel("Energy [eV]")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def get_settings(self):
        return {
            "cmap": self.cmap_combo.currentText(),
            "pmin": float(self.vmin_slider.value()),
            "pmax": float(self.vmax_slider.value()),
            "use_log": self.log_cb.isChecked(),
            "use_kspace": self.kspace_cb.isChecked(),
        }

    def is_preview_enabled(self):
        return self.save_preview_cb.isChecked()

    def _get_params_tab(self):
        w = self.window()
        if w and hasattr(w, "params_tab"):
            return w.params_tab
        return None

    def _nav_to(self, idx):
        w = self.window()
        if w and hasattr(w, "tabs"):
            w.tabs.setCurrentIndex(idx)
```

- [ ] **Step 2: Commit**

```bash
git add gui/preview_tab.py && git commit -m "feat: add PreviewTab with k-space toggle and contrast controls"
```

---

### Task 11: Implement convert_tab.py

**Files:**
- Create: `gui/convert_tab.py`

- [ ] **Step 1: Write the module**

```python
"""Tab 4: Conversion with progress bar, log output, and output folder selection."""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QProgressBar, QFileDialog, QLineEdit, QGroupBox,
    QMessageBox,
)
from PySide6.QtCore import Signal, Qt

from converter.engine import ConvertWorker


class ConvertTab(QWidget):
    request_params = Signal()
    request_preview_settings = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None

        layout = QVBoxLayout(self)

        title = QLabel("Convert")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Summary
        self.summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(self.summary_group)
        self.summary_label = QLabel("No files selected.")
        summary_layout.addWidget(self.summary_label)
        layout.addWidget(self.summary_group)

        # Output folder
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self.out_dir_edit = QLineEdit("data/converted_h5/")
        out_row.addWidget(self.out_dir_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output)
        out_row.addWidget(browse_btn)
        layout.addLayout(out_row)

        # Start button
        self.start_btn = QPushButton("Start Conversion")
        self.start_btn.setStyleSheet(
            "background-color: #27ae60; color: white; font-size: 12pt; padding: 8px 24px;"
        )
        self.start_btn.clicked.connect(self._start_conversion)
        layout.addWidget(self.start_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        # Log
        log_label = QLabel("Log:")
        layout.addWidget(log_label)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "background-color: #1a1a2e; color: #a0ffa0; font-family: Consolas, monospace; font-size: 10pt;"
        )
        self.log_view.setMaximumBlockCount(500)
        layout.addWidget(self.log_view, 1)

        # Navigation
        nav = QHBoxLayout()
        back_btn = QPushButton("← Back: Preview")
        back_btn.clicked.connect(lambda: self._nav_to(2))
        nav.addWidget(back_btn)
        nav.addStretch()
        layout.addLayout(nav)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.out_dir_edit.setText(d)

    def _start_conversion(self):
        w = self.window()
        if not w or not hasattr(w, "file_tab"):
            return
        file_paths = w.file_tab.get_files()
        if not file_paths:
            QMessageBox.warning(self, "No Files", "Please add files in the Select Files tab.")
            return

        params = w.params_tab.get_all_params()
        preview_enabled = w.preview_tab.is_preview_enabled()
        preview_settings = w.preview_tab.get_settings()
        output_dir = self.out_dir_edit.text().strip() or "data/converted_h5/"

        self._log(f"Starting conversion of {len(file_paths)} file(s)...")
        self._log(f"Output: {output_dir}")
        self._log(f"Preview: {'Yes' if preview_enabled else 'No'}")
        self._log("-" * 40)

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(file_paths))
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)

        self._worker = ConvertWorker(
            file_paths, params, output_dir, preview_enabled, preview_settings
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.start()

    def _on_progress(self, idx, msg):
        self.progress_bar.setValue(idx)
        self.progress_label.setText(msg)

    def _on_file_done(self, path, success, msg):
        color = "#a0ffa0" if success else "#ff6666"
        self._log(msg, color)

    def _on_all_done(self, success, fail):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_label.setText(f"Done — {success} success, {fail} failed")
        self.start_btn.setEnabled(True)
        self._log(f"\n{'=' * 40}")
        self._log(f"Conversion complete: {success} success, {fail} failed")
        if fail == 0:
            QMessageBox.information(self, "Done", f"All {success} file(s) converted successfully.")

    def _log(self, text, color="#a0ffa0"):
        self.log_view.append(f"<span style='color:{color};'>{text}</span>")

    def _nav_to(self, idx):
        w = self.window()
        if w and hasattr(w, "tabs"):
            w.tabs.setCurrentIndex(idx)
```

- [ ] **Step 2: Commit**

```bash
git add gui/convert_tab.py && git commit -m "feat: add ConvertTab with progress and log"
```

---

### Task 12: Create converter_app.py entry point and run.bat launcher

**Files:**
- Create: `converter_app.py`
- Create: `run.bat`

- [ ] **Step 1: Write converter_app.py**

```python
"""ARPES Data Converter — PySide6 GUI application entry point."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so converter/ and gui/ are importable
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ARPES Data Converter")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write run.bat**

```bat
@echo off
python converter_app.py
pause
```

- [ ] **Step 3: Smoke test — verify GUI launches**

Launch the app and verify the window appears with four tabs:
```bash
python converter_app.py
```
Expected: Window opens with title "ARPES Data Converter", 4 tabs visible. Close manually.

- [ ] **Step 4: End-to-end test — convert a real file through the GUI**

Manual test steps (documented, not automated):
1. Launch `python converter_app.py`
2. Tab 1: Browse → select `data/txtdata/MS30013.txt`
3. Tab 2: Enter sample name "Test", hv "40.8", temperature "300"
4. Tab 3: Verify preview renders, toggle k-space on/off, adjust contrast
5. Tab 4: Set output to a temp dir, click Start Conversion
6. Verify .h5 is created with all attributes
7. Verify preview .png is created (if enabled)

- [ ] **Step 5: Commit**

```bash
git add converter_app.py run.bat && git commit -m "feat: add entry point and launcher for ARPES converter GUI"
```
