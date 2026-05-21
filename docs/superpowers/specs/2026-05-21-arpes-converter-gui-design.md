# ARPES Data Converter GUI — Design Spec

## Overview

A standalone PySide6 GUI application that converts ARPES raw data (`.txt`, `.pxt`, `.bin`) into HDF5 format. Designed for non-programmer users — click-to-run, with a guided tabbed workflow. Supports batch file selection, manual entry of experimental parameters (with batch defaults + per-file overrides), SES header auto-extraction, optional k-space preview with adjustable contrast/colormap, and progress-tracked conversion.

## Supported Input Formats

| Format | Extension | Source | Parser |
|--------|-----------|--------|--------|
| Scienta DA30L text export | `.txt` | SES software | `readers/txt_reader.py` |
| Scienta PXT binary | `.pxt` | SES software | `readers/pxt_reader.py` |
| Raw binary cube | `.bin` | legacy/manual export | `readers/pxt_reader.py` (shared `load_bin`) |

Format detection is automatic — based on file extension and header inspection. No user selection needed.

## GUI Layout — Tabbed Workflow

Four tabs, left-to-right. User can jump back to adjust anything before converting.

### Tab 1: Select Files

- **Drag & drop zone** — accepts `.txt`, `.pxt`, `.bin` files
- **Browse button** — opens native file dialog with extension filter
- **File list** — shows selected files with detected format and shape (e.g., "MS30013.txt — DA30L txt, 1221×463")
- **Remove button** — remove individual files from the list
- Auto-detect: if PXT file has multiple channels, show channel info
- "Next" button proceeds to Tab 2

### Tab 2: Parameters

Split layout: batch defaults form (left) + per-file override table (right).

**Batch defaults form fields:**
- Sample name (text)
- Sample ID (text)
- Position: X, Y, Z, Polar, Tilt, Azimuth (6 float fields)
- Temperature T (float, K)
- Photon energy hv (float, eV)
- Polarization (text, e.g. "p", "s", "circular")
- Slit (text, e.g. "0.2mm", "0.5mm")
- Work function Φ (float, eV, **default 4.2**)

**Per-file override table:**
- Rows = selected files, columns = all manual parameters
- Empty cell = use batch default
- Click cell to enter override value
- SES parameters (pass energy, lens mode, step count, dwell time, etc.) auto-extracted from file headers — not in the table, but shown in a read-only info panel per file

### Tab 3: Preview

- **File selector dropdown** — pick which file to preview
- **k-space checkbox** — toggle between angle-space (θ vs E) and momentum-space (k∥ vs E)
  - Formula: `k∥ = 0.5123 × √(E_kin) × sin(θ)` where `E_kin = hv − Φ − E_binding`
  - Uses hv and Φ from Tab 2 parameters
- **Colormap dropdown** — inferno (default), viridis, plasma, gray, jet, turbo
- **Log scale checkbox** — on by default
- **Contrast sliders**:
  - vmin percentile (0–10%, default 1%)
  - vmax percentile (90–100%, default 99.5%)
  - Sliders update on "Refresh Preview" button click (not live, to avoid lag)
- **Embedded matplotlib figure** — renders the spectrum with current settings
- **Save preview checkbox** — whether to save a PNG alongside .h5 during conversion (uses current preview settings)

### Tab 4: Convert

- **Summary panel** — file count, formats, preview status
- **Output folder** — text field + Browse button, defaults to `data/converted_h5/`
- **Start Conversion button** — begins batch processing
- **Progress bar** — per-file progress with current filename
- **Log console** — scrollable text output showing each file result, extracted SES params, preview save confirmation, warnings/errors
- Back button to adjust settings before re-running

## Architecture

```
arpes-converter/
├── converter_app.py          # Entry point — QApplication + MainWindow
├── converter/
│   ├── __init__.py
│   ├── engine.py             # ConversionEngine: orchestrates format detect → read → write
│   ├── readers/
│   │   ├── __init__.py
│   │   ├── txt_reader.py     # DA30L .txt parser
│   │   └── pxt_reader.py     # .pxt / .bin parser
│   ├── writer.py             # HDF5 writer
│   └── preview.py            # Preview generation + k-space conversion
├── gui/
│   ├── __init__.py
│   ├── main_window.py        # QMainWindow + QTabWidget
│   ├── file_tab.py           # Tab 1
│   ├── params_tab.py         # Tab 2
│   ├── preview_tab.py        # Tab 3 (embeds matplotlib FigureCanvas)
│   └── convert_tab.py        # Tab 4 (progress, log, worker thread)
└── run.bat                   # Double-click launcher
```

### Key design principles

- **Separation of concerns**: GUI classes never touch file I/O directly — they call `ConversionEngine` methods. Readers/writers are pure data transforms with no UI dependency.
- **Worker thread**: Conversion runs in a `QThread` so the UI stays responsive. Progress signals update the progress bar and log via Qt signals/slots.
- **Reuse existing parsers**: Refactor logic from `convert_txt_to_h5.py` and `convert_arpes_to_h5.py` into the `readers/` package. Clean up the original scripts to delegate to the new package (or remove them).

## Data Flow

```
User drops files → Tab 1 stores file paths
User fills params → Tab 2 stores dict of batch defaults + per-file overrides
User configures preview → Tab 3 calls converter.preview.generate() on demand
User clicks Convert → Tab 4:
  1. Merge batch defaults + per-file overrides into final params per file
  2. For each file:
     a. Detect format → call appropriate reader
     b. Extract SES header params (auto)
     c. Merge manual + auto params
     d. Write .h5 via writer.write_h5()
     e. If preview enabled, generate .png via preview.generate()
  3. Report done
```

## HDF5 Output Structure

Each `.h5` file contains:
- `spectrum` — 2D float32 array [energy, angle]
- `energy` — 1D float32 energy axis
- `thetax` — 1D float32 angle axis
- `raw_channels` — spectrum reshaped to [1, H, W] for ML compatibility

Attributes:
- `source_format` — "txt", "pxt", or "bin"
- `source_path` — original file path
- `shape` — string representation of spectrum shape
- **Manual params**: `sample_name`, `sample_id`, `position_x`, `position_y`, `position_z`, `position_polar`, `position_tilt`, `position_azimuth`, `temperature_K`, `photon_energy_eV`, `polarization`, `slit`, `work_function_eV`
- **Auto-extracted SES params**: `pass_energy_eV`, `lens_mode`, `num_energy_steps`, `num_angle_steps`, `dwell_time_ms`, `energy_step_eV`, `angle_step_deg`, `frame_type`, `channel_count`, `channel_used` (plus any additional SES header fields present)

## k-Space Conversion

```
k∥ [Å⁻¹] = 0.5123 × √(E_kin [eV]) × sin(θ [rad])
  where E_kin = hv − Φ − E_binding
```

- `hv` comes from the user-entered photon energy parameter
- `Φ` comes from the user-entered work function (default 4.2 eV)
- `E_binding` = `E_fermi − energy_axis[i]` (referenced to Fermi level when available, otherwise raw energy)
- `θ` comes from the `thetax` angle axis in radians
- If `hv` is not yet provided, k-space mode shows a warning and falls back to angle-space

## Error Handling

- **Unreadable file**: skip with `[FAIL]` log entry, continue remaining files
- **Format detection failure**: show error in log, skip file
- **Missing required params** (hv for k-space): warn in preview tab, disable k-space toggle until filled
- **Disk full / write error**: abort remaining batch, show error dialog
- **Empty file list on Convert**: show warning, don't proceed

## Testing Strategy

- Unit tests for each reader (txt, pxt, bin) against sample files in `data/`
- Unit tests for k-space conversion math with known inputs/outputs
- Unit tests for HDF5 writer (round-trip: write then read back, verify datasets + attrs)
- Manual UI testing for each tab on Windows

## Packaging

- `run.bat` in the project root: `@python converter_app.py` — double-click to launch
- Dependencies: PySide6, h5py, numpy, matplotlib (pyyaml optional for future config support)
- Not packaged as .exe initially — run from source with `pip install -e .` to get project deps
