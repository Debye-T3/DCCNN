# DCCNN ARPES

Reproducible tooling for 2D ARPES-cut denoising. The supported interchange
format is a canonical xarray/HDF5 `DataArray` with dimensions `("eV", "alpha")`.

## Environment

This project uses its locked, project-local `uv` environment. Anaconda is not
required and should not be used for this workflow.

```powershell
uv sync --extra dev
```

Before CUDA training, verify that the resolved PyTorch build sees the intended
GPU (the current workstation target is RTX 5080):

```powershell
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"
```

The command must print a device name containing `RTX 5080`. This is an
environment check, not a claim that a scientific model checkpoint has been
accepted.

## Data boundary and workflow

`D:\Projects\convert` is the only supported raw-data-to-canonical-xarray/HDF5
converter. Convert raw experimental inputs there first; do not add another raw
converter to this repository. `D:\Data\ARPES` is read-only input data, and
scanning, training, evaluation, and inference must not write to it.

Validate a canonical cut, or explicitly use the read-only legacy adapter when
inspecting an older DCCNN H5 file:

```powershell
uv run dccnn-data validate D:\Projects\convert\data\converted_h5\NiNS0032.h5
uv run dccnn-data validate D:\Projects\dccnn\dccnn-arpes-main\data\converted_h5\nbvt0005_txt.h5 --allow-legacy
```

Legacy H5 compatibility is read-only. It adapts an old file into the common
in-memory data boundary; it never rewrites or upgrades that source file in
place.

Build the reviewed data inputs in order:

```powershell
uv run dccnn-data scan --source D:\Data\ARPES --converted D:\Projects\dccnn\workspace\converted --output D:\Projects\dccnn\workspace\manifests\records.csv
uv run dccnn-data pairs --manifest D:\Projects\dccnn\workspace\manifests\records.csv --output D:\Projects\dccnn\workspace\manifests\pairs.csv
uv run dccnn-data split --manifest D:\Projects\dccnn\workspace\manifests\records.csv --pairs D:\Projects\dccnn\workspace\manifests\pairs.csv --output D:\Projects\dccnn\workspace\splits
```

Review `pairs.csv` and `split_audit.json` before training. Train only after the
CPU and CUDA smoke checks are suitable for the installed hardware:

```powershell
uv run dccnn-train --config configs/train_cut_v1.yaml --smoke-test --device cpu
uv run dccnn-train --config configs/train_cut_v1.yaml --smoke-test --device cuda
uv run dccnn-train --config configs/train_cut_v1.yaml --device cuda
```

Evaluate a locked test split and run inference to a new output file:

```powershell
uv run dccnn-eval --config configs/train_cut_v1.yaml --split D:\Projects\dccnn\workspace\splits\test.csv --output D:\Projects\dccnn\outputs\evaluation
$denoiseCheckpoint = Get-ChildItem D:\Projects\dccnn\outputs\experiments -Recurse -Filter best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
uv run dccnn-denoise --input D:\Projects\convert\data\converted_h5\NiNS0032.h5 --checkpoint $denoiseCheckpoint --output D:\Projects\dccnn\outputs\inference
```

The repository does not claim that a research-grade checkpoint exists or that
the quantitative and physical-fidelity acceptance criteria have passed. Those
claims require a reviewed, locked evaluation run and its evidence artifacts.

## Local directories

`D:\Projects\dccnn\workspace` stores generated manifests, pairing review
artifacts, and leakage-safe split files. `D:\Projects\dccnn\outputs` stores
derived smoke runs, training checkpoints, evaluation reports, and denoised
copies. Both are intentionally separate from source archives and canonical
converter inputs.

## Legacy inventory

Create a review-only inventory before requesting any separate archival action:

```powershell
uv run python scripts/inventory_legacy.py --repo D:\Projects\dccnn\dccnn-arpes-main --archive D:\Projects\dccnn\legacy_archive --output D:\Projects\dccnn\workspace\manifests\legacy_inventory.csv
```

The CSV records path, type, size, modification time, SHA-256, duplicate group,
and a proposed destination. It does not create the archive directory, move
files, or delete files. Unsafe archive roots (the repository root, source root,
or a drive root) are rejected.

## Checks

```powershell
uv run ruff check src tests scripts
uv run pytest -v
uv run pytest --cov=dccnn_arpes --cov-report=term-missing
```
