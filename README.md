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
CPU and CUDA smoke checks are suitable for the installed hardware. Training
re-runs the `sample_id`, `acquisition_group`, `pair_id`, `source_path`, and
reviewed-pair connectivity audit before it creates a run directory; editing all
CSV files into a superficially consistent but leaking split is rejected.

The example config names two provenance records for the same locked manifest:

```yaml
paths:
  manifest: D:/Projects/dccnn/workspace/manifests/records.csv
  provenance_path: D:/Projects/dccnn/workspace/manifests/data_provenance.scientific.json
  smoke_provenance_path: D:/Projects/dccnn/workspace/manifests/data_provenance.smoke.json
```

Each JSON file has schema version 1:

```json
{
  "schema_version": 1,
  "classification": "reviewed_scientific_dataset",
  "scientific_use": true,
  "record_ids": ["record-id-1", "record-id-2"],
  "input_sha256": {
    "record-id-1": "<64-character SHA-256 of its converted_path>",
    "record-id-2": "<64-character SHA-256 of its converted_path>"
  }
}
```

Prepare both files only after the manifest and all three split CSVs are locked.
`record_ids` must be unique and exactly cover every partition input, and
`input_sha256` must contain the matching `Get-FileHash -Algorithm SHA256`
digest for every canonical H5. The smoke record uses
`classification: controlled_smoke_fixture` and `scientific_use: false`; the
scientific record uses the values shown above. The scientific record is a
review authority: it must be independently reviewed and approved. Neither the
training command nor a script in this repository generates that authority, and
copying a smoke record while changing its label is not a review.

These two commands select the two configured provenance paths without changing
the manifest:

```powershell
uv run dccnn-train --config configs/train_cut_v1.yaml --smoke-test --device cpu
uv run dccnn-train --config configs/train_cut_v1.yaml --device cuda
```

The CUDA smoke variant is also available after the CPU smoke:

```powershell
uv run dccnn-train --config configs/train_cut_v1.yaml --smoke-test --device cuda
```

## Evaluation artifacts

A standard `test.csv` alone describes the locked population, not a scientific
five-method comparison. Running it alone deliberately writes conservative
`not_evaluated` evidence:

```powershell
uv run dccnn-eval --split D:\Projects\dccnn\workspace\splits\test.csv --output D:\Projects\dccnn\outputs\evaluation\population-only
```

After producing and reviewing the comparison artifacts, create a CSV keyed by
the same unique `record_id` or `file_id` as the split:

```csv
record_id,denoised_path,reference_path,legacy_output_path
record-id-1,D:/Projects/dccnn/outputs/inference/record-id-1_denoised.h5,D:/Projects/dccnn/workspace/references/record-id-1.h5,D:/Projects/dccnn/workspace/legacy/record-id-1.h5
```

The join must be exactly one-to-one. Duplicate keys, missing split keys, extra
artifact keys, or empty artifact paths are rejected before a report directory
is created. Relative artifact paths are resolved beside the artifacts CSV.
With the reviewed table, the evaluator loads raw input from the standard split,
computes Gaussian and median baselines, and loads LegacyCCNN and
ResidualDenoiser2D outputs for all five methods:

```powershell
uv run dccnn-eval --split D:\Projects\dccnn\workspace\splits\test.csv --artifacts D:\Projects\dccnn\workspace\manifests\reviewed_evaluation_artifacts.csv --output D:\Projects\dccnn\outputs\evaluation\scientific
```

Every residual output in a scientific report must have a non-empty
`denoising_checkpoint_sha256`, and all rows must name the same checkpoint.
Missing or mixed checkpoint hashes make the whole acceptance result
`not_evaluated`.

Run inference to a new output file:

```powershell
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
converter inputs. Training and inference resolve filesystem redirects before
writing and reject destinations under `D:\Data\ARPES`, `D:\Projects\convert`,
or any input source path.

## Legacy inventory

Create a review-only inventory before requesting any separate archival action:

```powershell
uv run python scripts/inventory_legacy.py --repo D:\Projects\dccnn\dccnn-arpes-main --archive D:\Projects\dccnn\legacy_archive --output D:\Projects\dccnn\workspace\manifests\legacy_inventory.csv
```

The CSV records path, type, size, modification time, SHA-256, duplicate group,
and a proposed destination. It does not create the archive directory, move
files, or delete files. Unsafe archive roots (the repository root, source root,
or a drive root) are rejected. The default repository scan is bounded to
legacy checkpoints, H5, CSV, PNG, config files, and result directories. It
prunes hidden/environment/worktree/cache trees and the current `src`, `tests`,
`docs`, and `configs` trees; unknown files are not hashed or reported as
`other`. Use `--source <legacy-subtree>` only when a narrower legacy location
has been reviewed explicitly.

## Checks

```powershell
uv run ruff check src tests scripts
uv run pytest -v
uv run pytest --cov=dccnn_arpes --cov-report=term-missing
```
