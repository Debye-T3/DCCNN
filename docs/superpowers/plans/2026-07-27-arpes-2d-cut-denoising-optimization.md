# ARPES 二维 Cut 降噪优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一个以 convert 输出的 xarray/HDF5 为唯一标准格式、可复现训练并能验证物理结构保真的二维 ARPES cut 降噪流程。

**Architecture:** 保留现有代码和实验产物作为 legacy 参考，在 `src/dccnn_arpes` 中建立新的分层包。数据入口先把标准 xarray/HDF5 或旧 H5 适配为经过校验的 `DataArray(eV, alpha)`，后续清单、配对、裁剪、训练、评估和推理只依赖该对象；PyTorch 模型不直接解析 HDF5。

**Tech Stack:** Python 3.12、uv、PyTorch、xarray、h5netcdf、netCDF4、NumPy、SciPy、pandas、openpyxl、PyYAML、scikit-image、matplotlib、pytorch-msssim、pytest、ruff。

## Global Constraints

- `D:\Data\ARPES` 是只读原始实验档案，不得由扫描、训练、评估或推理代码写入。
- `D:\Projects\convert` 输出的 xarray/HDF5 是唯一标准交换格式；二维 cut 的规范维度顺序是 `("eV", "alpha")`。
- 标准 cut 必须可由 `xr.load_dataarray()` 直接读取，且 `eV`、`alpha` 是一维维度坐标。
- 旧 DCCNN H5 仅由显式只读适配器加载，不在原位修改或自动覆盖。
- 推理输出写入独立的 `<stem>_denoised.h5`，保留原坐标和实验属性，输入文件校验和保持不变。
- 第一阶段只处理二维 `cut` 和 `fineCut`；map、fastmap、DataTree 多节点选择不进入本计划的模型训练范围。
- 数据划分在裁剪前按样品和完整测量组完成；同一原图、配对组和所有派生裁剪只能属于一个 split。
- 训练默认采样比例为 A 级短/长配对 50%、B 级重复扫描 30%、C 级单张合成噪声 20%。
- Python 环境使用项目内 `.venv`、`pyproject.toml` 和提交到 Git 的 `uv.lock`。
- 不删除现有权重、H5、CSV、PNG、配置或旧脚本；归档操作先生成清单并默认 dry-run。
- 当前工作树中的用户修改不得混入各任务提交；每次只暂存任务列出的文件。

---

## 目标文件结构

```text
dccnn-arpes-main/
├─ pyproject.toml
├─ uv.lock
├─ README.md
├─ configs/
│  ├─ paths.example.yaml
│  ├─ metadata_aliases.yaml
│  └─ train_cut_v1.yaml
├─ src/dccnn_arpes/
│  ├─ __init__.py
│  ├─ cli/
│  │  ├─ data.py
│  │  ├─ train.py
│  │  ├─ eval.py
│  │  └─ denoise.py
│  ├─ io/
│  │  ├─ xarray_h5.py
│  │  └─ legacy_h5.py
│  ├─ data/
│  │  ├─ schema.py
│  │  ├─ discovery.py
│  │  ├─ metadata.py
│  │  ├─ pairing.py
│  │  ├─ splitting.py
│  │  ├─ transforms.py
│  │  ├─ noise.py
│  │  └─ dataset.py
│  ├─ models/
│  │  ├─ legacy_ccnn.py
│  │  └─ residual.py
│  ├─ training/
│  │  ├─ config.py
│  │  ├─ losses.py
│  │  ├─ checkpoints.py
│  │  └─ trainer.py
│  ├─ inference/
│  │  ├─ tiling.py
│  │  └─ pipeline.py
│  └─ evaluation/
│     ├─ metrics.py
│     ├─ baselines.py
│     └─ report.py
├─ scripts/
│  └─ inventory_legacy.py
└─ tests/
   ├─ conftest.py
   ├─ io/
   ├─ data/
   ├─ models/
   ├─ training/
   ├─ inference/
   └─ evaluation/
```

现有 `modules/`、`train/`、`converter/`、`gui/` 和旧 `scripts/` 在新流程验收前保持不变。

### Task 1: 建立可复现环境和新包骨架

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/dccnn_arpes/__init__.py`
- Create: `src/dccnn_arpes/cli/data.py`
- Create: `src/dccnn_arpes/cli/train.py`
- Create: `src/dccnn_arpes/cli/eval.py`
- Create: `src/dccnn_arpes/cli/denoise.py`
- Create: `tests/test_cli_smoke.py`
- Modify: `.gitignore`
- Create: `uv.lock`

**Interfaces:**
- Produces console commands `dccnn-data`, `dccnn-train`, `dccnn-eval`, `dccnn-denoise`.
- Establishes the import root `dccnn_arpes`.

- [ ] **Step 1: Write the failing CLI smoke test**

```python
from dccnn_arpes.cli import data, denoise, eval as eval_cli, train


def test_cli_modules_expose_main():
    assert callable(data.main)
    assert callable(train.main)
    assert callable(eval_cli.main)
    assert callable(denoise.main)
```

- [ ] **Step 2: Run the test and confirm the package does not yet exist**

Run: `uv run pytest tests/test_cli_smoke.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'dccnn_arpes'`.

- [ ] **Step 3: Add packaging and exact dependency groups**

Create a PEP 621 `pyproject.toml` with:

```toml
[project]
name = "dccnn-arpes"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "h5netcdf>=1.7",
  "h5py>=3.12",
  "matplotlib>=3.9",
  "netCDF4>=1.7",
  "numpy>=2.1",
  "openpyxl>=3.1",
  "pandas>=2.2",
  "pytorch-msssim>=1.0",
  "pyyaml>=6.0",
  "scikit-image>=0.24",
  "scipy>=1.14",
  "torch>=2.7",
  "tqdm>=4.66",
  "xarray>=2024.10",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-cov>=6.0", "ruff>=0.9"]

[project.scripts]
dccnn-data = "dccnn_arpes.cli.data:main"
dccnn-train = "dccnn_arpes.cli.train:main"
dccnn-eval = "dccnn_arpes.cli.eval:main"
dccnn-denoise = "dccnn_arpes.cli.denoise:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

Each initial CLI `main()` must use `argparse`, support `--help`, and return without filesystem mutation when only help is requested. Add `.venv/`, `workspace/`, `outputs/`, generated H5, checkpoints and metric artifacts to `.gitignore`; do not ignore `uv.lock`, `configs/`, tests or docs.

- [ ] **Step 4: Create and verify the locked environment**

Run:

```powershell
uv venv --python 3.12
uv lock
uv sync --extra dev
uv run pytest tests/test_cli_smoke.py -v
uv run ruff check src tests
```

Expected: the smoke test passes and Ruff reports no errors. Before CUDA training, install the official PyTorch wheel selected for the workstation driver and verify `torch.cuda.get_device_name(0)` contains `RTX 5080`; do not substitute a CPU-only wheel silently.

- [ ] **Step 5: Commit only the environment foundation**

```powershell
git add pyproject.toml uv.lock README.md .gitignore src/dccnn_arpes tests/test_cli_smoke.py
git commit -m "build: create reproducible dccnn arpes package"
```

### Task 2: 实现标准 xarray/HDF5 读取、校验和旧 H5 适配

**Files:**
- Create: `src/dccnn_arpes/io/__init__.py`
- Create: `src/dccnn_arpes/io/xarray_h5.py`
- Create: `src/dccnn_arpes/io/legacy_h5.py`
- Create: `tests/conftest.py`
- Create: `tests/io/test_xarray_h5.py`
- Create: `tests/io/test_legacy_h5.py`
- Modify: `src/dccnn_arpes/cli/data.py`

**Interfaces:**
- Produces `validate_cut(data: xr.DataArray) -> xr.DataArray`.
- Produces `load_cut(path: Path, *, allow_legacy: bool = False) -> xr.DataArray`.
- Produces `write_cut(data: xr.DataArray, path: Path, *, overwrite: bool = False) -> None`.
- Produces `load_legacy_cut(path: Path) -> xr.DataArray`.

- [ ] **Step 1: Write failing canonical-format tests**

Use a fixture:

```python
@pytest.fixture
def canonical_cut() -> xr.DataArray:
    return xr.DataArray(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        dims=("eV", "alpha"),
        coords={"eV": np.linspace(-0.3, 0.1, 4), "alpha": np.linspace(-10, 10, 5)},
        name="cut001",
        attrs={"temperature_K": 20.0, "sample_id": "sample-a"},
    )
```

Assert that `write_cut()` followed by `xr.load_dataarray()` preserves values, coordinates, dimension order and serializable attrs. Also assert:

```python
with pytest.raises(ValueError, match="missing required dimension"):
    validate_cut(canonical_cut.rename({"alpha": "x"}))

with pytest.raises(ValueError, match="strictly monotonic"):
    validate_cut(canonical_cut.assign_coords(eV=[0.0, 0.0, 0.1, 0.2]))
```

- [ ] **Step 2: Run tests and confirm missing interfaces**

Run: `uv run pytest tests/io/test_xarray_h5.py tests/io/test_legacy_h5.py -v`

Expected: FAIL because `dccnn_arpes.io` is absent.

- [ ] **Step 3: Implement the canonical contract**

`validate_cut()` must:

1. require an `xr.DataArray`, exactly two dimensions, and the set `{"eV", "alpha"}`;
2. transpose by dimension name to `("eV", "alpha")`;
3. require one-dimensional dimension coordinates whose lengths match the array;
4. reject empty arrays, nonnumeric data, NaN/Inf values, repeated coordinates and non-monotonic axes;
5. return `float32` data without changing coordinate values or attrs.

`load_cut()` first calls `xr.load_dataarray(path)`. If that fails and `allow_legacy=True`, call `load_legacy_cut()`; otherwise raise a message naming the path and explaining that convert xarray/HDF5 is required.

`write_cut()` validates first, writes to a sibling temporary file with `engine="h5netcdf"`, reopens it with `xr.load_dataarray()`, validates the reopened object, then atomically replaces the destination. It must refuse to overwrite an existing path unless an explicit `overwrite=True` parameter is passed.

`load_legacy_cut()` opens with `h5py.File(path, "r")`, requires `spectrum`, `energy`, `thetax`, constructs:

```python
xr.DataArray(
    spectrum,
    dims=("eV", "alpha"),
    coords={"eV": energy, "alpha": thetax},
    name=path.stem,
    attrs={"legacy_source": str(path), **serializable_root_attrs},
)
```

If `spectrum.T` is the only orientation matching both coordinates, transpose it explicitly; otherwise reject ambiguous shapes. Never open a legacy file in append or write mode.

- [ ] **Step 4: Add validation CLI and verify both paths**

Implement:

```powershell
dccnn-data validate D:\Projects\convert\data\converted_h5\NiNS0032.h5
dccnn-data validate D:\Projects\dccnn\dccnn-arpes-main\data\converted_h5\nbvt0005_txt.h5 --allow-legacy
```

The command prints path, object name, shape, dimension order, coordinate ranges and `standard` or `legacy-adapted`. It exits nonzero for invalid files. Run:

```powershell
uv run pytest tests/io -v
uv run dccnn-data validate D:\Projects\convert\data\converted_h5\NiNS0032.h5
```

Expected: all tests pass and the real convert file reports dimensions `eV, alpha`.

- [ ] **Step 5: Commit the unified data boundary**

```powershell
git add src/dccnn_arpes/io src/dccnn_arpes/cli/data.py tests/conftest.py tests/io
git commit -m "feat: add canonical xarray hdf5 data boundary"
```

### Task 3: 建立原始数据发现、实验记录解析和标准产物关联

**Files:**
- Create: `src/dccnn_arpes/data/__init__.py`
- Create: `src/dccnn_arpes/data/schema.py`
- Create: `src/dccnn_arpes/data/discovery.py`
- Create: `src/dccnn_arpes/data/metadata.py`
- Create: `configs/metadata_aliases.yaml`
- Create: `configs/paths.example.yaml`
- Create: `tests/data/test_discovery.py`
- Create: `tests/data/test_metadata.py`
- Modify: `src/dccnn_arpes/cli/data.py`

**Interfaces:**
- Produces immutable `ManifestRecord` with all fields in design section 4.
- Produces `scan_archive(root: Path) -> list[ManifestRecord]`.
- Produces `read_workbook_candidates(path: Path, aliases: Mapping) -> pd.DataFrame`.
- Produces `associate_converted(records, converted_root) -> list[ManifestRecord]`.
- Produces UTF-8 CSV manifest and a JSON audit report.

- [ ] **Step 1: Write failing discovery and metadata tests**

Create a temporary tree containing `.pxt`, `.txt`, `.bin` plus `.ini`, `.ibw`, `.zip`, `.xlsx`, an unrelated `.png`, and a convert H5. Assert only supported source files enter the manifest, `.bin` retains its `.ini` sidecar, paths are absolute, and no source file timestamp or hash changes.

Create a workbook where the second row has blank sample and temperature cells. Assert the parser emits candidate inherited values plus:

```python
assert rows.loc[1, "metadata_inherited"] is True
assert rows.loc[1, "review_status"] == "needs_review"
```

- [ ] **Step 2: Run tests and confirm discovery is absent**

Run: `uv run pytest tests/data/test_discovery.py tests/data/test_metadata.py -v`

Expected: FAIL on missing `dccnn_arpes.data.discovery`.

- [ ] **Step 3: Implement schema, scanning and workbook audit**

`ManifestRecord` must contain:

```text
record_id, source_path, converted_path, source_format, file_id,
sample_name, sample_id, session_id, acquisition_group, scan_type,
temperature_K, photon_energy_eV, polarization,
position_x, position_y, position_z, position_polar, position_tilt,
position_azimuth, energy_axis, angle_axis, acquisition_time_s,
sweep_count, pair_type, pair_id, review_status, split, quality_flag,
exclusion_reason, notes
```

Use SHA-256 of normalized absolute source path plus file size as stable `record_id`. Scanning is recursive and read-only. Store coordinate arrays in CSV as compact JSON arrays. `metadata_aliases.yaml` must include Chinese and English aliases for file ID, sample, temperature, photon energy, polarization, acquisition time and sweeps. Forward-filled workbook values are candidate metadata only and always set `review_status=needs_review`.

Association first matches explicit `source_path` attrs from the convert H5, then normalized exact stem, then unique file ID token. Ambiguous candidates remain unassociated and are recorded in `association_issues.json`; never choose the first match.

- [ ] **Step 4: Run a real read-only archive scan**

Run:

```powershell
uv run dccnn-data scan --source D:\Data\ARPES --converted D:\Projects\dccnn\workspace\converted --output D:\Projects\dccnn\workspace\manifests\records.csv
```

Verify the command creates only files under `D:\Projects\dccnn\workspace\manifests`. Compare pre/post source file counts and sampled SHA-256 values in the audit report. Review `unknown_excel_columns.json` and add only evidenced aliases to `configs/metadata_aliases.yaml`.

- [ ] **Step 5: Commit discovery without generated manifests**

```powershell
git add src/dccnn_arpes/data configs/metadata_aliases.yaml configs/paths.example.yaml tests/data
git commit -m "feat: index arpes sources and experiment metadata"
```

### Task 4: 实现配对审核和无泄漏 group-level 划分

**Files:**
- Create: `src/dccnn_arpes/data/pairing.py`
- Create: `src/dccnn_arpes/data/splitting.py`
- Create: `configs/data_cut_v1.yaml`
- Create: `tests/data/test_pairing.py`
- Create: `tests/data/test_splitting.py`
- Modify: `src/dccnn_arpes/cli/data.py`

**Interfaces:**
- Produces `classify_pair(left, right) -> PairDecision`.
- Produces `propose_pairs(records) -> tuple[list[PairRecord], list[PairDecision]]`.
- Produces `assign_group_splits(records, *, seed=20260727, ratios=(0.8, 0.1, 0.1))`.

- [ ] **Step 1: Write table-driven failing pairing tests**

Build records that differ one field at a time. Assert identical physical settings with different acquisition time/sweeps produce A, independent repeats produce B, and every prohibited difference—temperature, photon energy, polarization, position, surface treatment/state, coordinate range, shape, or cut/map type—produces a rejected decision with the exact differing field in `exclusion_reason`.

For splitting, assert all rows sharing any of `sample_id`, `acquisition_group`, `pair_id` or `source_path` remain in one split and that the same seed produces byte-identical CSV output.

- [ ] **Step 2: Run tests and confirm failures**

Run: `uv run pytest tests/data/test_pairing.py tests/data/test_splitting.py -v`

Expected: FAIL because pairing and splitting modules do not exist.

- [ ] **Step 3: Implement conservative pair decisions**

Compare numeric settings with tolerances from `configs/data_cut_v1.yaml`:

```yaml
pairing:
  position_atol: 1.0e-6
  coordinate_rtol: 1.0e-6
  coordinate_atol: 1.0e-8
split:
  seed: 20260727
  train: 0.80
  val: 0.10
  test: 0.10
```

Temperature and photon energy require exact equality after numeric parsing. Missing fields never count as equal automatically. A candidate using inherited Excel metadata remains `needs_review`.

Build split connected components over sample, acquisition group, pair and source relationships. Assign whole components deterministically to 80/10/10 while minimizing record-count error. Reserve at least one complete material/sample component for test when three or more samples exist.

- [ ] **Step 4: Add CLI outputs and leakage audit**

Implement:

```powershell
dccnn-data pairs --manifest D:\Projects\dccnn\workspace\manifests\records.csv --output D:\Projects\dccnn\workspace\manifests\pairs.csv
dccnn-data split --manifest D:\Projects\dccnn\workspace\manifests\records.csv --pairs D:\Projects\dccnn\workspace\manifests\pairs.csv --output D:\Projects\dccnn\workspace\splits
```

The split command writes `train.csv`, `val.csv`, `test.csv` and `split_audit.json`; it exits nonzero if a connected component appears in multiple splits.

- [ ] **Step 5: Commit pairing and splitting**

```powershell
git add src/dccnn_arpes/data/pairing.py src/dccnn_arpes/data/splitting.py src/dccnn_arpes/cli/data.py tests/data configs/data_cut_v1.yaml
git commit -m "feat: add reviewed pairing and leakage-safe splits"
```

### Task 5: 实现共享归一化、动态噪声和混合训练数据集

**Files:**
- Create: `src/dccnn_arpes/data/transforms.py`
- Create: `src/dccnn_arpes/data/noise.py`
- Create: `src/dccnn_arpes/data/dataset.py`
- Create: `tests/data/test_transforms.py`
- Create: `tests/data/test_noise.py`
- Create: `tests/data/test_dataset.py`

**Interfaces:**
- Produces `TransformStats(lower: float, scale: float)`.
- Produces `IntensityTransform.fit(input_array)`, `forward(array, stats)`, `inverse(array, stats)`.
- Produces `NoiseParameters` and `synthesize_noisy(clean, params, rng)`.
- Produces `ArpesCutDataset.set_epoch(epoch)` and samples `(input, target, metadata)`.

- [ ] **Step 1: Write failing numerical and sampling tests**

Assert normalization/inversion round-trip error is below `1e-5` relative tolerance, and the same input-derived statistics are applied to both input and target. Assert Poisson/background/stripe synthesis is deterministic for the same seed and changes across epochs. Assert a fixed synthetic manifest yields approximately 50/30/20 A/B/C samples over 1,000 draws and that paired crops use identical pixel origins.

- [ ] **Step 2: Run tests and confirm missing modules**

Run: `uv run pytest tests/data/test_transforms.py tests/data/test_noise.py tests/data/test_dataset.py -v`

Expected: FAIL on missing transform, noise and dataset APIs.

- [ ] **Step 3: Implement reversible preprocessing**

For every sample:

1. divide by `acquisition_time_s` when present, otherwise by `sweep_count` when present;
2. reject A-level pairing if neither scale is available and the pair has not been manually approved;
3. clip only negative count artifacts to zero;
4. apply `log1p`;
5. fit input-derived robust limits at quantiles 0.01 and 0.995;
6. use `scale=max(upper-lower, 1e-6)` and normalize both input and target with the same statistics;
7. retain `TransformStats` for inverse transformation.

Do not normalize input and target independently.

- [ ] **Step 4: Implement dynamic A/B/C sampling**

- A loads short acquisition as input and long acquisition as target after count-rate conversion.
- B loads one repeat as input and another repeat—or the mean of remaining repeats—as target.
- C treats the measured cut as target and adds calibrated Poisson, low-frequency background and row/column stripe noise to form input.
- Ten percent of accepted clean samples use input equal to target as an identity constraint.
- Random seed derives from `(base_seed, epoch, dataset_index, record_id)`; `set_epoch()` changes crops/noise while fixed seed and epoch remain reproducible.
- Return tensors shaped `[1, crop_eV, crop_alpha]` plus record ID, pair type, crop coordinates and transform stats.

Run: `uv run pytest tests/data -v`

Expected: all data tests pass.

- [ ] **Step 5: Commit the training data pipeline**

```powershell
git add src/dccnn_arpes/data tests/data
git commit -m "feat: add reproducible mixed arpes cut dataset"
```

### Task 6: 复现 LegacyCCNN 并实现 ResidualDenoiser2D

**Files:**
- Create: `src/dccnn_arpes/models/__init__.py`
- Create: `src/dccnn_arpes/models/legacy_ccnn.py`
- Create: `src/dccnn_arpes/models/residual.py`
- Create: `tests/models/test_legacy_ccnn.py`
- Create: `tests/models/test_residual.py`

**Interfaces:**
- Produces `LegacyCCNN(kernel_size=3, num_layers=7)`.
- Produces `load_legacy_checkpoint(model, path)`.
- Produces `ResidualDenoiser2D(channels=64, blocks=8)`.
- Produces `denoise_forward(model, input_tensor) -> tuple[Tensor, Tensor | None]`.

- [ ] **Step 1: Write failing architecture and compatibility tests**

Assert `LegacyCCNN` uses the same `layers.*` and `final.*` state-dict keys as `modules/models/ccnn.py`, preserves `[N,1,H,W]`, and produces identical output after copying a seeded old-model state dict. Assert `ResidualDenoiser2D` preserves arbitrary odd spatial sizes, contains no pooling, transpose convolution or BatchNorm, and returns:

```python
denoised, predicted_noise = model(x)
torch.testing.assert_close(denoised, x - predicted_noise)
```

Assert `denoise_forward()` returns `(legacy_output, None)` for LegacyCCNN and the model tuple unchanged for ResidualDenoiser2D so training, evaluation and inference share one prediction contract.

- [ ] **Step 2: Run tests and confirm failures**

Run: `uv run pytest tests/models -v`

Expected: FAIL because the new model modules do not exist.

- [ ] **Step 3: Implement both models**

Copy the legacy architecture exactly: first `Conv2d(1,64,3,padding=1)` plus PReLU, five `Conv2d(64,64,3,padding=1)` plus PReLU stages for `num_layers=7`, and final `Conv2d(64,1,3,padding=1)`.

Implement each residual block as two 3×3 64-channel convolutions with PReLU after the first convolution and a local skip. The candidate has an input convolution, eight blocks, a noise-output convolution, and global subtraction. Initialize the final noise convolution to zero so initial denoised output is identity.

`load_legacy_checkpoint()` accepts a raw state dict or dictionaries containing `state_dict` or `model_state_dict`, strips an optional `module.` prefix, and fails with explicit missing/unexpected keys.

All training and inference code calls `denoise_forward()` rather than branching on model class names.

- [ ] **Step 4: Verify models on CPU and one CUDA tensor**

Run:

```powershell
uv run pytest tests/models -v
uv run python -c "import torch; from dccnn_arpes.models.residual import ResidualDenoiser2D; m=ResidualDenoiser2D().cuda(); print(m(torch.zeros(1,1,257,259,device='cuda'))[0].shape)"
```

Expected: tests pass and CUDA prints `torch.Size([1, 1, 257, 259])`.

- [ ] **Step 5: Commit models**

```powershell
git add src/dccnn_arpes/models tests/models
git commit -m "feat: add legacy and residual denoising models"
```

### Task 7: 实现无静默降级的复合物理保真损失

**Files:**
- Create: `src/dccnn_arpes/training/__init__.py`
- Create: `src/dccnn_arpes/training/losses.py`
- Create: `tests/training/test_losses.py`

**Interfaces:**
- Produces `CompositeDenoisingLoss(charbonnier=0.80, ms_ssim=0.15, gradient=0.05)`.
- Returns total loss and detached component metrics.

- [ ] **Step 1: Write failing loss tests**

For seeded `[2,1,256,256]` tensors, assert total loss is finite, backward produces finite gradients, identical tensors have lower loss than shifted tensors, and:

```python
expected = 0.80 * parts["charbonnier"] + 0.15 * parts["ms_ssim"] + 0.05 * parts["gradient"]
torch.testing.assert_close(total.detach(), expected)
```

Patch import resolution so a missing `pytorch_msssim` raises `RuntimeError` naming the required dependency; a fallback approximation is forbidden.

- [ ] **Step 2: Run tests and confirm missing loss**

Run: `uv run pytest tests/training/test_losses.py -v`

Expected: FAIL on missing `CompositeDenoisingLoss`.

- [ ] **Step 3: Implement exact components**

- Charbonnier: `mean(sqrt((prediction-target)^2 + 1e-6))`.
- MS-SSIM: clamp only the two SSIM views to `[0, 1]`, then calculate `1 - pytorch_msssim.ms_ssim(..., data_range=1.0, size_average=True)`. Keep unclipped tensors for Charbonnier, gradient loss and inverse transformation.
- Gradient: mean Charbonnier error of first differences along both `eV` and `alpha`.
- Validate weights are nonnegative and sum to one within `1e-8`.
- Record each unweighted component separately for experiment analysis.

- [ ] **Step 4: Run focused and complete unit tests**

Run:

```powershell
uv run pytest tests/training/test_losses.py -v
uv run pytest tests/io tests/data tests/models tests/training -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit loss implementation**

```powershell
git add src/dccnn_arpes/training tests/training
git commit -m "feat: add structure-preserving denoising loss"
```

### Task 8: 实现配置、训练循环、checkpoint 和实验溯源

**Files:**
- Create: `src/dccnn_arpes/training/config.py`
- Create: `src/dccnn_arpes/training/checkpoints.py`
- Create: `src/dccnn_arpes/training/trainer.py`
- Create: `configs/train_cut_v1.yaml`
- Create: `tests/training/test_config.py`
- Create: `tests/training/test_checkpoint.py`
- Create: `tests/training/test_trainer_smoke.py`
- Modify: `src/dccnn_arpes/cli/train.py`

**Interfaces:**
- Produces validated `TrainConfig`.
- Produces `run_training(config: TrainConfig) -> TrainingResult`.
- Checkpoint schema includes model/optimizer/scaler state, epoch, best metric, config, hashes and versions.

- [ ] **Step 1: Write failing config, checkpoint and CPU smoke tests**

Use two tiny canonical H5 files and fixed manifest rows. Train a four-channel, one-block residual model for two CPU epochs. Assert:

- `best.pt` and `last.pt` exist;
- `metrics.csv` contains total and three component losses for train/val;
- `run.json` contains seed, manifest SHA-256, split SHA-256, Git commit, Python/PyTorch/CUDA versions and device;
- reloading `last.pt` restores epoch and produces identical model output;
- NaN input stops with `FloatingPointError`.

- [ ] **Step 2: Run tests and confirm training APIs are absent**

Run: `uv run pytest tests/training/test_config.py tests/training/test_checkpoint.py tests/training/test_trainer_smoke.py -v`

Expected: FAIL on missing config/trainer/checkpoint modules.

- [ ] **Step 3: Implement strict config and reproducibility**

Create `configs/train_cut_v1.yaml` with this baseline:

```yaml
paths:
  manifest: D:/Projects/dccnn/workspace/manifests/records.csv
  pairs: D:/Projects/dccnn/workspace/manifests/pairs.csv
  splits: D:/Projects/dccnn/workspace/splits
  output: D:/Projects/dccnn/outputs/experiments
seed: 20260727
model:
  name: residual_denoiser_2d
  channels: 64
  blocks: 8
data:
  crop_size: [256, 256]
  samples_per_epoch: 10000
  sampling:
    A: 0.50
    B: 0.30
    C: 0.20
  identity_probability: 0.10
  noise:
    poisson_peak_counts: [50.0, 5000.0]
    background_fraction: [0.0, 0.08]
    stripe_probability: 0.30
    stripe_fraction: [0.0, 0.05]
training:
  batch_size: 6
  epochs: 100
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  workers: 4
  device: cuda
  amp: true
loss:
  charbonnier: 0.80
  ms_ssim: 0.15
  gradient: 0.05
```

Unknown keys and missing required keys are errors. The `--smoke-test` option overrides only epochs, samples per epoch, model channels/blocks and output subdirectory; it does not rewrite the YAML file.

Seed Python, NumPy, CPU/CUDA PyTorch and DataLoader workers. At each epoch call `dataset.set_epoch(epoch)`. Use AdamW, mixed precision only on CUDA, validation composite loss for best checkpoint, and stop immediately on nonfinite data/loss/gradients.

- [ ] **Step 4: Verify CPU training and RTX 5080 smoke**

Run:

```powershell
uv run pytest tests/training -v
uv run dccnn-train --config configs/train_cut_v1.yaml --smoke-test --device cpu
uv run dccnn-train --config configs/train_cut_v1.yaml --smoke-test --device cuda
```

Expected: both smoke runs finish, CUDA metadata names the RTX 5080, and each run uses a separate timestamp-plus-config-hash output directory.

- [ ] **Step 5: Commit reproducible training**

```powershell
git add src/dccnn_arpes/training src/dccnn_arpes/cli/train.py configs/train_cut_v1.yaml tests/training
git commit -m "feat: add reproducible arpes denoising training"
```

### Task 9: 实现保持坐标和属性的安全推理

**Files:**
- Create: `src/dccnn_arpes/inference/__init__.py`
- Create: `src/dccnn_arpes/inference/tiling.py`
- Create: `src/dccnn_arpes/inference/pipeline.py`
- Create: `tests/inference/test_tiling.py`
- Create: `tests/inference/test_pipeline.py`
- Modify: `src/dccnn_arpes/cli/denoise.py`

**Interfaces:**
- Produces `tiled_predict(model, tensor, tile_size, overlap) -> Tensor`.
- Produces `denoise_file(input_path, checkpoint_path, output_dir) -> Path`.

- [ ] **Step 1: Write failing tiling and immutability tests**

Use an identity model to assert tiled and full-image outputs match within `1e-6`, including dimensions smaller than a tile and odd shapes. For `denoise_file()`, hash the input before/after and assert equality. Reopen output with `xr.load_dataarray()` and assert equal dims, coords and original attrs plus:

```text
denoising_model
denoising_checkpoint_sha256
denoising_timestamp_utc
denoising_transform
```

Assert output path is `<stem>_denoised.h5` and an existing destination is not overwritten by default.

- [ ] **Step 2: Run tests and confirm inference is absent**

Run: `uv run pytest tests/inference -v`

Expected: FAIL on missing inference modules.

- [ ] **Step 3: Implement tiled inference and atomic output**

Validate input through `load_cut()`, derive preprocessing stats from the input, predict normalized noise/denoised values, inverse-transform to physical count-rate scale, and rebuild a DataArray using the exact original coordinates and serializable attrs. Blend overlapping tiles with a separable Hann weight and reject `overlap >= tile_size`.

Write through `write_cut()` to a new destination. Never call `h5py.File(..., "a")` and never add `denoised_*` datasets to the input.

- [ ] **Step 4: Run file-level and real-file smoke tests**

Run:

```powershell
uv run pytest tests/inference -v
$denoiseCheckpoint = Get-ChildItem -Path D:\Projects\dccnn\outputs\experiments -Recurse -Filter best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
uv run dccnn-denoise --input D:\Projects\convert\data\converted_h5\NiNS0032.h5 --checkpoint $denoiseCheckpoint --output D:\Projects\dccnn\outputs\inference
uv run dccnn-data validate D:\Projects\dccnn\outputs\inference\NiNS0032_denoised.h5
```

The real-file command is run only after a selected checkpoint exists. Expected: standard validation passes and the source SHA-256 is unchanged.

- [ ] **Step 5: Commit inference**

```powershell
git add src/dccnn_arpes/inference src/dccnn_arpes/cli/denoise.py tests/inference
git commit -m "feat: add coordinate-preserving safe inference"
```

### Task 10: 实现定量指标、传统基线和物理保真报告

**Files:**
- Create: `src/dccnn_arpes/evaluation/__init__.py`
- Create: `src/dccnn_arpes/evaluation/metrics.py`
- Create: `src/dccnn_arpes/evaluation/baselines.py`
- Create: `src/dccnn_arpes/evaluation/report.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `tests/evaluation/test_report.py`
- Modify: `src/dccnn_arpes/cli/eval.py`

**Interfaces:**
- Produces `evaluate_pair(input_da, output_da, reference_da) -> dict[str, float]`.
- Produces Gaussian and median baseline DataArrays preserving coordinates.
- Produces per-file CSV, summary JSON and fixed-scale preview/EDC/MDC figures.

- [ ] **Step 1: Write failing analytical tests**

Use analytic Gaussian peaks with known center, FWHM and integral. Assert:

- identical arrays give MAE/NRMSE zero and SSIM one;
- a one-bin shift is reported as one coordinate sampling step;
- fitted FWHM is within 2% of the analytic value;
- integrated-intensity error is zero for identical arrays;
- all input/output/reference coordinate mismatches raise before metric calculation.

- [ ] **Step 2: Run tests and confirm evaluation is absent**

Run: `uv run pytest tests/evaluation -v`

Expected: FAIL on missing evaluation modules.

- [ ] **Step 3: Implement metrics and baselines**

Calculate MAE, NRMSE, PSNR, SSIM, EDC/MDC correlation, peak-position error, FWHM relative error, integrated-intensity relative error and noise-only-region reduction. Implement `scipy.ndimage.gaussian_filter` and `median_filter` baselines without changing coords/attrs.

Use coordinate values—not pixel indexes—for peak/FWHM results. When a fit is invalid, return a named status field and NaN for that metric; keep the file in reports rather than dropping it.

- [ ] **Step 4: Implement acceptance and temperature-series reporting**

The report compares raw input, Gaussian, median, LegacyCCNN and ResidualDenoiser2D. It writes every file, summary statistics, worst cases, fixed color-scale panels, difference maps and representative EDC/MDC curves.

For temperature groups, sort by `temperature_K` and report peak-position, FWHM and integrated-intensity trends without treating adjacent temperatures as denoising targets. For high-quality identity samples, flag any new peak/stripe or threshold violation.

`acceptance.json` must contain a pass/fail/evidence entry for each rule:

1. at least 80% of paired test cuts have lower NRMSE than their raw input;
2. mean paired-test NRMSE is at least 10% lower than LegacyCCNN;
3. peak-position error is no larger than one `eV` or `alpha` sampling step;
4. FWHM relative error is at most 10%;
5. count-rate-normalized integrated-intensity relative error is at most 5%;
6. high-quality identity cuts contain no new peak with prominence above 5% of the reference maximum and no new stripe above five background standard deviations; flagged cases require review rather than automatic removal;
7. temperature trend direction is not reversed and no adjacent output jump exceeds both three times the corresponding input-series median absolute jump and the measurement uncertainty;
8. the number of evaluated, failed-fit and manually flagged samples reconciles exactly to the locked test manifest row count.

Run:

```powershell
uv run pytest tests/evaluation -v
uv run dccnn-eval --config configs/train_cut_v1.yaml --split D:\Projects\dccnn\workspace\splits\test.csv --output D:\Projects\dccnn\outputs\evaluation
```

Expected: report artifacts exist and failed fits/samples remain visible.

- [ ] **Step 5: Commit evaluation**

```powershell
git add src/dccnn_arpes/evaluation src/dccnn_arpes/cli/eval.py tests/evaluation
git commit -m "feat: add physical-fidelity evaluation suite"
```

### Task 11: 建立 legacy 资产清单并完成端到端验收

**Files:**
- Create: `scripts/inventory_legacy.py`
- Create: `tests/test_inventory_legacy.py`
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces a read-only legacy inventory containing path, type, size, modification time, SHA-256 and proposed destination.
- Defaults to reporting only; this task does not delete or move legacy files.

- [ ] **Step 1: Write the failing inventory safety test**

Create a temporary legacy tree and assert inventory output is stable, duplicate hashes are grouped, proposed destinations remain under a supplied archive root, and source file hashes/timestamps are unchanged. Reject archive roots equal to the repository root, drive root or source root.

- [ ] **Step 2: Run test and confirm the script is absent**

Run: `uv run pytest tests/test_inventory_legacy.py -v`

Expected: FAIL because the inventory script is missing.

- [ ] **Step 3: Implement read-only inventory and documentation**

The script accepts:

```powershell
uv run python scripts/inventory_legacy.py --repo D:\Projects\dccnn\dccnn-arpes-main --archive D:\Projects\dccnn\legacy_archive --output D:\Projects\dccnn\workspace\manifests\legacy_inventory.csv
```

It classifies old checkpoints, H5, CSV, PNG, result directories and configs, but performs no move/delete. README must document:

1. `uv sync --extra dev`;
2. convert as the only raw-to-xarray/HDF5 converter;
3. data validation, scan, pair, split, train, evaluate and denoise commands;
4. workspace/output directory roles;
5. legacy H5 read-only compatibility;
6. RTX 5080 verification;
7. no-Anaconda requirement.

- [ ] **Step 4: Run the complete verification matrix**

Run:

```powershell
uv run ruff check src tests scripts
uv run pytest -v
uv run pytest --cov=dccnn_arpes --cov-report=term-missing
uv run dccnn-data validate D:\Projects\convert\data\converted_h5\NiNS0032.h5
uv run dccnn-train --config configs/train_cut_v1.yaml --smoke-test --device cpu
uv run dccnn-train --config configs/train_cut_v1.yaml --smoke-test --device cuda
```

Expected: lint and all tests pass, canonical H5 validates, both smoke trainings finish, and source archive/inference inputs retain their pre-run hashes.

- [ ] **Step 5: Commit the handoff documentation and inventory**

```powershell
git add scripts/inventory_legacy.py tests/test_inventory_legacy.py README.md .gitignore
git commit -m "docs: complete dccnn arpes workflow handoff"
```

## 实施检查点

1. Task 2 完成后，先用 convert 的真实 H5 和一个旧 DCCNN H5 验证统一入口。
2. Task 4 完成后，人工复核 `pairs.csv` 和 `split_audit.json`，锁定测试集后再进入训练数据实现。
3. Task 8 完成后，先进行 CPU 与 RTX 5080 smoke test，再启动长训练。
4. Task 10 完成后，只有同时通过定量降噪和物理保真标准，ResidualDenoiser2D 才能替代 LegacyCCNN。
5. legacy 资产移动或删除不属于本计划的自动步骤；先审阅 Task 11 的清单，再单独授权归档动作。
