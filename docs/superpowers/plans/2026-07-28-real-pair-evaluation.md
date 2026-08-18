# Real Paired ARPES Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible evaluator that runs the scientific checkpoint on reviewed A-type pairs and compares raw low-exposure and denoised outputs with count-rate-normalized high-exposure references.

**Architecture:** Put reusable pair orientation, normalization, metric aggregation, and report writing in `src/dccnn_arpes/evaluation/real_pairs.py`. Add a thin `scripts/evaluate_real_pairs.py` CLI wrapper that reads the existing manifest/pairs CSVs, invokes `dccnn_arpes.inference.denoise_file`, and writes pair-level CSV/JSON artifacts under `D:\Projects\dccnn\outputs` without overwriting existing files.

**Tech Stack:** Python 3.12, xarray, NumPy, existing `evaluate_pair` metrics, existing canonical HDF5 loader/writer, pytest.

## Global Constraints

- Only reviewed `pair_type=A` pairs are eligible.
- Exposure scale uses `acquisition_time_s` first, then `sweep_count`.
- Lower exposure is the input; higher exposure is the reference.
- Input, output, and reference are divided by their respective exposure scales before metrics.
- Existing H5 inputs and checkpoints are read-only; output must be under `D:\Projects\dccnn\outputs`.
- Existing output files must not be overwritten.

---

### Task 1: Add failing unit tests for pair orientation and normalization

**Files:**
- Create: `tests/evaluation/test_real_pairs.py`
- Test: `src/dccnn_arpes/evaluation/real_pairs.py` (not present yet)

**Interfaces:**
- Test `effective_exposure(record: ManifestRecord) -> float`.
- Test `orient_pair(left: ManifestRecord, right: ManifestRecord) -> tuple[ManifestRecord, ManifestRecord]`.
- Test `count_rate_normalize(data: xr.DataArray, scale: float) -> xr.DataArray`.

- [ ] **Step 1: Write the failing tests**

```python
def test_orient_pair_uses_acquisition_time_before_sweeps():
    short = _record("short", acquisition_time_s=2.0, sweep_count=20)
    long = _record("long", acquisition_time_s=8.0, sweep_count=1)
    assert orient_pair(long, short) == (short, long)


def test_orient_pair_falls_back_to_sweeps():
    one = _record("one", acquisition_time_s=None, sweep_count=1)
    ten = _record("ten", acquisition_time_s=None, sweep_count=10)
    assert orient_pair(ten, one) == (one, ten)


def test_count_rate_normalize_scales_values_and_neutralizes_attrs():
    data = xr.DataArray(
        np.full((2, 2), 6.0),
        dims=("eV", "alpha"),
        coords={"eV": [1.0, 2.0], "alpha": [3.0, 4.0]},
        attrs={"acquisition_time_s": 3.0},
    )
    normalized = count_rate_normalize(data, 3.0)
    np.testing.assert_allclose(normalized.values, 2.0)
    assert normalized.attrs["acquisition_time_s"] == 1.0
    assert normalized.attrs.get("sweep_count") in (None, "")
```

- [ ] **Step 2: Run the tests and verify they fail for the intended reason**

Run: `uv run pytest tests/evaluation/test_real_pairs.py -q`

Expected: collection/import failure because `dccnn_arpes.evaluation.real_pairs` does not exist yet.

### Task 2: Implement reusable pair evaluation helpers

**Files:**
- Create: `src/dccnn_arpes/evaluation/real_pairs.py`
- Modify: `src/dccnn_arpes/evaluation/__init__.py`

**Interfaces:**
- `effective_exposure(record) -> float` raises `ValueError` for missing/non-positive scales.
- `orient_pair(left, right) -> (input_record, reference_record)`.
- `count_rate_normalize(data, scale) -> xr.DataArray` preserves coordinates and resets scale attrs.
- `compare_pair(input_da, output_da, reference_da) -> dict[str, object]` returns existing `evaluate_pair` metrics for raw and denoised candidates.

- [ ] **Step 1: Implement the three helpers with the exact validation rules from the tests.**
- [ ] **Step 2: Re-export the public helpers from `dccnn_arpes.evaluation`.**
- [ ] **Step 3: Run `uv run pytest tests/evaluation/test_real_pairs.py -q` and verify all tests pass.**

### Task 3: Add the real-pair CLI evaluator

**Files:**
- Create: `scripts/evaluate_real_pairs.py`
- Modify: `tests/evaluation/test_real_pairs.py`

**Interfaces:**
- CLI arguments: `--manifest`, `--pairs`, `--checkpoint`, `--output`.
- Writes `<output>/denoised/<input_stem>_denoised.h5`, `<output>/pair_metrics.csv`, and `<output>/summary.json`.
- Refuses output paths outside `D:\Projects\dccnn\outputs` and refuses existing denoised files.

- [ ] **Step 1: Add a failing test for split propagation and report columns using two temporary canonical cuts and one A pair.**
- [ ] **Step 2: Run the focused test and verify it fails because the CLI/report function is absent.**
- [ ] **Step 3: Implement CSV loading, A-pair filtering, record-ID resolution, pair orientation, inference invocation, normalized raw/denoised comparisons, and CSV/JSON report writing.**
- [ ] **Step 4: Run focused tests and verify pass.**

### Task 4: Run the current real-pair evaluation and verify artifacts

**Files:**
- Create at runtime: `D:\Projects\dccnn\outputs\evaluation\real_pairs_best_epoch44\`

- [ ] **Step 1: Run:**

```powershell
uv run python scripts\evaluate_real_pairs.py `
  --manifest D:\Projects\dccnn\workspace\manifests\records.csv `
  --pairs D:\Projects\dccnn\workspace\manifests\pairs.csv `
  --checkpoint D:\Projects\dccnn\outputs\experiments\20260728T042313.895922Z-48e5215dd4db\best.pt `
  --output D:\Projects\dccnn\outputs\evaluation\real_pairs_best_epoch44
```

- [ ] **Step 2: Confirm three A pairs are evaluated, split labels are preserved, the denoised H5 files validate, and `pair_metrics.csv` contains raw-versus-denoised columns.**
- [ ] **Step 3: Run the full test suite: `uv run pytest -q`.**
