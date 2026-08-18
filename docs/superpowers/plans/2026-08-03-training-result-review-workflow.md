# Training Result Review Workflow Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan task-by-task with review checkpoints.

**Goal:** Complete a split-aware comparison of the two finished training runs, select the better checkpoint using real paired data and physical-fidelity checks, and decide whether more A-pair data or another training run is justified.

**Architecture:** Treat each finished run as an immutable experiment. First verify lifecycle and provenance artifacts, then compare the best checkpoints on the same reviewed A-pair manifest, and only afterward inspect representative images. The C population has no clean paired reference, so it is reported as qualitative/population evidence rather than a supervised denoising score.

**Tech Stack:** PowerShell, `uv`, Python, pytest, the existing `scripts/evaluate_real_pairs.py`, CSV/JSON artifacts, ImageTool Manager.

## Global Constraints

- Do not overwrite existing experiment or evaluation directories.
- Use `best.pt` for checkpoint selection; use `last.pt` only as a diagnostic comparison.
- Report train and validation A pairs separately; never combine them as one generalization score.
- Do not claim quantitative denoising quality for C records without a clean reference.
- Keep `amp: false` until a separate AMP stability experiment is explicitly justified.

---

### Task 1: Freeze and audit both formal runs

**Files/artifacts:**
- Read: `D:/Projects/dccnn/outputs/experiments/20260728T042313.895922Z-48e5215dd4db/run.json`
- Read: `D:/Projects/dccnn/outputs/experiments/20260728T042313.895922Z-48e5215dd4db/metrics.csv`
- Read: `D:/Projects/dccnn/outputs/experiments/20260728T133840.539098Z-649d7053890c/run.json`
- Read: `D:/Projects/dccnn/outputs/experiments/20260728T133840.539098Z-649d7053890c/metrics.csv`

- [ ] Confirm both `run.json` files have `status=completed`, no error fields, and `best.pt`/`last.pt` exist.
- [ ] Compute the minimum `val_total`, its epoch, final validation loss, and final train loss for each run.
- [ ] Record the run seed, config hash, data hash, split hash, checkpoint path, and start/end timestamps.

Expected current baseline:

| Run | Seed | Best epoch | Best val loss | Final val loss |
|---|---:|---:|---:|---:|
| `20260728T042313.895922Z-48e5215dd4db` | 20260727 | 44 | 0.03187245 | 0.03215693 |
| `20260728T133840.539098Z-649d7053890c` | 20260728 | 53 | 0.03197882 | 0.03224551 |

### Task 2: Evaluate both best checkpoints on the same A pairs

**Files/artifacts:**
- Use: `D:/Projects/dccnn/dccnn-arpes-main/scripts/evaluate_real_pairs.py`
- Use: `D:/Projects/dccnn/workspace/manifests/records.csv`
- Use: `D:/Projects/dccnn/workspace/manifests/pairs.csv`
- Create: `D:/Projects/dccnn/outputs/evaluation/real_pairs_seed20260727_best`
- Create: `D:/Projects/dccnn/outputs/evaluation/real_pairs_seed20260728_best`

- [ ] Keep the existing first-run evaluation as the seed-20260727 reference.
- [ ] Run the evaluator for the second checkpoint into a new directory:

```powershell
Set-Location D:\Projects\dccnn\dccnn-arpes-main
uv run python scripts\evaluate_real_pairs.py `
  --manifest D:\Projects\dccnn\workspace\manifests\records.csv `
  --pairs D:\Projects\dccnn\workspace\manifests\pairs.csv `
  --checkpoint D:\Projects\dccnn\outputs\experiments\20260728T133840.539098Z-649d7053890c\best.pt `
  --output D:\Projects\dccnn\outputs\evaluation\real_pairs_seed20260728_best
```

- [ ] Compare `pair_metrics.csv` row-by-row for the same three pair IDs.
- [ ] Use the validation row (`A_Ni033NbSe20028_Ni033NbSe20029`) as the primary model-selection signal.
- [ ] Use the two train rows only to detect fitting/overfitting behavior.

### Task 3: Visual and physical-fidelity review

**Artifacts:**
- Read both evaluation `summary.json` files and `pair_metrics.csv` files.
- Open the three raw/denoised/reference triplets in ImageTool Manager.

- [ ] Inspect MS3Co and nbvt for noise suppression, peak broadening, and loss of fine structure.
- [ ] Inspect Ni033 specifically for whether the small change preserves the broad peak and intensity.
- [ ] Check `ssim`, EDC/MDC correlation, peak-position error, FWHM error, and integrated-intensity error before accepting visually stronger smoothing.
- [ ] Record any case where NRMSE improves but a physical feature degrades.

### Task 4: Population-only C evaluation

- [ ] Keep the existing `population-only-20260728` report labeled `not_evaluated` for supervised metrics.
- [ ] Run both checkpoints on the 19 test C files only for qualitative side-by-side images and distribution summaries.
- [ ] Do not use C output alone to declare one checkpoint superior; there is no clean reference for C.
- [ ] Select 3 to 5 representative C files covering clean, moderate-noise, and severe-noise cases for manual review.

### Task 5: Decision gate

- [ ] Select seed 20260728 only if its validation A metrics improve or remain comparable while physical-fidelity errors do not worsen.
- [ ] If both seeds are similar and Ni033 remains near 1% improvement, stop changing epochs; the limiting factor is training-pair coverage/domain mismatch.
- [ ] If one seed is clearly better, keep its `best.pt` and record the checkpoint SHA-256 in the evaluation report.

### Task 6: Expand A-pair coverage before another training run

- [ ] Screen repeated-exposure C groups: `NiNbSe2` (19 files), `V3S4` (29 files), and `nbvt` (27 files).
- [ ] Confirm identical coordinates and acquisition conditions manually before adding any pair.
- [ ] Add only manually approved pairs to `pairs.csv`, preserving split boundaries and avoiding connected-component leakage.
- [ ] Put Ni033-like moderate-noise pairs in the training split only after adding separate validation pairs from the same noise regime.

### Task 7: Retrain only after the evidence review

- [ ] Keep model architecture, `amp: false`, and learning rate fixed for the first data-coverage experiment.
- [ ] Change one factor at a time, preferably adding reviewed A pairs before changing sampling weights.
- [ ] Give the new run a new seed/config and a new output directory; do not modify either completed run.
- [ ] Repeat Tasks 1–5 after the new run.

## Completion criteria

The review is complete when both best checkpoints have comparable A-pair CSV/JSON reports, the validation pair has been visually checked, C population outputs are explicitly labeled qualitative, and a written decision identifies the selected checkpoint or the exact data gap blocking selection.
