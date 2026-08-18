# Real Paired ARPES Evaluation Design

## Purpose

Provide a reproducible command-line evaluator for reviewed A-type ARPES pairs.
The evaluator will compare a lower-exposure input, the denoised checkpoint
output, and a higher-exposure reference after count-rate normalization.

## Inputs and outputs

The command accepts a manifest CSV, reviewed pairs CSV, a version-2 checkpoint,
and an output directory. It selects only A pairs, resolves record IDs through
the manifest, runs inference on the lower-exposure endpoint, and writes:

- one denoised canonical H5 per pair;
- `pair_metrics.csv` with raw and count-rate-normalized metrics;
- `summary.json` with pair counts, split labels, checkpoint hash, and aggregate
  metrics.

The output directory must be under `D:\Projects\dccnn\outputs` and existing
files are never overwritten.

## Pair direction and normalization

Each endpoint must have a positive `acquisition_time_s` or `sweep_count`.
Acquisition time takes precedence, matching the training dataset behavior.
The endpoint with the smaller effective scale is the input and the larger one
is the reference. Input, denoised output, and reference values are divided by
their respective effective scales before pixel and structural metrics are
calculated. Physical features are calculated on these normalized arrays.

## Metrics and review semantics

The evaluator will report MAE, NRMSE, PSNR, SSIM, profile correlations, peak
position errors, FWHM errors, and integrated-intensity error. It will preserve
the source split (`train`, `val`, or `test`) in every row so development-pair
results cannot be mistaken for independent test evidence.

Missing or invalid metadata, non-A pairs, shape/coordinate mismatches, and
existing output files produce actionable errors and no partial report.

## Testing

Unit tests will cover deterministic pair direction, count-rate normalization,
missing exposure metadata, split propagation, and duplicate-output refusal.
An integration smoke run will execute the evaluator against the current
reviewed A pairs and the completed scientific checkpoint.
