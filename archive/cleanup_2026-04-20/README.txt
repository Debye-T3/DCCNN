Cleanup Archive (2026-04-20) / 清理归档说明（2026-04-20）

Purpose / 目的
- EN: This archive stores historical experiment outputs and intermediate files moved from the project root to keep the active workspace clean.
- 中文：本归档保存从项目根目录迁移出的历史实验结果与中间文件，用于保持当前工作区简洁。

What was moved / 已迁移内容
- EN: `results_ms_try2`, `results_ms_txt`, `results_ms_txt_quick4`, `results_ms_txt_v2`, `results_ms_txt_v3`, `results_ms_txt_v4`, `results_ms_txt_v5`
- 中文：`results_ms_try2`、`results_ms_txt`、`results_ms_txt_quick4`、`results_ms_txt_v2`、`results_ms_txt_v3`、`results_ms_txt_v4`、`results_ms_txt_v5`

- EN: Legacy subfolders from `results/`:
  `backup_run1`, `inference_ms_run1`, `inference_ms_run1_use`, `inference_ms_run2`, `inference_ms_v1_fixed`,
  and later `csv`, `inference`, `lr_finder`, `models`
- 中文：从 `results/` 迁移的历史子目录：
  `backup_run1`、`inference_ms_run1`、`inference_ms_run1_use`、`inference_ms_run2`、`inference_ms_v1_fixed`，
  以及后续迁移的 `csv`、`inference`、`lr_finder`、`models`

- EN: Legacy configs from `config/` were moved to `archive/config_history/`.
- 中文：`config/` 中的历史配置已迁移到 `archive/config_history/`。

- EN: Non-`*_txt.h5` files from `data/converted_h5/` were moved to:
  `archive/cleanup_2026-04-20/data_converted_h5_legacy/`
- 中文：`data/converted_h5/` 中非 `*_txt.h5` 文件已迁移到：
  `archive/cleanup_2026-04-20/data_converted_h5_legacy/`

- EN: Non-`*_txt.png` preview files from `data/previews/` were moved to:
  `archive/cleanup_2026-04-20/data_previews_legacy/`
- 中文：`data/previews/` 中非 `*_txt.png` 预览图已迁移到：
  `archive/cleanup_2026-04-20/data_previews_legacy/`

- EN: Root temporary folders `__pycache__` and `dccnn_project.egg-info` were moved to:
  `archive/cleanup_2026-04-20/root_misc/`
- 中文：根目录临时文件夹 `__pycache__` 和 `dccnn_project.egg-info` 已迁移到：
  `archive/cleanup_2026-04-20/root_misc/`

Active folders kept in root / 根目录保留的活动目录
- EN: `results_ms_txt_quick4_v8`, `results_ms_txt_v6`, `results_nbvt_test`, `results` (minimal legacy remains)
- 中文：`results_ms_txt_quick4_v8`、`results_ms_txt_v6`、`results_nbvt_test`、`results`（仅保留少量历史文件）

How to restore / 如何恢复
- EN: Move any file/folder back from this archive to its original location using `Move-Item`.
- 中文：如需恢复，可使用 `Move-Item` 将本归档中的文件/文件夹移回原始路径。

Notes / 备注
- EN: No files were deleted in this cleanup pass; only moved.
- 中文：本次清理未删除文件，仅执行了移动归档。
