Archive Layout / 归档结构说明

This folder stores historical files moved out of the active workspace.
本目录用于保存从当前工作区迁移出的历史文件。

1) legacy/
- EN: Long-term historical data grouped by type.
- 中文：按类型长期保存的历史数据。

1.1) legacy/configs/
- EN: Old config versions no longer used for active training.
- 中文：不再用于当前训练的旧配置文件。

1.2) legacy/results/
- EN: Historical experiment result folders from previous rounds.
- 中文：过往轮次实验结果目录。

2) cleanup_2026-04-20/
- EN: Snapshot of the cleanup operation executed on 2026-04-20.
- 中文：2026-04-20 进行清理时生成的快照归档。

3) cleanup_2026-04-22/
- EN: Snapshot of the cleanup operation executed on 2026-04-22.
- 中文：2026-04-22 进行清理时生成的快照归档。

Policy / 策略
- EN: Files are moved, not deleted.
- 中文：归档仅执行移动，不删除文件。

Restore / 恢复
- EN: Move files back to the original location if needed.
- 中文：如需恢复，移动回原始路径即可。
