Archive Layout (Current)

Primary index:
- archive/by_batch/README.txt
- archive/by_batch/_batch_summary.csv

Storage rule:
- Files are organized by training batch tag:
  archive/by_batch/<model_tag>/<file_type>/...
- Non-batch files are under:
  archive/by_batch/unmatched/<file_type>/...

Notes:
- Each batch folder contains index.txt with file counts and file list.
- Duplicate names are suffixed with _dupN to avoid overwrite.
