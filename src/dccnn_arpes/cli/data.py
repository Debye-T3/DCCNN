"""Data-preparation command-line interface."""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import yaml

from dccnn_arpes.data.discovery import (
    associate_converted_with_issues,
    build_scan_audit,
    scan_archive,
    source_snapshot,
    write_json,
    write_manifest_csv,
)
from dccnn_arpes.data.metadata import read_workbook_candidates
from dccnn_arpes.data.pairing import propose_pairs, read_pairs_csv, write_pairs_csv
from dccnn_arpes.data.schema import ManifestRecord
from dccnn_arpes.data.splitting import assign_group_splits, write_split_csvs
from dccnn_arpes.io import load_cut

_WORKSPACE_ROOT = Path(r"D:\Projects\dccnn\workspace")


def _format_range(values) -> str:
    """Format the inclusive range of one validated coordinate."""
    return f"{values.min():g} to {values.max():g}"


def _validate_command(path: Path, allow_legacy: bool) -> None:
    """Load a cut and print the properties needed to verify its data boundary."""
    cut = load_cut(path, allow_legacy=allow_legacy)
    source = "legacy-adapted" if "legacy_source" in cut.attrs else "standard"
    print(f"path: {path}")
    print(f"object name: {cut.name}")
    print(f"shape: {cut.shape}")
    print(f"dimensions: {', '.join(cut.dims)}")
    print(f"eV range: {_format_range(cut.coords['eV'].values)}")
    print(f"alpha range: {_format_range(cut.coords['alpha'].values)}")
    print(f"format: {source}")


def _default_alias_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "metadata_aliases.yaml"


def _load_aliases(path: Path | None) -> dict[str, list[str]]:
    alias_path = path or _default_alias_path()
    with alias_path.open(encoding="utf-8") as stream:
        aliases = yaml.safe_load(stream)
    if not isinstance(aliases, dict):
        raise TypeError(f"metadata aliases must be a mapping: {alias_path}")
    return {str(key): [str(value) for value in values] for key, values in aliases.items()}


def _apply_workbook_candidates(
    records: list[ManifestRecord], candidates: list,
) -> list[ManifestRecord]:
    """Apply only uniquely identified workbook candidates to matching source records."""
    by_file_id: dict[str, list] = {}
    for row in candidates:
        file_id = str(row["file_id"] or "").casefold()
        if file_id:
            by_file_id.setdefault(file_id, []).append(row)
    applied: list[ManifestRecord] = []
    for record in records:
        matching = by_file_id.get(record.file_id.casefold(), [])
        if len(matching) != 1:
            applied.append(record)
            continue
        row = matching[0]
        applied.append(
            replace(
                record,
                sample_name=str(row["sample_name"] or ""),
                temperature_K=row["temperature_K"],
                photon_energy_eV=row["photon_energy_eV"],
                polarization=str(row["polarization"] or ""),
                acquisition_time_s=row["acquisition_time_s"],
                sweep_count=row["sweep_count"],
                review_status=str(row["review_status"]),
            )
        )
    return applied


def _validate_scan_output(source: Path, output: Path) -> None:
    """Refuse any manifest or audit path outside the workspace or inside source data."""
    source_root = Path(source).expanduser().resolve(strict=True)
    workspace_root = _WORKSPACE_ROOT.expanduser().resolve(strict=False)
    output = Path(output).expanduser().resolve(strict=False)
    targets = (
        output,
        output.parent / "scan_audit.json",
        output.parent / "unknown_excel_columns.json",
        output.parent / "association_issues.json",
    )
    for target in targets:
        try:
            target.relative_to(workspace_root)
        except ValueError as error:
            raise ValueError(f"scan outputs must be under workspace: {workspace_root}") from error
        try:
            target.relative_to(source_root)
        except ValueError:
            continue
        raise ValueError("scan outputs must not be under the source root")


def _scan_command(source: Path, converted: Path, output: Path, aliases_path: Path | None) -> None:
    """Create source manifest and audits while treating source and converted roots as inputs."""
    _validate_scan_output(source, output)
    aliases = _load_aliases(aliases_path)
    source_before = source_snapshot(source)
    records = scan_archive(source)
    unknown_columns: dict[str, list[str]] = {}
    candidate_rows: list = []
    workbook_count = 0
    for record in records:
        if record.source_format != "xlsx":
            continue
        workbook_count += 1
        try:
            candidates = read_workbook_candidates(Path(record.source_path), aliases)
        except (OSError, ValueError, KeyError) as error:
            unknown_columns[record.source_path] = [f"workbook read error: {error}"]
            continue
        unknown_columns[record.source_path] = candidates.attrs["unknown_columns"]
        candidate_rows.extend(candidates.to_dict("records"))
    records = _apply_workbook_candidates(records, candidate_rows)
    records, association_issues = associate_converted_with_issues(records, converted)
    write_manifest_csv(records, output)
    source_after = source_snapshot(source)
    audit = build_scan_audit(
        records,
        source_root=source,
        converted_root=converted,
        source_before=source_before,
        source_after=source_after,
        workbook_count=workbook_count,
    )
    audit_path = output.parent / "scan_audit.json"
    write_json(audit, audit_path)
    write_json(unknown_columns, output.parent / "unknown_excel_columns.json")
    write_json(association_issues, output.parent / "association_issues.json")
    print(f"manifest: {output.resolve()}")
    print(f"audit: {audit_path.resolve()}")


def _validate_workspace_output(output: Path) -> Path:
    """Keep generated review and split outputs inside the workspace boundary."""
    workspace_root = _WORKSPACE_ROOT.expanduser().resolve(strict=False)
    output = Path(output).expanduser().resolve(strict=False)
    try:
        output.relative_to(workspace_root)
    except ValueError as error:
        raise ValueError(f"generated outputs must be under workspace: {workspace_root}") from error
    return output


def _read_manifest_csv(path: Path) -> list[ManifestRecord]:
    """Load Task 3's UTF-8 manifest representation without mutating source records."""
    float_fields = {
        "temperature_K",
        "photon_energy_eV",
        "position_x",
        "position_y",
        "position_z",
        "position_polar",
        "position_tilt",
        "position_azimuth",
        "acquisition_time_s",
    }
    fieldnames = set(ManifestRecord.__dataclass_fields__)
    records: list[ManifestRecord] = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            values = {field: row.get(field, "") for field in fieldnames}
            for field in float_fields:
                value = str(values[field]).strip()
                values[field] = float(value) if value else None
            sweep = str(values["sweep_count"]).strip()
            values["sweep_count"] = int(sweep) if sweep else None
            for field in ("energy_axis", "angle_axis"):
                value = str(values[field]).strip()
                values[field] = tuple(float(item) for item in json.loads(value)) if value else ()
            records.append(ManifestRecord(**values))
    return records


def _pairs_command(manifest: Path, output: Path) -> None:
    output = _validate_workspace_output(output)
    pairs, decisions = propose_pairs(_read_manifest_csv(manifest))
    write_pairs_csv(pairs, output)
    print(f"pairs: {output.resolve()}")
    print(f"accepted: {len(pairs)}; rejected: {sum(not decision.accepted for decision in decisions)}")


def _split_command(manifest: Path, pairs_path: Path, output: Path) -> None:
    output = _validate_workspace_output(output)
    pairs = read_pairs_csv(pairs_path)
    links = [(pair.left_record_id, pair.right_record_id) for pair in pairs]
    assigned = assign_group_splits(_read_manifest_csv(manifest), pair_links=links)
    audit = write_split_csvs(assigned, output, pair_links=links)
    if not audit["leakage_free"]:
        raise ValueError("a connected component appears in multiple splits")
    print(f"splits: {output.resolve()}")
    print(f"audit: {(output / 'split_audit.json').resolve()}")


def main() -> None:
    """Run the data-preparation command-line interface."""
    parser = argparse.ArgumentParser(description="Prepare ARPES denoising data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="validate a 2D ARPES cut")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--allow-legacy", action="store_true")
    scan_parser = subparsers.add_parser("scan", help="index source files and associate converted outputs")
    scan_parser.add_argument("--source", type=Path, required=True)
    scan_parser.add_argument("--converted", type=Path, required=True)
    scan_parser.add_argument("--output", type=Path, required=True)
    scan_parser.add_argument("--aliases", type=Path)
    pairs_parser = subparsers.add_parser("pairs", help="propose conservative reviewed cut pairs")
    pairs_parser.add_argument("--manifest", type=Path, required=True)
    pairs_parser.add_argument("--output", type=Path, required=True)
    split_parser = subparsers.add_parser("split", help="create leakage-safe group-level splits")
    split_parser.add_argument("--manifest", type=Path, required=True)
    split_parser.add_argument("--pairs", type=Path, required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.command == "validate":
        try:
            _validate_command(arguments.path, arguments.allow_legacy)
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))
    elif arguments.command == "scan":
        try:
            _scan_command(arguments.source, arguments.converted, arguments.output, arguments.aliases)
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))
    elif arguments.command == "pairs":
        try:
            _pairs_command(arguments.manifest, arguments.output)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            parser.error(str(error))
    elif arguments.command == "split":
        try:
            _split_command(arguments.manifest, arguments.pairs, arguments.output)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
