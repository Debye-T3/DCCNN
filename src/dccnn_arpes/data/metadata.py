"""Conservative extraction of experiment-workbook metadata candidates."""

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

_TEXT_COLUMNS = {"file_id", "sample_name", "polarization"}
_FLOAT_COLUMNS = {"temperature_K", "photon_energy_eV", "acquisition_time_s"}


def _normalise_label(value: object) -> str:
    """Normalize a workbook heading without changing its displayed spelling."""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _clean_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    converted = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(converted) else float(converted)


def _clean_int(value: object) -> int | None:
    numeric = _clean_float(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _alias_columns(columns: list[object], aliases: Mapping) -> tuple[dict[str, object], list[str]]:
    aliases_by_name = {
        canonical: {_normalise_label(alias) for alias in values}
        for canonical, values in aliases.items()
    }
    selected: dict[str, object] = {}
    unknown: list[str] = []
    for column in columns:
        normalised = _normalise_label(column)
        matches = [
            canonical for canonical, known_aliases in aliases_by_name.items() if normalised in known_aliases
        ]
        if not matches:
            unknown.append(str(column))
            continue
        selected.setdefault(matches[0], column)
    return selected, unknown


def _candidate_rows(frame: pd.DataFrame, aliases: Mapping) -> tuple[pd.DataFrame, list[str]]:
    selected, unknown = _alias_columns(list(frame.columns), aliases)
    output = pd.DataFrame(index=frame.index)
    inherited = [False] * len(frame)
    for canonical in aliases:
        source = frame[selected[canonical]] if canonical in selected else pd.Series(None, index=frame.index)
        previous: object = None
        values: list[object] = []
        for index, value in source.items():
            cleaned = (
                _clean_text(value)
                if canonical in _TEXT_COLUMNS
                else _clean_int(value)
                if canonical == "sweep_count"
                else _clean_float(value)
                if canonical in _FLOAT_COLUMNS
                else value
            )
            is_missing = cleaned == "" if canonical in _TEXT_COLUMNS else cleaned is None
            if canonical != "file_id" and is_missing and previous is not None:
                cleaned = previous
                inherited[frame.index.get_loc(index)] = True
            if not is_missing:
                previous = cleaned
            values.append(cleaned)
        output[canonical] = pd.Series(values, index=frame.index, dtype=object)
    output["metadata_inherited"] = pd.Series(inherited, index=frame.index, dtype=object)
    output["review_status"] = pd.Series(
        ["needs_review" if value else "unreviewed" for value in inherited],
        index=frame.index,
        dtype=object,
    )
    return output, unknown


def read_workbook_candidates(path: Path, aliases: Mapping) -> pd.DataFrame:
    """Read workbook values as reviewable candidates, never as confirmed metadata."""
    path = Path(path)
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    candidates: list[pd.DataFrame] = []
    unknown_columns: list[str] = []
    for sheet_name, frame in sheets.items():
        rows, unknown = _candidate_rows(frame, aliases)
        rows["workbook_path"] = str(path.resolve())
        rows["sheet_name"] = str(sheet_name)
        rows["excel_row"] = [int(index) + 2 if isinstance(index, int) else None for index in frame.index]
        candidates.append(rows.reset_index(drop=True))
        unknown_columns.extend(column for column in unknown if column not in unknown_columns)
    result = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    result.attrs["unknown_columns"] = unknown_columns
    return result
