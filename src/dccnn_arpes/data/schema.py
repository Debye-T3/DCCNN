"""Stable manifest schema for source ARPES records."""

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One immutable source record and its conservative candidate metadata."""

    record_id: str = ""
    source_path: str = ""
    converted_path: str = ""
    source_format: str = ""
    file_id: str = ""
    sample_name: str = ""
    sample_id: str = ""
    session_id: str = ""
    acquisition_group: str = ""
    scan_type: str = ""
    temperature_K: float | None = None
    photon_energy_eV: float | None = None
    polarization: str = ""
    position_x: float | None = None
    position_y: float | None = None
    position_z: float | None = None
    position_polar: float | None = None
    position_tilt: float | None = None
    position_azimuth: float | None = None
    energy_axis: str = ""
    angle_axis: str = ""
    acquisition_time_s: float | None = None
    sweep_count: int | None = None
    pair_type: str = ""
    pair_id: str = ""
    review_status: str = "unreviewed"
    split: str = ""
    quality_flag: str = ""
    exclusion_reason: str = ""
    notes: str = ""


MANIFEST_FIELDNAMES = tuple(field.name for field in fields(ManifestRecord))
