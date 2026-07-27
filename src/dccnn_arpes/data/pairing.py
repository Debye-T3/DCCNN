"""Conservative reviewable pairing for ARPES manifest records."""

import csv
import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import yaml

from .schema import ManifestRecord


@dataclass(frozen=True, slots=True)
class PairDecision:
    """One reviewed candidate comparison, including rejected candidates."""

    left_record_id: str
    right_record_id: str
    accepted: bool
    pair_type: str = ""
    review_status: str = "reviewed"
    exclusion_reason: str = ""


@dataclass(frozen=True, slots=True)
class PairRecord:
    """One accepted pair suitable for human review and downstream splitting."""

    pair_id: str
    left_record_id: str
    right_record_id: str
    pair_type: str
    review_status: str = "reviewed"


PAIR_FIELDNAMES = tuple(PairRecord.__dataclass_fields__)


def _config() -> dict[str, float]:
    path = Path(__file__).resolve().parents[3] / "configs" / "data_cut_v1.yaml"
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    pairing = values.get("pairing", {})
    return {
        "position_atol": float(pairing["position_atol"]),
        "coordinate_rtol": float(pairing["coordinate_rtol"]),
        "coordinate_atol": float(pairing["coordinate_atol"]),
    }


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _same_exact(left: object, right: object) -> bool:
    if _missing(left) or _missing(right):
        return False
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def _same_position(left: object, right: object, *, atol: float) -> bool:
    if _missing(left) or _missing(right):
        return False
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _same_axis(left: tuple[float, ...], right: tuple[float, ...], *, rtol: float, atol: float) -> bool:
    if not left or not right or len(left) != len(right):
        return False
    return all(math.isclose(a, b, rel_tol=rtol, abs_tol=atol) for a, b in zip(left, right, strict=True))


def _surface_state(record: ManifestRecord) -> str:
    """Treat unstructured notes as a conservative physical-state declaration."""
    return record.notes.strip()


def _rejected(left: ManifestRecord, right: ManifestRecord, reason: str, *, review_status: str = "reviewed") -> PairDecision:
    return PairDecision(
        left_record_id=left.record_id,
        right_record_id=right.record_id,
        accepted=False,
        review_status=review_status,
        exclusion_reason=reason,
    )


def classify_pair(left: ManifestRecord, right: ManifestRecord) -> PairDecision:
    """Accept only candidates whose physical settings are fully comparable and equal."""
    if left.review_status == "needs_review" or right.review_status == "needs_review":
        return _rejected(left, right, "metadata_inherited", review_status="needs_review")

    for field in ("sample_id", "temperature_K", "photon_energy_eV", "polarization", "scan_type"):
        if not _same_exact(getattr(left, field), getattr(right, field)):
            return _rejected(left, right, field)

    settings = _config()
    for field in (
        "position_x",
        "position_y",
        "position_z",
        "position_polar",
        "position_tilt",
        "position_azimuth",
    ):
        if not _same_position(getattr(left, field), getattr(right, field), atol=settings["position_atol"]):
            return _rejected(left, right, field)

    if _missing(_surface_state(left)) or _missing(_surface_state(right)) or _surface_state(left) != _surface_state(right):
        return _rejected(left, right, "surface_state")
    if len(left.energy_axis) != len(right.energy_axis) or len(left.angle_axis) != len(right.angle_axis):
        return _rejected(left, right, "shape")
    for field in ("energy_axis", "angle_axis"):
        if not _same_axis(
            getattr(left, field),
            getattr(right, field),
            rtol=settings["coordinate_rtol"],
            atol=settings["coordinate_atol"],
        ):
            return _rejected(left, right, field)

    comparable_acquisition = (
        left.acquisition_time_s is not None
        and right.acquisition_time_s is not None
        and not _same_exact(left.acquisition_time_s, right.acquisition_time_s)
    )
    comparable_sweeps = (
        left.sweep_count is not None
        and right.sweep_count is not None
        and not _same_exact(left.sweep_count, right.sweep_count)
    )
    pair_type = "A" if comparable_acquisition or comparable_sweeps else "B"
    return PairDecision(left.record_id, right.record_id, accepted=True, pair_type=pair_type)


def _pair_id(decision: PairDecision) -> str:
    material = "\0".join((decision.pair_type, *sorted((decision.left_record_id, decision.right_record_id))))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def propose_pairs(records: Iterable[ManifestRecord]) -> tuple[list[PairRecord], list[PairDecision]]:
    """Compare every candidate deterministically and retain only accepted pair records."""
    ordered = sorted(records, key=lambda record: record.record_id)
    pairs: list[PairRecord] = []
    decisions: list[PairDecision] = []
    for left, right in combinations(ordered, 2):
        decision = classify_pair(left, right)
        decisions.append(decision)
        if decision.accepted:
            pairs.append(
                PairRecord(
                    pair_id=_pair_id(decision),
                    left_record_id=decision.left_record_id,
                    right_record_id=decision.right_record_id,
                    pair_type=decision.pair_type,
                    review_status=decision.review_status,
                )
            )
    return pairs, decisions


def write_pairs_csv(pairs: Sequence[PairRecord], path: Path) -> None:
    """Write accepted pairs as deterministic UTF-8 CSV for manual review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDNAMES)
        writer.writeheader()
        writer.writerows(asdict(pair) for pair in sorted(pairs, key=lambda pair: pair.pair_id))


def read_pairs_csv(path: Path) -> list[PairRecord]:
    """Read a pairs CSV previously emitted by :func:`write_pairs_csv`."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [PairRecord(**{field: row.get(field, "") for field in PAIR_FIELDNAMES}) for row in csv.DictReader(stream)]
