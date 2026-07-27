"""Deterministic training loop and experiment provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dccnn_arpes.data.dataset import ArpesCutDataset
from dccnn_arpes.data.noise import NoiseParameters
from dccnn_arpes.data.pairing import PAIR_FIELDNAMES, PairRecord
from dccnn_arpes.data.schema import MANIFEST_FIELDNAMES, ManifestRecord
from dccnn_arpes.data.splitting import leakage_audit
from dccnn_arpes.models import ResidualDenoiser2D, denoise_forward
from dccnn_arpes.safety import guard_output_path

from .checkpoints import save_checkpoint
from .config import NoiseConfig, TrainConfig
from .losses import CompositeDenoisingLoss

_METRIC_FIELDS = (
    "epoch",
    "train_total",
    "train_charbonnier",
    "train_ms_ssim",
    "train_gradient",
    "val_total",
    "val_charbonnier",
    "val_ms_ssim",
    "val_gradient",
)
_REVIEWED = {"reviewed", "approved", "manually_approved"}
_NUMERIC_FLOAT_FIELDS = {
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
_NUMERIC_INT_FIELDS = {"sweep_count"}
_AXIS_FIELDS = {"energy_axis", "angle_axis"}


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metrics_path: Path
    run_path: Path


def _parse_manifest_value(field: str, value: str) -> object:
    if field in _AXIS_FIELDS:
        return tuple(float(item) for item in json.loads(value or "[]"))
    if field in _NUMERIC_FLOAT_FIELDS:
        return None if value == "" else float(value)
    if field in _NUMERIC_INT_FIELDS:
        return None if value == "" else int(value)
    return value


def _read_manifest(path: Path) -> list[ManifestRecord]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != set(MANIFEST_FIELDNAMES):
            raise ValueError(f"manifest {path} does not have the canonical schema")
        return [
            ManifestRecord(
                **{
                    field: _parse_manifest_value(field, row.get(field, ""))
                    for field in MANIFEST_FIELDNAMES
                }
            )
            for row in reader
        ]


def _read_pairs(path: Path) -> list[PairRecord]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != set(PAIR_FIELDNAMES):
            raise ValueError(f"pairs file {path} does not have the canonical schema")
        return [
            PairRecord(**{field: row.get(field, "") for field in PAIR_FIELDNAMES}) for row in reader
        ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_sha256(splits: Path) -> str:
    digest = hashlib.sha256()
    for name in ("train.csv", "val.csv", "test.csv"):
        path = splits / name
        if not path.is_file():
            raise FileNotFoundError(f"required split file does not exist: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _data_hashes(records: list[ManifestRecord]) -> tuple[dict[str, str], str]:
    file_hashes = {
        record.record_id: _sha256(Path(record.converted_path))
        for record in sorted(records, key=lambda item: item.record_id)
    }
    digest = hashlib.sha256()
    for record_id, file_hash in file_hashes.items():
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
    return file_hashes, digest.hexdigest()


def _git_metadata() -> dict[str, object]:
    checkout = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256()
        digest.update(
            subprocess.run(
                ["git", "-C", str(checkout), "diff", "--binary", "HEAD"],
                check=True,
                capture_output=True,
            ).stdout
        )
        untracked = subprocess.run(
            ["git", "-C", str(checkout), "ls-files", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        for encoded_path in sorted(path for path in untracked if path):
            digest.update(encoded_path)
            digest.update(b"\0")
            path = checkout / encoded_path.decode("utf-8")
            if path.is_file():
                digest.update(path.read_bytes())
            digest.update(b"\0")
        return {
            "git_commit": commit,
            "git_dirty": bool(status),
            "source_diff_sha256": digest.hexdigest() if status else None,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": "unavailable",
            "git_dirty": None,
            "source_diff_sha256": None,
        }


def _device(config: TrainConfig) -> tuple[torch.device, str]:
    requested = config.training.device
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
        return device, torch.cuda.get_device_name(device)
    return torch.device("cpu"), "cpu"


def _versions(device_name: str) -> dict[str, object]:
    return {
        **_git_metadata(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": device_name,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_noise(
    noise: NoiseConfig, *, seed: int, epoch: int, dataset_index: int
) -> NoiseParameters:
    material = f"{seed}\0{epoch}\0{dataset_index}\0noise-parameters".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest()[:8], "little"))
    log_peak = rng.uniform(
        math.log(noise.poisson_peak_counts[0]),
        math.log(noise.poisson_peak_counts[1]),
    )
    return NoiseParameters(
        poisson_peak_counts=math.exp(log_peak),
        background_fraction=float(rng.uniform(*noise.background_fraction)),
        stripe_probability=noise.stripe_probability,
        stripe_fraction=float(rng.uniform(*noise.stripe_fraction)),
    )


class _ConfiguredNoiseDataset(ArpesCutDataset):
    """Select configured noise ranges reproducibly before each base sample."""

    def __init__(self, *args, noise_config: NoiseConfig, **kwargs) -> None:
        self._noise_config = noise_config
        super().__init__(*args, **kwargs)

    def __getitem__(self, dataset_index: int):
        self.noise_parameters = _sample_noise(
            self._noise_config,
            seed=self.base_seed,
            epoch=self.epoch,
            dataset_index=dataset_index,
        )
        return super().__getitem__(dataset_index)


def _validate_records(records: list[ManifestRecord], split: str) -> None:
    if not records:
        raise ValueError(f"{split} split has no reviewed records")
    for record in records:
        if record.review_status not in _REVIEWED:
            raise ValueError(f"{split} record {record.record_id!r} is not explicitly reviewed")
        if record.exclusion_reason:
            raise ValueError(f"{split} record {record.record_id!r} is excluded")
        if record.split != split:
            raise ValueError(
                f"record {record.record_id!r} is in {split}.csv but declares {record.split!r}"
            )
        if not record.converted_path or not Path(record.converted_path).is_file():
            raise FileNotFoundError(
                f"reviewed record {record.record_id!r} has no existing converted file"
            )


def _validate_pairs(pairs: list[PairRecord], master_records: list[ManifestRecord]) -> None:
    master = {record.record_id: record for record in master_records}
    pair_ids: set[str] = set()
    for pair in pairs:
        if not pair.pair_id or pair.pair_id in pair_ids:
            raise ValueError("pair IDs must be non-empty and unique")
        pair_ids.add(pair.pair_id)
        if pair.pair_type not in {"A", "B"}:
            raise ValueError(f"pair {pair.pair_id!r} has invalid pair type {pair.pair_type!r}")
        if (
            not pair.left_record_id
            or not pair.right_record_id
            or pair.left_record_id == pair.right_record_id
        ):
            raise ValueError(f"pair {pair.pair_id!r} must have two distinct non-empty endpoints")
        if pair.review_status not in _REVIEWED:
            raise ValueError(f"pair {pair.pair_id!r} is not explicitly reviewed")
        for record_id in (pair.left_record_id, pair.right_record_id):
            if record_id not in master:
                raise ValueError(
                    f"pair endpoint {record_id!r} is not present in the master manifest"
                )
            record = master[record_id]
            if record.review_status not in _REVIEWED or record.exclusion_reason:
                raise ValueError(f"pair endpoint {record_id!r} is not an eligible reviewed record")
        left = master[pair.left_record_id]
        right = master[pair.right_record_id]
        if left.split != right.split:
            raise ValueError(f"pair {pair.pair_id!r} crosses split boundaries")


def _master_by_id(master_records: list[ManifestRecord]) -> dict[str, ManifestRecord]:
    master_by_id: dict[str, ManifestRecord] = {}
    for record in master_records:
        if not record.record_id or record.record_id in master_by_id:
            raise ValueError("master manifest record IDs must be non-empty and unique")
        master_by_id[record.record_id] = record
    return master_by_id


def _validate_partitions(
    master_records: list[ManifestRecord],
    partitions: dict[str, list[ManifestRecord]],
) -> None:
    master_by_id = _master_by_id(master_records)
    observed: dict[str, str] = {}
    for split in ("train", "val", "test"):
        local_ids: set[str] = set()
        for record in partitions[split]:
            if not record.record_id or record.record_id in local_ids:
                raise ValueError(f"{split} partition record IDs must be non-empty and unique")
            local_ids.add(record.record_id)
            if record.record_id in observed:
                raise ValueError(
                    f"record {record.record_id!r} appears in multiple partitions: "
                    f"{observed[record.record_id]} and {split}"
                )
            observed[record.record_id] = split
            master = master_by_id.get(record.record_id)
            if master is None:
                raise ValueError(
                    f"split record {record.record_id!r} is not present in the master manifest"
                )
            if master != record:
                raise ValueError(
                    f"split record {record.record_id!r} does not match the master manifest"
                )
            if record.split != split:
                raise ValueError(
                    f"record {record.record_id!r} is in {split}.csv but declares {record.split!r}"
                )

    for record in master_records:
        eligible = (
            record.review_status in _REVIEWED
            and bool(record.converted_path)
            and not record.exclusion_reason
        )
        if not eligible:
            continue
        if record.split not in partitions:
            raise ValueError(
                f"eligible master record {record.record_id!r} has invalid split {record.split!r}"
            )
        actual = observed.get(record.record_id)
        if actual is None:
            raise ValueError(
                f"eligible master record {record.record_id!r} is missing from {record.split}"
            )


def _preflight_split_audit(config: TrainConfig) -> list[ManifestRecord]:
    """Reject self-consistent but connectivity-leaking splits before creating a run."""
    master_records = _read_manifest(config.paths.manifest)
    partitions = {
        split: _read_manifest(config.paths.splits / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    _validate_partitions(master_records, partitions)
    pairs = _read_pairs(config.paths.pairs)
    _validate_pairs(pairs, master_records)
    pair_links = [(pair.left_record_id, pair.right_record_id) for pair in pairs]
    audit = leakage_audit(master_records, pair_links)
    if not audit["leakage_free"]:
        raise ValueError("a connected component appears in multiple splits")
    return master_records


def _dataset(
    records: list[ManifestRecord], pairs: list[PairRecord], config: TrainConfig
) -> ArpesCutDataset:
    ids = {record.record_id for record in records}
    eligible_pairs = [
        pair for pair in pairs if pair.left_record_id in ids and pair.right_record_id in ids
    ]
    return _ConfiguredNoiseDataset(
        records,
        eligible_pairs,
        crop_size=config.data.crop_size,
        samples_per_epoch=config.data.samples_per_epoch,
        sampling=config.data.sampling.as_dict(),
        identity_probability=config.data.identity_probability,
        base_seed=config.seed,
        noise_config=config.data.noise,
    )


def _loader(
    dataset: ArpesCutDataset, config: TrainConfig, *, shuffle: bool, seed_offset: int
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed + seed_offset)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.workers,
        pin_memory=config.training.device == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _nonfinite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"nonfinite {name} detected")


def _epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: CompositeDenoisingLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "charbonnier": 0.0, "ms_ssim": 0.0, "gradient": 0.0}
    batches = 0
    try:
        for inputs, targets, _metadata in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            _nonfinite("input data", inputs)
            _nonfinite("target data", targets)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=device.type, enabled=amp):
                    predictions, _ = denoise_forward(model, inputs)
                    _nonfinite("model output", predictions)
                    loss, components = criterion(predictions, targets)
                _nonfinite("loss", loss)
                if optimizer is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            _nonfinite("gradient", parameter.grad)
                    scaler.step(optimizer)
                    scaler.update()
            batch_size = int(inputs.shape[0])
            totals["total"] += float(loss.detach()) * batch_size
            for name, value in components.items():
                totals[name] += float(value) * batch_size
            batches += batch_size
    except ValueError as error:
        if "finite" in str(error).lower() or "nan" in str(error).lower():
            raise FloatingPointError("nonfinite data detected") from error
        raise
    if batches == 0:
        raise ValueError("data loader produced no samples")
    return {name: value / batches for name, value in totals.items()}


def _provenance_path(config: TrainConfig) -> Path:
    return config.paths.provenance_path or config.paths.manifest.parent / "data_provenance.json"


def _provenance(config: TrainConfig, file_hashes: dict[str, str]) -> dict[str, object]:
    path = _provenance_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"required data provenance file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in data provenance file: {path}") from error
    required = {
        "schema_version",
        "classification",
        "scientific_use",
        "record_ids",
        "input_sha256",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"data provenance file is missing required fields: {path}")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("data provenance schema_version must be 1")
    expected_classification = (
        "controlled_smoke_fixture" if config.smoke_test else "reviewed_scientific_dataset"
    )
    if value["classification"] != expected_classification:
        raise ValueError(f"data provenance classification must be {expected_classification!r}")
    if (
        type(value["scientific_use"]) is not bool
        or value["scientific_use"] is not config.scientific_use
    ):
        raise ValueError("data provenance scientific_use does not match the run")
    record_ids = value["record_ids"]
    if (
        not isinstance(record_ids, list)
        or any(not isinstance(record_id, str) for record_id in record_ids)
        or len(record_ids) != len(set(record_ids))
        or set(record_ids) != set(file_hashes)
    ):
        raise ValueError("data provenance record_ids must exactly match partition inputs")
    declared_hashes = value["input_sha256"]
    if (
        not isinstance(declared_hashes, dict)
        or set(declared_hashes) != set(file_hashes)
        or any(
            not isinstance(declared, str) or len(declared) != 64 or declared.casefold() != actual
            for record_id, actual in file_hashes.items()
            for declared in [declared_hashes.get(record_id)]
        )
    ):
        raise ValueError("data provenance input_sha256 must exactly match partition inputs")
    return value


def _write_run_json(path: Path, run: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_training_body(
    config: TrainConfig,
    *,
    output: Path,
    run_path: Path,
    run: dict[str, object],
    config_hash: str,
) -> TrainingResult:
    device, device_name = _device(config)
    _seed_everything(config.seed)
    master_records = _read_manifest(config.paths.manifest)
    train_records = _read_manifest(config.paths.splits / "train.csv")
    val_records = _read_manifest(config.paths.splits / "val.csv")
    test_records = _read_manifest(config.paths.splits / "test.csv")
    partitions = {"train": train_records, "val": val_records, "test": test_records}
    _validate_partitions(master_records, partitions)
    _validate_records(train_records, "train")
    _validate_records(val_records, "val")
    if test_records:
        _validate_records(test_records, "test")
    pairs = _read_pairs(config.paths.pairs)
    _validate_pairs(pairs, master_records)

    manifest_hash = _sha256(config.paths.manifest)
    split_hash = _split_sha256(config.paths.splits)
    pairs_hash = _sha256(config.paths.pairs)
    file_hashes, data_hash = _data_hashes(train_records + val_records + test_records)
    hashes = {
        "manifest_sha256": manifest_hash,
        "split_sha256": split_hash,
        "pairs_sha256": pairs_hash,
        "data_sha256": data_hash,
    }
    provenance_path = _provenance_path(config)
    provenance = _provenance(config, file_hashes)
    hashes["provenance_sha256"] = _sha256(provenance_path)

    versions = _versions(device_name)
    run.update(
        {
            **hashes,
            "config_sha256": config_hash,
            "git_commit": versions["git_commit"],
            "git_dirty": versions["git_dirty"],
            "source_diff_sha256": versions["source_diff_sha256"],
            "python_version": versions["python"],
            "pytorch_version": versions["pytorch"],
            "cuda_version": versions["cuda"],
            "device": versions["device"],
            "data_provenance": provenance,
            "data_provenance_path": str(provenance_path),
            "data_file_sha256": file_hashes,
        }
    )
    _write_run_json(run_path, run)

    train_dataset = _dataset(train_records, pairs, config)
    val_dataset = _dataset(val_records, pairs, config)
    train_loader = _loader(train_dataset, config, shuffle=True, seed_offset=0)
    val_loader = _loader(val_dataset, config, shuffle=False, seed_offset=1)
    model = ResidualDenoiser2D(channels=config.model.channels, blocks=config.model.blocks).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    amp = config.training.amp and device.type == "cuda"
    # The composite MS-SSIM backward overflows at GradScaler's 65,536 default
    # before unscaling. Keep a fixed unit scale so fail-fast gradient checks
    # indicate real numerical failure rather than scaler calibration.
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp,
        init_scale=1.0,
        growth_interval=2**31 - 1,
    )
    criterion = CompositeDenoisingLoss(
        charbonnier=config.loss.charbonnier,
        ms_ssim=config.loss.ms_ssim,
        gradient=config.loss.gradient,
    ).to(device)

    metrics_path = output / "metrics.csv"
    best_path = output / "best.pt"
    last_path = output / "last.pt"
    best_metric = math.inf
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_METRIC_FIELDS)
        writer.writeheader()
        for epoch in range(1, config.training.epochs + 1):
            train_dataset.set_epoch(epoch)
            val_dataset.set_epoch(epoch)
            train_metrics = _epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp=amp,
            )
            val_metrics = _epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp=amp,
            )
            row: dict[str, object] = {"epoch": epoch}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            writer.writerow(row)
            stream.flush()
            if val_metrics["total"] < best_metric:
                best_metric = val_metrics["total"]
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    config=config,
                    hashes=hashes,
                    versions=versions,
                )
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                config=config,
                hashes=hashes,
                versions=versions,
            )
    return TrainingResult(
        output_dir=output,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        metrics_path=metrics_path,
        run_path=run_path,
    )


def run_training(config: TrainConfig) -> TrainingResult:
    """Train a residual denoiser and atomically record the complete run lifecycle."""
    master_records = _preflight_split_audit(config)
    output_root = guard_output_path(
        config.paths.output,
        input_sources=(
            config.paths.manifest,
            config.paths.pairs,
            config.paths.splits,
            *(record.source_path for record in master_records if record.source_path),
            *(record.converted_path for record in master_records if record.converted_path),
        ),
    )
    config_material = json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    config_hash = hashlib.sha256(config_material).hexdigest()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = guard_output_path(
        output_root / f"{timestamp}-{config_hash[:12]}",
        input_sources=(config.paths.manifest, config.paths.pairs, config.paths.splits),
    )
    output.mkdir(parents=True, exist_ok=False)
    run_path = output / "run.json"
    run: dict[str, object] = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "output_dir": str(output),
        "seed": config.seed,
        "config_sha256": config_hash,
        "config": config.as_dict(),
        "smoke_test": config.smoke_test,
        "scientific_use": config.scientific_use,
        "requested_device": config.training.device,
    }
    _write_run_json(run_path, run)
    try:
        result = _run_training_body(
            config,
            output=output,
            run_path=run_path,
            run=run,
            config_hash=config_hash,
        )
    except Exception as error:
        run.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "error_class": type(error).__name__,
                "error_message": str(error),
            }
        )
        _write_run_json(run_path, run)
        raise
    run.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    _write_run_json(run_path, run)
    return result
