"""Validation-safe, local-only YOLO11s fine-tuning entry point."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tomato_det.common.config import load_yaml
from tomato_det.common.device import DeviceInfo, detect_device
from tomato_det.common.meta import runtime_metadata
from tomato_det.common.paths import project_root
from tomato_det.common.seed import seed_everything


@dataclass(frozen=True, slots=True)
class YoloTrainingConfig:
    experiment_id: str
    model_id: str
    pretrained_weights: Path
    data_yaml: Path
    run_dir: Path
    image_size: int
    epochs: int
    patience: int
    batch: int
    workers: int
    optimizer: str
    seed: int
    device: str
    cache: bool


@dataclass(frozen=True, slots=True)
class TrainingPreflight:
    ready: bool
    device: DeviceInfo
    errors: tuple[str, ...]


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def load_training_config(path: Path) -> YoloTrainingConfig:
    raw = load_yaml(path)
    required = {
        "experiment_id",
        "model_id",
        "pretrained_weights",
        "data_yaml",
        "run_dir",
        "image_size",
        "epochs",
        "patience",
        "batch",
        "workers",
        "optimizer",
        "seed",
        "device",
        "cache",
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Training config missing: {', '.join(sorted(missing))}")
    config = YoloTrainingConfig(
        experiment_id=str(raw["experiment_id"]),
        model_id=str(raw["model_id"]),
        pretrained_weights=_project_path(str(raw["pretrained_weights"])),
        data_yaml=_project_path(str(raw["data_yaml"])),
        run_dir=_project_path(str(raw["run_dir"])),
        image_size=int(raw["image_size"]),
        epochs=int(raw["epochs"]),
        patience=int(raw["patience"]),
        batch=int(raw["batch"]),
        workers=int(raw["workers"]),
        optimizer=str(raw["optimizer"]),
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        cache=bool(raw["cache"]),
    )
    if config.image_size <= 0 or config.epochs <= 0 or config.batch <= 0 or config.workers < 0:
        raise ValueError(
            "image_size, epochs, and batch must be positive; workers cannot be negative"
        )
    return config


def training_preflight(
    config: YoloTrainingConfig, requested_device: str | None = None
) -> TrainingPreflight:
    """Check all prerequisites before importing or loading a large model."""
    errors: list[str] = []
    device = detect_device(requested_device or config.device)
    if not device.torch_available:
        errors.append("PyTorch is not installed")
    elif device.reason:
        errors.append(device.reason)
    if importlib.util.find_spec("ultralytics") is None:
        errors.append("Ultralytics is not installed")
    if not config.data_yaml.is_file():
        errors.append(f"Processed dataset.yaml is missing: {config.data_yaml}")
    if not config.pretrained_weights.is_file():
        errors.append(f"Local COCO pretrained weights are missing: {config.pretrained_weights}")
    if config.run_dir.exists() and any(config.run_dir.iterdir()):
        errors.append(f"Run directory already contains files: {config.run_dir}")
    return TrainingPreflight(not errors, device, tuple(errors))


def run_training(
    config: YoloTrainingConfig, *, seed: int | None = None, device: str | None = None
) -> Path:
    """Train against train/val defined in dataset.yaml and store only real run artifacts."""
    effective_seed = config.seed if seed is None else seed
    preflight = training_preflight(config, requested_device=device)
    if not preflight.ready:
        raise RuntimeError("Training preflight failed:\n- " + "\n- ".join(preflight.errors))
    from ultralytics import YOLO

    seed_everything(effective_seed)
    config.run_dir.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(config.pretrained_weights), task="detect")
    model.train(
        data=str(config.data_yaml),
        imgsz=config.image_size,
        epochs=config.epochs,
        patience=config.patience,
        batch=config.batch,
        workers=config.workers,
        optimizer=config.optimizer,
        seed=effective_seed,
        device=preflight.device.resolved,
        cache=config.cache,
        project=str(config.run_dir.parent),
        name=config.run_dir.name,
        exist_ok=False,
        pretrained=True,
    )
    save_dir_value = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir_value is None:
        raise RuntimeError("Ultralytics training ended without a run directory")
    save_dir = Path(str(save_dir_value))
    best_checkpoint = save_dir / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise RuntimeError(f"Ultralytics training ended without best.pt at {best_checkpoint}")
    metadata: dict[str, Any] = {
        **runtime_metadata(project_root()),
        "experiment_id": config.experiment_id,
        "model_id": config.model_id,
        "seed": effective_seed,
        "device": preflight.device.resolved,
        "data_yaml": str(config.data_yaml),
        "pretrained_weights": str(config.pretrained_weights),
        "best_checkpoint": str(best_checkpoint),
        "training_args": {
            "image_size": config.image_size,
            "epochs": config.epochs,
            "patience": config.patience,
            "batch": config.batch,
            "workers": config.workers,
            "optimizer": config.optimizer,
        },
    }
    (save_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return best_checkpoint
