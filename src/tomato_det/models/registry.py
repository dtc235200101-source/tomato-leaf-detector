"""Allowlisted local model registry; it never accepts a user-supplied checkpoint path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tomato_det.common.config import load_yaml
from tomato_det.common.paths import config_path, project_root


class ModelUnavailableError(RuntimeError):
    """Raised before inference/training when a configured local model cannot be used."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    display_name: str
    architecture: str
    checkpoint: Path


def load_model_specs(app_config_path: Path | None = None) -> dict[str, ModelSpec]:
    config = load_yaml(app_config_path or config_path("app.yaml"))
    raw_models = config.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("app.yaml must define a non-empty models mapping")
    specs: dict[str, ModelSpec] = {}
    for model_id, raw in raw_models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Model {model_id} must be a mapping")
        required = {"display_name", "architecture", "checkpoint"}
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"Model {model_id} missing: {', '.join(sorted(missing))}")
        checkpoint = Path(str(raw["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = project_root() / checkpoint
        specs[str(model_id)] = ModelSpec(
            model_id=str(model_id),
            display_name=str(raw["display_name"]),
            architecture=str(raw["architecture"]),
            checkpoint=checkpoint,
        )
    return specs


def model_availability(spec: ModelSpec) -> tuple[bool, str | None]:
    if not spec.checkpoint.is_file():
        return False, "Local checkpoint is missing"
    return True, None


def build_detector(model_id: str, *, app_config_path: Path | None = None) -> Any:
    specs = load_model_specs(app_config_path)
    if model_id not in specs:
        raise ModelUnavailableError(f"Model ID is not allowlisted: {model_id}")
    spec = specs[model_id]
    available, reason = model_availability(spec)
    if not available:
        raise ModelUnavailableError(f"{model_id} is unavailable: {reason}")
    if spec.architecture != "yolo11s":
        raise ModelUnavailableError(f"No detector adapter is implemented for {spec.architecture}")
    from tomato_det.models.yolo_detector import YoloDetector

    return YoloDetector(model_id=spec.model_id, checkpoint=spec.checkpoint)
