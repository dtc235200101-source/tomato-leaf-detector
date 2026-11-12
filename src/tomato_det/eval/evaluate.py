"""Validation-only Ultralytics evaluator that persists only actual measured metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tomato_det.common.config import load_yaml
from tomato_det.common.meta import runtime_metadata
from tomato_det.common.paths import config_path, project_root
from tomato_det.eval.artifacts import evaluate_prediction_artifacts
from tomato_det.train.train_yolo import YoloTrainingConfig


def _metric_payload(results: Any) -> dict[str, float]:
    """Extract numeric metrics from the public Ultralytics results dictionary."""
    values = getattr(results, "results_dict", None)
    if not isinstance(values, dict):
        raise RuntimeError("Ultralytics validation returned no results_dict")
    return {
        str(key): float(value) for key, value in values.items() if isinstance(value, (int, float))
    }


def evaluate_validation(
    config: YoloTrainingConfig,
    checkpoint: Path,
    *,
    device: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Evaluate validation only; Phase 2 deliberately leaves final TEST evaluation unavailable."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
    if not config.data_yaml.is_file():
        raise FileNotFoundError(f"Processed dataset.yaml is missing: {config.data_yaml}")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is required for validation") from error
    eval_config = load_yaml(config_path("eval.yaml"))
    model = YOLO(str(checkpoint), task="detect")
    results = model.val(
        data=str(config.data_yaml),
        split="val",
        imgsz=config.image_size,
        conf=float(eval_config["confidence_threshold"]),
        iou=float(eval_config["nms_iou"]),
        max_det=int(eval_config["max_detections"]),
        device=device or config.device,
    )
    destination = output_dir or config.run_dir
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        **runtime_metadata(project_root()),
        "split": "val",
        "checkpoint": str(checkpoint),
        "data_yaml": str(config.data_yaml),
        "metrics": _metric_payload(results),
    }
    # Keep this separate from the unified evaluator's metrics_val.json: Ultralytics'
    # internal validator applies NMS at conf=0.001 differently, so the two files are
    # not interchangeable inputs for tdet report.
    output_path = destination / "metrics_val_ultralytics.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation artifact: {output_path}")
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def evaluate_saved_predictions(
    *,
    annotations: Path,
    predictions: Path,
    class_ids: list[int],
    split: str,
    confidence_threshold: float,
    operating_iou: float,
    output_path: Path,
    operating_confidence: float | None = None,
    checkpoint: Path | None = None,
    data_yaml: Path | None = None,
    evaluation_lock: Path | None = None,
) -> Path:
    """Evaluate prediction JSON produced by a real model, refusing unlocked TEST use."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite metrics artifact: {output_path}")
    payload = evaluate_prediction_artifacts(
        annotations=annotations,
        predictions=predictions,
        class_ids=class_ids,
        split=split,
        confidence_threshold=confidence_threshold,
        operating_iou=operating_iou,
        operating_confidence=operating_confidence,
        checkpoint=checkpoint,
        data_yaml=data_yaml,
        evaluation_lock=evaluation_lock,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
