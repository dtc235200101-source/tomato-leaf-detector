"""Read COCO annotations/predictions and enforce an explicit final-test lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tomato_det.common.meta import runtime_metadata
from tomato_det.common.paths import project_root
from tomato_det.eval.metrics import GroundTruth, Prediction, evaluate_detections


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_coco_ground_truth(path: Path) -> list[GroundTruth]:
    payload = _load_json(path)
    images = {
        str(item["id"])
        for item in payload.get("images", [])
        if isinstance(item, dict) and "id" in item
    }
    records: list[GroundTruth] = []
    for item in payload.get("annotations", []):
        if not isinstance(item, dict) or not {"image_id", "category_id", "bbox"} <= item.keys():
            raise ValueError("COCO annotations require image_id, category_id, and bbox")
        image_id = str(item["image_id"])
        if image_id not in images:
            raise ValueError(f"Annotation refers to unknown image_id: {image_id}")
        x, y, width, height = (float(value) for value in item["bbox"])
        records.append(
            GroundTruth(image_id, int(item["category_id"]), (x, y, x + width, y + height))
        )
    return records


def read_predictions(path: Path) -> list[Prediction]:
    payload = _load_json(path)
    items = payload.get("predictions", payload if isinstance(payload, list) else None)
    if not isinstance(items, list):
        raise ValueError("Prediction artifact must contain a 'predictions' list")
    records: list[Prediction] = []
    for item in items:
        if (
            not isinstance(item, dict)
            or not {"image_id", "class_id", "score", "box"} <= item.keys()
        ):
            raise ValueError("Predictions require image_id, class_id, score, and pixel xyxy box")
        box = tuple(float(value) for value in item["box"])
        if len(box) != 4:
            raise ValueError("Prediction box must contain four xyxy values")
        records.append(
            Prediction(str(item["image_id"]), int(item["class_id"]), float(item["score"]), box)
        )
    return records


def create_evaluation_lock(
    *,
    checkpoint: Path,
    data_yaml: Path,
    validation_metrics: Path,
    output_path: Path,
) -> Path:
    """Freeze the exact checkpoint/data pair only after a real validation artifact exists."""
    validation = _load_json(validation_metrics)
    if validation.get("split") != "val":
        raise ValueError("Only a validation metrics artifact may freeze final-test configuration")
    for artifact in (checkpoint, data_yaml):
        if not artifact.is_file():
            raise FileNotFoundError(f"Cannot freeze missing artifact: {artifact}")
    lock = {
        "schema_version": 1,
        "purpose": "final_test_only",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": _sha256(data_yaml),
        "validation_metrics": str(validation_metrics.resolve()),
        "validation_metrics_sha256": _sha256(validation_metrics),
        **runtime_metadata(project_root()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return output_path


def _verify_lock(lock_path: Path, checkpoint: Path, data_yaml: Path) -> None:
    lock = _load_json(lock_path)
    if lock.get("purpose") != "final_test_only":
        raise ValueError("Evaluation lock has an unknown purpose")
    if lock.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("Checkpoint differs from the frozen evaluation lock")
    if lock.get("data_yaml_sha256") != _sha256(data_yaml):
        raise ValueError("Dataset configuration differs from the frozen evaluation lock")


def evaluate_prediction_artifacts(
    *,
    annotations: Path,
    predictions: Path,
    class_ids: list[int],
    split: str,
    confidence_threshold: float,
    operating_iou: float,
    operating_confidence: float | None = None,
    checkpoint: Path | None = None,
    data_yaml: Path | None = None,
    evaluation_lock: Path | None = None,
) -> dict[str, object]:
    """Evaluate persisted model predictions; TEST requires an immutable matching lock."""
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    if split == "test":
        if not evaluation_lock or not checkpoint or not data_yaml:
            raise ValueError("TEST evaluation requires checkpoint, data_yaml, and evaluation_lock")
        _verify_lock(evaluation_lock, checkpoint, data_yaml)
    metrics = evaluate_detections(
        read_coco_ground_truth(annotations),
        read_predictions(predictions),
        class_ids=class_ids,
        confidence_threshold=confidence_threshold,
        operating_iou=operating_iou,
        operating_confidence=operating_confidence,
    )
    return {
        **runtime_metadata(project_root()),
        "split": split,
        "annotations": str(annotations),
        "predictions": str(predictions),
        "metrics": metrics,
    }
