"""Export real detector predictions as pixel-space artifacts for the unified evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tomato_det.common.config import load_yaml
from tomato_det.common.paths import config_path, project_root
from tomato_det.models.yolo_detector import YoloDetector


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root() / path


def _image_ids(coco_annotations: Path) -> dict[str, int]:
    payload = json.loads(coco_annotations.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{coco_annotations} has no images list")
    mapping: dict[str, int] = {}
    for item in images:
        if not isinstance(item, dict) or "id" not in item or "file_name" not in item:
            raise ValueError(f"{coco_annotations} contains an invalid image record")
        mapping[Path(str(item["file_name"])).name] = int(item["id"])
    return mapping


def export_predictions(
    *,
    checkpoint: Path,
    split: str,
    output_path: Path,
    confidence_threshold: float = 0.001,
    nms_iou: float = 0.6,
    image_size: int = 640,
    max_detections: int = 100,
    device: str = "auto",
    model_id: str | None = None,
    detector: Any | None = None,
    data_config_path: Path | None = None,
) -> Path:
    """Run a local checkpoint over one split and write pixel xyxy predictions.

    Inference happens only over already-exported local images. The output schema is the
    one consumed by ``eval-artifacts``; no metric is computed here.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite prediction artifact: {output_path}")

    data_config = load_yaml(data_config_path or config_path("data.yaml"))
    yolo_dir = _resolve(str(data_config["processed_yolo_dir"]))
    coco_dir = _resolve(str(data_config["processed_coco_dir"]))
    images_dir = yolo_dir / "images" / split
    annotations = coco_dir / "annotations" / f"{split}.json"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Processed image directory is missing: {images_dir}")
    if not annotations.is_file():
        raise FileNotFoundError(f"COCO annotation file is missing: {annotations}")

    image_ids = _image_ids(annotations)
    active_detector = detector or YoloDetector(
        model_id=model_id or str(load_yaml(config_path("app.yaml"))["default_model"]),
        checkpoint=checkpoint,
        device=device,
    )
    predictions: list[dict[str, object]] = []
    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
    for image_path in image_paths:
        image_id = image_ids.get(image_path.name)
        if image_id is None:
            raise ValueError(f"Image is not present in COCO annotations: {image_path.name}")
        with Image.open(image_path) as source:
            image_rgb = np.asarray(source.convert("RGB"))
            width, height = source.size
        detections = active_detector.predict(
            image_rgb,
            conf=confidence_threshold,
            iou=nms_iou,
            imgsz=image_size,
            max_det=max_detections,
        )
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            predictions.append(
                {
                    "image_id": image_id,
                    "class_id": detection.class_id,
                    "score": detection.score,
                    "box": [x1 * width, y1 * height, x2 * width, y2 * height],
                }
            )

    payload = {
        "schema_version": 1,
        "coordinate_space": "pixel_xyxy",
        "split": split,
        "checkpoint": str(checkpoint),
        "confidence_threshold": confidence_threshold,
        "nms_iou": nms_iou,
        "image_size": image_size,
        "max_detections": max_detections,
        "images": len(image_paths),
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
