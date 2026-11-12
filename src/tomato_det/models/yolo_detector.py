"""Ultralytics YOLO adapter that only loads an already-local checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tomato_det.common.classes import class_lookup, load_classes
from tomato_det.common.paths import config_path
from tomato_det.models.base import Detection
from tomato_det.models.registry import ModelUnavailableError


class YoloDetector:
    """Adapt Ultralytics results to the normalized project `Detection` contract."""

    def __init__(self, *, model_id: str, checkpoint: Path, device: str | None = None) -> None:
        if not checkpoint.is_file():
            raise ModelUnavailableError(f"Local checkpoint does not exist: {checkpoint}")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ModelUnavailableError("Ultralytics is not installed") from error
        self.model_id = model_id
        self._device = device
        self._model = YOLO(str(checkpoint), task="detect")
        self._classes_by_id = {
            spec.id: spec
            for spec in class_lookup(load_classes(config_path("classes.yaml"))).values()
        }

    def predict(
        self,
        image_rgb: Any,
        *,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        agnostic_nms: bool = False,
    ) -> list[Detection]:
        # Ultralytics interprets numpy images as BGR (OpenCV convention), while the
        # project Detector contract is RGB. Convert explicitly so every caller gets
        # the same channel order the checkpoint was trained with.
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is a runtime dependency
            np = None
        source = image_rgb
        if np is not None and isinstance(image_rgb, np.ndarray):
            if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
                raise ValueError("YOLO input must be an HxWx3 RGB array")
            source = np.ascontiguousarray(image_rgb[:, :, ::-1])
        arguments = {
            "source": source,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "max_det": max_det,
            "agnostic_nms": agnostic_nms,
            "verbose": False,
        }
        device = getattr(self, "_device", None)
        if device and device != "auto":
            arguments["device"] = device
        results = self._model.predict(**arguments)
        if not results:
            return []
        result = results[0]
        height, width = result.orig_shape
        if height <= 0 or width <= 0:
            raise RuntimeError("YOLO returned an invalid source image shape")
        detections: list[Detection] = []
        for xyxy, class_id, score in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            strict=True,
        ):
            category = self._classes_by_id.get(int(class_id))
            if category is None:
                raise RuntimeError(
                    f"Checkpoint emitted class outside the configured mapping: {class_id}"
                )
            x1, y1, x2, y2 = xyxy
            detections.append(
                Detection(
                    class_id=category.id,
                    class_name=category.slug,
                    score=float(score),
                    box=(x1 / width, y1 / height, x2 / width, y2 / height),
                )
            )
        return detections
