"""Shared serving representation for all detectors."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    score: float
    box: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.box
        if self.class_id < 0 or not self.class_name:
            raise ValueError(
                "Detection class_id must be non-negative and class_name must not be empty"
            )
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Detection score must be finite and in [0, 1]")
        if not all(isfinite(value) for value in self.box) or not (
            0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
        ):
            raise ValueError("Detection box must be finite normalized xyxy coordinates")


class Detector(Protocol):
    model_id: str

    def predict(
        self,
        image_rgb: Any,
        *,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        agnostic_nms: bool = False,
    ) -> list[Detection]: ...
