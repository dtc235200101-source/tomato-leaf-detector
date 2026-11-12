"""Typed, dataset-independent records used by the preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BoxRecord:
    """A pixel-space xyxy annotation after structural validation."""

    source_label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_id: int | None = None
    class_slug: str | None = None

    def is_valid_for(self, width: int, height: int) -> bool:
        return (
            width > 0
            and height > 0
            and all(isfinite(value) for value in (self.xmin, self.ymin, self.xmax, self.ymax))
            and 0 <= self.xmin < self.xmax <= width
            and 0 <= self.ymin < self.ymax <= height
        )

    def to_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        if not self.is_valid_for(width, height):
            raise ValueError("Cannot normalize an invalid bounding box")
        box_width = (self.xmax - self.xmin) / width
        box_height = (self.ymax - self.ymin) / height
        x_center = (self.xmin + self.xmax) / (2 * width)
        y_center = (self.ymin + self.ymax) / (2 * height)
        return x_center, y_center, box_width, box_height


@dataclass(slots=True)
class ImageRecord:
    """One readable source image and all valid source annotations attached to it."""

    image_id: str
    filename: str
    path: Path
    width: int
    height: int
    source_split: str
    boxes: list[BoxRecord] = field(default_factory=list)

    def with_boxes(self, boxes: list[BoxRecord]) -> ImageRecord:
        return ImageRecord(
            image_id=self.image_id,
            filename=self.filename,
            path=self.path,
            width=self.width,
            height=self.height,
            source_split=self.source_split,
            boxes=boxes,
        )


@dataclass(frozen=True, slots=True)
class DataIssue:
    """A non-silent data problem retained in an inspection report."""

    code: str
    message: str
    split: str
    filename: str | None = None
    row_number: int | None = None
