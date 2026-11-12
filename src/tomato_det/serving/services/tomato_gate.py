"""Optional local gate that rejects images before tomato disease detection.

The detector is trained only with tomato labels, so it cannot reliably decide
that a visually similar leaf belongs to another crop.  This gate is deliberately
separate and requires a locally trained binary classifier with the classes
``tomato_leaf`` and ``other``.  Without that checkpoint the API marks the input
as unverified instead of making a false claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class TomatoAssessmentStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class TomatoAssessment:
    status: TomatoAssessmentStatus
    message: str
    confidence: float | None = None


class TomatoGate(Protocol):
    def assess(self, image_rgb: Any) -> TomatoAssessment: ...


class UnavailableTomatoGate:
    """Honest fallback used until an explicit crop-eligibility checkpoint exists."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def assess(self, image_rgb: Any) -> TomatoAssessment:
        del image_rgb
        return TomatoAssessment(
            TomatoAssessmentStatus.UNVERIFIED,
            f"Chưa xác minh loại cây: {self._reason}",
        )


class YoloTomatoGate:
    """Run a local Ultralytics classification checkpoint before detector inference."""

    def __init__(
        self, checkpoint: Path, *, positive_class: str, threshold: float, image_size: int
    ) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Tomato gate checkpoint is missing: {checkpoint}")
        if not 0 < threshold <= 1:
            raise ValueError("Tomato gate threshold must be in (0, 1]")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Ultralytics is required for tomato gate") from error
        self._model = YOLO(str(checkpoint), task="classify")
        self._positive_class = positive_class
        self._threshold = threshold
        self._image_size = image_size

    def assess(self, image_rgb: Any) -> TomatoAssessment:
        results = self._model.predict(source=image_rgb, imgsz=self._image_size, verbose=False)
        if not results or results[0].probs is None:
            raise RuntimeError("Tomato gate returned no classification probabilities")
        probabilities = results[0].probs
        class_names = results[0].names
        class_id = int(probabilities.top1)
        label = str(class_names[class_id])
        confidence = float(probabilities.top1conf)
        accepted = label == self._positive_class and confidence >= self._threshold
        if accepted:
            return TomatoAssessment(
                TomatoAssessmentStatus.ACCEPTED, "Đã xác minh lá cà chua", confidence
            )
        return TomatoAssessment(
            TomatoAssessmentStatus.REJECTED,
            "Không xác nhận đây là lá cà chua. Vui lòng nhập ảnh lá cà chua rõ nét, "
            "không lẫn cây khác.",
            confidence,
        )


def build_tomato_gate(config: dict[str, object], project_dir: Path) -> TomatoGate:
    """Build a local gate or a truthful unavailable fallback from app configuration."""
    raw = config.get("tomato_gate")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return UnavailableTomatoGate("chưa cấu hình mô hình kiểm tra lá cà chua")
    checkpoint_value = raw.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        return UnavailableTomatoGate("thiếu đường dẫn checkpoint kiểm tra lá cà chua")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = project_dir / checkpoint
    if not checkpoint.is_file():
        return UnavailableTomatoGate("chưa có checkpoint kiểm tra lá cà chua")
    try:
        return YoloTomatoGate(
            checkpoint,
            positive_class=str(raw.get("positive_class", "tomato_leaf")),
            threshold=float(raw.get("threshold", 0.8)),
            image_size=int(raw.get("image_size", 224)),
        )
    except (ImportError, RuntimeError, ValueError) as error:
        return UnavailableTomatoGate(f"không thể nạp mô hình kiểm tra ({error})")
