"""Dependency-light detection metrics for persisted, real prediction artifacts.

The evaluator deliberately consumes boxes in pixel ``xyxy`` coordinates.  Serving
uses normalized boxes, so callers must explicitly convert before creating a
``Prediction``.  This prevents accidental mixing of the two coordinate spaces.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

Box = tuple[float, float, float, float]


def _valid_box(box: Box) -> bool:
    return all(isfinite(value) for value in box) and box[0] < box[2] and box[1] < box[3]


@dataclass(frozen=True, slots=True)
class GroundTruth:
    image_id: str
    class_id: int
    box: Box

    def __post_init__(self) -> None:
        if self.class_id < 0 or not _valid_box(self.box):
            raise ValueError("Ground-truth class_id and pixel xyxy box must be valid")


@dataclass(frozen=True, slots=True)
class Prediction:
    image_id: str
    class_id: int
    score: float
    box: Box

    def __post_init__(self) -> None:
        if self.class_id < 0 or not 0 <= self.score <= 1 or not _valid_box(self.box):
            raise ValueError("Prediction class_id, score, and pixel xyxy box must be valid")


def box_iou(first: Box, second: Box) -> float:
    """Return IoU for two pixel xyxy boxes."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _average_precision(matches: list[bool], total_ground_truth: int) -> float | None:
    if not total_ground_truth:
        return None
    if not matches:
        return 0.0
    precision: list[float] = []
    true_positive = 0
    for index, matched in enumerate(matches, start=1):
        true_positive += int(matched)
        precision.append(true_positive / index)
    recall = [sum(matches[: index + 1]) / total_ground_truth for index in range(len(matches))]
    # COCO's 101-point interpolated AP.  This is deterministic and has no external dependency.
    return (
        sum(
            max((p for p, r in zip(precision, recall, strict=True) if r >= point), default=0.0)
            for point in (step / 100 for step in range(101))
        )
        / 101
    )


def _matches_for_class(
    ground_truth: list[GroundTruth], predictions: list[Prediction], iou_threshold: float
) -> list[bool]:
    unmatched: dict[str, set[int]] = {}
    for index, item in enumerate(ground_truth):
        unmatched.setdefault(item.image_id, set()).add(index)
    matches: list[bool] = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = unmatched.get(prediction.image_id, set())
        best = max(
            candidates,
            key=lambda index: box_iou(prediction.box, ground_truth[index].box),
            default=None,
        )
        if best is not None and box_iou(prediction.box, ground_truth[best].box) >= iou_threshold:
            candidates.remove(best)
            matches.append(True)
        else:
            matches.append(False)
    return matches


def evaluate_detections(
    ground_truth: Iterable[GroundTruth],
    predictions: Iterable[Prediction],
    *,
    class_ids: Iterable[int],
    confidence_threshold: float = 0.001,
    operating_iou: float = 0.5,
    operating_confidence: float | None = None,
) -> dict[str, object]:
    """Compute mAP at a low threshold and P/R at the selected operating threshold."""
    if not 0 <= confidence_threshold <= 1 or not 0 < operating_iou <= 1:
        raise ValueError("confidence_threshold must be in [0, 1] and operating_iou in (0, 1]")
    operating_confidence = (
        confidence_threshold if operating_confidence is None else operating_confidence
    )
    if not 0 <= operating_confidence <= 1:
        raise ValueError("operating_confidence must be in [0, 1]")
    ids = list(class_ids)
    if ids != list(range(len(ids))):
        raise ValueError("class_ids must be contiguous from zero")
    truth = list(ground_truth)
    predicted = [item for item in predictions if item.score >= confidence_threshold]
    operating_predictions = [
        item for item in predicted if item.score >= operating_confidence
    ]
    if any(item.class_id not in ids for item in [*truth, *predicted]):
        raise ValueError("Annotations contain a class outside class_ids")

    per_class: list[dict[str, object]] = []
    ap50_values: list[float] = []
    ap5095_values: list[float] = []
    for class_id in ids:
        class_truth = [item for item in truth if item.class_id == class_id]
        class_predictions = [item for item in predicted if item.class_id == class_id]
        ap50 = _average_precision(
            _matches_for_class(class_truth, class_predictions, 0.5), len(class_truth)
        )
        ap_values = [
            _average_precision(
                _matches_for_class(class_truth, class_predictions, threshold), len(class_truth)
            )
            for threshold in (0.5 + step * 0.05 for step in range(10))
        ]
        ap5095 = (
            None if not class_truth else sum(value or 0.0 for value in ap_values) / len(ap_values)
        )
        if ap50 is not None:
            ap50_values.append(ap50)
        if ap5095 is not None:
            ap5095_values.append(ap5095)
        per_class.append(
            {"class_id": class_id, "support": len(class_truth), "ap50": ap50, "ap50_95": ap5095}
        )

    matrix = [[0 for _ in range(len(ids) + 1)] for _ in range(len(ids) + 1)]
    unmatched_truth = set(range(len(truth)))
    ordered_predictions = sorted(operating_predictions, key=lambda item: item.score, reverse=True)
    for prediction in ordered_predictions:
        candidates = [
            index for index in unmatched_truth if truth[index].image_id == prediction.image_id
        ]
        match = max(
            candidates, key=lambda index: box_iou(prediction.box, truth[index].box), default=None
        )
        if match is not None and box_iou(prediction.box, truth[match].box) >= operating_iou:
            matrix[truth[match].class_id][prediction.class_id] += 1
            unmatched_truth.remove(match)
        else:
            matrix[-1][prediction.class_id] += 1
    for index in unmatched_truth:
        matrix[truth[index].class_id][-1] += 1
    true_positive = sum(matrix[index][index] for index in ids)
    false_positive = sum(matrix[-1][index] for index in ids) + sum(
        matrix[actual][predicted_id]
        for actual in ids
        for predicted_id in ids
        if actual != predicted_id
    )
    false_negative = sum(matrix[index][-1] for index in ids) + sum(
        matrix[actual][predicted_id]
        for actual in ids
        for predicted_id in ids
        if actual != predicted_id
    )
    return {
        "schema_version": 1,
        "coordinate_space": "pixel_xyxy",
        "confidence_threshold": confidence_threshold,
        "operating_confidence_threshold": operating_confidence,
        "operating_iou": operating_iou,
        "map50": sum(ap50_values) / len(ap50_values) if ap50_values else None,
        "map50_95": sum(ap5095_values) / len(ap5095_values) if ap5095_values else None,
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0,
        "recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0,
        "per_class": per_class,
        "confusion_matrix": {"labels": [*ids, "background"], "counts": matrix},
    }
