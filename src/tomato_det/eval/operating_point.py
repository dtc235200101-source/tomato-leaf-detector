"""Select an operating confidence threshold using validation data only."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from tomato_det.eval.metrics import GroundTruth, Prediction, evaluate_detections


def select_operating_point(
    ground_truth: Iterable[GroundTruth],
    predictions: Iterable[Prediction],
    *,
    class_ids: Sequence[int],
    operating_iou: float = 0.5,
    thresholds: Sequence[float] | None = None,
) -> dict[str, float]:
    """Return the threshold with the best validation F1 on a fixed, documented grid.

    AP itself is computed from all predictions at the evaluator's low confidence
    threshold. This helper only chooses the deployment threshold, so it is kept
    separate from the final metrics artifact.
    """
    truth = list(ground_truth)
    predicted = list(predictions)
    candidate_thresholds = list(thresholds) if thresholds is not None else [
        step / 20 for step in range(1, 20)
    ]
    if not predicted:
        return {"threshold": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    best: dict[str, float] | None = None
    for threshold in candidate_thresholds:
        result = evaluate_detections(
            truth,
            predicted,
            class_ids=class_ids,
            confidence_threshold=threshold,
            operating_iou=operating_iou,
        )
        precision = float(result["precision"])
        recall = float(result["recall"])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if best is None or (candidate["f1"], candidate["precision"], candidate["threshold"]) > (
            best["f1"],
            best["precision"],
            best["threshold"],
        ):
            best = candidate
    if best is None:
        raise RuntimeError("Operating point search produced no candidate")
    return best
