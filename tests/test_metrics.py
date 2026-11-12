import json
from pathlib import Path

import pytest

from tomato_det.eval.artifacts import create_evaluation_lock, evaluate_prediction_artifacts
from tomato_det.eval.benchmark_latency import LatencySummary, write_benchmark
from tomato_det.eval.metrics import GroundTruth, Prediction, evaluate_detections
from tomato_det.eval.report import render_report


def test_unified_metrics_reports_map_operating_point_and_confusion() -> None:
    truth = [GroundTruth("one", 0, (0, 0, 10, 10)), GroundTruth("two", 1, (0, 0, 10, 10))]
    predictions = [
        Prediction("one", 0, 0.9, (0, 0, 10, 10)),
        Prediction("one", 0, 0.8, (20, 20, 30, 30)),
    ]

    result = evaluate_detections(truth, predictions, class_ids=[0, 1])

    assert result["map50"] == 0.5
    assert result["map50_95"] == 0.5
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["confusion_matrix"]["counts"] == [[1, 0, 0], [0, 0, 1], [1, 0, 0]]


def test_operating_confidence_changes_only_precision_and_recall() -> None:
    truth = [GroundTruth("one", 0, (0, 0, 10, 10))]
    predictions = [
        Prediction("one", 0, 0.9, (0, 0, 10, 10)),
        Prediction("one", 0, 0.2, (20, 20, 30, 30)),
    ]

    low = evaluate_detections(truth, predictions, class_ids=[0])
    high = evaluate_detections(
        truth, predictions, class_ids=[0], operating_confidence=0.5
    )

    assert high["map50"] == low["map50"]
    assert high["precision"] == pytest.approx(1.0)
    assert high["recall"] == pytest.approx(1.0)
    assert low["precision"] == pytest.approx(0.5)


def test_test_evaluation_requires_matching_frozen_config(tmp_path: Path) -> None:
    annotations = tmp_path / "test.json"
    predictions = tmp_path / "predictions.json"
    checkpoint = tmp_path / "best.pt"
    data_yaml = tmp_path / "dataset.yaml"
    val_metrics = tmp_path / "metrics_val.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1}],
                "annotations": [{"image_id": 1, "category_id": 0, "bbox": [0, 0, 4, 4]}],
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            {"predictions": [{"image_id": 1, "class_id": 0, "score": 0.9, "box": [0, 0, 4, 4]}]}
        ),
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"checkpoint")
    data_yaml.write_text("path: data", encoding="utf-8")
    val_metrics.write_text(json.dumps({"split": "val", "metrics": {}}), encoding="utf-8")
    lock = create_evaluation_lock(
        checkpoint=checkpoint,
        data_yaml=data_yaml,
        validation_metrics=val_metrics,
        output_path=tmp_path / "lock.json",
    )

    result = evaluate_prediction_artifacts(
        annotations=annotations,
        predictions=predictions,
        class_ids=[0],
        split="test",
        confidence_threshold=0.001,
        operating_iou=0.5,
        checkpoint=checkpoint,
        data_yaml=data_yaml,
        evaluation_lock=lock,
    )

    assert result["split"] == "test"
    data_yaml.write_text("path: changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        evaluate_prediction_artifacts(
            annotations=annotations,
            predictions=predictions,
            class_ids=[0],
            split="test",
            confidence_threshold=0.001,
            operating_iou=0.5,
            checkpoint=checkpoint,
            data_yaml=data_yaml,
            evaluation_lock=lock,
        )


def test_report_and_benchmark_are_artifact_based(tmp_path: Path) -> None:
    metrics = {
        "split": "val",
        "metrics": {
            "map50": 0.5,
            "map50_95": 0.4,
            "precision": 0.6,
            "recall": 0.7,
            "per_class": [{"class_id": 0, "support": 2, "ap50": 0.5, "ap50_95": 0.4}],
            "confusion_matrix": {"labels": [0, "background"], "counts": [[1, 1], [0, 1]]},
        },
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    output = render_report(metrics_path, tmp_path / "report", figures=False)
    benchmark = write_benchmark(
        LatencySummary(3, 2.0, 3.0),
        tmp_path / "latency.json",
        model_id="unit-test",
        device="cpu",
        warmup_runs=1,
        measured_runs=3,
        input_description="synthetic unit-test input",
    )

    assert output["report"].is_file()
    assert json.loads(benchmark.read_text(encoding="utf-8"))["latency"]["p95_ms"] == 3.0
