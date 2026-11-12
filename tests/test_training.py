from dataclasses import replace
from pathlib import Path

from tomato_det.eval.benchmark_latency import summarize_latency_ms
from tomato_det.eval.evaluate import _metric_payload
from tomato_det.train.train_yolo import load_training_config, training_preflight

ROOT = Path(__file__).resolve().parents[1]


class _ValidationResults:
    results_dict = {"metrics/mAP50(B)": 0.25, "metrics/mAP50-95(B)": 0.12, "ignored": "value"}


def test_training_preflight_reports_missing_real_prerequisites(tmp_path: Path) -> None:
    config = replace(
        load_training_config(ROOT / "configs" / "experiments" / "yolo11s_coco.yaml"),
        data_yaml=tmp_path / "missing_dataset.yaml",
        pretrained_weights=tmp_path / "missing_yolo11s.pt",
        run_dir=tmp_path / "new_run",
    )
    status = training_preflight(config)

    assert not status.ready
    assert any("dataset.yaml is missing" in error for error in status.errors)
    assert any("pretrained weights are missing" in error for error in status.errors)


def test_evaluation_metrics_are_extracted_from_actual_results_object() -> None:
    metrics = _metric_payload(_ValidationResults())

    assert metrics == {"metrics/mAP50(B)": 0.25, "metrics/mAP50-95(B)": 0.12}


def test_latency_summary_has_deterministic_percentiles() -> None:
    summary = summarize_latency_ms([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary.samples == 5
    assert summary.p50_ms == 3.0
    assert summary.p95_ms == 5.0
