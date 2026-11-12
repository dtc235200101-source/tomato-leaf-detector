"""Generate tables and optional plots strictly from measured artifacts.

Every number written to ``REPORT.md`` is read from a JSON artifact produced by a real
run. Missing optional artifacts simply omit their section; nothing is estimated or
filled in by hand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tomato_det.common.classes import load_classes
from tomato_det.common.paths import config_path


def _read_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise ValueError("Metrics artifact must contain a 'metrics' object")
    return payload


def _read_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Optional artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must contain a JSON object: {path}")
    return payload


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _class_names() -> dict[int, tuple[str, str]]:
    try:
        classes = load_classes(config_path("classes.yaml"))
    except (FileNotFoundError, ValueError):
        return {}
    return {spec.id: (spec.slug, spec.name_vi) for spec in classes}


def _summary_table(title: str, payload: dict[str, Any]) -> list[str]:
    metrics = payload["metrics"]
    return [
        f"### {title}",
        "",
        f"Nguồn: `{payload.get('annotations', '—')}` + `{payload.get('predictions', '—')}`",
        "",
        "| mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |",
        "| ---: | ---: | ---: | ---: |",
        f"| {_number(metrics['map50'])} | {_number(metrics['map50_95'])} | "
        f"{_number(metrics['precision'])} | {_number(metrics['recall'])} |",
        "",
    ]


def _per_class_table(payload: dict[str, Any]) -> list[str]:
    names = _class_names()
    rows = payload["metrics"]["per_class"]
    lines = [
        "### Per-class metrics (test)",
        "",
        "| id | lớp | tên tiếng Việt | support | AP@0.5 | AP@0.5:0.95 |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    zero_support = []
    for row in rows:
        class_id = int(row["class_id"])
        slug, name_vi = names.get(class_id, ("—", "—"))
        if int(row.get("support", 0)) == 0:
            zero_support.append(f"{class_id} ({slug})")
        lines.append(
            f"| {class_id} | `{slug}` | {name_vi} | {row.get('support', '—')} | "
            f"{_number(row.get('ap50'))} | {_number(row.get('ap50_95'))} |"
        )
    lines.append("")
    if zero_support:
        lines.append(
            "**Cảnh báo:** các lớp sau không có ground-truth box trong tập test, "
            f"nên AP không có ý nghĩa: {', '.join(zero_support)}."
        )
        lines.append("")
    return lines


def _operating_point_section(payload: dict[str, Any] | None) -> list[str]:
    if payload is None:
        return []
    point = payload.get("operating_point", {})
    return [
        "### Operating point (chọn trên validation)",
        "",
        "| threshold | precision | recall | F1 |",
        "| ---: | ---: | ---: | ---: |",
        f"| {_number(point.get('threshold'), 2)} | {_number(point.get('precision'))} | "
        f"{_number(point.get('recall'))} | {_number(point.get('f1'))} |",
        "",
        "Ngưỡng này được chọn bằng F1 trên `val` và không dùng test để điều chỉnh.",
        "",
    ]


def _latency_section(payload: dict[str, Any] | None) -> list[str]:
    if payload is None:
        return []
    latency = payload.get("latency", {})
    return [
        "### Inference latency",
        "",
        "| thiết bị | imgsz | warm-up | số lượt đo | P50 (ms) | P95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| {payload.get('device', '—')} | {payload.get('image_size', '—')} | "
        f"{payload.get('warmup_runs', '—')} | {payload.get('measured_runs', '—')} | "
        f"{_number(latency.get('p50_ms'), 2)} | {_number(latency.get('p95_ms'), 2)} |",
        "",
        f"Input: `{payload.get('input', '—')}`. Đây là latency inference-only trên local, "
        "không gồm encode JPEG, WebSocket hay vẽ overlay.",
        "",
    ]


def _dataset_section(payload: dict[str, Any] | None) -> list[str]:
    if payload is None:
        return []
    splits = payload.get("splits", {})
    dedup = payload.get("dedup", {})
    lines = [
        "### Dataset splits",
        "",
        "| split | ảnh | box | ảnh nhiều box |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("train", "val", "test"):
        split = splits.get(name, {})
        lines.append(
            f"| {name} | {split.get('images', '—')} | {split.get('boxes', '—')} | "
            f"{split.get('images_with_multiple_boxes', '—')} |"
        )
    lines += [
        "",
        f"Near-duplicate groups: {dedup.get('groups', '—')}; "
        f"nhóm cắt qua train/test: {dedup.get('groups_crossing_train_test', '—')}; "
        f"ảnh train bị loại vì trùng test: "
        f"{dedup.get('excluded_train_images_crossing_test', '—')}.",
        "",
    ]
    return lines


def render_report(
    metrics_path: Path,
    output_dir: Path,
    *,
    figures: bool = True,
    validation_metrics: Path | None = None,
    operating_point: Path | None = None,
    latency: Path | None = None,
    dataset_report: Path | None = None,
) -> dict[str, Path]:
    """Write CSV/Markdown and, when installed, plots from real evaluation JSON files."""
    payload = _read_metrics(metrics_path)
    metrics = payload["metrics"]
    required = {"map50", "map50_95", "precision", "recall", "per_class", "confusion_matrix"}
    missing = required.difference(metrics)
    if missing:
        raise ValueError(
            f"Metrics artifact lacks unified evaluator fields: {', '.join(sorted(missing))}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["split", "map50", "map50_95", "precision", "recall"]
        )
        writer.writeheader()
        writer.writerow(
            {"split": payload.get("split"), **{key: metrics[key] for key in writer.fieldnames[1:]}}
        )
    per_class_path = output_dir / "per_class_metrics.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class_id", "support", "ap50", "ap50_95"])
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    confusion = metrics["confusion_matrix"]
    confusion_path = output_dir / "confusion_matrix.csv"
    with confusion_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *confusion["labels"]])
        for label, row in zip(confusion["labels"], confusion["counts"], strict=True):
            writer.writerow([label, *row])

    validation = _read_optional(validation_metrics)
    point = _read_optional(operating_point)
    timings = _read_optional(latency)
    dataset = _read_optional(dataset_report)

    report_lines = [
        "# Evaluation report — Tomato Leaf Detector",
        "",
        f"Sinh tự động từ artifact: `{metrics_path}`  ",
        f"Split được chấm: `{payload.get('split')}`  ",
        f"Thời điểm tạo artifact: `{payload.get('created_at_utc', '—')}`",
        "",
        "> Mọi số liệu dưới đây được đọc trực tiếp từ JSON của lần chạy thật. "
        "Không có giá trị nào được nhập tay hoặc ước lượng.",
        "",
    ]
    if validation is not None:
        report_lines += _summary_table("Validation", validation)
    report_lines += _summary_table("Test (chỉ chạy sau evaluation lock)", payload)
    report_lines += _per_class_table(payload)
    report_lines += _operating_point_section(point)
    report_lines += _latency_section(timings)
    report_lines += _dataset_section(dataset)
    report_lines += [
        "## Phạm vi và hạn chế",
        "",
        "PlantDoc gồm ảnh thực địa lấy từ Internet với độ phủ lớp không đều; nhãn đánh dấu "
        "một lá hoặc cụm lá, không phân đoạn từng vết bệnh. Ảnh webcam khác về ánh sáng, "
        "ống kính, khoảng cách, độ mờ, giống cây, nền và giai đoạn bệnh, nên các số liệu này "
        "không chứng minh hiệu năng ngoài split đã đánh giá.",
        "",
        "MVP chạy một seed trên CPU với cấu hình ngắn (`imgsz=320`, 10 epoch); chưa có "
        "bootstrap confidence interval và chưa chạy ma trận 6 cấu hình × 3 seed. "
        "Không nên dùng kết quả này để kết luận kiến trúc nào tốt hơn.",
        "",
    ]
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    result = {
        "summary": summary_path,
        "per_class": per_class_path,
        "confusion": confusion_path,
        "report": report_path,
    }
    if figures:
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required for figures; install requirements-train.txt"
            ) from error
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        classes = metrics["per_class"]
        figure_path = figure_dir / "per_class_ap.png"
        plt.figure(figsize=(9, 4))
        plt.bar(
            [str(item["class_id"]) for item in classes], [item["ap50"] or 0 for item in classes]
        )
        plt.xlabel("Class ID")
        plt.ylabel("AP@0.5")
        plt.title("Per-class AP@0.5")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=160)
        plt.close()
        result["per_class_figure"] = figure_path
    return result
