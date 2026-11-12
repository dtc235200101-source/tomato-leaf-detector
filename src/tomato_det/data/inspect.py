"""High-level PlantDoc inspection and preparation orchestration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from tomato_det.common.classes import load_classes
from tomato_det.common.config import load_yaml
from tomato_det.common.paths import config_path, project_root
from tomato_det.common.types import ImageRecord
from tomato_det.data.dedup import group_near_duplicates, groups_crossing_source_splits
from tomato_det.data.export_coco import export_coco
from tomato_det.data.export_yolo import export_yolo
from tomato_det.data.parse_plantdoc import ParseResult, parse_plantdoc_csv
from tomato_det.data.split import assert_no_group_leakage, make_splits
from tomato_det.data.stats import issue_payload, summarize_records, write_json_report
from tomato_det.data.tomato_subset import select_tomato_subset
from tomato_det.data.visualize import write_audit_preview


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root() / path


def _load_data_config(data_config_path: Path | None) -> dict[str, Any]:
    config = load_yaml(data_config_path or config_path("data.yaml"))
    required = {
        "raw_dir",
        "train_images_dir",
        "test_images_dir",
        "train_labels",
        "test_labels",
        "processed_yolo_dir",
        "processed_coco_dir",
        "splits_dir",
        "val_fraction",
        "seed",
        "phash_distance",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"data.yaml missing fields: {', '.join(sorted(missing))}")
    return config


def _parse_sources(config: dict[str, Any]) -> tuple[ParseResult, ParseResult]:
    raw_dir = _resolve(str(config["raw_dir"]))
    train = parse_plantdoc_csv(
        raw_dir / str(config["train_labels"]), raw_dir / str(config["train_images_dir"]), "train"
    )
    test = parse_plantdoc_csv(
        raw_dir / str(config["test_labels"]), raw_dir / str(config["test_images_dir"]), "test"
    )
    return train, test


def inspect_dataset(data_config_path: Path | None = None) -> dict[str, Any]:
    """Inspect raw source CSVs/images without writing processed image data."""
    config = _load_data_config(data_config_path)
    train, test = _parse_sources(config)
    classes = load_classes(config_path("classes.yaml"))
    train_subset, train_report = select_tomato_subset(train.records, classes)
    test_subset, test_report = select_tomato_subset(test.records, classes)
    result = {
        "source": {
            "train": {
                "records": summarize_records(train.records),
                "labels": dict(train.source_labels),
            },
            "test": {
                "records": summarize_records(test.records),
                "labels": dict(test.source_labels),
            },
        },
        "tomato_subset": {
            "train": {"records": summarize_records(train_subset), **asdict(train_report)},
            "test": {"records": summarize_records(test_subset), **asdict(test_report)},
        },
        "issues": issue_payload(train.issues + test.issues),
    }
    return result


def prepare_dataset(data_config_path: Path | None = None) -> dict[str, Any]:
    """Filter, deduplicate, split, and export PlantDoc data using explicit source data only."""
    config = _load_data_config(data_config_path)
    classes = load_classes(config_path("classes.yaml"))
    train_parse, test_parse = _parse_sources(config)
    initial_issues = train_parse.issues + test_parse.issues
    if any(issue.code in {"MISSING_CSV", "INVALID_CSV_HEADER"} for issue in initial_issues):
        raise ValueError("Cannot prepare data: source CSV is missing or has an invalid header")
    train_records, train_subset_report = select_tomato_subset(train_parse.records, classes)
    test_records, test_subset_report = select_tomato_subset(test_parse.records, classes)
    dedup = group_near_duplicates(train_records + test_records, int(config["phash_distance"]))
    crossing = groups_crossing_source_splits(train_records + test_records, dedup.groups)
    split = make_splits(
        train_records,
        test_records,
        dedup.groups,
        float(config["val_fraction"]),
        int(config["seed"]),
    )
    assert_no_group_leakage(split, dedup.groups)
    by_id = {record.image_id: record for record in train_records + test_records}
    split_records: dict[str, list[ImageRecord]] = {
        "train": [by_id[image_id] for image_id in split.train_ids],
        "val": [by_id[image_id] for image_id in split.val_ids],
        "test": [by_id[image_id] for image_id in split.test_ids],
    }
    yolo_dir = _resolve(str(config["processed_yolo_dir"]))
    coco_dir = _resolve(str(config["processed_coco_dir"]))
    occupied_outputs = [
        path for path in (yolo_dir, coco_dir) if path.exists() and any(path.iterdir())
    ]
    if occupied_outputs:
        rendered = ", ".join(str(path) for path in occupied_outputs)
        raise FileExistsError(f"Refusing to overwrite an existing export: {rendered}")
    yolo_counts = export_yolo(split_records, yolo_dir, classes)
    coco_counts = export_coco(split_records, coco_dir, classes)
    splits_dir = _resolve(str(config["splits_dir"]))
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, image_ids in (
        ("train", split.train_ids),
        ("val", split.val_ids),
        ("test", split.test_ids),
    ):
        (splits_dir / f"{split_name}.txt").write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    report = {
        "config": {
            "seed": config["seed"],
            "val_fraction": config["val_fraction"],
            "phash_distance": config["phash_distance"],
        },
        "splits": {name: summarize_records(records) for name, records in split_records.items()},
        "subset": {"train": asdict(train_subset_report), "test": asdict(test_subset_report)},
        "dedup": {
            "groups": len(set(dedup.groups.values())),
            "groups_crossing_train_test": len(crossing),
            "excluded_train_images_crossing_test": len(split.excluded_train_ids),
            "hash_failures": issue_payload(dedup.issues),
        },
        "exports": {"yolo_images": yolo_counts, "coco_images": coco_counts},
        "issues": issue_payload(initial_issues + dedup.issues),
    }
    interim_dir = project_root() / "data" / "interim"
    write_json_report(interim_dir / "prepare_report.json", report)
    write_json_report(
        interim_dir / "image_groups.json", {"groups": dedup.groups, "hashes": dedup.hashes}
    )
    write_audit_preview(split_records["train"], classes, interim_dir / "audit_train.jpg")
    return report
