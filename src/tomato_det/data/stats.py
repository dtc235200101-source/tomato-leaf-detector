"""Reproducible JSON reports for inspection and preparation runs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tomato_det.common.types import DataIssue, ImageRecord


def summarize_records(records: list[ImageRecord]) -> dict[str, Any]:
    return {
        "images": len(records),
        "boxes": sum(len(record.boxes) for record in records),
        "images_with_multiple_boxes": sum(len(record.boxes) > 1 for record in records),
        "boxes_by_class_id": dict(
            sorted(Counter(box.class_id for record in records for box in record.boxes).items())
        ),
    }


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def issue_payload(issues: list[DataIssue]) -> dict[str, Any]:
    return {
        "count": len(issues),
        "by_code": dict(sorted(Counter(issue.code for issue in issues).items())),
        "items": [asdict(issue) for issue in issues],
    }


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
