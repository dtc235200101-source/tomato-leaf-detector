"""YOLO export with explicit, validated conversion from pixel xyxy boxes."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from tomato_det.common.classes import ClassSpec
from tomato_det.common.types import ImageRecord


def _safe_filename(record: ImageRecord) -> str:
    name = Path(record.filename).name
    if not name:
        raise ValueError(f"Image record {record.image_id} has no filename")
    return name


def export_yolo(
    split_records: dict[str, list[ImageRecord]], output_dir: Path, classes: list[ClassSpec]
) -> dict[str, int]:
    """Copy images and write matching YOLO labels and `dataset.yaml`.

    The destination must be empty or absent, preventing accidental mixing with an earlier export.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix output with existing directory: {output_dir}")
    names = {spec.id: spec.slug for spec in classes}
    counts: dict[str, int] = {}
    for split_name, records in split_records.items():
        images_dir = output_dir / "images" / split_name
        labels_dir = output_dir / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        for record in records:
            filename = _safe_filename(record)
            if filename in seen:
                raise ValueError(f"Filename collision in {split_name}: {filename}")
            seen.add(filename)
            shutil.copy2(record.path, images_dir / filename)
            label_path = labels_dir / f"{Path(filename).stem}.txt"
            lines: list[str] = []
            for box in record.boxes:
                if box.class_id not in names:
                    raise ValueError(f"Box in {record.image_id} has no configured class ID")
                x_center, y_center, width, height = box.to_yolo(record.width, record.height)
                if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
                    raise ValueError(
                        f"Normalization produced an out-of-range box for {record.image_id}"
                    )
                lines.append(
                    f"{box.class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"
                )
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[split_name] = len(records)
    dataset = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {spec.id: spec.slug for spec in classes},
    }
    (output_dir / "dataset.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
    )
    return counts
