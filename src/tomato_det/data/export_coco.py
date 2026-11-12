"""COCO annotation export sharing the same validated split records as YOLO."""

from __future__ import annotations

import json
from pathlib import Path

from tomato_det.common.classes import ClassSpec
from tomato_det.common.types import ImageRecord


def export_coco(
    split_records: dict[str, list[ImageRecord]], output_dir: Path, classes: list[ClassSpec]
) -> dict[str, int]:
    """Write one COCO JSON file per split; image paths point to the sibling YOLO export."""
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    categories = [
        {"id": spec.id, "name": spec.slug, "supercategory": spec.group} for spec in classes
    ]
    counts: dict[str, int] = {}
    for split_name, records in split_records.items():
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        annotation_id = 1
        for image_id, record in enumerate(records, start=1):
            images.append(
                {
                    "id": image_id,
                    "file_name": f"../tomato_yolo/images/{split_name}/{Path(record.filename).name}",
                    "width": record.width,
                    "height": record.height,
                }
            )
            for box in record.boxes:
                if box.class_id is None:
                    raise ValueError(f"Box in {record.image_id} is missing a mapped class")
                if not box.is_valid_for(record.width, record.height):
                    raise ValueError(f"Invalid box reached COCO export: {record.image_id}")
                width, height = box.xmax - box.xmin, box.ymax - box.ymin
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": box.class_id,
                        "bbox": [box.xmin, box.ymin, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        payload = {"images": images, "annotations": annotations, "categories": categories}
        (annotations_dir / f"{split_name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        counts[split_name] = len(records)
    return counts
