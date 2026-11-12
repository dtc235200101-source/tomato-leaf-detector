from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from tomato_det.eval.metrics import GroundTruth, Prediction
from tomato_det.eval.operating_point import select_operating_point
from tomato_det.eval.predictions import export_predictions
from tomato_det.models.base import Detection


class _FakeDetector:
    def predict(self, image_rgb: object, **_: object) -> list[Detection]:
        return [Detection(0, "early_blight", 0.9, (0.1, 0.2, 0.8, 0.8))]


def test_export_predictions_converts_normalized_boxes_to_pixel_xyxy(tmp_path: Path) -> None:
    yolo_dir = tmp_path / "tomato_yolo"
    coco_dir = tmp_path / "tomato_coco"
    images_dir = yolo_dir / "images" / "val"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (20, 10), "green").save(images_dir / "leaf.jpg")
    annotations_dir = coco_dir / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "val.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 7, "file_name": "../tomato_yolo/images/val/leaf.jpg"},
                ],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "processed_yolo_dir": str(yolo_dir),
                "processed_coco_dir": str(coco_dir),
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "predictions_val.json"

    result = export_predictions(
        checkpoint=checkpoint,
        split="val",
        output_path=output,
        detector=_FakeDetector(),
        data_config_path=data_config,
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["split"] == "val"
    assert payload["coordinate_space"] == "pixel_xyxy"
    assert payload["predictions"] == [
        {"image_id": 7, "class_id": 0, "score": 0.9, "box": [2.0, 2.0, 16.0, 8.0]}
    ]


def test_operating_point_selects_best_validation_f1() -> None:
    truth = [GroundTruth("1", 0, (0, 0, 10, 10))]
    predictions = [
        Prediction("1", 0, 0.9, (0, 0, 10, 10)),
        Prediction("2", 0, 0.6, (0, 0, 10, 10)),
    ]

    result = select_operating_point(
        truth,
        predictions,
        class_ids=[0],
        thresholds=[0.5, 0.85, 0.95],
    )

    assert result == {"threshold": 0.85, "precision": 1.0, "recall": 1.0, "f1": 1.0}
