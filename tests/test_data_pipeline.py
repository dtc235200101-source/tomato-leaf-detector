import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from tomato_det.common.classes import load_classes
from tomato_det.common.types import BoxRecord, ImageRecord
from tomato_det.data.dedup import group_near_duplicates
from tomato_det.data.export_coco import export_coco
from tomato_det.data.export_yolo import export_yolo
from tomato_det.data.parse_plantdoc import parse_plantdoc_csv
from tomato_det.data.split import assert_no_group_leakage, make_splits
from tomato_det.data.tomato_subset import select_tomato_subset

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _record(tmp_path: Path, image_id: str, split: str, class_id: int = 0) -> ImageRecord:
    image_path = tmp_path / f"{Path(image_id).stem}.jpg"
    image = Image.new("RGB", (20, 10), "red")
    if "two" in image_id:
        ImageDraw.Draw(image).rectangle((0, 0, 9, 9), fill="black")
    image.save(image_path)
    return ImageRecord(
        image_id=image_id,
        filename=image_path.name,
        path=image_path,
        width=20,
        height=10,
        source_split=split,
        boxes=[BoxRecord("Tomato Early blight leaf", 1, 2, 11, 8, class_id, "early_blight")],
    )


def test_parser_reports_bad_rows_and_keeps_only_valid_boxes(tmp_path: Path) -> None:
    images = tmp_path / "TRAIN"
    images.mkdir()
    Image.new("RGB", (20, 10), "red").save(images / "valid.jpg")
    (images / "broken.jpg").write_text("not an image", encoding="utf-8")
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [
            {
                "filename": "valid.jpg",
                "width": 20,
                "height": 10,
                "class": "Tomato leaf",
                "xmin": 1,
                "ymin": 2,
                "xmax": 11,
                "ymax": 8,
            },
            {
                "filename": "valid.jpg",
                "width": 20,
                "height": 10,
                "class": "Tomato leaf",
                "xmin": 9,
                "ymin": 2,
                "xmax": 3,
                "ymax": 8,
            },
            {
                "filename": "missing.jpg",
                "width": 20,
                "height": 10,
                "class": "Tomato leaf",
                "xmin": 1,
                "ymin": 2,
                "xmax": 11,
                "ymax": 8,
            },
            {
                "filename": "broken.jpg",
                "width": 20,
                "height": 10,
                "class": "Tomato leaf",
                "xmin": 1,
                "ymin": 2,
                "xmax": 11,
                "ymax": 8,
            },
        ],
    )

    parsed = parse_plantdoc_csv(csv_path, images, "train")

    assert len(parsed.records) == 1
    assert len(parsed.records[0].boxes) == 1
    assert {issue.code for issue in parsed.issues} == {
        "INVALID_BOX",
        "MISSING_IMAGE",
        "CORRUPT_IMAGE",
    }


def test_subset_maps_only_explicit_tomato_labels(tmp_path: Path) -> None:
    record = _record(tmp_path, "train/a.jpg", "train")
    record.boxes.append(BoxRecord("Apple leaf", 2, 2, 10, 8))
    classes = load_classes(ROOT / "configs" / "classes.yaml")

    selected, report = select_tomato_subset([record], classes)

    assert selected[0].boxes[0].class_id == 0
    assert selected[0].boxes[0].class_slug == "early_blight"
    assert report.unmapped_labels == {"Apple leaf": 1}


def test_dedup_and_split_exclude_train_duplicate_of_test(tmp_path: Path) -> None:
    train_one = _record(tmp_path, "train/one.jpg", "train")
    train_two = _record(tmp_path, "train/two.jpg", "train")
    train_three = _record(tmp_path, "train/three.jpg", "train")
    test = _record(tmp_path, "test/copy.jpg", "test")
    test.path.write_bytes(train_one.path.read_bytes())

    dedup = group_near_duplicates([train_one, train_two, test], max_distance=0)
    assert dedup.groups[train_one.image_id] == dedup.groups[test.image_id]
    groups = {
        train_one.image_id: "duplicate-of-test",
        train_two.image_id: "train-group-one",
        train_three.image_id: "train-group-two",
        test.image_id: "duplicate-of-test",
    }
    split = make_splits(
        [train_one, train_two, train_three], [test], groups, val_fraction=0.5, seed=42
    )

    assert train_one.image_id in split.excluded_train_ids
    assert split.test_ids == [test.image_id]
    assert_no_group_leakage(split, dedup.groups)


def test_yolo_and_coco_exports_use_validated_coordinates(tmp_path: Path) -> None:
    record = _record(tmp_path, "train/one.jpg", "train")
    classes = load_classes(ROOT / "configs" / "classes.yaml")
    splits = {"train": [record], "val": [], "test": []}

    export_yolo(splits, tmp_path / "tomato_yolo", classes)
    export_coco(splits, tmp_path / "tomato_coco", classes)

    label = (tmp_path / "tomato_yolo" / "labels" / "train" / "one.txt").read_text(encoding="utf-8")
    assert label == "0 0.30000000 0.50000000 0.50000000 0.60000000\n"
    coco = json.loads(
        (tmp_path / "tomato_coco" / "annotations" / "train.json").read_text(encoding="utf-8")
    )
    assert coco["annotations"][0]["bbox"] == [1, 2, 10, 6]
    assert [category["id"] for category in coco["categories"]] == list(range(9))
