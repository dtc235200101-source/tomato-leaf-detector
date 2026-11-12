"""CSV parser for the documented PlantDoc object-detection annotation schema."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from tomato_det.common.types import BoxRecord, DataIssue, ImageRecord

REQUIRED_COLUMNS = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}


@dataclass(slots=True)
class ParseResult:
    records: list[ImageRecord] = field(default_factory=list)
    issues: list[DataIssue] = field(default_factory=list)
    source_labels: Counter[str] = field(default_factory=Counter)


def _read_number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "").strip()
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _resolve_image(image_dir: Path, filename: str) -> Path:
    candidate = Path(filename)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("filename must be a relative path inside the image directory")
    direct = image_dir / candidate
    if direct.is_file():
        return direct
    basename = image_dir / candidate.name
    if basename.is_file():
        return basename
    raise FileNotFoundError(filename)


def parse_plantdoc_csv(csv_path: Path, image_dir: Path, source_split: str) -> ParseResult:
    """Read one PlantDoc CSV and retain only images and boxes that pass validation.

    Malformed inputs are returned as structured issues, never silently clipped or repaired.
    The actual image dimensions are used when the image can be decoded.
    """
    result = ParseResult()
    pending: dict[str, ImageRecord] = {}
    try:
        stream = csv_path.open("r", encoding="utf-8-sig", newline="")
    except FileNotFoundError:
        result.issues.append(DataIssue("MISSING_CSV", f"CSV not found: {csv_path}", source_split))
        return result

    with stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS.difference(fields)
        if missing:
            result.issues.append(
                DataIssue(
                    "INVALID_CSV_HEADER",
                    f"Missing columns: {', '.join(sorted(missing))}",
                    source_split,
                )
            )
            return result

        for row_number, row in enumerate(reader, start=2):
            filename = row.get("filename", "").strip()
            label = row.get("class", "").strip()
            if not filename or not label:
                result.issues.append(
                    DataIssue(
                        "INVALID_ROW",
                        "filename and class are required",
                        source_split,
                        filename,
                        row_number,
                    )
                )
                continue
            result.source_labels[label] += 1
            try:
                declared_width = int(_read_number(row, "width"))
                declared_height = int(_read_number(row, "height"))
                values = [_read_number(row, name) for name in ("xmin", "ymin", "xmax", "ymax")]
            except (TypeError, ValueError) as error:
                result.issues.append(
                    DataIssue("INVALID_ROW", str(error), source_split, filename, row_number)
                )
                continue
            if declared_width <= 0 or declared_height <= 0:
                result.issues.append(
                    DataIssue(
                        "INVALID_IMAGE_SIZE",
                        "CSV width and height must be positive",
                        source_split,
                        filename,
                        row_number,
                    )
                )
                continue
            try:
                image_path = _resolve_image(image_dir, filename)
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    actual_width, actual_height = image.size
            except FileNotFoundError:
                result.issues.append(
                    DataIssue(
                        "MISSING_IMAGE",
                        "Referenced image is missing",
                        source_split,
                        filename,
                        row_number,
                    )
                )
                continue
            except (UnidentifiedImageError, OSError) as error:
                result.issues.append(
                    DataIssue("CORRUPT_IMAGE", str(error), source_split, filename, row_number)
                )
                continue
            if (declared_width, declared_height) != (actual_width, actual_height):
                dimension_message = (
                    f"CSV={declared_width}x{declared_height}, actual={actual_width}x{actual_height}"
                )
                result.issues.append(
                    DataIssue(
                        "IMAGE_DIMENSION_MISMATCH",
                        dimension_message,
                        source_split,
                        filename,
                        row_number,
                    )
                )
            box = BoxRecord(label, *values)
            if not box.is_valid_for(actual_width, actual_height):
                result.issues.append(
                    DataIssue(
                        "INVALID_BOX",
                        "Box must have finite xyxy coordinates inside decoded image bounds "
                        "and positive area",
                        source_split,
                        filename,
                        row_number,
                    )
                )
                continue
            normalized_filename = filename.replace("\\", "/")
            image_id = f"{source_split}/{normalized_filename}"
            record = pending.get(image_id)
            if record is None:
                record = ImageRecord(
                    image_id=image_id,
                    filename=filename,
                    path=image_path,
                    width=actual_width,
                    height=actual_height,
                    source_split=source_split,
                )
                pending[image_id] = record
            record.boxes.append(box)
    result.records = sorted(pending.values(), key=lambda item: item.image_id)
    return result
