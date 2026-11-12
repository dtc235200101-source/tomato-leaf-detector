"""Small visual audit artifact for manually checking exported annotations."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from tomato_det.common.classes import ClassSpec
from tomato_det.common.types import ImageRecord


def write_audit_preview(
    records: list[ImageRecord], classes: list[ClassSpec], output_path: Path, limit: int = 12
) -> None:
    """Create a contact sheet of source images with mapped boxes; never modifies source images."""
    if limit < 1:
        raise ValueError("limit must be positive")
    colors = {item.id: item.color for item in classes}
    thumbnails: list[Image.Image] = []
    for record in records[:limit]:
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        drawing = ImageDraw.Draw(image)
        for box in record.boxes:
            drawing.rectangle(
                (box.xmin, box.ymin, box.xmax, box.ymax), outline=colors[box.class_id], width=3
            )
            drawing.text(
                (box.xmin, max(0, box.ymin - 14)),
                box.class_slug or "unmapped",
                fill=colors[box.class_id],
            )
        image.thumbnail((320, 240))
        thumbnails.append(image)
    if not thumbnails:
        return
    width = 640
    cell_width, cell_height = 320, 240
    rows = (len(thumbnails) + 1) // 2
    canvas = Image.new("RGB", (width, rows * cell_height), "white")
    for index, image in enumerate(thumbnails):
        x, y = (index % 2) * cell_width, (index // 2) * cell_height
        canvas.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
