"""Explicit mapping from PlantDoc source labels to the nine tomato classes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from tomato_det.common.classes import ClassSpec, class_lookup, normalize_source_label
from tomato_det.common.types import BoxRecord, ImageRecord


@dataclass(frozen=True, slots=True)
class SubsetReport:
    input_images: int
    output_images: int
    input_boxes: int
    output_boxes: int
    unmapped_labels: dict[str, int]


def select_tomato_subset(
    records: list[ImageRecord], classes: list[ClassSpec]
) -> tuple[list[ImageRecord], SubsetReport]:
    """Keep only explicitly configured tomato labels, reporting all other source labels."""
    lookup = class_lookup(classes)
    output: list[ImageRecord] = []
    unmapped: Counter[str] = Counter()
    input_boxes = 0
    output_boxes = 0
    for record in records:
        selected: list[BoxRecord] = []
        for box in record.boxes:
            input_boxes += 1
            spec = lookup.get(normalize_source_label(box.source_label))
            if spec is None:
                unmapped[box.source_label] += 1
                continue
            selected.append(
                BoxRecord(
                    source_label=box.source_label,
                    xmin=box.xmin,
                    ymin=box.ymin,
                    xmax=box.xmax,
                    ymax=box.ymax,
                    class_id=spec.id,
                    class_slug=spec.slug,
                )
            )
        if selected:
            output_boxes += len(selected)
            output.append(record.with_boxes(selected))
    report = SubsetReport(
        input_images=len(records),
        output_images=len(output),
        input_boxes=input_boxes,
        output_boxes=output_boxes,
        unmapped_labels=dict(sorted(unmapped.items())),
    )
    return output, report
