"""Loading and validating the single source of truth for tomato class labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


def normalize_source_label(value: str) -> str:
    """Normalize harmless spelling differences without guessing semantic equivalence."""
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ").strip()).casefold()


@dataclass(frozen=True, slots=True)
class ClassSpec:
    id: int
    name_source: str
    slug: str
    name_vi: str
    group: str
    color: str


def load_classes(path: Path) -> list[ClassSpec]:
    """Read class definitions and reject non-contiguous or ambiguous mappings."""
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or not isinstance(raw.get("classes"), list):
        raise ValueError(f"{path} must contain a 'classes' list")

    required = {"id", "name_source", "slug", "name_vi", "group", "color"}
    classes: list[ClassSpec] = []
    for index, item in enumerate(raw["classes"]):
        if not isinstance(item, dict):
            raise ValueError(f"classes[{index}] must be a mapping")
        missing = required.difference(item)
        if missing:
            raise ValueError(f"classes[{index}] missing fields: {', '.join(sorted(missing))}")
        classes.append(ClassSpec(**{field: item[field] for field in required}))

    classes.sort(key=lambda item: item.id)
    expected_ids = list(range(len(classes)))
    ids = [item.id for item in classes]
    if ids != expected_ids:
        raise ValueError(f"Class IDs must be continuous from 0; found {ids}")
    normalized = [normalize_source_label(item.name_source) for item in classes]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Two class definitions normalize to the same source label")
    if len({item.slug for item in classes}) != len(classes):
        raise ValueError("Class slugs must be unique")
    return classes


def class_lookup(classes: list[ClassSpec]) -> dict[str, ClassSpec]:
    return {normalize_source_label(spec.name_source): spec for spec in classes}
