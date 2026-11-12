"""Perceptual-hash grouping for exact and near-duplicate source images."""

from __future__ import annotations

from dataclasses import dataclass

import imagehash
from PIL import Image, UnidentifiedImageError

from tomato_det.common.types import DataIssue, ImageRecord


@dataclass(slots=True)
class DedupResult:
    groups: dict[str, str]
    hashes: dict[str, str]
    issues: list[DataIssue]


class _DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = self.parent[value]
        while root != self.parent[root]:
            root = self.parent[root]
        while value != root:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def group_near_duplicates(records: list[ImageRecord], max_distance: int) -> DedupResult:
    """Hash readable images and join every pair whose pHash distance is within the threshold."""
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    hashes: dict[str, imagehash.ImageHash] = {}
    issues: list[DataIssue] = []
    for record in records:
        try:
            with Image.open(record.path) as image:
                hashes[record.image_id] = imagehash.phash(image.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
            issues.append(
                DataIssue("HASH_FAILED", str(error), record.source_split, record.filename)
            )
    ids = sorted(hashes)
    union_find = _DisjointSet(ids)
    for position, left in enumerate(ids):
        for right in ids[position + 1 :]:
            if hashes[left] - hashes[right] <= max_distance:
                union_find.union(left, right)
    groups = {image_id: f"group-{union_find.find(image_id)}" for image_id in ids}
    return DedupResult(groups, {key: str(value) for key, value in hashes.items()}, issues)


def groups_crossing_source_splits(records: list[ImageRecord], groups: dict[str, str]) -> set[str]:
    """Return groups represented in both TRAIN and TEST source splits."""
    membership: dict[str, set[str]] = {}
    for record in records:
        group = groups.get(record.image_id)
        if group:
            membership.setdefault(group, set()).add(record.source_split)
    return {group for group, splits in membership.items() if {"train", "test"}.issubset(splits)}
