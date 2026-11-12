"""Deterministic, group-preserving train/validation split construction."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from tomato_det.common.types import ImageRecord


@dataclass(frozen=True, slots=True)
class SplitResult:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    excluded_train_ids: list[str]


def _group_records(
    records: list[ImageRecord], groups: dict[str, str]
) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[groups.get(record.image_id, f"singleton-{record.image_id}")].append(record)
    return grouped


def make_splits(
    train_records: list[ImageRecord],
    test_records: list[ImageRecord],
    groups: dict[str, str],
    val_fraction: float,
    seed: int,
) -> SplitResult:
    """Keep source TEST intact and make an approximately stratified validation group split.

    Any train image whose near-duplicate group appears in TEST is excluded from train/validation.
    The allocation is greedy by multi-label group histogram and deterministic for a given seed.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    test_groups = {groups[record.image_id] for record in test_records if record.image_id in groups}
    allowed_train = [
        record for record in train_records if groups.get(record.image_id) not in test_groups
    ]
    excluded = sorted({record.image_id for record in train_records if record not in allowed_train})
    grouped = _group_records(allowed_train, groups)
    if len(grouped) < 2:
        raise ValueError(
            "Need at least two non-leaking TRAIN groups to create train and validation splits"
        )

    target_images = max(1, round(sum(len(items) for items in grouped.values()) * val_fraction))
    totals: Counter[int] = Counter(
        box.class_id
        for records in grouped.values()
        for record in records
        for box in record.boxes
        if box.class_id is not None
    )
    target_labels = {class_id: count * val_fraction for class_id, count in totals.items()}
    tie_breaker = random.Random(seed)
    candidates = list(grouped)
    tie_breaker.shuffle(candidates)
    candidates.sort(
        key=lambda group: (
            -sum(len(record.boxes) for record in grouped[group]),
            -len(grouped[group]),
        )
    )
    chosen: set[str] = set()
    val_count = 0
    val_labels: Counter[int] = Counter()
    for group in candidates:
        records = grouped[group]
        image_count = len(records)
        histogram: Counter[int] = Counter(
            box.class_id for record in records for box in record.boxes if box.class_id is not None
        )
        without_score = sum((val_labels[key] - target_labels[key]) ** 2 for key in target_labels)
        with_labels = val_labels + histogram
        with_score = sum((with_labels[key] - target_labels[key]) ** 2 for key in target_labels)
        image_distance_without = abs(val_count - target_images)
        image_distance_with = abs(val_count + image_count - target_images)
        if (with_score, image_distance_with) < (without_score, image_distance_without):
            chosen.add(group)
            val_count += image_count
            val_labels += histogram
    if not chosen:
        chosen.add(candidates[-1])
    if len(chosen) == len(grouped):
        chosen.remove(candidates[-1])
    val_ids = sorted(record.image_id for group in chosen for record in grouped[group])
    train_ids = sorted(
        record.image_id for group in set(grouped).difference(chosen) for record in grouped[group]
    )
    test_ids = sorted(record.image_id for record in test_records)
    return SplitResult(train_ids, val_ids, test_ids, excluded)


def assert_no_group_leakage(split: SplitResult, groups: dict[str, str]) -> None:
    split_groups = {
        "train": {groups.get(image_id, f"singleton-{image_id}") for image_id in split.train_ids},
        "val": {groups.get(image_id, f"singleton-{image_id}") for image_id in split.val_ids},
        "test": {groups.get(image_id, f"singleton-{image_id}") for image_id in split.test_ids},
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_groups[left].intersection(split_groups[right])
        if overlap:
            raise ValueError(
                f"Near-duplicate group leakage between {left} and {right}: {sorted(overlap)}"
            )
