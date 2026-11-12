from pathlib import Path

import pytest

from tomato_det.common.classes import load_classes, normalize_source_label

ROOT = Path(__file__).resolve().parents[1]


def test_project_class_map_has_exactly_nine_contiguous_classes() -> None:
    classes = load_classes(ROOT / "configs" / "classes.yaml")

    assert [item.id for item in classes] == list(range(9))
    assert [item.slug for item in classes] == [
        "early_blight",
        "septoria_leaf_spot",
        "bacterial_spot",
        "late_blight",
        "mosaic_virus",
        "yellow_virus",
        "leaf_mold",
        "spider_mites",
        "healthy",
    ]


def test_label_normalization_does_not_change_words() -> None:
    assert normalize_source_label(" Tomato_leaf   late-blight ") == "tomato leaf late blight"


def test_rejects_non_contiguous_class_ids(tmp_path: Path) -> None:
    path = tmp_path / "classes.yaml"
    path.write_text(
        "classes:\n"
        "  - {id: 0, name_source: a, slug: a, name_vi: a, group: a, color: '#000'}\n"
        "  - {id: 2, name_source: b, slug: b, name_vi: b, group: b, color: '#111'}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="continuous"):
        load_classes(path)
