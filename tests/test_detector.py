from pathlib import Path

import pytest

from tomato_det.common.classes import ClassSpec
from tomato_det.models.base import Detection
from tomato_det.models.registry import (
    ModelUnavailableError,
    build_detector,
    load_model_specs,
    model_availability,
)
from tomato_det.models.yolo_detector import YoloDetector

ROOT = Path(__file__).resolve().parents[1]


def test_detection_requires_normalized_finite_geometry() -> None:
    detection = Detection(0, "healthy", 0.8, (0.1, 0.2, 0.9, 0.8))

    assert detection.class_name == "healthy"
    with pytest.raises(ValueError, match="normalized"):
        Detection(0, "healthy", 0.8, (0.2, 0.2, 0.1, 0.8))
    with pytest.raises(ValueError, match="score"):
        Detection(0, "healthy", 1.1, (0.1, 0.2, 0.9, 0.8))


def test_missing_allowlisted_checkpoint_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tomato_det.models import registry

    monkeypatch.setattr(registry, "project_root", lambda: tmp_path)
    specs = load_model_specs(ROOT / "configs" / "app.yaml")
    available, reason = model_availability(specs["yolo11s_coco"])

    assert not available
    assert reason == "Local checkpoint is missing"
    with pytest.raises(ModelUnavailableError, match="unavailable"):
        build_detector("yolo11s_coco", app_config_path=ROOT / "configs" / "app.yaml")


def test_registry_rejects_model_id_outside_allowlist() -> None:
    with pytest.raises(ModelUnavailableError, match="not allowlisted"):
        build_detector("arbitrary-path")


class _FakeTensor:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def tolist(self) -> list[object]:
        return self._values


class _FakeBoxes:
    xyxy = _FakeTensor([[10.0, 20.0, 50.0, 80.0]])
    cls = _FakeTensor([0.0])
    conf = _FakeTensor([0.75])


class _FakeResult:
    orig_shape = (100, 200)
    boxes = _FakeBoxes()


class _FakeModel:
    def predict(self, **_: object) -> list[_FakeResult]:
        return [_FakeResult()]


def test_yolo_adapter_feeds_bgr_numpy_arrays_to_ultralytics() -> None:
    import numpy as np

    captured: dict[str, object] = {}

    class _CapturingModel:
        def predict(self, **kwargs: object) -> list[_FakeResult]:
            captured["source"] = kwargs["source"]
            return [_FakeResult()]

    detector = object.__new__(YoloDetector)
    detector.model_id = "yolo11s_coco"
    detector._model = _CapturingModel()
    detector._classes_by_id = {
        0: ClassSpec(0, "Tomato Early blight leaf", "early_blight", "Úa sớm", "fungal", "#000000")
    }
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb[..., 0] = 255

    detections = detector.predict(image_rgb=rgb, conf=0.25, iou=0.5, imgsz=640, max_det=50)

    source = captured["source"]
    assert isinstance(source, np.ndarray)
    assert source.flags["C_CONTIGUOUS"]
    # RGB red must land in the last (blue in BGR naming) channel of the Ultralytics input.
    assert int(source[0, 0, 2]) == 255
    assert int(source[0, 0, 0]) == 0
    assert detections == [Detection(0, "early_blight", 0.75, (0.05, 0.2, 0.25, 0.8))]


def test_yolo_adapter_normalizes_pixel_boxes_without_loading_ultralytics() -> None:
    detector = object.__new__(YoloDetector)
    detector.model_id = "yolo11s_coco"
    detector._model = _FakeModel()
    detector._classes_by_id = {
        0: ClassSpec(0, "Tomato Early blight leaf", "early_blight", "Úa sớm", "fungal", "#000000")
    }

    detections = detector.predict(image_rgb=object(), conf=0.25, iou=0.5, imgsz=640, max_det=50)

    assert detections == [Detection(0, "early_blight", 0.75, (0.05, 0.2, 0.25, 0.8))]
