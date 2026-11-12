from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tomato_det.models.base import Detection
from tomato_det.serving.main import create_app
from tomato_det.serving.routers.stream import FrameJob, LatestFrameQueue
from tomato_det.serving.schemas import StreamConfig
from tomato_det.serving.services.model_manager import ModelNotFoundError
from tomato_det.serving.settings import Settings


class StreamFakeManager:
    loaded_model_ids: list[str] = []

    def is_allowlisted(self, model_id: str) -> bool:
        return model_id == "yolo11s_coco"

    def predict(self, model_id: str, image_rgb: object, **_: object) -> list[Detection]:
        if not self.is_allowlisted(model_id):
            raise ModelNotFoundError(f"Model is not allowlisted: {model_id}")
        return [Detection(3, "late_blight", 0.8, (0.1, 0.2, 0.7, 0.9))]


def _jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (20, 10), "green").save(buffer, format="JPEG")
    return buffer.getvalue()


def _frame(frame_id: int, jpeg: bytes) -> bytes:
    return frame_id.to_bytes(4, byteorder="big") + jpeg


def _config(**overrides: object) -> str:
    payload: dict[str, object] = {
        "type": "config",
        "model": "yolo11s_coco",
        "conf": 0.25,
        "iou": 0.5,
        "imgsz": 640,
        "max_det": 50,
        "agnostic_nms": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def client() -> TestClient:
    settings = Settings(max_frame_bytes=10_000, max_image_pixels=1_000, max_detections=50)
    with TestClient(
        create_app(settings=settings, model_manager=StreamFakeManager())
    ) as test_client:
        yield test_client


def test_websocket_config_and_frame_id_round_trip(client: TestClient) -> None:
    with client.websocket_connect("/ws/detect") as socket:
        socket.send_text(_config())
        socket.send_bytes(_frame(513, _jpeg()))
        response = socket.receive_json()

    assert response["type"] == "result"
    assert response["frame_id"] == 513
    assert response["model"] == "yolo11s_coco"
    assert response["image"] == {"width": 20, "height": 10}
    assert response["detections"][0]["box"] == [0.1, 0.2, 0.7, 0.9]
    assert response["inference_ms"] >= 0


def test_websocket_rejects_invalid_config_short_frame_and_bad_jpeg(client: TestClient) -> None:
    with client.websocket_connect("/ws/detect") as socket:
        socket.send_text(_config(conf=1.5))
        invalid_config = socket.receive_json()
        socket.send_text(_config())
        socket.send_bytes(b"bad")
        short_frame = socket.receive_json()
        socket.send_bytes(_frame(1, b"not-a-jpeg"))
        invalid_image = socket.receive_json()

    assert invalid_config["code"] == "INVALID_CONFIG"
    assert short_frame["code"] == "INVALID_FRAME"
    assert invalid_image["code"] == "INVALID_IMAGE"


def test_websocket_rejects_payload_too_large() -> None:
    settings = Settings(max_frame_bytes=10, max_image_pixels=1_000, max_detections=50)
    with TestClient(create_app(settings=settings, model_manager=StreamFakeManager())) as client:
        with client.websocket_connect("/ws/detect") as socket:
            socket.send_text(_config())
            socket.send_bytes(_frame(9, b"x" * 11))
            response = socket.receive_json()

    assert response["code"] == "PAYLOAD_TOO_LARGE"


def test_websocket_reports_model_unavailable_with_real_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tomato_det.models import registry

    monkeypatch.setattr(registry, "project_root", lambda: tmp_path)
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/detect") as socket:
            socket.send_text(_config())
            socket.send_bytes(_frame(9, _jpeg()))
            response = socket.receive_json()

    assert response["code"] == "MODEL_UNAVAILABLE"


def test_latest_frame_queue_replaces_pending_frame() -> None:
    async def scenario() -> tuple[int, int | None]:
        queue = LatestFrameQueue()
        config = StreamConfig.model_validate_json(_config())
        first = FrameJob(1, b"first", config)
        second = FrameJob(2, b"second", config)
        assert queue.offer(first) is None
        discarded = queue.offer(second)
        selected = await queue.get()
        return selected.frame_id, discarded.frame_id if discarded else None

    selected_id, discarded_id = asyncio.run(scenario())

    assert selected_id == 2
    assert discarded_id == 1
