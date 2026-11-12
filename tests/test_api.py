from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tomato_det.models.base import Detection
from tomato_det.serving.main import create_app
from tomato_det.serving.services.model_manager import ModelNotFoundError, ModelStatus
from tomato_det.serving.services.tomato_gate import TomatoAssessment, TomatoAssessmentStatus
from tomato_det.serving.settings import Settings


class FakeModelManager:
    loaded_model_ids: list[str] = []

    def list_models(self) -> list[ModelStatus]:
        return [
            ModelStatus(
                model_id="yolo11s_coco",
                display_name="YOLO11s fine-tuned from COCO",
                available=True,
                reason=None,
            )
        ]

    def predict(self, model_id: str, image_rgb: object, **_: object) -> list[Detection]:
        if model_id != "yolo11s_coco":
            raise ModelNotFoundError(f"Model is not allowlisted: {model_id}")
        return [Detection(0, "early_blight", 0.9, (0.1, 0.2, 0.8, 0.9))]


class RejectingTomatoGate:
    def assess(self, image_rgb: object) -> TomatoAssessment:
        del image_rgb
        return TomatoAssessment(
            TomatoAssessmentStatus.REJECTED,
            "Không xác nhận đây là lá cà chua. Vui lòng nhập ảnh lá cà chua rõ nét.",
            0.94,
        )


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (20, 10), "green").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def client() -> TestClient:
    settings = Settings(max_upload_bytes=1_000, max_image_pixels=1_000, max_detections=50)
    with TestClient(create_app(settings=settings, model_manager=FakeModelManager())) as test_client:
        yield test_client


def test_health_classes_and_models(client: TestClient) -> None:
    health = client.get("/api/health")
    classes = client.get("/api/classes")
    models = client.get("/api/models")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert len(classes.json()) == 9
    assert models.json()[0]["available"] is True


def test_detect_returns_normalized_contract_from_explicit_fake_manager(client: TestClient) -> None:
    response = client.post(
        "/api/detect",
        files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "yolo11s_coco"
    assert body["image"] == {"width": 20, "height": 10}
    assert body["detections"][0]["box"] == [0.1, 0.2, 0.8, 0.9]
    assert body["input_assessment"]["status"] == "unverified"
    assert body["recommendations"][0]["class_name"] == "early_blight"


def test_detect_rejects_non_tomato_before_disease_detector() -> None:
    app = create_app(
        settings=Settings(max_upload_bytes=1_000, max_image_pixels=1_000, max_detections=50),
        model_manager=FakeModelManager(),
        tomato_gate=RejectingTomatoGate(),
    )
    with TestClient(app) as local_client:
        response = local_client.post(
            "/api/detect", files={"file": ("other-leaf.jpg", _image_bytes(), "image/jpeg")}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["input_assessment"]["status"] == "rejected"
    assert body["detections"] == []
    assert body["recommendations"] == []


def test_detect_reports_invalid_image_unknown_model_and_invalid_query(client: TestClient) -> None:
    invalid_image = client.post(
        "/api/detect", files={"file": ("broken.jpg", b"not-an-image", "image/jpeg")}
    )
    unknown_model = client.post(
        "/api/detect?model=outside-allowlist",
        files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
    )
    invalid_config = client.post(
        "/api/detect?conf=1.1", files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")}
    )

    assert invalid_image.status_code == 422
    assert invalid_image.json()["code"] == "INVALID_IMAGE"
    assert unknown_model.status_code == 404
    assert unknown_model.json()["code"] == "MODEL_NOT_FOUND"
    assert invalid_config.status_code == 422
    assert invalid_config.json()["code"] == "INVALID_CONFIG"


def test_detect_rejects_large_payload(client: TestClient) -> None:
    response = client.post("/api/detect", files={"file": ("large.jpg", b"x" * 1_001, "image/jpeg")})

    assert response.status_code == 413
    assert response.json()["code"] == "PAYLOAD_TOO_LARGE"


def test_real_manager_reports_missing_checkpoint_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tomato_det.models import registry

    monkeypatch.setattr(registry, "project_root", lambda: tmp_path)
    with TestClient(create_app()) as real_client:
        models = real_client.get("/api/models")
        detect = real_client.post(
            "/api/detect", files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")}
        )

    assert models.status_code == 200
    assert models.json()[0]["available"] is False
    assert models.json()[0]["reason"] == "Local checkpoint is missing"
    assert detect.status_code == 503
    assert detect.json()["code"] == "MODEL_UNAVAILABLE"


def test_production_app_serves_built_frontend_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tomato_det.serving import main

    frontend_dist = tmp_path / "frontend" / "dist"
    frontend_dist.mkdir(parents=True)
    (frontend_dist / "index.html").write_text("<title>Tomato UI</title>", encoding="utf-8")
    monkeypatch.setattr(main, "project_root", lambda: tmp_path)

    app = main.create_app(settings=Settings(), model_manager=FakeModelManager())
    with TestClient(app) as local_client:
        root = local_client.get("/")
        models = local_client.get("/api/models")

    assert root.status_code == 200
    assert "Tomato UI" in root.text
    assert models.status_code == 200
