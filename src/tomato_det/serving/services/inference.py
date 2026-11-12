"""Synchronous inference service designed to be executed in a FastAPI thread pool."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from tomato_det.models.base import Detection
from tomato_det.models.registry import ModelUnavailableError
from tomato_det.serving.services.model_manager import ModelManager, ModelNotFoundError
from tomato_det.serving.services.tomato_gate import (
    TomatoAssessment,
    TomatoAssessmentStatus,
    TomatoGate,
)
from tomato_det.serving.utils.image_io import ImageValidationError, decode_image_rgb


@dataclass(frozen=True, slots=True)
class InferenceResult:
    model_id: str
    width: int
    height: int
    inference_ms: float
    detections: list[Detection]
    input_assessment: TomatoAssessment


class InferenceServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def run_inference(
    manager: ModelManager,
    tomato_gate: TomatoGate,
    *,
    payload: bytes,
    model_id: str,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    max_bytes: int,
    max_pixels: int,
    agnostic_nms: bool = False,
) -> InferenceResult:
    try:
        image_rgb, width, height = decode_image_rgb(
            payload, max_bytes=max_bytes, max_pixels=max_pixels
        )
    except ImageValidationError as error:
        raise InferenceServiceError("INVALID_IMAGE", str(error)) from error
    try:
        start = perf_counter()
        assessment = tomato_gate.assess(image_rgb)
        if assessment.status is TomatoAssessmentStatus.REJECTED:
            elapsed_ms = (perf_counter() - start) * 1000
            return InferenceResult(model_id, width, height, elapsed_ms, [], assessment)
        detections = manager.predict(
            model_id,
            image_rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            agnostic_nms=agnostic_nms,
        )
        elapsed_ms = (perf_counter() - start) * 1000
    except ModelNotFoundError as error:
        raise InferenceServiceError("MODEL_NOT_FOUND", str(error)) from error
    except ModelUnavailableError as error:
        raise InferenceServiceError("MODEL_UNAVAILABLE", str(error)) from error
    except Exception as error:
        raise InferenceServiceError("INFERENCE_FAILED", "Inference failed") from error
    return InferenceResult(model_id, width, height, elapsed_ms, detections, assessment)
