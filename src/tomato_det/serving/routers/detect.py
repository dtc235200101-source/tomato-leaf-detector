from __future__ import annotations

import logging

from fastapi import APIRouter, File, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from tomato_det.serving.schemas import (
    DetectionResponse,
    DetectResponse,
    ImageInfo,
    InputAssessmentResponse,
    RecommendationResponse,
)
from tomato_det.serving.services.inference import InferenceServiceError, run_inference
from tomato_det.serving.services.recommendations import recommendations_for

logger = logging.getLogger(__name__)
router = APIRouter(tags=["detect"])


@router.post("/api/detect", response_model=DetectResponse)
async def detect(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008 - FastAPI declares multipart input this way.
    model: str | None = Query(default=None),
    conf: float = Query(default=0.25, ge=0, le=1),
    iou: float = Query(default=0.5, ge=0, le=1),
    imgsz: int = Query(default=640),
    max_det: int = Query(default=50, ge=1),
) -> DetectResponse:
    settings = request.app.state.settings
    if imgsz not in request.app.state.allowed_image_sizes:
        raise InferenceServiceError("INVALID_CONFIG", "imgsz is not supported")
    if max_det > settings.max_detections:
        raise InferenceServiceError("INVALID_CONFIG", "max_det exceeds the configured limit")
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = 0
        if declared_size > settings.max_upload_bytes:
            raise InferenceServiceError(
                "PAYLOAD_TOO_LARGE", "Upload exceeds the configured byte limit"
            )
    payload = await file.read(settings.max_upload_bytes + 1)
    if len(payload) > settings.max_upload_bytes:
        raise InferenceServiceError("PAYLOAD_TOO_LARGE", "Upload exceeds the configured byte limit")
    try:
        result = await run_in_threadpool(
            run_inference,
            request.app.state.model_manager,
            request.app.state.tomato_gate,
            payload=payload,
            model_id=model or settings.default_model,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            max_bytes=settings.max_upload_bytes,
            max_pixels=settings.max_image_pixels,
        )
    except InferenceServiceError:
        raise
    except Exception as error:
        logger.exception("Unexpected inference route failure")
        raise InferenceServiceError("INFERENCE_FAILED", "Inference failed") from error
    return DetectResponse(
        model=result.model_id,
        image=ImageInfo(width=result.width, height=result.height),
        inference_ms=result.inference_ms,
        detections=[
            DetectionResponse(
                class_id=item.class_id,
                class_name=item.class_name,
                score=item.score,
                box=item.box,
            )
            for item in result.detections
        ],
        input_assessment=InputAssessmentResponse(
            status=result.input_assessment.status,
            message=result.input_assessment.message,
            confidence=result.input_assessment.confidence,
        ),
        recommendations=[
            RecommendationResponse(
                class_id=item.class_id,
                class_name=item.class_name,
                title=item.title,
                actions=list(item.actions),
                caution=item.caution,
            )
            for item in recommendations_for(result.detections)
        ],
    )
