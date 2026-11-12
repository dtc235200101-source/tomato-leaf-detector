"""Pydantic API response schemas shared by REST and later WebSocket routes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorResponse(ApiModel):
    code: str
    message: str


class ClassResponse(ApiModel):
    id: int = Field(ge=0)
    slug: str
    name_vi: str
    group: str
    color: str


class ModelInfo(ApiModel):
    model_id: str
    display_name: str
    available: bool
    reason: str | None = None


class SplitSummary(ApiModel):
    name: str
    count: int = Field(ge=0)
    samples: list[str]


class DatasetSplitsResponse(ApiModel):
    splits: list[SplitSummary]


class HealthResponse(ApiModel):
    status: str
    device: str
    loaded_models: list[str]


class ImageInfo(ApiModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DetectionResponse(ApiModel):
    class_id: int = Field(ge=0)
    class_name: str
    score: float = Field(ge=0, le=1)
    box: tuple[float, float, float, float]


class InputAssessmentResponse(ApiModel):
    status: Literal["accepted", "rejected", "unverified"]
    message: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class RecommendationResponse(ApiModel):
    class_id: int = Field(ge=0)
    class_name: str
    title: str
    actions: list[str]
    caution: str


class DetectResponse(ApiModel):
    model: str
    image: ImageInfo
    inference_ms: float = Field(ge=0)
    detections: list[DetectionResponse]
    input_assessment: InputAssessmentResponse
    recommendations: list[RecommendationResponse]


class StreamConfig(ApiModel):
    type: Literal["config"]
    model: str
    conf: float = Field(ge=0, le=1)
    iou: float = Field(ge=0, le=1)
    imgsz: int = Field(gt=0)
    max_det: int = Field(ge=1)
    agnostic_nms: bool = False


class StreamResult(DetectResponse):
    type: Literal["result"] = "result"
    frame_id: int = Field(ge=0)
