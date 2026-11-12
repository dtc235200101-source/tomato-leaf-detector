"""WebSocket JPEG-frame inference with a one-item latest-frame queue per client."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from tomato_det.serving.schemas import (
    DetectionResponse,
    ErrorResponse,
    ImageInfo,
    InputAssessmentResponse,
    RecommendationResponse,
    StreamConfig,
    StreamResult,
)
from tomato_det.serving.services.inference import InferenceServiceError, run_inference
from tomato_det.serving.services.recommendations import recommendations_for

logger = logging.getLogger(__name__)
router = APIRouter(tags=["stream"])


@dataclass(frozen=True, slots=True)
class FrameJob:
    frame_id: int
    jpeg: bytes
    config: StreamConfig


class LatestFrameQueue:
    """Queue at most one pending frame; offering a new frame replaces the stale pending one."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[FrameJob] = asyncio.Queue(maxsize=1)

    def offer(self, frame: FrameJob) -> FrameJob | None:
        discarded: FrameJob | None = None
        if self._queue.full():
            discarded = self._queue.get_nowait()
        self._queue.put_nowait(frame)
        return discarded

    async def get(self) -> FrameJob:
        return await self._queue.get()


async def _send_json(websocket: WebSocket, lock: asyncio.Lock, payload: dict[str, object]) -> None:
    async with lock:
        await websocket.send_json(payload)


async def _send_error(websocket: WebSocket, lock: asyncio.Lock, code: str, message: str) -> None:
    await _send_json(websocket, lock, ErrorResponse(code=code, message=message).model_dump())


def _validate_config(websocket: WebSocket, config: StreamConfig) -> str | None:
    if config.imgsz not in websocket.app.state.allowed_image_sizes:
        return "imgsz is not supported"
    if config.max_det > websocket.app.state.settings.max_detections:
        return "max_det exceeds the configured limit"
    if not websocket.app.state.model_manager.is_allowlisted(config.model):
        return "Model is not allowlisted"
    return None


async def _inference_worker(
    websocket: WebSocket, queue: LatestFrameQueue, send_lock: asyncio.Lock
) -> None:
    settings = websocket.app.state.settings
    manager = websocket.app.state.model_manager
    tomato_gate = websocket.app.state.tomato_gate
    while True:
        frame = await queue.get()
        try:
            result = await run_in_threadpool(
                run_inference,
                manager,
                tomato_gate,
                payload=frame.jpeg,
                model_id=frame.config.model,
                conf=frame.config.conf,
                iou=frame.config.iou,
                imgsz=frame.config.imgsz,
                max_det=frame.config.max_det,
                agnostic_nms=frame.config.agnostic_nms,
                max_bytes=settings.max_frame_bytes,
                max_pixels=settings.max_image_pixels,
            )
            response = StreamResult(
                frame_id=frame.frame_id,
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
            await _send_json(websocket, send_lock, response.model_dump())
        except InferenceServiceError as error:
            await _send_error(websocket, send_lock, error.code, error.message)
        except WebSocketDisconnect:
            return
        except Exception:
            logger.exception("Unexpected WebSocket inference failure")
            await _send_error(websocket, send_lock, "INFERENCE_FAILED", "Inference failed")


@router.websocket("/ws/detect")
async def detect_stream(websocket: WebSocket) -> None:
    """Receive text config then binary `[uint32 big-endian frame_id][JPEG]` frames."""
    await websocket.accept()
    send_lock = asyncio.Lock()
    queue = LatestFrameQueue()
    worker = asyncio.create_task(_inference_worker(websocket, queue, send_lock))
    current_config: StreamConfig | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            text = message.get("text")
            binary = message.get("bytes")
            if text is not None:
                try:
                    candidate = StreamConfig.model_validate_json(text)
                except ValidationError:
                    await _send_error(
                        websocket, send_lock, "INVALID_CONFIG", "Invalid stream config"
                    )
                    continue
                error = _validate_config(websocket, candidate)
                if error:
                    await _send_error(websocket, send_lock, "INVALID_CONFIG", error)
                    continue
                current_config = candidate
                continue
            if binary is None:
                await _send_error(
                    websocket, send_lock, "INVALID_FRAME", "Expected text config or binary frame"
                )
                continue
            if current_config is None:
                await _send_error(
                    websocket, send_lock, "INVALID_CONFIG", "Send a valid config before frames"
                )
                continue
            if len(binary) < 4:
                await _send_error(
                    websocket, send_lock, "INVALID_FRAME", "Frame must include a 4-byte frame_id"
                )
                continue
            frame_id = int.from_bytes(binary[:4], byteorder="big", signed=False)
            jpeg = binary[4:]
            if len(jpeg) > websocket.app.state.settings.max_frame_bytes:
                await _send_error(
                    websocket,
                    send_lock,
                    "PAYLOAD_TOO_LARGE",
                    "Frame exceeds the configured byte limit",
                )
                continue
            queue.offer(FrameJob(frame_id=frame_id, jpeg=jpeg, config=current_config))
    except WebSocketDisconnect:
        return
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
