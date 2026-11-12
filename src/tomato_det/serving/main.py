"""FastAPI app factory for local REST inference."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from tomato_det.common.config import load_yaml
from tomato_det.common.paths import config_path, project_root
from tomato_det.serving.routers import classes, dataset, detect, health, models, stream
from tomato_det.serving.schemas import ErrorResponse
from tomato_det.serving.services.inference import InferenceServiceError
from tomato_det.serving.services.model_manager import ModelManager
from tomato_det.serving.services.tomato_gate import TomatoGate, build_tomato_gate
from tomato_det.serving.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
    tomato_gate: TomatoGate | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    app_config = load_yaml(config_path("app.yaml"))
    allowed_sizes = app_config.get("allowed_image_sizes", [])
    if not isinstance(allowed_sizes, list) or not all(
        isinstance(size, int) for size in allowed_sizes
    ):
        raise ValueError("app.yaml allowed_image_sizes must be a list of integers")
    app = FastAPI(title="Tomato Leaf Detector API", version="0.1.0")
    app.state.settings = settings
    app.state.allowed_image_sizes = frozenset(allowed_sizes)
    app.state.model_manager = model_manager or ModelManager(
        max_loaded_models=settings.max_loaded_models
    )
    app.state.tomato_gate = tomato_gate or build_tomato_gate(app_config, project_root())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(InferenceServiceError)
    async def inference_error_handler(_: Request, error: InferenceServiceError) -> JSONResponse:
        status = {
            "INVALID_IMAGE": 422,
            "PAYLOAD_TOO_LARGE": 413,
            "MODEL_NOT_FOUND": 404,
            "MODEL_UNAVAILABLE": 503,
            "INVALID_CONFIG": 422,
            "INFERENCE_FAILED": 500,
        }.get(error.code, 500)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(code=error.code, message=error.message).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(code="INVALID_CONFIG", message=str(error.errors())).model_dump(),
        )

    app.include_router(health.router)
    app.include_router(classes.router)
    app.include_router(dataset.router)
    app.include_router(models.router)
    app.include_router(detect.router)
    app.include_router(stream.router)
    frontend_dist = project_root() / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    return app


app = create_app()
