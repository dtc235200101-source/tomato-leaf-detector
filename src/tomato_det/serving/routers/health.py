from fastapi import APIRouter, Request

from tomato_det.common.device import detect_device
from tomato_det.serving.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    device = detect_device(settings.device)
    return HealthResponse(
        status="ok",
        device=device.resolved or "unavailable",
        loaded_models=request.app.state.model_manager.loaded_model_ids,
    )
