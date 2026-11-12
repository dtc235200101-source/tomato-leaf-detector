from fastapi import APIRouter, Request

from tomato_det.serving.schemas import ModelInfo

router = APIRouter(tags=["models"])


@router.get("/api/models", response_model=list[ModelInfo])
def models(request: Request) -> list[ModelInfo]:
    return [
        ModelInfo(
            model_id=status.model_id,
            display_name=status.display_name,
            available=status.available,
            reason=status.reason,
        )
        for status in request.app.state.model_manager.list_models()
    ]
