from fastapi import APIRouter

from tomato_det.common.classes import load_classes
from tomato_det.common.paths import config_path
from tomato_det.serving.schemas import ClassResponse

router = APIRouter(tags=["classes"])


@router.get("/api/classes", response_model=list[ClassResponse])
def classes() -> list[ClassResponse]:
    return [
        ClassResponse(
            id=spec.id,
            slug=spec.slug,
            name_vi=spec.name_vi,
            group=spec.group,
            color=spec.color,
        )
        for spec in load_classes(config_path("classes.yaml"))
    ]
