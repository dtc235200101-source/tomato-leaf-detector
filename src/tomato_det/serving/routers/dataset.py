"""Dataset explorer endpoints: split stats, image manifests and image files.

Reads the generated split manifests (data/splits/*.txt) and the processed
YOLO layout (data/processed/tomato_yolo/images/<split>) so the web UI can
browse train/val/test images without touching the filesystem directly.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from tomato_det.common.paths import project_root
from tomato_det.serving.schemas import DatasetSplitsResponse, SplitSummary

router = APIRouter(tags=["dataset"])

PROCESSED_IMAGES = project_root() / "data" / "processed" / "tomato_yolo" / "images"
SPLITS_DIR = project_root() / "data" / "splits"
SPLIT_NAMES = ("train", "val", "test")


def _read_split(name: str) -> list[str]:
    manifest = SPLITS_DIR / f"{name}.txt"
    if not manifest.is_file():
        return []
    image_ids: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        image_id = line.strip()
        if image_id:
            image_ids.append(image_id)
    return image_ids


def _image_file(split: str, name: str) -> Path:
    safe = Path(name).name
    candidate = PROCESSED_IMAGES / split / safe
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return candidate


@router.get("/api/dataset/splits", response_model=DatasetSplitsResponse)
def splits() -> DatasetSplitsResponse:
    summary = [
        SplitSummary(
            name=name,
            count=len(_read_split(name)),
            samples=_read_split(name)[:24],
        )
        for name in SPLIT_NAMES
    ]
    return DatasetSplitsResponse(splits=summary)


@router.get("/api/dataset/image/{split}")
def image(split: str, name: str) -> FileResponse:
    if split not in SPLIT_NAMES:
        raise HTTPException(status_code=404, detail="unknown split")
    path = _image_file(split, name)
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return FileResponse(path, media_type=media_type)
