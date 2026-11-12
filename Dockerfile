# Build the React client first; FastAPI serves this static output in production.
FROM node:22-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# Render production image for the all-in-one FastAPI + React inference service.
# The model file is intentionally not copied from Git. Supply it during image build
# or mount/provision it separately at weights/yolo11s_coco/best.pt.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt requirements-train.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt -r requirements-train.txt \
    && pip install --no-deps -e .

COPY configs ./configs
COPY src ./src
COPY data/splits ./data/splits
COPY --from=frontend-build /frontend/dist ./frontend/dist

# A deployment image needs a real checkpoint at this path. It is intentionally
# absent here because checkpoints must not be committed to the repository.
RUN mkdir -p weights/yolo11s_coco

CMD ["sh", "-c", "uvicorn tomato_det.serving.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
