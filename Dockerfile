# Build the React client first; FastAPI serves this static output in production.
FROM node:22-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# Render production image for the all-in-one FastAPI + React inference service.
# The trained checkpoint ships from deploy/model/best.pt because weights/*.pt is
# git-ignored; it is copied to the path app.yaml expects at runtime.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    TDET_DEVICE=cpu \
    TDET_MAX_LOADED_MODELS=1

WORKDIR /app

# libgl1/libglib2.0-0 are required by the OpenCV wheel that ultralytics pulls in.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt requirements-serve.txt ./
COPY src ./src
COPY configs ./configs
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements-serve.txt \
    && pip install --no-deps -e .

COPY data/splits ./data/splits
COPY --from=frontend-build /frontend/dist ./frontend/dist
COPY deploy/model/best.pt ./weights/yolo11s_coco/best.pt

# Render injects PORT; fall back to 8000 when running the image locally.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn tomato_det.serving.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
