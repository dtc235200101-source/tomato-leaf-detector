# Local REST API

Start the backend after installing the runtime requirements:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn tomato_det.serving.main:app --host 127.0.0.1 --port 8000
```

If `frontend/dist` exists, `GET /` returns the production React build. During development, run Vite on
`127.0.0.1:5173`; its proxy forwards `/api` and `/ws` to this server.

The server has no model download path. If the local checkpoint is missing, `/api/models` reports the
model as unavailable and `/api/detect` returns `MODEL_UNAVAILABLE`.

| Method | Route | Result |
|---|---|---|
| GET | `/api/health` | Local API health, selected device, loaded model IDs |
| GET | `/api/classes` | Nine configured tomato classes |
| GET | `/api/models` | Allowlisted model availability and reason |
| POST | `/api/detect` | Multipart field `file`; returns normalized `xyxy` detections |

`POST /api/detect` accepts optional query values `model`, `conf`, `iou`, `imgsz`, and `max_det`.
`conf` and `iou` are within `[0, 1]`; `imgsz` must be one of `320`, `416`, or `640`; `max_det` is
limited by `TDET_MAX_DETECTIONS`.

Errors use the form:

```json
{"code": "MODEL_UNAVAILABLE", "message": "yolo11s_coco is unavailable: Local checkpoint is missing"}
```

Defined codes are `INVALID_IMAGE`, `PAYLOAD_TOO_LARGE`, `MODEL_NOT_FOUND`, `MODEL_UNAVAILABLE`,
`INVALID_CONFIG`, and `INFERENCE_FAILED`.

## WebSocket inference

`WS /ws/detect` accepts one JSON config message before binary frames:

```json
{"type":"config","model":"yolo11s_coco","conf":0.25,"iou":0.5,"imgsz":640,"max_det":50,"agnostic_nms":false}
```

Each binary frame is four big-endian bytes for unsigned `frame_id`, followed by JPEG bytes. The API
returns `{ "type": "result", "frame_id": ..., "detections": [...] }`; boxes are normalized xyxy
coordinates in `[0, 1]` against the submitted JPEG. Errors retain `{ "code", "message" }`.

For every connection the server runs at most one inference and keeps at most one pending frame. A new
frame replaces a pending frame, so the most recent frame wins instead of building a latency queue.
