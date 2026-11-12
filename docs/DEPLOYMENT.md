# Đưa dịch vụ từ localhost lên online (Render)

## 1. Stack thực tế của dự án

| Lớp | Công nghệ | Ghi chú |
|---|---|---|
| Backend | **Python 3.12 + FastAPI + Uvicorn** | `src/tomato_det/serving/`, REST `/api/*` + WebSocket `/ws/detect` |
| Model AI | **YOLO11s (Ultralytics) trên PyTorch CPU** | checkpoint thật ở `weights/yolo11s_coco/best.pt` (~19 MB) |
| Frontend | **React 19 + Vite 7 + TypeScript** | build tĩnh vào `frontend/dist`, được FastAPI phục vụ tại `/` |
| Dữ liệu | Không có database | chỉ đọc manifest `data/splits/*.txt` |

Vì có **backend Python + model AI nặng (torch/ultralytics) + WebSocket**, hai hướng sau không dùng được:

- **GitHub Pages**: chỉ phục vụ file tĩnh, không chạy được Python/torch/WebSocket.
- **Vercel**: serverless function giới hạn dung lượng bundle (không chứa nổi torch) và không phù hợp WebSocket stateful.

→ Dùng **Render Web Service (Docker, plan free)**: build một image duy nhất chứa cả React build, FastAPI và checkpoint.

## 2. File cấu hình đã có trong repo

| File | Vai trò |
|---|---|
| `Dockerfile` | multi-stage: build React bằng Node 22 → runtime Python 3.12 + torch CPU + checkpoint |
| `requirements-serve.txt` | dependency chỉ để chạy inference (không kéo torchmetrics/pycocotools/sklearn/pandas/matplotlib) |
| `render.yaml` | Blueprint: Docker, `plan: free`, health check `/api/health`, env `TDET_*` |
| `.dockerignore` | loại `data/`, `runs/`, `reports/`, `.venv`, `*.pt`… nhưng **giữ** `deploy/model/best.pt` |
| `deploy/model/best.pt` | bản checkpoint được đóng vào image (vì `weights/**/*.pt` bị git-ignore) |

`weights/` vẫn là đường dẫn app đọc lúc chạy; Dockerfile copy `deploy/model/best.pt` vào `weights/yolo11s_coco/best.pt`.

## 3. Deploy bằng Blueprint (khuyến nghị)

1. Đẩy code lên GitHub (nhánh `main`) — đã có trong bước commit của agent.
2. Vào <https://dashboard.render.com> → **New** → **Blueprint**.
3. Chọn repository `tomato-leaf-detector` → Render đọc `render.yaml` → **Apply**.
4. Chờ build **5–12 phút** (tải wheel torch CPU ~200 MB + `npm ci` + `vite build`).
5. Mở `https://<tên-service>.onrender.com`.

Render tự inject biến `PORT`; container chạy
`uvicorn tomato_det.serving.main:app --host 0.0.0.0 --port $PORT`.

## 4. Thông số khi tạo Web Service thủ công

**Cách A — Docker (giống `render.yaml`):**

| Trường | Giá trị |
|---|---|
| Language / Environment | `Docker` |
| Dockerfile Path | `./Dockerfile` |
| Docker Build Context | `.` |
| Build Command | *(để trống — Docker tự build)* |
| Start Command | *(để trống — dùng `CMD` trong Dockerfile)* |
| Health Check Path | `/api/health` |
| Instance Type | `Free` |

Biến môi trường: `TDET_DEVICE=cpu`, `TDET_MAX_LOADED_MODELS=1`, `TDET_LOG_LEVEL=INFO`.

**Cách B — Python runtime (không dùng Docker):** cần tự build sẵn `frontend/dist` và commit nó, hoặc chấp nhận chỉ có API không có UI.

| Trường | Giá trị |
|---|---|
| Build Command | `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && pip install -r requirements-serve.txt && pip install --no-deps -e .` |
| Start Command | `uvicorn tomato_det.serving.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/api/health` |

Start Command trên chính là lệnh chuẩn cần dán vào Render: bind `0.0.0.0` và đọc `$PORT`.

## 5. Giới hạn của plan free (đọc trước khi deploy)

- **RAM 512 MB.** `import torch` + YOLO11s có thể vượt ngưỡng này, đặc biệt khi `imgsz=640`; nếu Render báo OOM/restart liên tục thì nâng lên **Starter (2 GB, ~7 USD/tháng)** hoặc hạ `imgsz` xuống `416`/`320` trên giao diện.
- **Spin down sau ~15 phút không có request** → request đầu tiên chậm khoảng 30–60 giây (cold start, phải nạp lại model).
- **Không có persistent disk** trên free plan → checkpoint phải nằm trong image (đó là lý do dùng `deploy/model/best.pt`).
- **Image lớn** (~1,5–2 GB) nên mỗi lần deploy mất vài phút.
- Một số tài khoản Render yêu cầu xác minh thẻ trước khi tạo service free.
- Ảnh upload và frame webcam **rời khỏi máy bạn** và được xử lý trên server Render — khác với thiết kế local ban đầu.

## 6. Kiểm tra sau khi deploy

1. `GET https://<service>/api/health` → `status: "ok"`.
2. `GET https://<service>/api/models` → `available: true` (nếu `false`, checkpoint chưa vào image).
3. Mở `https://<service>/` → tải ảnh lá cà chua → có bbox + nhãn + `inference_ms`.
4. Webcam: HTTPS bắt buộc cho `getUserMedia`; socket phải là `wss://<service>/ws/detect`.
5. Trang "Dữ liệu dataset" chỉ hiện số lượng split; ảnh mẫu không được đóng vào image nên sẽ trả 404 (hành vi có chủ đích).

## 7. Cập nhật model sau này

```powershell
# sau khi train xong
Copy-Item runs/<exp_id>/weights/best.pt deploy/model/best.pt -Force
git add deploy/model/best.pt
git commit -m "Update deployed checkpoint"
git push
```

Sau đó bấm **Manual Deploy** trên Render (Blueprint mặc định không auto-deploy khi chưa bật).

## 8. Ghi chú license

Checkpoint được phân phối kèm image kế thừa **AGPL-3.0** từ Ultralytics. Không đưa dataset PlantDoc lên repo/image công khai (CC BY 4.0, ảnh gốc thu thập từ internet).
