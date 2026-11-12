# PROMPT.md — Bộ prompt giao việc cho AI Agent

Tài liệu này dành cho Cursor, Claude Code, GitHub Copilot Agent hoặc AI coding agent tương tự. Chạy **từng phase**, không đưa toàn bộ dự án vào một lần yêu cầu sinh code.

## 0. Prompt điều phối chính

Sao chép prompt dưới đây vào AI Agent tại thư mục gốc repo:

```text
Bạn là Senior ML Engineer + Full-stack Engineer phụ trách dự án Tomato Leaf Detector.

BẮT BUỘC đọc trước khi sửa code:
- README.md
- PLAN.md
- AGENT.md
- PROMPT.md (phase được yêu cầu)

Mục tiêu: xây dựng ứng dụng web AI chạy local, nhận diện 9 lớp lá cà chua từ PlantDoc Object Detection Dataset, hỗ trợ upload ảnh và live webcam preview có vẽ bounding box.

Quyết định đã chốt:
- Model chính serving: YOLO11s fine-tune từ COCO, imgsz 640.
- Baseline nghiên cứu: YOLOv8s và Faster R-CNN ResNet50-FPN v2; không bắt buộc hoàn thành trước MVP.
- Backend: FastAPI + Uvicorn; frontend: React + TypeScript + Vite.
- Webcam: WebSocket với frame_id + JPEG binary; trả normalized bbox, class, confidence, inference_ms.
- Chạy local, mặc định bind 127.0.0.1; không gửi ảnh lên cloud.
- configs/classes.yaml là nguồn sự thật cho 9 class.

Quy tắc:
1. Tuân thủ AGENT.md và tiêu chí nghiệm thu trong PLAN.md.
2. Trước khi sửa, kiểm tra repo hiện có và tóm tắt ngắn các file dự định sửa.
3. Thực hiện đúng phase được chỉ định, không tạo code giả hoặc placeholder tuyên bố đã chạy.
4. Viết test cho logic mới; chạy test/lint/build liên quan.
5. Không commit dataset PlantDoc raw, weights .pt, .venv, node_modules, secrets hoặc file lớn.
6. Khi thiếu dataset, checkpoint, GPU hoặc dependency, tiếp tục phần có thể làm được; nêu rõ phần chưa kiểm chứng.
7. Không bịa kết quả mAP/FPS. Các số trong ví dụ chỉ dùng minh họa giao thức.
8. Không tải model/dữ liệu từ Internet tự động khi web app khởi động. Checkpoint phải nằm ở vị trí local được cấu hình.
9. Không đổi kiến trúc/model chính nếu chưa trình bày lý do và cập nhật docs.

Đầu ra cuối phase:
- Tóm tắt file tạo/sửa.
- Test/lint/build đã chạy, cùng kết quả thật.
- Cách chạy/kiểm tra.
- Blocker và phần chưa làm.
- Cập nhật docs nếu API, CLI, cấu hình hoặc hành vi thay đổi.

Bắt đầu bằng phase được giao tiếp theo sau. Không tự triển khai phase tiếp theo nếu chưa được yêu cầu.
```

## 1. Phase 0 — Khảo sát repository và dựng khung

```text
Thực hiện PHASE 0 theo PLAN.md.

Trước tiên kiểm tra cây thư mục, Python/Node versions, pyproject.toml, requirements và frontend package hiện có. Không ghi đè file hiện hữu một cách mù quáng.

Hãy tạo/tối ưu cấu trúc tối thiểu để dự án có:
- pyproject.toml với package src layout, CLI entry point `tdet`, cấu hình Ruff và pytest;
- requirements.txt, requirements-train.txt, requirements-dev.txt có phân nhóm rõ;
- configs/classes.yaml với đúng 9 lớp và ID 0..8;
- configs/data.yaml, eval.yaml, app.yaml cùng các config experiment khởi đầu;
- .env.example, .gitignore, data/README.md, weights/README.md;
- CLI `tdet --help` và các command group/placeholder có thông báo “chưa triển khai” trung thực (không giả vờ xử lý xong data/train).

Không cài PyTorch bằng một wheel ngẫu nhiên; README phải hướng dẫn cài torch/torchvision theo CPU hoặc CUDA từ pytorch.org trước. Không thêm dependencies trùng hoặc xung đột OpenCV.

Viết test cho class mapping và load YAML. Chạy ruff/pytest nếu có thể. Cập nhật README chỉ với lệnh thật sự đã được implement.
```

## 2. Phase 1 — Parser và pipeline PlantDoc

```text
Thực hiện PHASE 1 theo PLAN.md, tuân thủ mục data của AGENT.md.

Tạo:
- src/tomato_det/common/types.py: ImageRecord, BoxRecord và kiểu chuẩn hóa;
- src/tomato_det/data/inspect.py;
- parse_plantdoc.py đọc cấu trúc CSV của bản PlantDoc Object Detection (filename,width,height,class,xmin,ymin,xmax,ymax);
- tomato_subset.py ánh xạ có kiểm soát vào 9 class đích;
- dedup.py dùng perceptual hash và nhóm near-duplicate;
- split.py giữ split TEST nguồn, tách validation từ TRAIN theo group và tránh group rò rỉ;
- export_yolo.py / export_coco.py;
- stats.py / visualize.py để báo cáo và tạo ảnh minh họa box.

Trước khi hard-code tên class, mở metadata/CSV thật nếu dữ liệu đã có; nếu không có dataset, tạo fixture test nhỏ với schema tài liệu hóa và không tuyên bố đã xác nhận số liệu của bộ thật. Parser phải phát hiện class lạ, ảnh thiếu/hỏng, tọa độ sai và ghi thống kê.

Không tải raw dataset tự động trong pipeline. Cung cấp lệnh/script download riêng có nguồn rõ ràng. Không sửa nhãn thầm lặng. Dùng seed cố định và lưu split manifest.

Viết pytest cho parser, mapping, tọa độ, export YOLO, khử trùng lặp và split leakage. Kết thúc bằng các lệnh kiểm tra và ví dụ đầu ra.
```

## 3. Phase 2 — Detector và huấn luyện YOLO11s

```text
Thực hiện PHASE 2 theo PLAN.md.

Triển khai tối thiểu:
- models/base.py định nghĩa Detector protocol và Detection chuẩn hóa;
- models/yolo_detector.py bọc Ultralytics YOLO;
- models/registry.py map model_id sang class/config/path allowlisted;
- train/train_yolo.py đọc configs/experiments/yolo11s_coco.yaml, ghi runs/<exp_id>/, best checkpoint, logs và meta.json;
- eval/evaluate.py tính metrics validation và lưu output có cấu trúc;
- eval/benchmark_latency.py đo warm-up, latency P50/P95 và ghi device/runtime/config.

MVP chỉ bắt buộc YOLO11s COCO-pretrained. Không bắt buộc train scratch, YOLOv8 hoặc Faster R-CNN ở phase này. Thiết kế code để baseline có thể thêm sau mà không làm rối interface.

Ràng buộc:
- Không tuyên bố có checkpoint đã train nếu chưa thực sự chạy train.
- Nếu thiếu data/weights hoặc không có GPU, tạo test bằng mock rõ ràng và hướng dẫn lệnh chạy thật; không tính metric từ dữ liệu giả vào report.
- Ghi seed, versions, git commit/hash (nếu có), data split, hyperparameters, device và thời điểm chạy.
- Threshold hoạt động được chọn từ validation, không từ test.

Viết test cho normalized boxes, confidence, registry và hành vi model unavailable. Chạy test có thể chạy trong môi trường hiện tại.
```

## 4. Phase 3 — Backend FastAPI và REST

```text
Thực hiện PHASE 3 theo PLAN.md và API contract trong AGENT.md.

Triển khai:
- serving/settings.py đọc TDET_* từ environment;
- serving/schemas.py định nghĩa response/request schemas;
- serving/services/model_manager.py lazy-load model theo allowlist, bảo vệ truy cập model và hỗ trợ trạng thái unavailable;
- serving/services/inference.py decode/validate image và gọi detector;
- serving/utils/image_io.py kiểm tra bytes, decode ảnh, giới hạn kích thước;
- routers/health.py, classes.py, models.py, detect.py;
- serving/main.py với lifespan và frontend dist mount tùy điều kiện.

Endpoints cần có: GET /api/health, GET /api/classes, GET /api/models, POST /api/detect. Dùng file multipart `file`. Trả normalized bbox xyxy, slug, score, image width/height và inference_ms.

Ràng buộc:
- App phải start được khi chưa có weights; endpoint models nêu rõ model unavailable và lý do.
- Không nhận đường dẫn checkpoint tự do từ người dùng.
- Validate conf/iou/imgsz/max_det, file size và image decode.
- Không chạy inference blocking trực tiếp trên event loop.
- Mặc định host 127.0.0.1; CORS chỉ cấu hình origins cần thiết cho Vite dev.

Tạo FastAPI TestClient tests cho health/classes/models, ảnh hợp lệ, ảnh hỏng, model missing, tham số sai và payload quá lớn. Không giả vờ inference thật nếu chỉ dùng mock.
```

## 5. Phase 4 — WebSocket và xử lý frame

```text
Thực hiện PHASE 4: WebSocket /ws/detect.

Implement protocol đúng AGENT.md:
- Client gửi text config `{type, model, conf, iou, imgsz, max_det, agnostic_nms}`.
- Client gửi binary gồm 4 byte uint32 big-endian frame_id + JPEG bytes.
- Server trả JSON result cùng frame_id, model, image width/height, inference_ms và detections normalized.
- Lỗi trả schema `{type:"error", code, message}` và không làm crash server.

Bắt buộc xử lý config không hợp lệ, frame binary ngắn, JPEG hỏng, payload quá lớn, model unavailable và disconnect. Không tạo queue không giới hạn; dùng tối đa một inference đang chạy và một frame pending/latest-frame-wins cho mỗi kết nối hoặc cơ chế tương đương được tài liệu hóa. Inference chạy ngoài event loop. Không log binary/frame ảnh.

Viết tests WebSocket cho config, frame hợp lệ (dùng fixture ảnh), frame hỏng, frame_id round-trip, giới hạn payload, disconnect và model unavailable. Làm rõ trong docs rằng latency là số đo thật khi chạy backend, không phải giá trị ví dụ.
```

## 6. Phase 5 — Frontend upload + webcam live preview

```text
Thực hiện PHASE 5 theo PLAN.md.

Dựng frontend React + TypeScript + Vite, tạo:
- LivePage: Start/Stop camera, WebSocket state, model selector, confidence/IoU/imgsz/max_det controls;
- UploadPage: upload ảnh, hiển thị boxes/result list, save snapshot phía client;
- VideoStage + OverlayCanvas: video và canvas overlay;
- useWebcam, useDetectionStream, useFps;
- api/client.ts, api/wsClient.ts, api/types.ts;
- drawBoxes.ts, frameEncoder.ts, colors.ts;
- StatsBar, Legend, Disclaimer và UI trạng thái lỗi/unavailable/loading.

Client webcam:
- Chỉ gọi getUserMedia sau click Start.
- Resize frame bảo toàn tỷ lệ, cạnh dài tối đa 640 px, encode JPEG.
- Gửi header 4 byte frame_id big-endian + JPEG bytes.
- Giữ tối đa một frame đang bay hoặc cơ chế backpressure tương đương; bỏ kết quả cũ theo frame_id.
- Stop/unmount đóng MediaStream tracks, timer/RAF và WebSocket.

Overlay:
- box normalized xyxy theo ảnh đã gửi, không phải CSS pixels;
- tính vùng video thực đang hiển thị, xử lý object-fit/letterbox, resize, DPR;
- không làm lệch box khi video container rộng/hẹp hoặc đổi chiều.

Viết Vitest cho encoder/protocol/drawBoxes; kiểm tra npm run build. Không dùng CDN, font ngoài hay cloud endpoint. Giao diện phải hiển thị rõ khi chưa có weights; không vẽ box giả.
```

## 7. Phase 6 — Tích hợp, báo cáo và đóng gói

```text
Thực hiện PHASE 6 theo PLAN.md.

Kiểm tra end-to-end và hoàn thiện tài liệu:
- ruff + pytest;
- npm ci + npm test -- --run + npm run build;
- scripts/ws_smoke_test.py kiểm tra WebSocket protocol khi server chạy;
- README hướng dẫn Windows PowerShell và terminal Unix, setup torch CPU/CUDA, dataset path, chuẩn bị data, train, đưa weights vào local, chạy backend/frontend;
- docs/API.md ghi request/response/errors;
- reports/REPORT.md có template khoa học, metrics sinh từ artifacts thực, per-class support và hạn chế.

Tạo CI tối thiểu chạy lint/test và frontend build (không yêu cầu dataset lớn hay checkpoint thật trên CI). Kiểm tra .gitignore để không commit data/raw, data/processed ảnh, runs, *.pt, .venv, node_modules hoặc file `.env`.

Không tự điền số mAP/FPS. Phân biệt kết quả chưa chạy, kết quả mock test và kết quả inference/train thật. Tổng kết lệnh đã chạy và test nào còn phụ thuộc vào data/weights/GPU.
```

## 8. Prompt sửa lỗi khi Agent gặp vấn đề

### 8.1 Lỗi môi trường hoặc thư viện

```text
Chẩn đoán lỗi môi trường theo stack trace và phiên bản đang cài. Trước khi đề xuất thay đổi, kiểm tra Python/torch/torchvision/ultralytics và OS. Không nâng cấp toàn bộ dependency theo kiểu thử vận may. Đề xuất thay đổi nhỏ nhất, cập nhật lock/requirements nếu cần, rồi chạy lại đúng lệnh gây lỗi. Báo rõ nguyên nhân đã xác minh và phần còn là giả thuyết.
```

### 8.2 Live preview bị trễ

```text
Đo riêng: frame encode, network/WebSocket, preprocessing, model inference, serialize response và overlay render. Không kết luận bottleneck từ FPS tổng duy nhất. Kiểm tra backpressure/frame_id và queue có bị tích lũy không. Chỉ đề xuất giảm imgsz/JPEG quality/send rate sau khi có baseline đo. Không sửa ngưỡng/model ảnh hưởng độ chính xác mà không ghi lại cấu hình.
```

### 8.3 Box vẽ sai vị trí

```text
Kiểm tra lần lượt: ảnh gửi server có bị crop không; kích thước ảnh trong response; bbox là normalized xyxy hay pixel; aspect ratio; object-fit/letterbox; CSS size so với canvas backing size; devicePixelRatio; frame_id có trộn kết quả cũ không. Viết test đơn vị cho các trường hợp corner/letterbox/portrait/resize trước khi sửa.
```

### 8.4 Metric bất thường

```text
Không sửa metric bằng cách làm đẹp số. Kiểm tra class mapping, split leakage, format bbox, image dimensions, confidence/NMS, box coordinate convention, class ID offset của Faster R-CNN và evaluator. Tạo fixture nhỏ có ground truth dự đoán biết trước; so sánh metric implementation với fixture rồi ghi nguyên nhân.
```

## 9. Definition of Done toàn dự án

- [x] Chuẩn hóa PlantDoc tạo được split và báo cáo có thể tái lập.
  Train 620 ảnh/2.206 box, val 30/439, test 69/192; `configs/data.yaml` cố định seed 42 và ngưỡng pHash.
- [x] YOLO11s checkpoint thật có metadata.
  `runs/yolo11s_coco_smoke/weights/best.pt` + `meta.json` (seed, versions, args, device). Đây là run
  CPU 10 epoch `imgsz=320`; cấu hình 100 epoch `imgsz=640` chưa chạy vì không có GPU.
- [x] REST upload trả bbox hợp lệ (đã kiểm tra `/api/detect` với checkpoint thật, bbox normalized).
- [x] Live preview nhận kết quả theo `frame_id` và vẽ đúng tọa độ
  (`scripts/ws_smoke_test.py` trả `type=result`, `frame_id=1`; unit test `drawBoxes`).
- [x] Camera/socket cleanup hoạt động khi Stop/unmount (`useWebcam.stop` dừng mọi track và gỡ
  `srcObject`; `useDetectionStream.disconnect` đóng socket; cả hai chạy trong cleanup của `useEffect`).
  Đã review code, **chưa có test hook tự động**.
- [x] Server chạy khi model unavailable và đưa thông báo hữu ích (`/api/models` trả `available=false`
  kèm lý do; `/api/detect` trả `MODEL_UNAVAILABLE`; có test tự động).
- [x] Backend tests (33 passed), frontend tests (7 passed) và production build chạy được.
- [x] Báo cáo tách validation và test, có per-class metrics và hạn chế domain gap
  (`reports/REPORT.md`, sinh từ JSON thật của run).
- [x] README/AGENT/PLAN/PROMPT đồng bộ với code sau khi cập nhật phạm vi MVP.
- [x] Không commit raw dataset, checkpoint nặng, secret hay dữ liệu webcam
  (`.gitignore` chặn `data/raw`, `data/processed`, `runs`, `*.pt`; không có secret trong repo).

**Chưa kiểm chứng tự động:** webcam thật trong trình duyệt (cần quyền camera của người dùng) và ảnh
chụp/video demo. Không tạo media giả để thay thế.
