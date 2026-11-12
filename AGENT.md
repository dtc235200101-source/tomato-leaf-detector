# AGENT.md — Quy tắc làm việc cho AI coding agent

Đây là quy ước bắt buộc khi Agent triển khai Tomato Leaf Detector. Đọc `README.md`, `PLAN.md`, file này và prompt của phase hiện tại trước khi sửa code.

## 1. Mục tiêu dự án

Xây dựng ứng dụng web AI chạy local để phát hiện lá cà chua khỏe/bệnh trong ảnh upload và webcam, dựa trên PlantDoc Object Detection Dataset. Mô hình serving chính là YOLO11s fine-tune từ COCO; YOLOv8s và Faster R-CNN là baseline nghiên cứu mở rộng.

- Bài toán là **object detection ở mức lá/cụm lá**. Không tuyên bố mô hình khoanh từng đốm bệnh hoặc đo phần trăm diện tích bệnh.
- Inference mặc định chạy trên máy local, backend bind `127.0.0.1`.
- Không được bịa metric, kết quả train, benchmark, coverage test hoặc trạng thái triển khai.
- Khi thiếu checkpoint, phải báo `unavailable`; không dựng dự đoán giả lập trong đường chạy production.

## 2. Thứ tự ưu tiên và quy tắc thay đổi

1. Tuân thủ quyết định đã chốt trong `PLAN.md`.
2. Giữ MVP chạy được: chuẩn hóa data → train/infer YOLO11s → API → live preview.
3. Chỉ thêm baseline/scope phụ sau khi MVP và tests nền đã chạy.
4. Không thay stack/framework/model chính nếu chưa có lý do kỹ thuật và cập nhật tài liệu.
5. Nếu tài liệu mâu thuẫn, `configs/classes.yaml` là nguồn chuẩn cho class mapping; `AGENT.md` là quy tắc code; `PLAN.md` là phạm vi/nghiệm thu; `README.md` là hướng dẫn người dùng. Báo cáo mâu thuẫn trước khi tự ý xóa chức năng.
6. Ưu tiên thay đổi nhỏ, cô lập, có test; không viết lại toàn bộ repo để xử lý lỗi cục bộ.

## 3. Cách làm việc bắt buộc

Trước khi code:

- Kiểm tra cây thư mục và file đã tồn tại; không ghi đè dữ liệu/người dùng nếu chưa hiểu nội dung.
- Nêu ngắn gọn các file sẽ thay đổi, giả định quan trọng và test sẽ chạy.
- Tìm hiểu version API thư viện từ package/lock hiện có, không giả định API cũ/mới.

Trong khi code:

- Triển khai theo một phase nhỏ trong `PROMPT.md`.
- Nếu chưa thể chạy test vì thiếu dataset, checkpoint, GPU hoặc package hệ thống, nói rõ test nào chưa chạy và lý do.
- Không tạo output “thành công” giả để vượt qua test; chỉ mock ở unit test với tên rõ ràng.
- Không tạo một file hàng nghìn dòng nếu có thể chia theo trách nhiệm.

Sau khi code:

- Chạy test/lint/build phù hợp với các file thay đổi.
- Báo danh sách file sửa, lệnh đã chạy, kết quả thật, phần còn thiếu và bước chạy tiếp theo.
- Cập nhật `README.md`/`PLAN.md` nếu hành vi hoặc lệnh CLI/API thay đổi.

## 4. Quy tắc dữ liệu và lớp

- `configs/classes.yaml` là single source of truth, với đúng 9 lớp, ID liên tục `0..8`.
- Tên label trong CSV nguồn có thể khác viết hoa/thường/dấu gạch dưới; chỉ chuẩn hóa bằng quy tắc minh bạch, sau đó đối chiếu map tường minh.
- Nếu phát hiện lớp không rõ, dừng hoặc ghi vào báo cáo `unmapped_classes`; không tự gán lớp gần giống.
- Chỉ giữ box thuộc 9 lớp đích và giữ ảnh nếu còn ít nhất một box đích.
- YOLO text: `class_id x_center y_center width height`; tọa độ chuẩn hóa theo width/height và phải hữu hạn, width/height dương, mọi giá trị trong `[0,1]` sau khi xử lý hợp lệ.
- Giữ split TEST nguồn; không dùng test để chọn epoch, model, threshold hoặc hyperparameter.
- Near-duplicate group không được chia qua train/val. Khi phát hiện train/val gần trùng test, loại bản sao trong train/val và ghi log.
- Không commit ảnh raw PlantDoc, file dataset lớn, `.pt`, cache, log nhiều GB, `.venv`, `node_modules` hoặc `frontend/dist` nếu không có lý do phát hành rõ ràng.

## 5. Quy tắc Python

- Hỗ trợ Python 3.10–3.12; target Python 3.11 nếu dependency tương thích.
- Dùng type hints cho API/module lõi, tên biến/hàm rõ nghĩa và docstring nơi logic không hiển nhiên.
- Dùng `pathlib.Path`; không hard-code đường dẫn Windows/Linux trong source.
- Dùng `logging`, không dùng `print()` cho logging trong backend/library. CLI có thể in tiến độ qua Typer.
- Khởi tạo seed tập trung; ghi seed và môi trường vào metadata.
- Không khởi tạo model nặng ở import-time; nạp trong lifespan/ModelManager để tests có thể import app.
- Không giữ inference nặng trên event loop FastAPI. Dùng thread/executor thích hợp và tránh nhiều request đồng thời gọi cùng model/device nếu không an toàn.
- Xử lý lỗi có nghĩa: `INVALID_IMAGE`, `MODEL_NOT_FOUND`, `PAYLOAD_TOO_LARGE`, `INFERENCE_FAILED`, `INVALID_CONFIG`, `MODEL_UNAVAILABLE`.
- Không bắt `Exception` rồi bỏ qua. Nếu catch rộng để giữ server sống, phải log stack trace và trả lỗi schema ổn định.
- Với Windows, không yêu cầu `num_workers > 0` cho pipeline smoke test; tránh lỗi spawn và giữ cấu hình workers rõ ràng.

## 6. Hợp đồng lớp/mô hình

### `ClassSpec`

Tối thiểu gồm `id`, `name_source`, `slug`, `name_vi`, `group`, `color`. ID liên tục theo file `classes.yaml`.

### `Detection`

Kết quả serving chuẩn hóa:

```python
Detection(
    class_id: int,
    class_name: str,      # slug chuẩn
    score: float,         # [0, 1]
    box: tuple[float, float, float, float],  # x1,y1,x2,y2 normalized [0,1]
)
```

Điều kiện: `0 <= x1 < x2 <= 1`, `0 <= y1 < y2 <= 1`, `0 <= score <= 1`. Nơi nào cần pixel coordinates cho evaluator phải chuyển đổi rõ ràng, không trộn pixel và normalized trong cùng một field.

### Detector interface

Một detector cung cấp tối thiểu:

- `model_id` và metadata/availability.
- `predict(image_rgb, conf, iou, imgsz, max_det, ...) -> list[Detection]`.
- Không đọc/ghi file ảnh trong interface inference nếu đầu vào đã là ảnh đã decode.
- Không tự tải checkpoint từ nguồn mạng khi ứng dụng khởi động.

`ModelManager` chỉ nạp `model_id` trong allowlist/config. API tuyệt đối không nhận arbitrary filesystem path từ người dùng.

## 7. Hợp đồng API

### REST

| Endpoint | Hợp đồng |
|---|---|
| `GET /api/health` | trả `status`, `device`, danh sách/tên model đã nạp ở mức an toàn |
| `GET /api/classes` | đúng class list theo YAML |
| `GET /api/models` | `model_id`, `display_name`, `available`, `reason` nếu unavailable, metric nếu có |
| `POST /api/detect` | field upload là `file`; tham số tùy chọn `model`, `conf`, `iou`, `imgsz`, `max_det` |
| `WS /ws/detect` | protocol ở mục 8 |

Mọi detect response phải chứa kích thước ảnh, `model`, `inference_ms` và danh sách `detections`. Không trả nhãn/ID tùy ý ngoài danh mục.

### Validation

- `conf`, `iou` trong `[0,1]`; `imgsz` trong danh sách cấu hình hợp lệ; `max_det` có giới hạn trên.
- Kiểm tra số byte trước khi decode; xác minh ảnh thực bằng decoder chứ không tin MIME/extension.
- Giới hạn độ phân giải cực lớn để giảm rủi ro hết RAM.
- Không ghi ảnh webcam/upload ra đĩa trừ khi người dùng chủ động chọn lưu snapshot phía client.

## 8. Hợp đồng WebSocket

- Route `/ws/detect`.
- Client gửi JSON text config:

```json
{"type":"config","model":"yolo11s_coco","conf":0.25,"iou":0.5,"imgsz":640,"max_det":50,"agnostic_nms":false}
```

- Client gửi frame binary: 4-byte unsigned big-endian `frame_id` theo sau bởi bytes JPEG.
- Server trả JSON text `type=result` có cùng `frame_id`, `model`, `image.width`, `image.height`, `inference_ms`, `detections`.
- `box` là `[x1,y1,x2,y2]` normalized theo ảnh JPEG đã gửi, không phải tọa độ CSS.
- Lỗi protocol trả `{type:"error", code:"...", message:"..."}`; không crash toàn server.
- Mỗi client có tối đa một inference đang chạy và tối đa một frame pending (latest-frame-wins). Không cho queue tăng vô hạn.
- Frontend bỏ kết quả lỗi thời, không vẽ kết quả lên video frame không liên quan nếu `frame_id` cũ hơn frame đang hiển thị.
- Stop/unmount phải đóng camera tracks và WebSocket.

## 9. Frontend và tọa độ overlay

- TypeScript strict; response từ backend được parse/validate trước khi dùng.
- Tách UI components: VideoStage, OverlayCanvas, ControlsPanel, ModelSelector, ResultList, StatsBar, Legend, Disclaimer, UploadDropzone.
- Dùng `getUserMedia()` chỉ sau thao tác Start có chủ đích của người dùng. Camera không tự bật khi vào trang.
- Canvas frame gửi server giữ nguyên tỷ lệ; cạnh dài mục tiêu tối đa 640 px trong cấu hình live.
- Overlay dùng normalized bbox và kích thước thực hiển thị của video. Xử lý `object-fit`, letterbox, resize container và `devicePixelRatio`; không dùng trực tiếp tọa độ ảnh làm CSS pixel.
- Test `drawBoxes.ts` cho box ở góc, box toàn khung, letterbox, video portrait/landscape và resize.
- Accessible: nút có label, hiển thị trạng thái camera/socket/model, lỗi rõ ràng, màu box không phải cách duy nhất để phân biệt class.
- Không dùng CDN, font ngoài hoặc API cloud cho app offline-first.

## 10. An toàn, privacy và giấy phép

- Mặc định host `127.0.0.1`; không tự mở lên `0.0.0.0`.
- Không log nội dung ảnh/video, dữ liệu cá nhân hoặc binary websocket.
- Không tải tài nguyên runtime từ Internet. Tải weights là bước setup riêng, có hướng dẫn và nguồn tin cậy.
- PlantDoc cần ghi attribution/nguồn, theo license của bản dataset thực tế được tải. Không commit/rehost dataset nguyên bản.
- Ultralytics YOLO được cung cấp theo AGPL-3.0 hoặc Enterprise; đồ án cần xem xét tuân thủ license trước khi phân phối mã/weights. Không diễn giải đây là tư vấn pháp lý.
- Giữ disclaimer: “Kết quả chỉ phục vụ mục đích học thuật/tham khảo; không thay thế chẩn đoán của chuyên gia bảo vệ thực vật.”

## 11. Tests bắt buộc

### Python

- Class map duy nhất, ID liên tục, xử lý label không biết.
- CSV parser: box hợp lệ, đảo tọa độ, out-of-bounds, file thiếu, ảnh hỏng.
- Export YOLO: tọa độ chuẩn hóa đúng, không âm/vượt phạm vi và class id đúng.
- Split: group không chồng train/val; TEST không bị đổi; không rò rỉ near-duplicate đã phát hiện.
- Detector output: normalized box/score hợp lệ; lỗi model-unavailable không làm sập app.
- REST: health/classes/models; detect valid/invalid file; validation query; model thiếu.
- WebSocket: config hợp lệ/sai, frame hợp lệ/sai, frame id trả về, payload quá lớn.
- Metrics: test evaluator với fixture synthetic đã biết kết quả.

### Frontend

- frame binary header đúng endianness và bytes.
- WebSocket response typing và xử lý lỗi.
- Canvas box geometry normalized → display coordinates.
- Webcam start/stop cleanup.

### Chạy kiểm tra

```bash
ruff check .
pytest -q
cd frontend && npm ci && npm test -- --run && npm run build
```

Chạy lệnh phù hợp với thay đổi; nếu một lệnh không khả dụng, báo rõ và không tuyên bố đã pass.

## 12. Quy trình phản hồi của Agent

Kết thúc mỗi phase bằng:

1. **Đã làm:** tính năng/file triển khai.
2. **Đã kiểm tra:** lệnh và kết quả cụ thể.
3. **Chưa làm / blocker:** dataset/checkpoint/dependency/phần còn thiếu.
4. **Cách chạy:** lệnh cụ thể để kiểm tra nhanh.
5. **Ảnh hưởng hợp đồng:** API/schema/config nào thay đổi và tài liệu nào đã cập nhật.
