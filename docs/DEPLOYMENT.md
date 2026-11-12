# Deploy một dịch vụ lên Render

## Phạm vi và cảnh báo riêng tư

Thiết kế gốc của dự án là inference local. Khi deploy, ảnh upload và frame webcam sẽ được gửi từ trình duyệt đến Render qua Internet. Chỉ deploy khi người dùng đã đồng ý với thay đổi này. Ứng dụng không có database. Docker build React/Vite, sau đó FastAPI phục vụ frontend, REST và WebSocket trong **một Render Web Service**.

## Điều kiện trước khi deploy

1. Tạo Git repository và đẩy source lên GitHub/GitLab. Không đẩy `data/`, `runs/`, `.venv`, `node_modules` hoặc checkpoint `.pt`.
2. Backend cần checkpoint thật ở `weights/yolo11s_coco/best.pt`. Render build từ Git sẽ **không có checkpoint** vì checkpoint bị bỏ qua có chủ đích. Khi thiếu checkpoint, `/api/models` báo unavailable và nhận diện trả `MODEL_UNAVAILABLE`.
3. Cần một cách cung cấp checkpoint an toàn khi build/chạy: private container image có checkpoint, Render persistent disk đã nạp checkpoint, hoặc object storage private với cơ chế tải có kiểm soát. Không để URL tải công khai trong source code.
4. Render CPU có thể chậm cho YOLO/PyTorch và Docker image khá lớn. Dùng plan có đủ RAM/dung lượng; thử health endpoint và upload ảnh sau deploy.

## Render all-in-one

1. Tạo Web Service từ repository, chọn Blueprint và `render.yaml`.
2. Provision checkpoint theo một trong các cách ở phần điều kiện. Không deploy dịch vụ nhận diện khi checkpoint chưa có.
3. Deploy, sau đó kiểm tra `https://<render-service>/api/health` và `https://<render-service>/api/models`. Chỉ khi `available: true` mới xem là deploy nhận diện thành công.

## Xác minh sau deploy

1. Mở `https://<render-service>/`; kiểm tra `GET /api/classes` và `/api/models` qua DevTools Network.
2. Upload một ảnh. Kết quả phải có model, image dimensions, inference time và detections hợp lệ; nếu checkpoint không có, lỗi phải là `MODEL_UNAVAILABLE`.
3. Mở webcam, cấp quyền camera và kiểm tra socket `wss://<render-service>/ws/detect` hoạt động.
4. Không dùng test set để tinh chỉnh confidence/model sau khi deploy.
