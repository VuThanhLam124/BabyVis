# BabyVis — Tạo ảnh em bé từ ảnh siêu âm

BabyVis là một công cụ tạo ảnh em bé từ ảnh siêu âm dựa trên Stable Diffusion + ControlNet (Canny), kèm giao diện Gradio thân thiện và chế độ xử lý theo lô.

## Cài đặt
- Yêu cầu Python 3.9+
- Cài thư viện:
  - `pip install -r requirements.txt`

## Chạy nhanh (Gradio UI)
- Chạy giao diện: `python apps/gradio_app.py`
- Quy trình: Tải ảnh siêu âm → Nhấn “Bắt đầu xử lý” → Tải ảnh kết quả.
- Ảnh kết quả mặc định lưu tại: `outputs/gradio/`

## Xử lý theo lô (Batch)
- Ảnh mẫu nằm trong `samples/`
- Có thể chuẩn bị danh sách ảnh tại `data/image_list.txt` (mỗi dòng 1 đường dẫn)
- Chạy: `python apps/batch_processor.py`
- Kết quả: `outputs/batch/`; Log: `logs/batch.log`

## Tuỳ chọn mô hình
- Mặc định SD v1.5 + ControlNet Canny (CPU nếu không có GPU).
- Có thể đổi sang SDXL qua biến môi trường hoặc override ID:
  - Ví dụ SDXL: 
    - `PREFER_SDXL=1 python apps/gradio_app.py`
    - Hoặc:
      - `BASE_MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0" \\
         CONTROLNET_ID="diffusers/controlnet-canny-sdxl-1.0" \\
         python apps/gradio_app.py`
- Các tham số gợi ý:
  - SD v1.5: steps 40–60, CFG 7–9, control 0.7–1.0
  - SDXL: steps 30–40, CFG 5–7, control 0.6–0.9

## Cấu trúc thư mục (đã cơ cấu lại)
- `src/babyvis/`: mã nguồn chính (inference, model_utils)
- `apps/`: điểm chạy
  - `gradio_app.py`: giao diện Gradio
  - `batch_processor.py`: xử lý theo lô
- `web/`: giao diện web tĩnh (index.html, style.css, app.js) — đã loại bỏ toàn bộ icon SVG
- `outputs/`: ảnh kết quả (`gradio/`, `batch/`, `tmp/`)
- `samples/`: ảnh mẫu
- `config/`: cấu hình batch (`config/batch_config.json`)
- `data/`: dữ liệu vào (vd. `data/image_list.txt`)
- `logs/`: log chạy batch

Ghi chú: Các script trong `apps/` tự thêm `src/` vào `PYTHONPATH` khi chạy trực tiếp, nên không cần cài đặt gói.

## Bảo trì và lưu ý
- Repo đã được dọn dẹp và loại bỏ tất cả icon (SVG/biểu tượng) khỏi giao diện tĩnh.
- Hình ảnh sinh ra chỉ mang tính tham khảo, không thay thế ý kiến y khoa.
