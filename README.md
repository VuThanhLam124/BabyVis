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

## Chạy trên CPU (khắc phục lỗi CUDA/cuBLAS)
- Nếu gặp lỗi kiểu `libcublasLt.so.11`/`CUDA` hoặc chưa cài driver/phụ thuộc CUDA đúng bản:
  - Tạm thời ép chạy CPU: `CUDA_VISIBLE_DEVICES="" python apps/batch_processor.py` hoặc `FORCE_CPU=1 python apps/batch_processor.py`
  - Code có cơ chế tự fallback GPU→CPU nếu lỗi CUDA xảy ra trong lúc suy luận.

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

### Dùng Qwen Image Edit (Qwen/Qwen-Image-Edit)
- Bật dùng Qwen Image Edit thay cho Stable Diffusion + ControlNet:
  - `USE_QWEN_IMAGE_EDIT=1 python apps/gradio_app.py`
  - hoặc trong batch: `USE_QWEN_IMAGE_EDIT=1 python apps/batch_processor.py`
- Yêu cầu cài thêm thư viện: `transformers`, `accelerate`, `sentencepiece`, `safetensors` (đã thêm sẵn trong `requirements.txt`).
- Ghi chú: Mã tải Qwen dùng `trust_remote_code=True` theo hướng dẫn từ Hugging Face. Đầu ra ảnh được decode tự động từ response của model.

### Dùng bản nhẹ GGUF (QuantStack/Qwen-Image-Edit-GGUF)
- Bật backend GGUF (chạy bằng llama.cpp, phù hợp CPU/GPU VRAM ~4GB):
  - `USE_QWEN_GGUF=1 python apps/gradio_app.py`
  - hoặc `USE_QWEN_GGUF=1 python apps/batch_processor.py`
- Tuỳ chọn tải model:
  - Dùng file cục bộ: đặt `QWEN_GGUF_PATH=/path/to/model.gguf`
  - Hoặc để tự tải từ HF: `QWEN_GGUF_REPO=QuantStack/Qwen-Image-Edit-GGUF` và (tuỳ chọn) `QWEN_GGUF_FILENAME=qwen-image-edit-q4_k_m.gguf`
- Tinh chỉnh hiệu năng/thông số thấp VRAM:
  - `QWEN_N_GPU_LAYERS=12` (mặc định, 4GB VRAM nên để 8–16; đặt 0 để chạy CPU-only)
  - `QWEN_N_THREADS=<số luồng CPU>`; `QWEN_N_CTX=2048`
- Yêu cầu cài `llama-cpp-python` (đã có trong `requirements.txt`). Cần bản llama.cpp hỗ trợ multi-modal (VL). Nếu backend GGUF chỉ trả về văn bản thay vì ảnh, code sẽ tự động dùng văn bản đó làm prompt tăng cường cho ControlNet để vẫn tạo được ảnh.

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
