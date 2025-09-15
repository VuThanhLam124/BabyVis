# 🤖 BabyVis - Qwen Edition# BabyVis — Tạo ảnh em bé từ ảnh siêu âm



**Dự đoán hình ảnh em bé từ siêu âm sử dụng Qwen Image Edit AI**BabyVis là một công cụ tạo ảnh em bé từ ảnh siêu âm dựa trên Stable Diffusion + ControlNet (Canny), kèm giao diện Gradio thân thiện và chế độ xử lý theo lô.



[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)## Cài đặt

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)- Yêu cầu Python 3.9+

- Cài thư viện:

## 🚀 Tính năng  - `pip install -r requirements.txt`



- **🧠 Pure Qwen AI**: Sử dụng hoàn toàn Qwen Image Edit - loại bỏ ControlNet để hiệu suất tối ưu## Chạy nhanh (Gradio UI)

- **⚡ Auto-Detection**: Tự động chọn backend tốt nhất (GGUF hoặc Transformers)- Chạy giao diện: `python apps/gradio_app.py`

- **🔧 Adaptive VRAM**: Tự động tối ưu theo VRAM có sẵn (4GB personal → 10GB+ server)- Quy trình: Tải ảnh siêu âm → Nhấn “Bắt đầu xử lý” → Tải ảnh kết quả.

- **🎯 High Performance**: Tối ưu hóa cho tốc độ và chất lượng- Ảnh kết quả mặc định lưu tại: `outputs/gradio/`

- **💻 Cross-Platform**: Hỗ trợ GPU CUDA, CPU, và các môi trường hybrid

## Xử lý theo lô (Batch)

## 📦 Cài đặt nhanh- Ảnh mẫu nằm trong `samples/`

- Có thể chuẩn bị danh sách ảnh tại `data/image_list.txt` (mỗi dòng 1 đường dẫn)

```bash- Chạy: `python apps/batch_processor.py`

# Clone repo- Kết quả: `outputs/batch/`; Log: `logs/batch.log`

git clone https://github.com/VuThanhLam124/BabyVis.git

cd BabyVis## Chạy trên CPU (khắc phục lỗi CUDA/cuBLAS)

- Nếu gặp lỗi kiểu `libcublasLt.so.11`/`CUDA` hoặc chưa cài driver/phụ thuộc CUDA đúng bản:

# Tạo conda environment  - Tạm thời ép chạy CPU: `CUDA_VISIBLE_DEVICES="" python apps/batch_processor.py` hoặc `FORCE_CPU=1 python apps/batch_processor.py`

conda create -n babyvis python=3.9 -y  - Code có cơ chế tự fallback GPU→CPU nếu lỗi CUDA xảy ra trong lúc suy luận.

conda activate babyvis

## Tuỳ chọn mô hình

# Cài đặt dependencies- Mặc định SD v1.5 + ControlNet Canny (CPU nếu không có GPU).

pip install -r requirements.txt- Có thể đổi sang SDXL qua biến môi trường hoặc override ID:

  - Ví dụ SDXL: 

# Chạy ngay lập tức    - `PREFER_SDXL=1 python apps/gradio_app.py`

./run_qwen_auto.sh    - Hoặc:

```      - `BASE_MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0" \\

         CONTROLNET_ID="diffusers/controlnet-canny-sdxl-1.0" \\

## 🎮 Cách sử dụng         python apps/gradio_app.py`

- Các tham số gợi ý:

### Chế độ tự động (Khuyến nghị)  - SD v1.5: steps 40–60, CFG 7–9, control 0.7–1.0

```bash  - SDXL: steps 30–40, CFG 5–7, control 0.6–0.9

./run_qwen_auto.sh    # Tự động chọn backend tốt nhất

```### Dùng Qwen Image Edit (Qwen/Qwen-Image-Edit)

- Bật dùng Qwen Image Edit thay cho Stable Diffusion + ControlNet:

### Chế độ cụ thể  - `USE_QWEN_IMAGE_EDIT=1 python apps/gradio_app.py`

```bash  - hoặc trong batch: `USE_QWEN_IMAGE_EDIT=1 python apps/batch_processor.py`

./run_qwen_gguf.sh         # Force GGUF (tương thích cao)- Yêu cầu cài thêm thư viện: `transformers`, `accelerate`, `sentencepiece`, `safetensors` (đã thêm sẵn trong `requirements.txt`).

./run_qwen_transformers.sh # Force Transformers (chất lượng cao)- Ghi chú: Mã tải Qwen dùng `trust_remote_code=True` theo hướng dẫn từ Hugging Face. Đầu ra ảnh được decode tự động từ response của model.

./run_qwen_cpu.sh          # CPU only (máy yếu)

```### Dùng bản nhẹ GGUF (QuantStack/Qwen-Image-Edit-GGUF)

- Bật backend GGUF (chạy bằng llama.cpp, phù hợp CPU/GPU VRAM ~4GB):

### Python API  - `USE_QWEN_GGUF=1 python apps/gradio_app.py`

```python  - hoặc `USE_QWEN_GGUF=1 python apps/batch_processor.py`

from babyvis.inference import generate_predict_auto- Tuỳ chọn tải model:

  - Dùng file cục bộ: đặt `QWEN_GGUF_PATH=/path/to/model.gguf`

# Tạo ảnh dự đoán  - Hoặc để tự tải từ HF: `QWEN_GGUF_REPO=QuantStack/Qwen-Image-Edit-GGUF` và (tuỳ chọn) `QWEN_GGUF_FILENAME=qwen-image-edit-q4_k_m.gguf`

generate_predict_auto(- Tinh chỉnh hiệu năng/thông số thấp VRAM:

    input_path="samples/ultrasound.jpg",  - `QWEN_N_GPU_LAYERS=12` (mặc định, 4GB VRAM nên để 8–16; đặt 0 để chạy CPU-only)

    output_path="output/baby.png",  - `QWEN_N_THREADS=<số luồng CPU>`; `QWEN_N_CTX=2048`

    backend="auto",  # hoặc "qwen", "qwen_gguf"- Yêu cầu cài `llama-cpp-python` (đã có trong `requirements.txt`). Cần bản llama.cpp hỗ trợ multi-modal (VL). Nếu backend GGUF chỉ trả về văn bản thay vì ảnh, code sẽ tự động dùng văn bản đó làm prompt tăng cường cho ControlNet để vẫn tạo được ảnh.

    ethnicity="mixed ethnicity"

)## Cấu trúc thư mục (đã cơ cấu lại)

```- `src/babyvis/`: mã nguồn chính (inference, model_utils)

- `apps/`: điểm chạy

## 🔧 Cấu hình  - `gradio_app.py`: giao diện Gradio

  - `batch_processor.py`: xử lý theo lô

### Environment Variables- `web/`: giao diện web tĩnh (index.html, style.css, app.js) — đã loại bỏ toàn bộ icon SVG

```bash- `outputs/`: ảnh kết quả (`gradio/`, `batch/`, `tmp/`)

# Backend selection- `samples/`: ảnh mẫu

export QWEN_BACKEND=auto          # auto, qwen, qwen_gguf- `config/`: cấu hình batch (`config/batch_config.json`)

export USE_QWEN_IMAGE_EDIT=1      # Force transformers- `data/`: dữ liệu vào (vd. `data/image_list.txt`)

export USE_QWEN_GGUF=1            # Force GGUF- `logs/`: log chạy batch



# Performance tuningGhi chú: Các script trong `apps/` tự thêm `src/` vào `PYTHONPATH` khi chạy trực tiếp, nên không cần cài đặt gói.

export QWEN_4BIT=true             # 4-bit quantization

export QWEN_FLASH_ATTN=true       # Flash attention## Bảo trì và lưu ý

export QWEN_N_GPU_LAYERS=20       # GPU layers for GGUF- Repo đã được dọn dẹp và loại bỏ tất cả icon (SVG/biểu tượng) khỏi giao diện tĩnh.

- Hình ảnh sinh ra chỉ mang tính tham khảo, không thay thế ý kiến y khoa.

# Local models
export QWEN_LOCAL_PATH=/path/to/local/model
export QWEN_GGUF_PATH=/path/to/model.gguf
```

### Batch Processing
```python
from apps.batch_processor import BatchImageProcessor

processor = BatchImageProcessor()
results = processor.process_image_list([
    "samples/1.jpeg",
    "samples/2.png",
    "samples/3.jpg"
])
```

## 🎯 Backends

| Backend | Ưu điểm | Nhược điểm | Khuyến nghị |
|---------|---------|------------|-------------|
| **Auto** | Tự động chọn tốt nhất | - | ✅ **Default** |
| **GGUF** | Tương thích cao, ít VRAM | Chậm hơn | Personal PC |
| **Transformers** | Chất lượng cao, nhanh | Nhiều VRAM | Server/GPU mạnh |
| **CPU** | Không cần GPU | Rất chậm | Backup mode |

## 📊 Hiệu suất

### VRAM Requirements
- **4GB**: GGUF Q4_K_M, 4-bit quantization
- **6GB**: GGUF Q5_K_M, optimal balance  
- **8GB+**: Transformers full precision
- **10GB+**: Flash attention enabled

### Tốc độ ước tính
- **GGUF CPU**: ~60-120s/image
- **GGUF GPU**: ~15-30s/image  
- **Transformers GPU**: ~5-15s/image

## 🔍 Troubleshooting

### Common Issues

**CUDA errors:**
```bash
export FORCE_CPU=1
./run_qwen_cpu.sh
```

**Out of memory:**
```bash
export QWEN_4BIT=true
export QWEN_8BIT=true
./run_qwen_gguf.sh
```

**Model not found:**
```bash
# Thử các repo fallback
export QWEN_GGUF_REPO=bartowski/Qwen-Image-Edit-GGUF
./run_qwen_gguf.sh
```

## 📁 Project Structure

```
BabyVis/
├── src/babyvis/           # Core Qwen processing
│   ├── model_utils.py     # Model loaders & VRAM detection
│   └── inference.py       # Image generation pipeline
├── apps/                  # Applications
│   ├── batch_processor.py # Batch processing
│   └── gradio_app.py      # Web interface
├── samples/               # Sample images
├── outputs/               # Generated results
└── run_qwen_*.sh         # Execution scripts
```

## 🎨 Tùy chỉnh

### Custom Instructions
```python
generate_predict_qwen_only(
    input_path="input.jpg",
    output_path="output.png",
    instruction="Transform this ultrasound into a realistic Asian newborn baby photo"
)
```

### Quality Settings
```python
# High quality (server)
export QWEN_N_CTX=4096
export QWEN_FLASH_ATTN=true

# Balanced (personal)  
export QWEN_N_CTX=2048
export QWEN_4BIT=true

# Fast (minimal)
export QWEN_N_CTX=1024
export QWEN_N_GPU_LAYERS=8
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **Qwen Team**: Qwen Image Edit model
- **llama.cpp**: GGUF inference engine
- **Hugging Face**: Model hosting & transformers
- **CUDA/PyTorch**: GPU acceleration

---

**🚀 Ready to generate baby predictions with Qwen AI!**

For support: Create an issue or contact [@VuThanhLam124](https://github.com/VuThanhLam124)