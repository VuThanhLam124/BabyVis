# 🍼 BabyVis - Professional Ultrasound to Baby Face Generation# BabyVis - GGUF Edition# BabyVis - GGUF Edition# 🤖 BabyVis - Qwen Edition# BabyVis — Tạo ảnh em bé từ ảnh siêu âm



**AI-powered ultrasound analysis and baby face prediction using Qwen-Image-Edit with Diffusers**



[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)🤖 **AI-powered ultrasound to baby face visualization using Qwen Image Edit GGUF model**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![GPU](https://img.shields.io/badge/GPU-4GB%20VRAM%20Optimized-green.svg)]()



## 🚀 Quick Start## 🚀 Quick Start🤖 **AI-powered ultrasound to baby face visualization using Qwen Image Edit GGUF model**



### Option 1: Automatic Installation (Recommended)

```bash

# Clone repository### Option 1: App Interface (Recommended)

git clone <your-repo>

cd BabyVis```bash



# Install optimized for your hardware./run_app.sh## 🚀 Quick Start**Dự đoán hình ảnh em bé từ siêu âm sử dụng Qwen Image Edit AI**BabyVis là một công cụ tạo ảnh em bé từ ảnh siêu âm dựa trên Stable Diffusion + ControlNet (Canny), kèm giao diện Gradio thân thiện và chế độ xử lý theo lô.

./install_optimized.sh

```

# Run with sample images

python main.py --input samples/ --output outputs/baby_faces/Choose from:

```

- 🖥️ **Desktop App** - Native GUI with drag & drop

### Option 2: Manual Installation

```bash- 🌐 **Web Interface** - Browser-based interface ### Option 1: GPU 4GB VRAM

# Install dependencies

pip install -r requirements.txt- 📦 **Batch Processing** - Process multiple images



# Single image processing- 🚀 **App Launcher** - GUI selector```bash

python main.py --input samples/1.jpeg --output outputs/ --prompt "realistic baby face"



# Batch processing

python main.py --input samples/ --output outputs/baby_faces/### Option 2: Hardware-Specific Scripts./run_4gb.sh[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)## Cài đặt

```



## 🎯 Features

**GPU 4GB VRAM:**```

- ✅ **Professional Diffusers Pipeline**: Uses standard `DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-Edit")`

- ✅ **4GB VRAM Optimized**: Quantized models, CPU offloading, memory-efficient attention```bash

- ✅ **Auto Hardware Detection**: Automatically chooses optimal configuration for your hardware

- ✅ **Batch Processing**: Process multiple ultrasound images at once./run_4gb.sh[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)- Yêu cầu Python 3.9+

- ✅ **Memory Management**: Advanced VRAM optimization with cleanup and monitoring

- ✅ **Production Ready**: Clean, modular codebase with proper error handling```



## 📋 Requirements### Option 2: GPU 12GB VRAM  



### Minimum Hardware**GPU 12GB VRAM:**

- **GPU**: NVIDIA RTX 3050 Ti (4GB VRAM) or better

- **CPU**: Intel i5/AMD Ryzen 5 (for CPU-only mode)```bash```bash- Cài thư viện:

- **RAM**: 8GB system memory

- **Storage**: 10GB free space (for model cache)./run_12gb.sh



### Software```./run_12gb.sh

- Python 3.8+

- CUDA 11.8+ (for GPU acceleration)

- PyTorch 2.0+

- Diffusers 0.21+**CPU Only:**```## 🚀 Tính năng  - `pip install -r requirements.txt`



## ⚙️ Hardware Configurations```bash



| Configuration | VRAM Usage | Speed | Quality | Recommended For |./run_cpu.sh

|---------------|------------|-------|---------|-----------------|

| **Quantized GPU** | ~3.5GB | Fast | High | RTX 3050 Ti, GTX 1660 |```

| **Full GPU** | ~6-8GB | Fastest | Highest | RTX 3060+, RTX 4060+ |

| **CPU Only** | 0GB | Slower | Good | No GPU available |### Option 3: CPU Only



## 📁 Project Structure## 📋 Requirements



``````bash

BabyVis/

├── main.py                     # Main application entry point- Python 3.8+

├── install_optimized.sh        # Optimized installation script

├── requirements.txt            # Dependencies- For GPU: NVIDIA GPU with 4GB+ VRAM./run_cpu.sh- **🧠 Pure Qwen AI**: Sử dụng hoàn toàn Qwen Image Edit - loại bỏ ControlNet để hiệu suất tối ưu## Chạy nhanh (Gradio UI)

├── src/babyvis/               # Core processing modules

│   ├── __init__.py- For CPU: 8GB+ RAM

│   ├── processor.py           # Main processor class

│   ├── inference.py           # Inference pipeline```

│   ├── batch_processor.py     # Batch processing utilities

│   ├── model_downloader.py    # Model management## 🔧 Setup

│   └── model_utils.py         # Model utilities

├── samples/                   # Sample ultrasound images- **⚡ Auto-Detection**: Tự động chọn backend tốt nhất (GGUF hoặc Transformers)- Chạy giao diện: `python apps/gradio_app.py`

├── outputs/                   # Generated baby faces

│   ├── diffusers_baby_faces/  # Main output directory1. **Clone the repository:**

│   └── final_intelligent/     # Previous analysis results

└── models/                    # Model cache (auto-created)```bash## 📋 Requirements

```

git clone <your-repo>

## 🛠️ Usage Examples

cd BabyVis- **🔧 Adaptive VRAM**: Tự động tối ưu theo VRAM có sẵn (4GB personal → 10GB+ server)- Quy trình: Tải ảnh siêu âm → Nhấn “Bắt đầu xử lý” → Tải ảnh kết quả.

### Command Line Interface

```

**Single Image:**

```bash- Python 3.8+

python main.py \

  --input samples/ultrasound.jpg \2. **Add ultrasound images:**

  --output outputs/ \

  --prompt "Transform this ultrasound into a beautiful realistic baby face" \```bash- For GPU: NVIDIA GPU with 4GB+ VRAM- **🎯 High Performance**: Tối ưu hóa cho tốc độ và chất lượng- Ảnh kết quả mặc định lưu tại: `outputs/gradio/`

  --steps 25 \

  --guidance 7.5mkdir -p samples

```

# Copy your ultrasound images to samples/ folder- For CPU: 8GB+ RAM

**Batch Processing:**

```bash```

python main.py \

  --input samples/ \- **💻 Cross-Platform**: Hỗ trợ GPU CUDA, CPU, và các môi trường hybrid

  --output outputs/baby_faces/ \

  --model-variant quantized \3. **Run with your preferred interface:**

  --cpu-offload

``````bash## 🔧 Setup



**Advanced Options:**# App interface (recommended)

```bash

python main.py \./run_app.sh## Xử lý theo lô (Batch)

  --input samples/ \

  --output outputs/ \

  --device cuda \

  --model-variant quantized \# Or specific hardware config1. **Clone the repository:**

  --steps 30 \

  --guidance 8.0 \./run_4gb.sh    # For 4GB GPU

  --cpu-offload

```./run_12gb.sh   # For 12GB GPU  ```bash## 📦 Cài đặt nhanh- Ảnh mẫu nằm trong `samples/`



### Python API./run_cpu.sh    # For CPU only



```python```git clone <your-repo>

from src.babyvis.processor import OptimizedBabyVisProcessor



# Initialize processor

processor = OptimizedBabyVisProcessor(enable_cpu_offload=True)## 📁 Project Structurecd BabyVis- Có thể chuẩn bị danh sách ảnh tại `data/image_list.txt` (mỗi dòng 1 đường dẫn)



# Load quantized model for 4GB VRAM

processor.load_model("quantized")

``````

# Process single image

result = processor.process_ultrasound_to_baby(BabyVis/

    "samples/ultrasound.jpg",

    prompt="realistic baby face from ultrasound",├── run_app.sh         # App interface launcher```bash- Chạy: `python apps/batch_processor.py`

    num_inference_steps=25

)├── run_4gb.sh         # GPU 4GB configuration



# Save result├── run_12gb.sh        # GPU 12GB configuration  2. **Add ultrasound images:**

if result:

    result.save("output/baby.png")├── run_cpu.sh         # CPU configuration



# Batch process├── app_desktop.py     # Desktop GUI app```bash# Clone repo- Kết quả: `outputs/batch/`; Log: `logs/batch.log`

outputs = processor.batch_process(

    "samples/", ├── app_web.py         # Web interface app

    "outputs/baby_faces/",

    prompt="beautiful baby face prediction"├── app_launcher.py    # GUI app selectormkdir -p samples

)

├── batch_processor.py # Batch processing

print(f"Generated {len(outputs)} baby faces")

├── model_downloader.py # Auto-download GGUF model# Copy your ultrasound images to samples/ foldergit clone https://github.com/VuThanhLam124/BabyVis.git

# Check memory usage

memory_info = processor.get_memory_usage()├── samples/           # Input ultrasound images

print(f"VRAM usage: {memory_info}")

├── outputs/           # Generated baby faces```

# Cleanup

processor.cleanup()├── models/           # GGUF model storage (auto-created)

```

└── src/babyvis/      # Core inference codecd BabyVis## Chạy trên CPU (khắc phục lỗi CUDA/cuBLAS)

## 🔧 Configuration Options

```

### Model Variants

- `default`: Full precision model (6-8GB VRAM)3. **Run with your preferred configuration:**

- `quantized`: 4-bit quantized model (3-4GB VRAM) - **Recommended for RTX 3050 Ti**

- `cpu`: CPU-optimized model## 🖥️ App Interfaces



### Performance Tuning```bash- Nếu gặp lỗi kiểu `libcublasLt.so.11`/`CUDA` hoặc chưa cài driver/phụ thuộc CUDA đúng bản:

```bash

# For 4GB VRAM (RTX 3050 Ti)### Desktop App (app_desktop.py)

python main.py --model-variant quantized --cpu-offload --steps 20

- **Native GUI** with Tkinter# For 4GB GPU

# For 8GB+ VRAM  

python main.py --model-variant default --steps 30- **Drag & drop** image selection



# CPU-only mode- **Real-time preview** of input/output./run_4gb.sh# Tạo conda environment  - Tạm thời ép chạy CPU: `CUDA_VISIBLE_DEVICES="" python apps/batch_processor.py` hoặc `FORCE_CPU=1 python apps/batch_processor.py`

python main.py --device cpu --model-variant cpu --steps 15

```- **Progress bar** and status updates



## 📊 Performance Benchmarks- **Save functionality** with file dialog



### RTX 3050 Ti (4GB VRAM)

- **Quantized model**: ~25-35 seconds per image

- **VRAM usage**: ~3.2GB peak### Web Interface (app_web.py)  # For 12GB GPU  conda create -n babyvis python=3.9 -y  - Code có cơ chế tự fallback GPU→CPU nếu lỗi CUDA xảy ra trong lúc suy luận.

- **Quality**: High (comparable to full model)

- **Browser-based** interface using Gradio

### RTX 3060 (12GB VRAM)

- **Full model**: ~15-20 seconds per image  - **Upload images** via web form./run_12gb.sh

- **VRAM usage**: ~6.5GB peak

- **Quality**: Highest- **Live processing** with progress



### CPU Only (Intel i7)- **Download results** directlyconda activate babyvis

- **Processing time**: ~2-4 minutes per image

- **RAM usage**: ~8GB peak- **Responsive design** for mobile/desktop

- **Quality**: Good

# For CPU only

## 🛠️ Troubleshooting

### Batch Processing (batch_processor.py)

### Memory Issues

```bash- **Command-line** interface./run_cpu.sh## Tuỳ chọn mô hình

# Out of VRAM error

python main.py --model-variant quantized --cpu-offload- **Process entire** samples/ folder



# Still running out of memory- **Automatic output** naming```

python main.py --device cpu --model-variant cpu

```- **Progress logging**



### Model Loading Issues# Cài đặt dependencies- Mặc định SD v1.5 + ControlNet Canny (CPU nếu không có GPU).

```bash

# Clear cache and retry## 🤖 Model Information

rm -rf ~/.cache/huggingface/

python main.py --input samples/1.jpeg --output outputs/## 📁 Project Structure

```

- **Model**: Qwen_Image_Edit-Q4_K_M.gguf (7.5GB)

### Performance Issues

```bash- **Source**: QuantStack/Qwen-Image-Edit-GGUFpip install -r requirements.txt- Có thể đổi sang SDXL qua biến môi trường hoặc override ID:

# Install XFormers for faster inference

pip install xformers- **Auto-download**: Model downloads automatically on first run



# Enable compilation (PyTorch 2.0+)```

export TORCH_COMPILE=1

python main.py --input samples/ --output outputs/## ⚙️ Hardware Configurations

```

BabyVis/  - Ví dụ SDXL: 

### CUDA Issues

```bash| Configuration | GPU Layers | VRAM Usage | RAM Usage | Speed |

# Check CUDA installation

python -c "import torch; print(torch.cuda.is_available())"|---------------|------------|------------|-----------|-------|├── run_4gb.sh         # GPU 4GB configuration



# Force CPU mode if CUDA problematic| 4GB GPU       | 20         | ~3.5GB     | ~4GB      | Fast  |

python main.py --device cpu --input samples/ --output outputs/

```| 12GB GPU      | 35         | ~11GB      | ~2GB      | Fastest |├── run_12gb.sh        # GPU 12GB configuration  # Chạy ngay lập tức    - `PREFER_SDXL=1 python apps/gradio_app.py`



## 🔍 Memory Optimization Features| CPU Only      | 0          | 0          | ~8GB      | Slower |



- **CPU Offloading**: Automatically moves models between GPU/CPU├── run_cpu.sh         # CPU configuration

- **Attention Slicing**: Reduces peak memory usage

- **VAE Tiling**: Handles large images efficiently## 🎯 Features

- **Automatic Cleanup**: Frees memory after each inference

- **Memory Monitoring**: Real-time VRAM usage tracking├── batch_processor.py # Main processing script./run_qwen_auto.sh    - Hoặc:

- **Gradient Checkpointing**: Trades compute for memory

- ✅ **Multiple interfaces** (Desktop, Web, Batch)

## 📈 Quality Enhancement Tips

- ✅ **Single GGUF model** (simplified)├── model_downloader.py # Auto-download GGUF model

### Optimal Prompts

```python- ✅ **Auto-download** from HuggingFace

# Generic prompt

"Transform this ultrasound into a realistic baby face"- ✅ **Hardware auto-detection**├── samples/           # Input ultrasound images```      - `BASE_MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0" \\



# Detailed prompt for better results- ✅ **Real-time processing** with progress

"Transform this ultrasound image into a beautiful, realistic newborn baby face with soft features, clear skin, and natural proportions"

- ✅ **3 optimization profiles**├── outputs/           # Generated baby faces

# Ethnicity-specific

"Transform this ultrasound into a realistic Asian baby face with delicate features"- ✅ **Simple configuration**

```

├── models/           # GGUF model storage (auto-created)         CONTROLNET_ID="diffusers/controlnet-canny-sdxl-1.0" \\

### Parameter Tuning

- **Steps**: 20-30 (balance speed/quality)## 📊 Output

- **Guidance Scale**: 7.0-8.5 (higher = more prompt adherence)

- **Image Size**: 512x512 (optimal for memory)└── src/babyvis/      # Core inference code



## 🎯 RoadmapGenerated baby faces will be saved in the `outputs/` directory with filenames like:



- [ ] **Web Interface**: Gradio/Streamlit web interface- `predicted_1.png````## 🎮 Cách sử dụng         python apps/gradio_app.py`

- [ ] **Mobile Support**: Optimized mobile processing

- [ ] **Advanced Features**: Gender prediction, trait analysis  - `predicted_2.png` 

- [ ] **Cloud Integration**: AWS/GCP deployment options

- [ ] **API Server**: REST API for integration- etc.

- [ ] **Real-time Processing**: Live camera feed support



## 📝 License

## 🛠️ Usage Examples## 🤖 Model Information- Các tham số gợi ý:

MIT License - see [LICENSE](LICENSE) file for details.



## 🤝 Contributing

### Desktop App

1. Fork the repository

2. Create feature branch: `git checkout -b feature/amazing-feature`1. Run `./run_app.sh` and choose "Desktop App"

3. Commit changes: `git commit -m 'Add amazing feature'`

4. Push to branch: `git push origin feature/amazing-feature`2. Click "Browse Image" to select ultrasound- **Model**: Qwen_Image_Edit-Q4_K_M.gguf (7.5GB)### Chế độ tự động (Khuyến nghị)  - SD v1.5: steps 40–60, CFG 7–9, control 0.7–1.0

5. Open Pull Request

3. Choose hardware configuration

## 🙏 Acknowledgments

4. Click "Generate Baby Face"- **Source**: QuantStack/Qwen-Image-Edit-GGUF

- **Qwen Team**: For the amazing Qwen-Image-Edit model

- **Hugging Face**: For Diffusers library and model hosting5. Save result when processing complete

- **PyTorch Team**: For the deep learning framework

- **NVIDIA**: For CUDA acceleration support- **Auto-download**: Model downloads automatically on first run```bash  - SDXL: steps 30–40, CFG 5–7, control 0.6–0.9



## 📞 Support### Web Interface



- **Issues**: [GitHub Issues](https://github.com/yourusername/BabyVis/issues)1. Run `./run_app.sh` and choose "Web Interface"

- **Discussions**: [GitHub Discussions](https://github.com/yourusername/BabyVis/discussions)

- **Email**: support@babyvis.ai2. Open browser to http://localhost:7860



---3. Upload ultrasound image## ⚙️ Hardware Configurations./run_qwen_auto.sh    # Tự động chọn backend tốt nhất



**🚀 Transform ultrasounds into beautiful baby faces with professional AI technology!**4. Configure settings



**💡 Optimized for RTX 3050 Ti and 4GB VRAM setups - no compromises on quality!**5. Click "Generate" and download result



### Batch Processing| Configuration | GPU Layers | VRAM Usage | RAM Usage | Speed |```### Dùng Qwen Image Edit (Qwen/Qwen-Image-Edit)

1. Put ultrasound images in `samples/` folder

2. Run `./run_app.sh` and choose "Batch Processing"|---------------|------------|------------|-----------|-------|

3. All images processed automatically

4. Check `outputs/` folder for results| 4GB GPU       | 20         | ~3.5GB     | ~4GB      | Fast  |- Bật dùng Qwen Image Edit thay cho Stable Diffusion + ControlNet:



## 🛠️ Troubleshooting| 12GB GPU      | 35         | ~11GB      | ~2GB      | Fastest |



### Model Download Issues| CPU Only      | 0          | 0          | ~8GB      | Slower |### Chế độ cụ thể  - `USE_QWEN_IMAGE_EDIT=1 python apps/gradio_app.py`

```bash

python3 -c "

from model_downloader import ensure_model

ensure_model()## 🎯 Features```bash  - hoặc trong batch: `USE_QWEN_IMAGE_EDIT=1 python apps/batch_processor.py`

"

```



### Memory Issues- ✅ Single GGUF model (simplified)./run_qwen_gguf.sh         # Force GGUF (tương thích cao)- Yêu cầu cài thêm thư viện: `transformers`, `accelerate`, `sentencepiece`, `safetensors` (đã thêm sẵn trong `requirements.txt`).

- Use CPU mode if GPU memory is insufficient

- Close other applications to free RAM- ✅ Auto-download from HuggingFace

- Try 4GB mode instead of 12GB mode

- ✅ Hardware auto-detection./run_qwen_transformers.sh # Force Transformers (chất lượng cao)- Ghi chú: Mã tải Qwen dùng `trust_remote_code=True` theo hướng dẫn từ Hugging Face. Đầu ra ảnh được decode tự động từ response của model.

### GUI Issues

```bash- ✅ Batch processing

# For desktop app, ensure tkinter is installed

sudo apt-get install python3-tk  # Ubuntu/Debian- ✅ 3 optimization profiles./run_qwen_cpu.sh          # CPU only (máy yếu)



# For web interface, check gradio installation- ✅ Simple configuration

pip install gradio

``````### Dùng bản nhẹ GGUF (QuantStack/Qwen-Image-Edit-GGUF)



### CUDA Issues## 📊 Output

```bash

# Check CUDA availability- Bật backend GGUF (chạy bằng llama.cpp, phù hợp CPU/GPU VRAM ~4GB):

python3 -c "

import subprocessGenerated baby faces will be saved in the `outputs/` directory with filenames like:

try:

    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)- `predicted_1.png`### Python API  - `USE_QWEN_GGUF=1 python apps/gradio_app.py`

    print(result.stdout)

except:- `predicted_2.png` 

    print('CUDA not available, will use CPU')

"- etc.```python  - hoặc `USE_QWEN_GGUF=1 python apps/batch_processor.py`

```



## 📝 License

## 🛠️ Troubleshootingfrom babyvis.inference import generate_predict_auto- Tuỳ chọn tải model:

See LICENSE file for details.



## 🤝 Contributing

### Model Download Issues  - Dùng file cục bộ: đặt `QWEN_GGUF_PATH=/path/to/model.gguf`

This is a simplified GGUF-only version focused on ease of use and performance with multiple interface options.
```bash

python3 -c "# Tạo ảnh dự đoán  - Hoặc để tự tải từ HF: `QWEN_GGUF_REPO=QuantStack/Qwen-Image-Edit-GGUF` và (tuỳ chọn) `QWEN_GGUF_FILENAME=qwen-image-edit-q4_k_m.gguf`

from model_downloader import ModelDownloader

downloader = ModelDownloader()generate_predict_auto(- Tinh chỉnh hiệu năng/thông số thấp VRAM:

downloader.download_if_needed()

"    input_path="samples/ultrasound.jpg",  - `QWEN_N_GPU_LAYERS=12` (mặc định, 4GB VRAM nên để 8–16; đặt 0 để chạy CPU-only)

```

    output_path="output/baby.png",  - `QWEN_N_THREADS=<số luồng CPU>`; `QWEN_N_CTX=2048`

### Memory Issues

- Use CPU mode if GPU memory is insufficient    backend="auto",  # hoặc "qwen", "qwen_gguf"- Yêu cầu cài `llama-cpp-python` (đã có trong `requirements.txt`). Cần bản llama.cpp hỗ trợ multi-modal (VL). Nếu backend GGUF chỉ trả về văn bản thay vì ảnh, code sẽ tự động dùng văn bản đó làm prompt tăng cường cho ControlNet để vẫn tạo được ảnh.

- Close other applications to free RAM

- Try 4GB mode instead of 12GB mode    ethnicity="mixed ethnicity"



### CUDA Issues)## Cấu trúc thư mục (đã cơ cấu lại)

```bash

# Check CUDA availability```- `src/babyvis/`: mã nguồn chính (inference, model_utils)

python3 -c "

import subprocess- `apps/`: điểm chạy

try:

    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)## 🔧 Cấu hình  - `gradio_app.py`: giao diện Gradio

    print(result.stdout)

except:  - `batch_processor.py`: xử lý theo lô

    print('CUDA not available, will use CPU')

"### Environment Variables- `web/`: giao diện web tĩnh (index.html, style.css, app.js) — đã loại bỏ toàn bộ icon SVG

```

```bash- `outputs/`: ảnh kết quả (`gradio/`, `batch/`, `tmp/`)

## 📝 License

# Backend selection- `samples/`: ảnh mẫu

See LICENSE file for details.

export QWEN_BACKEND=auto          # auto, qwen, qwen_gguf- `config/`: cấu hình batch (`config/batch_config.json`)

## 🤝 Contributing

export USE_QWEN_IMAGE_EDIT=1      # Force transformers- `data/`: dữ liệu vào (vd. `data/image_list.txt`)

This is a simplified GGUF-only version focused on ease of use and performance.
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