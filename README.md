# 🍼 BabyVis - AI-Powered Baby Face Generator

Convert ultrasound images into beautiful baby faces using state-of-the-art AI technology powered by **ComfyUI** and **Qwen-Image-Edit**.

## ✨ Features

- 🖼️ **Image-to-Image Transformation**: Convert ultrasound scans to realistic baby portraits
- 🤖 **Advanced AI Models**: Powered by Qwen-Image-Edit quantized model (12GB)
- 🌐 **Modern Web Interface**: Beautiful, responsive web UI with drag & drop
- ⚡ **GPU Optimized**: Optimized for RTX 3050 Ti (4GB VRAM)
- 🎛️ **Customizable Settings**: Adjust quality, transformation strength, and more
- 📱 **Mobile Friendly**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Ubuntu/Linux system
- NVIDIA GPU with 4GB+ VRAM (RTX 3050 Ti recommended)
- Conda/Anaconda installed
- 15GB+ free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/VuThanhLam124/BabyVis.git
cd BabyVis
```

2. **Setup environment**
```bash
conda create -n babyvis python=3.9
conda activate babyvis
cd ComfyUI && pip install -r requirements.txt
cd .. && pip install -r requirements.txt && pip install -r web_requirements.txt
```

3. **Verify model installation**
```bash
ls -la ComfyUI/models/checkpoints/Qwen_Image_Edit-Q4_K_M.gguf
# Should show ~12GB file
```

4. **Start the application**
```bash
./start.sh
```

5. **Open your browser**
Navigate to: `http://localhost:8000`

## 💻 Usage

1. **Upload Image**: Drag & drop or click to upload your ultrasound image
2. **Adjust Settings**: 
   - Quality Steps: 15-30 (higher = better quality, slower)
   - Transformation Strength: 0.5-1.0 (higher = more transformation)
3. **Generate**: Click "Generate Baby Face" and wait 30-60 seconds
4. **Download**: Save your beautiful baby portrait

## 🏗️ Architecture

```
BabyVis/
├── web_app.py              # FastAPI web interface
├── ComfyUI/                # ComfyUI framework
│   ├── models/checkpoints/  # AI models (Qwen-Image-Edit)
│   └── workflows/          # Image processing workflows
├── samples/                # Example ultrasound images
├── uploads/                # Temporary upload storage
├── outputs/                # Generated baby faces
└── start.sh               # Startup script
```

## 🔀 Backends

- Default backend: `diffusers` (runs Stable Diffusion img2img via `diffusers`).
- Optional backend: `comfyui` (queues a ComfyUI workflow and downloads the result).

Switch backend via env var:

```bash
BABYVIS_BACKEND=comfyui ./start.sh
```

Notes:
- Diffusers backend tries `Qwen/Qwen-Image-Edit` first (if available), then falls back to `runwayml/stable-diffusion-v1-5`.
- ComfyUI backend expects the Qwen checkpoint (GGUF) at `ComfyUI/models/checkpoints/Qwen_Image_Edit-Q4_K_M.gguf`.

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 4GB VRAM | RTX 3050 Ti+ |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 15GB free | 20GB+ SSD |
| **CPU** | 4 cores | 8+ cores |

## 🎨 Model Details

- **Base Model**: Qwen-Image-Edit (Alibaba)
- **Format**: GGUF Quantized (Q4_K_M)
- **Size**: 12.17GB
- **Optimization**: 4GB VRAM compatible
- **Quality**: Professional baby portrait generation

## 📈 Performance

| Setting | Quality | Speed | VRAM Usage |
|---------|---------|--------|------------|
| Steps: 15 | Good | 30s | 3.2GB |
| Steps: 25 | High | 45s | 3.5GB |
| Steps: 30 | Excellent | 60s | 3.8GB |

## 🛠️ Development

### API Endpoints
- `GET /`: Web interface
- `POST /generate`: Generate baby face from ultrasound
- `GET /download/{filename}`: Download generated images
- `GET /status`: System health check

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ComfyUI**: Node-based UI framework
- **Qwen-Image-Edit**: Image editing model by Alibaba
- **Hugging Face**: Model hosting platform
- **FastAPI**: Modern web framework

---

Made with ❤️ for expecting parents worldwide
