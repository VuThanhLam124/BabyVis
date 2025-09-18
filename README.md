# 🍼 BabyVis v2.0 - AI Baby Face Generator

Transform ultrasound images into beautiful baby faces using state-of-the-art **Qwen-Image-Edit** model and **Diffusers** library.

## ✨ Features

- 🤖 **Advanced AI Model**: Powered by Qwen/Qwen-Image-Edit via Diffusers
- 🖼️ **Image-to-Image Transformation**: Convert ultrasound scans to realistic baby portraits  
- 🌐 **Modern Web Interface**: Beautiful, responsive UI with drag & drop
- 📱 **Mobile Friendly**: Works on desktop and mobile devices
- 🎛️ **Customizable Settings**: Adjust quality levels, steps, and transformation strength
- ⚡ **GPU Optimized**: Efficient memory usage and GPU acceleration
- � **Image Validation**: Automatic ultrasound image quality assessment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- NVIDIA GPU with 4GB+ VRAM (recommended)
- 8GB+ RAM
- 5GB+ free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/VuThanhLam124/BabyVis.git
cd BabyVis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python main.py --mode web
```

4. **Open your browser**
Navigate to: `http://localhost:8000`

## 💻 Usage Modes

### Web Interface (Recommended)
```bash
# Start web application
python main.py --mode web

# Custom host and port
python main.py --mode web --host 0.0.0.0 --port 8080

# Development mode with auto-reload
python main.py --mode web --reload
```

### Command Line Interface
```bash
# Interactive CLI mode
python main.py --mode cli
```

### Test Mode
```bash
# Test all components
python main.py --mode test

# Check dependencies only
python main.py --check-deps
```

## 🏗️ Architecture

```
BabyVis/
├── main.py                    # Main application entry point
├── app.py                     # FastAPI web application
├── qwen_image_edit_model.py   # Core AI model handler
├── image_utils.py             # Image processing utilities
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
├── samples/                   # Example ultrasound images
├── uploads/                   # Temporary uploads (auto-created)
├── outputs/                   # Generated images (auto-created)
└── static/                    # Static web assets (auto-created)
```

## 🎛️ Configuration

### Quality Levels
- **Base**: Fast generation, good quality (15-20 steps)
- **Enhanced**: Balanced speed/quality (25-35 steps) - Recommended
- **Premium**: Best quality, slower (40-50 steps)

### Parameters
- **Steps**: Number of denoising steps (15-50)
- **Strength**: Transformation intensity (0.3-1.0)
- **Guidance Scale**: How closely to follow prompts (1.0-20.0)

### Environment Variables
```bash
# Force CPU usage (if no GPU)
export CUDA_VISIBLE_DEVICES=""

# Custom model (if available)
export QWEN_MODEL_ID="your-custom-model-id"
```

## 📊 Performance

| Quality Level | Steps | Time (GPU) | Time (CPU) | VRAM Usage |
|---------------|-------|------------|------------|------------|
| Base          | 15    | ~20s       | ~2min      | 2-3GB      |
| Enhanced      | 30    | ~40s       | ~4min      | 3-4GB      |
| Premium       | 50    | ~70s       | ~7min      | 4-5GB      |

*Times are approximate and depend on hardware*

## 🔧 API Reference

### REST API Endpoints

- `GET /` - Web interface
- `POST /validate` - Validate ultrasound image
- `POST /generate` - Generate baby face
- `GET /download/{filename}` - Download generated image
- `GET /status` - System status
- `GET /health` - Health check

### Python API
```python
from qwen_image_edit_model import QwenImageEditModel
from PIL import Image

# Initialize model
model = QwenImageEditModel()

# Load ultrasound image
image = Image.open("ultrasound.jpg")

# Generate baby face
success, baby_image, message = model.generate_baby_face(
    image,
    quality_level="enhanced",
    num_inference_steps=30,
    strength=0.8
)

if success:
    baby_image.save("baby_face.png")
```

## 🛠️ Development

### Setup Development Environment
```bash
# Install with development dependencies
pip install -r requirements.txt

# Run in development mode
python main.py --mode web --reload

# Run tests
python main.py --mode test
```

### Adding Custom Models
```python
# In qwen_image_edit_model.py
def __init__(self, model_id: str = "your-model-id"):
    # Custom model initialization
```

## 🧪 Testing

The application includes comprehensive testing:

```bash
# Test all components
python main.py --mode test

# Test individual modules
python qwen_image_edit_model.py
python image_utils.py
```

## 📋 System Requirements

| Component     | Minimum       | Recommended   |
|---------------|---------------|---------------|
| **Python**   | 3.8           | 3.9+          |
| **GPU**       | 4GB VRAM      | 8GB+ VRAM     |
| **RAM**       | 8GB           | 16GB+         |
| **Storage**   | 5GB free      | 10GB+ SSD     |
| **CPU**       | 4 cores       | 8+ cores      |

## 🚨 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Check dependencies
python main.py --check-deps

# Try CPU mode
export CUDA_VISIBLE_DEVICES=""
python main.py --mode test
```

**Out of Memory**
- Reduce steps to 15-20
- Use "base" quality level
- Close other applications
- Try CPU mode

**Poor Image Quality**
- Use higher quality ultrasound images
- Increase steps to 40-50
- Use "premium" quality level
- Adjust strength parameter

## 🔬 Model Details

- **Base Model**: Qwen/Qwen-Image-Edit (Alibaba)
- **Framework**: Diffusers + Transformers
- **Fallback Models**: Stable Diffusion v1.5, v2.1
- **Architecture**: Conditional diffusion model
- **Input**: 512x512 RGB images
- **Output**: High-quality baby portraits

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Alibaba DAMO Academy**: Qwen-Image-Edit model
- **Hugging Face**: Diffusers library and model hosting
- **Stability AI**: Stable Diffusion fallback models
- **FastAPI**: Modern web framework
- **Bootstrap**: Beautiful UI components

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/VuThanhLam124/BabyVis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VuThanhLam124/BabyVis/discussions)
- **Email**: [Contact](mailto:vuthanhlam124@example.com)

---

<div align="center">

**Made with ❤️ for expecting parents worldwide**

*Transform your ultrasound memories into beautiful baby portraits*

</div>
