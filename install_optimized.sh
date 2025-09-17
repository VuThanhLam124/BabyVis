#!/bin/bash
# install_optimized.sh - Installation script tối ưu cho RTX 3050 Ti (4GB VRAM)

echo "🚀 Installing BabyVis optimized for 4GB VRAM..."

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "❌ No NVIDIA GPU detected, will use CPU-only installation"
fi

echo "📦 Installing PyTorch with CUDA support (tested compatible versions)..."
# Install PyTorch 2.1.0 with CUDA 11.8 (tested working)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo "🤗 Installing Diffusers and Transformers (compatible versions)..."
pip install diffusers==0.18.0 transformers==4.34.0 accelerate==0.20.3

echo "🖼️ Installing image processing libraries..."
pip install Pillow opencv-python "numpy<2" scipy

echo "⚡ Installing utilities..."
pip install huggingface-hub==0.16.4 safetensors psutil

echo "🌐 Installing Gradio for web interface..."
pip install gradio

echo "🔧 Installing development tools..."
pip install pytest black flake8

echo "⚠️ Note: xformers is skipped due to compatibility issues with PyTorch 2.1.0"

echo "✨ Installation complete!"

# Test installation
echo "🧪 Testing installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'VRAM: {vram:.1f}GB')

try:
    from diffusers import StableDiffusionPipeline
    print(f'✅ Diffusers available')
except ImportError:
    print('❌ Diffusers not installed properly')

try:
    from PIL import Image
    print('✅ Pillow available')
except ImportError:
    print('❌ Pillow not available')
"

echo "🎉 Setup complete! Run with:"
echo "python main.py --input samples/ --output outputs/baby_faces/"