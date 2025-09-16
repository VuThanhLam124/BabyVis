#!/bin/bash
# BabyVis - GPU 4GB VRAM Configuration

echo "🚀 BabyVis - GPU 4GB VRAM Setup"
echo "================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download model if needed
echo "⬇️ Checking model..."
python3 -c "
from model_downloader import ModelDownloader
downloader = ModelDownloader()
downloader.download_if_needed()
"

# Set GPU configuration for 4GB
export MODEL_TYPE="4gb"
export CUDA_VISIBLE_DEVICES=0

echo "🎮 GPU 4GB Mode: 20 layers on GPU"
echo "💾 VRAM Usage: ~3.5GB"
echo ""

# Run batch processor
echo "🤖 Starting BabyVis with 4GB GPU config..."
python3 batch_processor.py

echo ""
echo "✅ Processing completed!"
echo "📂 Check 'outputs' directory for results"