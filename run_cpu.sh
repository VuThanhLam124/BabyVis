#!/bin/bash
# BabyVis - CPU Only Configuration

echo "🚀 BabyVis - CPU Only Setup"
echo "==========================="

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

# Set CPU configuration
export MODEL_TYPE="cpu"
export CUDA_VISIBLE_DEVICES=""

echo "🖥️ CPU Mode: All processing on CPU"
echo "💾 RAM Usage: ~8GB"
echo "⏱️ Processing time: Slower but works without GPU"
echo ""

# Run batch processor
echo "🤖 Starting BabyVis with CPU config..."
python3 batch_processor.py

echo ""
echo "✅ Processing completed!"
echo "📂 Check 'outputs' directory for results"