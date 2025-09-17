#!/bin/bash
# BabyVis Startup Script

echo "🍼 Starting BabyVis - Ultrasound to Baby Face Converter"
echo "================================================="

# Navigate to project directory
cd "$(dirname "$0")"

# Activate conda environment
echo "📦 Activating conda environment..."
source /home/ubuntu/anaconda3/bin/activate
conda activate babyvis

# Install Python dependencies (both web and core), ensures Diffusers is available
echo "📦 Ensuring Python dependencies..."
pip -q install -r requirements.txt || true
pip -q install -r web_requirements.txt || true

# Backend selection: diffusers (default) or comfyui
BACKEND=${BABYVIS_BACKEND:-diffusers}
echo "🧠 Selected backend: $BACKEND"

# Check Qwen model only if we intend to use ComfyUI backend
if [ "$BACKEND" = "comfyui" ]; then
    DIFFUSERS_DIR="ComfyUI/models/diffusers/${COMFYUI_DIFFUSERS:-Qwen-Image-Edit}"
    GGUF_FILE="ComfyUI/models/checkpoints/Qwen_Image_Edit-Q4_K_M.gguf"
    if [ -f "$DIFFUSERS_DIR/model_index.json" ]; then
        echo "✅ Found diffusers model at $DIFFUSERS_DIR"
    elif [ -f "$GGUF_FILE" ]; then
        echo "✅ Found GGUF model at $GGUF_FILE"
    else
        echo "⚠️  Qwen model not found in diffusers ($DIFFUSERS_DIR) or GGUF ($GGUF_FILE)"
        echo "    ComfyUI backend will start but generation may fail until you install one."
    fi
fi

echo "🤖 Using AI via $BACKEND backend"

# Start web application directly (no need for ComfyUI server)
echo "🚀 Starting web application on http://0.0.0.0:8000"
echo "📱 Access the app at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================="

BABYVIS_BACKEND="$BACKEND" python3 web_app.py
