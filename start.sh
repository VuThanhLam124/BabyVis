#!/bin/bash#!/bin/bash

# BabyVis Startup Script

# BabyVis v2.0 - Quick Start Script

# Ultrasound to Baby Face Generator using Qwen-Image-Edit & Diffusersecho "🍼 Starting BabyVis - Ultrasound to Baby Face Converter"

echo "================================================="

echo "🍼 BabyVis v2.0 - AI Baby Face Generator"

echo "========================================="# Navigate to project directory

cd "$(dirname "$0")"

# Check if Python is available

if ! command -v python &> /dev/null; then# Activate conda environment

    echo "❌ Python not found. Please install Python 3.8+ first."echo "📦 Activating conda environment..."

    exit 1source /home/ubuntu/anaconda3/bin/activate

ficonda activate babyvis



# Check Python version# Install Python dependencies (both web and core), ensures Diffusers is available

python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")echo "📦 Ensuring Python dependencies..."

echo "🐍 Python version: $python_version"pip -q install -r requirements.txt || true

pip -q install -r web_requirements.txt || true

# Check if virtual environment exists

if [ ! -d "venv" ]; then# Backend selection: diffusers (default) or comfyui

    echo "📦 Creating virtual environment..."BACKEND=${BABYVIS_BACKEND:-diffusers}

    python -m venv venvecho "🧠 Selected backend: $BACKEND"

fi

# Check Qwen model only if we intend to use ComfyUI backend

# Activate virtual environmentif [ "$BACKEND" = "comfyui" ]; then

echo "🔄 Activating virtual environment..."    DIFFUSERS_DIR="ComfyUI/models/diffusers/${COMFYUI_DIFFUSERS:-Qwen-Image-Edit}"

source venv/bin/activate    GGUF_FILE="ComfyUI/models/checkpoints/Qwen_Image_Edit-Q4_K_M.gguf"

    if [ -f "$DIFFUSERS_DIR/model_index.json" ]; then

# Check if requirements are installed        echo "✅ Found diffusers model at $DIFFUSERS_DIR"

if [ ! -f ".requirements_installed" ]; then    elif [ -f "$GGUF_FILE" ]; then

    echo "📥 Installing dependencies..."        echo "✅ Found GGUF model at $GGUF_FILE"

    pip install --upgrade pip    else

    pip install -r requirements.txt        echo "⚠️  Qwen model not found in diffusers ($DIFFUSERS_DIR) or GGUF ($GGUF_FILE)"

    touch .requirements_installed        echo "    ComfyUI backend will start but generation may fail until you install one."

    echo "✅ Dependencies installed successfully!"    fi

elsefi

    echo "✅ Dependencies already installed"

fiecho "🤖 Using AI via $BACKEND backend"



# Check dependencies# Start web application directly (no need for ComfyUI server)

echo "🔍 Checking dependencies..."echo "🚀 Starting web application on http://0.0.0.0:8000"

python main.py --check-depsecho "📱 Access the app at: http://localhost:8000"

echo ""

if [ $? -ne 0 ]; thenecho "Press Ctrl+C to stop the server"

    echo "❌ Dependency check failed. Please check the installation."echo "================================================="

    exit 1

fiBABYVIS_BACKEND="$BACKEND" python3 web_app.py


# Run application
echo "🚀 Starting BabyVis application..."
echo "🌐 Web interface will be available at: http://localhost:8000"
echo "📚 API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Start the web application
python main.py --mode web --host 0.0.0.0 --port 8000