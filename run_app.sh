#!/bin/bash
# BabyVis - App Launcher

echo "🚀 BabyVis - App Interface"
echo "=========================="

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
from model_downloader import ensure_model
ensure_model()
"

echo ""
echo "🎯 Choose your interface:"
echo "1. 🖥️  Desktop App (recommended)"
echo "2. 🌐 Web Interface" 
echo "3. 📦 Batch Processing"
echo "4. 🚀 App Launcher (GUI selector)"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🖥️ Starting Desktop App..."
        python3 app_desktop.py
        ;;
    2)
        echo "🌐 Starting Web Interface..."
        python3 app_web.py
        ;;
    3)
        echo "📦 Starting Batch Processing..."
        python3 batch_processor.py
        ;;
    4)
        echo "🚀 Starting App Launcher..."
        python3 app_launcher.py
        ;;
    *)
        echo "❌ Invalid choice. Starting Desktop App..."
        python3 app_desktop.py
        ;;
esac