#!/bin/bash
# run_qwen_auto.sh - Auto-detect và sử dụng backend Qwen tốt nhất

echo "🤖 BabyVis - Auto Qwen Detection"
echo "================================"

# Detect system info
echo "🖥️ System Information:"
echo "   CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    echo "   VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
else
    echo "   GPU: Not detected or no NVIDIA GPU"
fi

# Auto-detect best backend
export QWEN_BACKEND=auto
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo ""
echo "🚀 Starting BabyVis with Auto Qwen Backend..."
echo "   Backend: Auto-detection (GGUF → Transformers fallback)"
echo "   Process: Adaptive VRAM optimization"
echo ""

# Run batch processor
python3 apps/batch_processor.py

echo ""
echo "✅ Auto Qwen processing completed!"