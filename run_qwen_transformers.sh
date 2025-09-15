#!/bin/bash
# run_qwen_transformers.sh - Force sử dụng Qwen Image Edit (transformers)

echo "🧠 BabyVis - Qwen Transformers Mode"
echo "==================================="

# Detect system info
echo "🖥️ System Information:"
echo "   CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    echo "   VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
else
    echo "   GPU: Not detected or no NVIDIA GPU"
fi

# Force transformers backend
export QWEN_BACKEND=qwen
export USE_QWEN_IMAGE_EDIT=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Performance settings
export QWEN_4BIT=true
export QWEN_FLASH_ATTN=true

echo ""
echo "🚀 Starting BabyVis with Qwen Transformers..."
echo "   Backend: Qwen Image Edit (transformers)"
echo "   Model: Qwen/Qwen-Image-Edit"
echo "   Optimizations: 4-bit quantization, Flash Attention"
echo ""

# Run batch processor
python3 apps/batch_processor.py

echo ""
echo "✅ Qwen Transformers processing completed!"