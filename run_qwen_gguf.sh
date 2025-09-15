#!/bin/bash
# run_qwen_gguf.sh - Force sử dụng Qwen GGUF (llama.cpp)

echo "🔧 BabyVis - Qwen GGUF Mode"
echo "==========================="

# Detect system info
echo "🖥️ System Information:"
echo "   CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    echo "   VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
else
    echo "   GPU: Not detected or no NVIDIA GPU"
fi

# Force GGUF backend
export QWEN_BACKEND=qwen_gguf
export USE_QWEN_GGUF=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Performance settings for GGUF
export QWEN_N_THREADS=$(nproc)
export QWEN_N_CTX=4096

echo ""
echo "🚀 Starting BabyVis with Qwen GGUF..."
echo "   Backend: Qwen GGUF (llama.cpp)"
echo "   Threads: $QWEN_N_THREADS"
echo "   Context: $QWEN_N_CTX"
echo "   Repo: QuantStack/Qwen-Image-Edit-GGUF"
echo ""

# Run batch processor
python3 apps/batch_processor.py

echo ""
echo "✅ Qwen GGUF processing completed!"