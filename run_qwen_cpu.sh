#!/bin/bash
# run_qwen_cpu.sh - Chế độ CPU cho máy yếu hoặc không có GPU

echo "💻 BabyVis - Qwen CPU Mode"
echo "=========================="

# Detect system info
echo "🖥️ System Information:"
echo "   CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "   Cores: $(nproc)"
echo "   Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"

# Force CPU mode
export FORCE_CPU=1
export CUDA_VISIBLE_DEVICES=""
export QWEN_BACKEND=qwen_gguf
export USE_QWEN_GGUF=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# CPU-optimized settings
export QWEN_N_GPU_LAYERS=0
export QWEN_N_THREADS=$(nproc)
export QWEN_N_CTX=2048

echo ""
echo "🚀 Starting BabyVis in CPU mode..."
echo "   Backend: Qwen GGUF (CPU only)"
echo "   Threads: $QWEN_N_THREADS"
echo "   Context: $QWEN_N_CTX"
echo "   GPU Layers: 0"
echo ""

# Run batch processor
python3 apps/batch_processor.py

echo ""
echo "✅ Qwen CPU processing completed!"