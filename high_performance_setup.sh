#!/bin/bash
# high_performance_setup.sh - Setup for maximum Qwen Image Edit performance

echo "🚀 Setting up BabyVis for HIGH PERFORMANCE with Qwen Image Edit..."

# Memory and performance settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export HF_HUB_CACHE=/tmp/hf_cache  # Use faster temp storage if available

# Qwen Image Edit performance settings
export USE_QWEN_GGUF=1  # Force GGUF backend
export QWEN_N_GPU_LAYERS=35  # Aggressive GPU usage
export QWEN_N_CTX=4096  # Large context for quality
export QWEN_N_THREADS=8  # Optimize threads
export QWEN_CHAT_FORMAT=qwen2_vl

# Model preferences (performance-focused)
export QWEN_GGUF_FILENAME=Qwen_Image_Edit-Q5_K_M.gguf  # Best quality/speed ratio

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "✅ High performance environment configured!"
echo "💡 Available performance models:"
echo "   - Q5_K_M: Best quality/speed balance (recommended)"
echo "   - Q4_K_M: Faster, good quality"
echo "   - Q6_K: Higher quality, slower"
echo "   - Q8_0: Highest quality, slowest"

echo ""
echo "🎯 To run with maximum performance:"
echo "   source high_performance_setup.sh"
echo "   python3 apps/batch_processor.py"

echo ""
echo "⚙️ Current GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "   No NVIDIA GPU detected"