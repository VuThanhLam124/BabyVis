#!/usr/bin/env python3
"""
BabyVis Model Utils - Simplified for GGUF only
"""

import os
import torch
from pathlib import Path
from typing import Optional


def detect_device():
    """Phát hiện device tốt nhất có sẵn"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def detect_vram():
    """Phát hiện VRAM có sẵn (GB)"""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
        return 0.0
    except Exception:
        return 0.0


def get_model_config(vram_gb: float, device: str = "auto"):
    """
    Lấy cấu hình model dựa trên VRAM và device
    
    Args:
        vram_gb: VRAM available in GB
        device: Target device ("auto", "cuda", "cpu")
    
    Returns:
        Dict with configuration
    """
    if device == "auto":
        device = detect_device()
    
    # CPU configuration
    if device == "cpu" or vram_gb == 0:
        return {
            "device": "cpu",
            "n_gpu_layers": 0,
            "n_threads": max(1, os.cpu_count() - 1),
            "n_ctx": 2048,
            "profile": "cpu"
        }
    
    # GPU configurations based on VRAM
    if vram_gb >= 12:  # High-end GPU
        return {
            "device": "cuda", 
            "n_gpu_layers": 35,
            "n_threads": 4,
            "n_ctx": 4096,
            "profile": "high_end"
        }
    elif vram_gb >= 4:  # Mid-range GPU
        return {
            "device": "cuda",
            "n_gpu_layers": 20,
            "n_threads": 6,
            "n_ctx": 3072,
            "profile": "mid_range"
        }
    else:  # Low VRAM
        return {
            "device": "cuda",
            "n_gpu_layers": 10,
            "n_threads": 8,
            "n_ctx": 2048,
            "profile": "low_vram"
        }


def get_model_path():
    """Lấy đường dẫn model GGUF"""
    return Path("models") / "Qwen_Image_Edit-Q4_K_M.gguf"


def ensure_model():
    """Đảm bảo model có sẵn (auto-download nếu cần)"""
    model_path = get_model_path()
    
    if model_path.exists():
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 6000:  # Valid size check
            return str(model_path)
    
    # Import and run downloader
    print("🤖 Model not found, starting download...")
    try:
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from model_downloader import ensure_model as download_model
        if download_model():
            return str(model_path)
        else:
            raise RuntimeError("Failed to download model")
    except ImportError:
        raise RuntimeError("model_downloader.py not found")


def load_qwen_model(config: Optional[dict] = None):
    """
    Load Qwen GGUF model với llama-cpp-python
    
    Args:
        config: Model configuration dict
        
    Returns:
        Loaded llama model
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError("llama-cpp-python is required. Install with: pip install llama-cpp-python")
    
    # Ensure model is available
    model_path = ensure_model()
    
    # Get config if not provided
    if config is None:
        vram = detect_vram()
        config = get_model_config(vram)
    
    print(f"🚀 Loading Qwen model...")
    print(f"   Model: {model_path}")
    print(f"   Profile: {config['profile']}")
    print(f"   Device: {config['device']}")
    print(f"   GPU Layers: {config['n_gpu_layers']}")
    print(f"   Context: {config['n_ctx']}")
    
    try:
        model = Llama(
            model_path=model_path,
            n_gpu_layers=config['n_gpu_layers'],
            n_threads=config['n_threads'],
            n_ctx=config['n_ctx'],
            chat_format="qwen2_vl",  # Qwen vision format
            verbose=False,
            use_mmap=True,
            use_mlock=False
        )
        
        print(f"✅ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        
        # CPU fallback
        if config['device'] != 'cpu':
            print("🔄 Trying CPU fallback...")
            cpu_config = get_model_config(0, "cpu")
            return load_qwen_model(cpu_config)
        else:
            raise


def create_model_auto():
    """Tạo model với auto-detection"""
    vram = detect_vram()
    device = detect_device()
    config = get_model_config(vram, device)
    
    print(f"🔍 Auto-detection:")
    print(f"   VRAM: {vram:.1f}GB")
    print(f"   Device: {device}")
    print(f"   Profile: {config['profile']}")
    
    return load_qwen_model(config)


# Profile shortcuts
def create_model_4gb():
    """Tạo model cho GPU 4GB VRAM"""
    config = get_model_config(4.0, "cuda")
    return load_qwen_model(config)


def create_model_12gb():
    """Tạo model cho GPU 12GB VRAM"""
    config = get_model_config(12.0, "cuda")
    return load_qwen_model(config)


def create_model_cpu():
    """Tạo model cho CPU"""
    config = get_model_config(0.0, "cpu")
    return load_qwen_model(config)