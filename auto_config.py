#!/usr/bin/env python3
"""
Auto-detect VRAM and configure optimal settings for different environments
"""
import os
import sys
import torch

def detect_vram_and_configure():
    """Detect available VRAM and set optimal configuration"""
    
    try:
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            total_vram_gb = gpu_props.total_memory / (1024**3)
            
            # Clear cache to get accurate free memory
            torch.cuda.empty_cache()
            
            print(f"🖥️ Detected GPU: {gpu_props.name}")
            print(f"💾 Total VRAM: {total_vram_gb:.1f}GB")
            
            # Configure based on available VRAM
            if total_vram_gb >= 9:  # Server environment (10GB+)
                return configure_server_environment(total_vram_gb)
            elif total_vram_gb >= 3:  # Personal computer (4GB)
                return configure_personal_environment(total_vram_gb)
            else:  # Very limited VRAM
                return configure_minimal_environment(total_vram_gb)
                
        else:
            print("⚠️ No CUDA GPU detected - using CPU configuration")
            return configure_cpu_environment()
            
    except Exception as e:
        print(f"❌ Error detecting GPU: {e}")
        return configure_cpu_environment()

def configure_server_environment(vram_gb):
    """Configure for server environment with 10GB+ VRAM"""
    print(f"🚀 Configuring for SERVER environment ({vram_gb:.1f}GB VRAM)")
    
    config = {
        # Use direct Qwen Image Edit for best quality
        'USE_QWEN_IMAGE_EDIT': '1',
        'USE_QWEN_GGUF': '0',
        
        # High performance settings
        'QWEN_4BIT': 'false',  # Use full precision for best quality
        'QWEN_8BIT': 'false',
        'QWEN_FLASH_ATTN': 'true',  # Enable flash attention
        'QWEN_DEVICE_MAP': 'auto',
        
        # GPU optimizations
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'CUDA_VISIBLE_DEVICES': '0',
        
        # Model cache
        'HF_CACHE_DIR': '/tmp/hf_cache_server',
        
        # Inference settings
        'QWEN_BATCH_SIZE': '4',  # Can handle larger batches
        'QWEN_MAX_LENGTH': '2048',
        
        'profile': 'server'
    }
    
    print("✅ Server configuration:")
    print("   - Full precision Qwen Image Edit")
    print("   - Flash Attention enabled")
    print("   - Large batch processing")
    print("   - Maximum quality settings")
    
    return config

def configure_personal_environment(vram_gb):
    """Configure for personal computer with 4GB VRAM"""
    print(f"💻 Configuring for PERSONAL computer ({vram_gb:.1f}GB VRAM)")
    
    config = {
        # Use 4-bit quantized model for efficiency
        'USE_QWEN_IMAGE_EDIT': '1',
        'USE_QWEN_GGUF': '0',
        
        # Memory efficient settings
        'QWEN_4BIT': 'true',  # Essential for 4GB VRAM
        'QWEN_8BIT': 'false',
        'QWEN_FLASH_ATTN': 'false',  # May cause OOM on limited VRAM
        'QWEN_DEVICE_MAP': 'auto',
        
        # Conservative GPU settings
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        'CUDA_VISIBLE_DEVICES': '0',
        
        # Model cache
        'HF_CACHE_DIR': '/tmp/hf_cache_personal',
        
        # Conservative inference settings
        'QWEN_BATCH_SIZE': '1',  # Process one at a time
        'QWEN_MAX_LENGTH': '1024',
        
        'profile': 'personal'
    }
    
    print("✅ Personal computer configuration:")
    print("   - 4-bit quantized Qwen Image Edit")
    print("   - Memory-efficient processing")
    print("   - Single image batching")
    print("   - Optimized for 4GB VRAM")
    
    return config

def configure_minimal_environment(vram_gb):
    """Configure for very limited VRAM (<3GB)"""
    print(f"⚡ Configuring for MINIMAL VRAM ({vram_gb:.1f}GB)")
    
    config = {
        # Use GGUF for maximum efficiency
        'USE_QWEN_IMAGE_EDIT': '0',
        'USE_QWEN_GGUF': '1',
        
        # Minimal GPU usage
        'QWEN_N_GPU_LAYERS': '8',  # Very few layers on GPU
        'QWEN_N_CTX': '2048',
        'QWEN_N_THREADS': '4',
        
        # Conservative settings
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:64',
        'CUDA_VISIBLE_DEVICES': '0',
        
        'profile': 'minimal'
    }
    
    print("✅ Minimal VRAM configuration:")
    print("   - GGUF quantized model")
    print("   - Minimal GPU layers")
    print("   - CPU/GPU hybrid processing")
    
    return config

def configure_cpu_environment():
    """Configure for CPU-only processing"""
    print("🖥️ Configuring for CPU-only processing")
    
    config = {
        # Force CPU mode
        'FORCE_CPU': '1',
        'USE_QWEN_IMAGE_EDIT': '0',
        'USE_QWEN_GGUF': '0',
        
        # Use ControlNet fallback
        'DEVICE': 'cpu',
        'CUDA_VISIBLE_DEVICES': '-1',
        
        'profile': 'cpu'
    }
    
    print("✅ CPU configuration:")
    print("   - ControlNet stable diffusion")
    print("   - CPU-optimized processing")
    
    return config

def apply_environment_config(config):
    """Apply configuration to environment variables"""
    print(f"\n🔧 Applying {config['profile']} configuration...")
    
    for key, value in config.items():
        if key != 'profile':
            os.environ[key] = value
            print(f"   {key}={value}")
    
    print("✅ Environment configured!")
    return config['profile']

def get_recommended_command(profile):
    """Get recommended command for the current profile"""
    commands = {
        'server': [
            "# Server environment (10GB+ VRAM) - Maximum Performance",
            "export USE_QWEN_IMAGE_EDIT=1",
            "export QWEN_4BIT=false",
            "export QWEN_FLASH_ATTN=true",
            "python3 apps/batch_processor.py"
        ],
        'personal': [
            "# Personal computer (4GB VRAM) - Balanced Performance",
            "export USE_QWEN_IMAGE_EDIT=1", 
            "export QWEN_4BIT=true",
            "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128",
            "python3 apps/batch_processor.py"
        ],
        'minimal': [
            "# Minimal VRAM (<3GB) - GGUF Efficiency",
            "export USE_QWEN_GGUF=1",
            "export QWEN_N_GPU_LAYERS=8",
            "python3 apps/batch_processor.py"
        ],
        'cpu': [
            "# CPU-only - Stable Processing",
            "export FORCE_CPU=1",
            "python3 apps/batch_processor.py"
        ]
    }
    
    return commands.get(profile, commands['cpu'])

if __name__ == "__main__":
    print("🎯 BabyVis Auto-Configuration for Different VRAM Environments")
    print("=" * 65)
    
    # Detect and configure
    config = detect_vram_and_configure()
    profile = apply_environment_config(config)
    
    print(f"\n💡 Recommended commands for your environment:")
    for cmd in get_recommended_command(profile):
        print(f"   {cmd}")
    
    print(f"\n🚀 To run BabyVis with auto-detected settings:")
    print(f"   python3 apps/batch_processor.py")
    
    print(f"\n📊 Performance expectations:")
    if profile == 'server':
        print("   ⚡ Fastest processing, highest quality")
        print("   🎯 Full precision models")
        print("   📈 Can handle large batches")
    elif profile == 'personal':
        print("   ⚖️ Good balance of speed and quality")
        print("   💾 4-bit quantization saves 75% VRAM")
        print("   🔄 Single image processing")
    elif profile == 'minimal':
        print("   🐌 Slower but works on limited hardware")
        print("   💿 GGUF models for efficiency")
        print("   🔄 CPU/GPU hybrid processing")
    else:
        print("   🖥️ CPU processing, stable but slow")
        print("   ✅ Works on any hardware")