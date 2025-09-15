#!/usr/bin/env python3
"""
Test adaptive VRAM configuration for different environments
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_adaptive_configuration():
    """Test adaptive configuration detection"""
    try:
        from babyvis.model_utils import _detect_vram_gb, _get_adaptive_config
        
        vram_gb = _detect_vram_gb()
        config = _get_adaptive_config()
        
        print(f"🎯 Adaptive Configuration Test")
        print(f"=" * 50)
        print(f"📊 Detected VRAM: {vram_gb:.1f}GB")
        print(f"🎪 Profile: {config['profile'].upper()}")
        print(f"⚙️ Settings:")
        print(f"   - 4-bit quantization: {config['use_4bit']}")
        print(f"   - 8-bit quantization: {config['use_8bit']}")
        print(f"   - Flash Attention: {config['flash_attention']}")
        print(f"   - GPU Layers (GGUF): {config['gpu_layers']}")
        print(f"   - Batch Size: {config['batch_size']}")
        print(f"   - Max Length: {config['max_length']}")
        
        # Provide recommendations
        print(f"\n💡 Recommendations for your {config['profile']} setup:")
        
        if config['profile'] == 'server':
            print("   🚀 Maximum performance mode!")
            print("   ✅ Use direct Qwen Image Edit with full precision")
            print("   ✅ Flash Attention enabled for speed")
            print("   ✅ Can process multiple images in parallel")
            print("   📋 Command: export USE_QWEN_IMAGE_EDIT=1")
            
        elif config['profile'] == 'personal':
            print("   ⚖️ Balanced performance for 4GB VRAM")
            print("   ✅ Use 4-bit quantized Qwen Image Edit")
            print("   ✅ Optimized memory management")
            print("   ✅ Single image processing recommended")
            print("   📋 Commands:")
            print("      export USE_QWEN_IMAGE_EDIT=1")
            print("      export QWEN_4BIT=true")
            
        elif config['profile'] == 'minimal':
            print("   💾 Memory-efficient mode for limited VRAM")
            print("   ✅ Use GGUF quantized models")
            print("   ✅ Minimal GPU layers, CPU fallback")
            print("   ✅ Most memory efficient")
            print("   📋 Commands:")
            print("      export USE_QWEN_GGUF=1")
            print("      export QWEN_N_GPU_LAYERS=8")
            
        else:  # cpu
            print("   🖥️ CPU-only processing")
            print("   ✅ Use ControlNet stable diffusion")
            print("   ✅ Reliable but slower processing")
            print("   📋 Commands:")
            print("      export FORCE_CPU=1")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_environment_profiles():
    """Test different environment profile simulations"""
    profiles = [
        ('Server (10GB VRAM)', 10.0),
        ('Personal (4GB VRAM)', 4.0), 
        ('Minimal (2GB VRAM)', 2.0),
        ('CPU Only (0GB VRAM)', 0.0)
    ]
    
    print(f"\n🧪 Testing Different Environment Profiles")
    print(f"=" * 50)
    
    # Mock different VRAM scenarios
    original_cuda_available = None
    try:
        import torch
        original_cuda_available = torch.cuda.is_available
        
        for name, vram_gb in profiles:
            print(f"\n🎯 Simulating: {name}")
            print(f"   VRAM: {vram_gb}GB")
            
            # Mock VRAM detection
            def mock_cuda_available():
                return vram_gb > 0
            
            def mock_get_device_properties(device_id):
                class MockProps:
                    def __init__(self, memory_gb):
                        self.total_memory = int(memory_gb * 1024**3)
                        self.name = f"Mock GPU ({memory_gb}GB)"
                return MockProps(vram_gb)
            
            # Temporarily patch torch functions
            torch.cuda.is_available = mock_cuda_available
            if vram_gb > 0:
                torch.cuda.get_device_properties = mock_get_device_properties
            
            from babyvis.model_utils import _detect_vram_gb, _get_adaptive_config
            
            # Force reload of config
            detected_vram = _detect_vram_gb()
            config = _get_adaptive_config()
            
            print(f"   Profile: {config['profile']}")
            print(f"   4-bit: {config['use_4bit']}, 8-bit: {config['use_8bit']}")
            print(f"   GPU Layers: {config['gpu_layers']}")
            
    except Exception as e:
        print(f"⚠️ Profile simulation failed: {e}")
    
    finally:
        # Restore original functions
        if original_cuda_available:
            try:
                import torch
                torch.cuda.is_available = original_cuda_available
            except:
                pass

def main():
    print("🔧 BabyVis Adaptive VRAM Configuration Test")
    print("=" * 60)
    
    # Test current configuration
    if test_adaptive_configuration():
        print("\n✅ Adaptive configuration working!")
    else:
        print("\n❌ Configuration test failed")
        return
    
    # Test different profile simulations
    test_environment_profiles()
    
    print(f"\n🚀 Ready to run BabyVis with adaptive settings!")
    print(f"   python3 apps/batch_processor.py")
    
    print(f"\n📖 Manual overrides available:")
    print(f"   Server mode:   export USE_QWEN_IMAGE_EDIT=1 && export QWEN_4BIT=false")
    print(f"   Personal mode: export USE_QWEN_IMAGE_EDIT=1 && export QWEN_4BIT=true") 
    print(f"   GGUF mode:     export USE_QWEN_GGUF=1")
    print(f"   CPU mode:      export FORCE_CPU=1")

if __name__ == "__main__":
    main()