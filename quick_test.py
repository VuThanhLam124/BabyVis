#!/usr/bin/env python3
"""
Quick test script cho Qwen - không download model
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test cơ bản không cần model thực"""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test imports
        from babyvis.model_utils import _detect_vram_gb, _get_adaptive_config
        from babyvis.inference import get_professional_baby_prompt
        
        # Test VRAM detection
        vram = _detect_vram_gb()
        print(f"   VRAM detected: {vram:.1f}GB")
        
        # Test adaptive config
        config = _get_adaptive_config()
        print(f"   Profile: {config['profile']}")
        print(f"   GPU layers: {config['gpu_layers']}")
        
        # Test prompt generation
        prompts = get_professional_baby_prompt("Asian")
        print(f"   Prompt length: {len(prompts['positive'])} chars")
        
        print("✅ Basic functionality working")
        return True
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def test_batch_processor():
    """Test batch processor logic"""
    print("\n📦 Testing batch processor...")
    
    try:
        from apps.batch_processor import BatchImageProcessor
        
        # Create processor
        processor = BatchImageProcessor()
        
        # Test config loading
        processor.load_config()
        print(f"   Config loaded: {processor.config.get('output_directory', 'default')}")
        
        # Test với empty list
        results = processor.process_image_list([])
        print(f"   Empty list handled: {len(results)} results")
        
        print("✅ Batch processor working")
        return True
    except Exception as e:
        print(f"❌ Batch processor test failed: {e}")
        return False

def test_image_processing():
    """Test image processing functions"""
    print("\n🖼️ Testing image processing...")
    
    try:
        from babyvis.inference import enhanced_canny_detection, adaptive_canny_thresholds
        import numpy as np
        
        # Test với dummy image
        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test canny detection
        canny = enhanced_canny_detection(dummy_img, method="adaptive")
        print(f"   Canny detection: {canny.shape}")
        
        # Test adaptive thresholds
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        low, high = adaptive_canny_thresholds(gray)
        print(f"   Adaptive thresholds: {low}, {high}")
        
        print("✅ Image processing working")
        return True
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def main():
    print("⚡ BabyVis Qwen Edition - Quick Test")
    print("="*45)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Batch Processor", test_batch_processor),
        ("Image Processing", test_image_processing),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 BabyVis Qwen Edition - Core functionality ready!")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install transformers llama-cpp-python")
        print("2. Run: ./run_qwen_cpu.sh (CPU mode)")
        print("3. Or: ./run_qwen_auto.sh (auto-detect)")
    else:
        print("\n⚠️ Some core issues detected. Check your installation.")

if __name__ == "__main__":
    main()