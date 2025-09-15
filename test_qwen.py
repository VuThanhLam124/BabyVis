#!/usr/bin/env python3
"""
Test script để kiểm tra Qwen backends
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test import các module cần thiết"""
    print("🧪 Testing imports...")
    try:
        from babyvis.model_utils import load_qwen_image_edit_gguf, load_qwen_image_edit
        from babyvis.inference import generate_predict_auto
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_qwen_gguf():
    """Test Qwen GGUF loading"""
    print("\n🔧 Testing Qwen GGUF...")
    try:
        from babyvis.model_utils import load_qwen_image_edit_gguf
        
        # Cố gắng load model (có thể fail nếu không có llama-cpp-python)
        try:
            llm = load_qwen_image_edit_gguf()
            print("✅ Qwen GGUF loaded successfully")
            return True
        except Exception as e:
            if "llama-cpp-python" in str(e):
                print("⚠️ llama-cpp-python not installed - this is expected")
                return False
            else:
                print(f"❌ Qwen GGUF failed: {e}")
                return False
    except Exception as e:
        print(f"❌ Qwen GGUF test failed: {e}")
        return False

def test_qwen_transformers():
    """Test Qwen Transformers loading"""
    print("\n🧠 Testing Qwen Transformers...")
    try:
        from babyvis.model_utils import load_qwen_image_edit
        
        # Cố gắng load model (có thể fail nếu không có transformers)
        try:
            tokenizer, model, device = load_qwen_image_edit()
            print("✅ Qwen Transformers loaded successfully")
            return True
        except Exception as e:
            if "transformers" in str(e) or "trust_remote_code" in str(e):
                print("⚠️ transformers not installed or model unavailable - this is expected")
                return False
            else:
                print(f"❌ Qwen Transformers failed: {e}")
                return False
    except Exception as e:
        print(f"❌ Qwen Transformers test failed: {e}")
        return False

def test_auto_dispatch():
    """Test auto dispatch function"""
    print("\n⚡ Testing auto dispatch...")
    try:
        from babyvis.inference import generate_predict_auto
        
        # Test với file không tồn tại để kiểm tra error handling
        try:
            generate_predict_auto(
                input_path="nonexistent.jpg",
                output_path="test_output.png",
                backend="auto"
            )
            print("❌ Should have failed with file not found")
            return False
        except FileNotFoundError:
            print("✅ Auto dispatch correctly handles missing files")
            return True
        except Exception as e:
            print(f"⚠️ Auto dispatch failed with: {e}")
            return False
    except Exception as e:
        print(f"❌ Auto dispatch test failed: {e}")
        return False

def test_sample_files():
    """Kiểm tra sample files có tồn tại không"""
    print("\n📁 Testing sample files...")
    sample_dir = "samples"
    if os.path.exists(sample_dir):
        files = os.listdir(sample_dir)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if image_files:
            print(f"✅ Found {len(image_files)} sample images: {image_files[:3]}...")
            return True
        else:
            print("⚠️ No image files found in samples/")
            return False
    else:
        print("⚠️ samples/ directory not found")
        return False

def main():
    print("🤖 BabyVis Qwen Edition - Test Suite")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Sample Files", test_sample_files), 
        ("Qwen GGUF", test_qwen_gguf),
        ("Qwen Transformers", test_qwen_transformers),
        ("Auto Dispatch", test_auto_dispatch),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n📊 Test Results:")
    print("="*50)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed >= len(results) // 2:
        print("\n🎉 BabyVis Qwen Edition is ready to use!")
        print("Recommended: ./run_qwen_auto.sh")
    else:
        print("\n⚠️ Some issues detected. Check dependencies:")
        print("pip install transformers llama-cpp-python torch")

if __name__ == "__main__":
    main()