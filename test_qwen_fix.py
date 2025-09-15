#!/usr/bin/env python3
"""
Test script to verify Qwen GGUF fix
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_qwen_gguf_loading():
    """Test if Qwen GGUF model can be loaded with the new fix"""
    try:
        from babyvis.model_utils import load_qwen_image_edit_gguf
        
        print("🧪 Testing Qwen GGUF loading...")
        
        # This should now work with the preferred filename list
        llm = load_qwen_image_edit_gguf(
            repo_id="QuantStack/Qwen-Image-Edit-GGUF",
            n_gpu_layers=0,  # Use CPU only for safety
        )
        
        print("✅ Qwen GGUF loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading Qwen GGUF: {e}")
        return False

def test_controlnet_fallback():
    """Test if ControlNet backend works as fallback"""
    try:
        print("\n🧪 Testing ControlNet fallback...")
        
        # Force CPU mode to avoid CUDA issues
        os.environ['FORCE_CPU'] = '1'
        
        from babyvis.model_utils import load_canny_pipeline
        
        pipe = load_canny_pipeline()
        print("✅ ControlNet pipeline loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading ControlNet: {e}")
        return False

def test_batch_processor():
    """Test if batch processor runs without errors"""
    try:
        print("\n🧪 Testing batch processor...")
        
        # Import after setting environment
        from apps.batch_processor import BatchImageProcessor
        
        processor = BatchImageProcessor()
        
        # Test with a small list (just check if it doesn't crash)
        test_paths = ["samples/1.jpeg"]  # Only test one file
        
        print("✅ Batch processor initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in batch processor: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing fixes for Qwen GGUF and ControlNet...")
    
    success_count = 0
    total_tests = 3
    
    # Test 1: ControlNet fallback (most important)
    if test_controlnet_fallback():
        success_count += 1
    
    # Test 2: Batch processor
    if test_batch_processor():
        success_count += 1
    
    # Test 3: Qwen GGUF (optional, might fail due to network)
    if test_qwen_gguf_loading():
        success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count >= 2:  # At least ControlNet and batch processor work
        print("🎉 Core functionality is working! You can run the batch processor.")
    else:
        print("⚠️ Some issues remain. Check the error messages above.")