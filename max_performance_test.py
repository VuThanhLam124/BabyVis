#!/usr/bin/env python3
"""
High-performance test with direct Qwen Image Edit (not GGUF)
"""
import os
import sys
import time

# Performance environment settings for direct model
os.environ['USE_QWEN_IMAGE_EDIT'] = '1'  # Use direct transformers model
os.environ['QWEN_4BIT'] = 'true'  # Use 4-bit quantization for speed
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_qwen_direct_performance():
    """Test direct Qwen Image Edit model for maximum performance"""
    try:
        print("🚀 Testing Direct Qwen Image Edit - MAXIMUM PERFORMANCE MODE")
        print("⚙️ Environment settings:")
        print(f"   USE_QWEN_IMAGE_EDIT: {os.getenv('USE_QWEN_IMAGE_EDIT')}")
        print(f"   4-bit quantization: {os.getenv('QWEN_4BIT')}")
        print("")
        
        start_time = time.time()
        
        from apps.batch_processor import BatchImageProcessor
        
        processor = BatchImageProcessor()
        
        # Update config to use direct Qwen backend
        processor.config.update({
            "backend": "qwen"  # Use direct transformers backend
        })
        
        # Find test images
        test_files = [
            "samples/1.jpeg",
        ]
        
        existing_files = [f for f in test_files if os.path.exists(f)]
        
        if not existing_files:
            print("❌ No test files found")
            return False
            
        print(f"📁 Testing with: {existing_files[0]}")
        
        # Process one image for performance test
        load_start = time.time()
        results = processor.process_image_list([existing_files[0]])
        total_time = time.time() - start_time
        inference_time = time.time() - load_start
        
        print(f"")
        print(f"📊 PERFORMANCE RESULTS:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Inference time: {inference_time:.2f}s")
        print(f"   Output: {results}")
        
        if results:
            print("🎉 MAXIMUM-PERFORMANCE Qwen Image Edit working!")
            return True
        else:
            print("⚠️ No outputs generated")
            return False
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_gpu_memory():
    """Check GPU memory availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"🖥️ GPU Info:")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
            # Check available memory
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            free_memory = memory_gb - allocated
            print(f"   Memory: {allocated:.2f}GB allocated, {free_memory:.2f}GB free")
            
            if free_memory < 2.0:
                print("⚠️ Low GPU memory! Using 4-bit quantization recommended")
                os.environ['QWEN_4BIT'] = 'true'
            
            return True
        else:
            print("⚠️ No CUDA GPU available - will use CPU")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

if __name__ == "__main__":
    print("🔥 BabyVis MAXIMUM-PERFORMANCE Test with Direct Qwen Image Edit")
    print("=" * 70)
    
    # Check system
    has_gpu = check_gpu_memory()
    print("")
    
    if has_gpu:
        # Run performance test
        success = test_qwen_direct_performance()
        
        print("")
        if success:
            print("🎯 MAXIMUM-PERFORMANCE setup is working!")
            print("💡 To run full batch processing with maximum performance:")
            print("   export USE_QWEN_IMAGE_EDIT=1")
            print("   export QWEN_4BIT=true")
            print("   python3 apps/batch_processor.py")
        else:
            print("❌ Performance test failed - check errors above")
    else:
        print("💡 No GPU detected. For CPU performance, try:")
        print("   export USE_QWEN_IMAGE_EDIT=0  # Use ControlNet")
        print("   python3 apps/batch_processor.py")