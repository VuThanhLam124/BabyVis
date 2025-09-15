#!/usr/bin/env python3
"""
High-performance test for Qwen Image Edit
"""
import os
import sys
import time

# Performance environment settings
os.environ['USE_QWEN_GGUF'] = '1'
os.environ['QWEN_N_GPU_LAYERS'] = '30'  # Aggressive GPU usage
os.environ['QWEN_N_CTX'] = '4096'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_qwen_performance():
    """Test Qwen Image Edit with performance optimizations"""
    try:
        print("🚀 Testing Qwen Image Edit - HIGH PERFORMANCE MODE")
        print("⚙️ Environment settings:")
        print(f"   USE_QWEN_GGUF: {os.getenv('USE_QWEN_GGUF')}")
        print(f"   GPU Layers: {os.getenv('QWEN_N_GPU_LAYERS')}")
        print(f"   Context: {os.getenv('QWEN_N_CTX')}")
        print("")
        
        start_time = time.time()
        
        from apps.batch_processor import BatchImageProcessor
        
        processor = BatchImageProcessor()
        
        # Find test images
        test_files = [
            "samples/1.jpeg",
            "samples/1B.png",
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
            print("🎉 HIGH-PERFORMANCE Qwen Image Edit working!")
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
            print(f"   Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            return True
        else:
            print("⚠️ No CUDA GPU available - will use CPU")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

if __name__ == "__main__":
    print("🔥 BabyVis HIGH-PERFORMANCE Test with Qwen Image Edit")
    print("=" * 60)
    
    # Check system
    check_gpu_memory()
    print("")
    
    # Run performance test
    success = test_qwen_performance()
    
    print("")
    if success:
        print("🎯 HIGH-PERFORMANCE setup is working!")
        print("💡 To run full batch processing:")
        print("   source high_performance_setup.sh")
        print("   python3 apps/batch_processor.py")
    else:
        print("❌ Performance test failed - check errors above")
        print("💡 You can still use ControlNet backend:")
        print("   export USE_QWEN_GGUF=0")
        print("   python3 apps/batch_processor.py")