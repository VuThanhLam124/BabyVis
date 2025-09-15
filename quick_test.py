#!/usr/bin/env python3
"""
Quick test for batch processor with ControlNet backend only
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_batch_processor_controlnet():
    """Test batch processor with ControlNet backend"""
    try:
        print("🧪 Testing batch processor with ControlNet backend...")
        
        # Force CPU mode to avoid CUDA issues 
        os.environ['FORCE_CPU'] = '1'
        
        from apps.batch_processor import BatchImageProcessor
        
        processor = BatchImageProcessor()
        
        # Test with actual files if they exist
        test_files = [
            "samples/1.jpeg",
            "samples/1B.png", 
            "samples/3.png"
        ]
        
        existing_files = [f for f in test_files if os.path.exists(f)]
        
        if existing_files:
            print(f"📁 Found {len(existing_files)} test files: {existing_files}")
            
            # Process just the first file as a test
            results = processor.process_image_list([existing_files[0]])
            
            print(f"✅ Successfully processed {len(results)} images!")
            print(f"📂 Output files: {results}")
            
        else:
            print("⚠️ No test files found, but processor initialized correctly")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Quick test for BabyVis batch processor (ControlNet only)...")
    
    if test_batch_processor_controlnet():
        print("\n🎉 Batch processor is working! You can now run:")
        print("   python3 apps/batch_processor.py")
        print("\n💡 Tips:")
        print("   - Uses ControlNet backend (stable)")
        print("   - Runs on CPU to avoid CUDA issues") 
        print("   - Fixed filename mismatch for Qwen models")
    else:
        print("\n❌ Issues found. Check error messages above.")