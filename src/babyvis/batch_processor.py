#!/usr/bin/env python3
"""
BabyVis Batch Processor - Simplified
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from babyvis.inference import batch_process_images, process_single_image


def find_images(directory: str) -> list:
    """Tìm tất cả ảnh trong thư mục"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_paths = []
    
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            image_paths.append(str(file_path))
    
    return sorted(image_paths)


def main():
    """Main function"""
    print("🤖 BabyVis - Simple Batch Processor")
    print("=" * 50)
    
    # Get configuration from environment
    model_type = os.environ.get("MODEL_TYPE", "auto")
    input_dir = "samples"
    output_dir = "outputs"
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🤖 Model type: {model_type}")
    
    # Find images
    image_paths = find_images(input_dir)
    
    if not image_paths:
        print(f"❌ No images found in {input_dir}")
        print(f"💡 Add ultrasound images to the {input_dir} directory")
        return
    
    print(f"✅ Found {len(image_paths)} images")
    
    # Process images
    try:
        print("🚀 Starting processing...")
        results = batch_process_images(
            image_paths=image_paths,
            output_dir=output_dir,
            model_type=model_type,
            ethnicity="mixed"
        )
        
        if results:
            print(f"\n🎉 Success! Generated {len(results)} baby images")
            print(f"📂 Check output directory: {output_dir}")
        else:
            print(f"\n❌ No images were generated successfully")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")


if __name__ == "__main__":
    main()