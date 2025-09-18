#!/usr/bin/env python3
"""
BabyVis Main Application
Entry point for the BabyVis ultrasound to baby face generation application
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'torch',
        'diffusers',
        'transformers',
        'fastapi',
        'uvicorn',
        'PIL',
        'cv2',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            logger.info(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} - Missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies are available")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = ['uploads', 'outputs', 'static']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")

def run_web_app(host="0.0.0.0", port=8000, reload=False):
    """Run the web application"""
    try:
        import uvicorn
        from app import app
        
        logger.info(f"🚀 Starting BabyVis web application...")
        logger.info(f"🌐 Server will be available at: http://{host}:{port}")
        logger.info(f"📚 API docs available at: http://{host}:{port}/docs")
        
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running web application: {e}")
        sys.exit(1)

def run_cli_mode():
    """Run in CLI mode for batch processing"""
    try:
        from qwen_image_edit_model import QwenImageEditModel
        from image_utils import ImageValidator
        from PIL import Image
        import os
        
        print("🍼 BabyVis CLI Mode")
        print("=" * 50)
        
        # Get input image path
        input_path = input("Enter path to ultrasound image: ").strip()
        
        if not os.path.exists(input_path):
            print(f"❌ File not found: {input_path}")
            return
        
        # Get output path
        output_path = input("Enter output path (press Enter for auto): ").strip()
        if not output_path:
            output_path = f"baby_face_{os.path.basename(input_path)}.png"
        
        # Load and validate image
        print("📋 Loading and validating image...")
        image = Image.open(input_path)
        
        validator = ImageValidator()
        is_valid, message = validator.validate_ultrasound(image)
        quality = validator.check_image_quality(image)
        
        print(f"✅ Validation: {message}")
        print(f"📊 Quality score: {quality['quality_score']:.2f}")
        print(f"📐 Dimensions: {quality['width']}x{quality['height']}")
        
        # Get generation settings
        print("\n🎛️ Generation Settings:")
        quality_level = input("Quality level (base/enhanced/premium) [enhanced]: ").strip() or "enhanced"
        steps = int(input("Number of steps (15-50) [30]: ").strip() or "30")
        strength = float(input("Transformation strength (0.3-1.0) [0.8]: ").strip() or "0.8")
        
        # Initialize model and generate
        print(f"\n🤖 Initializing model...")
        model = QwenImageEditModel()
        
        print(f"🎨 Generating baby face...")
        success, baby_image, result_message = model.generate_baby_face(
            image,
            quality_level=quality_level,
            num_inference_steps=steps,
            strength=strength
        )
        
        if success and baby_image:
            baby_image.save(output_path)
            print(f"✅ Success! Baby face saved to: {output_path}")
            print(f"📄 Result: {result_message}")
        else:
            print(f"❌ Generation failed: {result_message}")
        
        # Clean up
        model.unload_model()
        
    except KeyboardInterrupt:
        print("\n🛑 CLI mode stopped by user")
    except Exception as e:
        print(f"❌ CLI error: {e}")

def run_test_mode():
    """Run test mode to verify everything works"""
    try:
        from qwen_image_edit_model import QwenImageEditModel
        from image_utils import UltrasoundProcessor, ImageValidator
        from PIL import Image
        import time
        
        print("🧪 BabyVis Test Mode")
        print("=" * 50)
        
        # Check for test image
        test_image_path = "samples/1.jpeg"
        if not os.path.exists(test_image_path):
            print(f"⚠️ Test image not found: {test_image_path}")
            print("Creating a test image...")
            
            # Create a simple test image
            test_image = Image.new("RGB", (512, 512), color="gray")
            os.makedirs("samples", exist_ok=True)
            test_image.save(test_image_path)
            print(f"✅ Created test image: {test_image_path}")
        
        # Load test image
        print("📋 Loading test image...")
        image = Image.open(test_image_path)
        
        # Test image validation
        print("🔍 Testing image validation...")
        validator = ImageValidator()
        is_valid, message = validator.validate_ultrasound(image)
        quality = validator.check_image_quality(image)
        print(f"   Validation: {message}")
        print(f"   Quality: {quality['quality_score']:.2f}")
        
        # Test ultrasound processing
        print("🔧 Testing ultrasound processing...")
        processor = UltrasoundProcessor()
        enhanced = processor.enhance_ultrasound(image)
        features = processor.detect_fetal_features(image)
        print(f"   Features detected: {len(features.get('head_candidates', []))}")
        
        # Test model loading
        print("🤖 Testing model loading...")
        start_time = time.time()
        model = QwenImageEditModel()
        
        # Test generation (with minimal settings for speed)
        print("🎨 Testing baby face generation...")
        success, baby_image, result_message = model.generate_baby_face(
            image,
            quality_level="base",
            num_inference_steps=15,
            strength=0.7
        )
        
        generation_time = time.time() - start_time
        
        if success and baby_image:
            output_path = "outputs/test_baby_face.png"
            os.makedirs("outputs", exist_ok=True)
            baby_image.save(output_path)
            print(f"✅ Test successful!")
            print(f"   Generation time: {generation_time:.1f}s")
            print(f"   Output saved: {output_path}")
            print(f"   Result: {result_message}")
        else:
            print(f"❌ Test failed: {result_message}")
        
        # Clean up
        model.unload_model()
        print("🧹 Cleaned up resources")
        
    except Exception as e:
        print(f"❌ Test error: {e}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="BabyVis - AI Baby Face Generator")
    parser.add_argument("--mode", choices=["web", "cli", "test"], default="web",
                       help="Application mode (default: web)")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host address for web mode (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for web mode (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check dependencies and exit")
    
    args = parser.parse_args()
    
    print("🍼 BabyVis - AI Baby Face Generator v2.0")
    print("=" * 50)
    
    # Check dependencies
    if args.check_deps or args.mode in ["web", "cli", "test"]:
        print("🔍 Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
    
    if args.check_deps:
        print("✅ All dependencies are satisfied!")
        return
    
    # Setup directories
    setup_directories()
    
    # Run in requested mode
    if args.mode == "web":
        run_web_app(host=args.host, port=args.port, reload=args.reload)
    elif args.mode == "cli":
        run_cli_mode()
    elif args.mode == "test":
        run_test_mode()

if __name__ == "__main__":
    main()