#!/usr/bin/env python3
"""
BabyVis Main Application
Entry point cho BabyVis ultrasound to baby face generation
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from babyvis.processor import OptimizedBabyVisProcessor


def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import diffusers
    except ImportError:
        missing_deps.append("diffusers")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
    
    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("pip install torch torchvision diffusers transformers Pillow")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="BabyVis: Ultrasound to Baby Face Generation")
    parser.add_argument("--input", "-i", required=True, help="Input ultrasound image or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for baby faces")
    parser.add_argument("--prompt", "-p", default="professional newborn baby photography, sleeping peacefully, soft natural lighting, ultra realistic, high detail, studio quality, beautiful skin texture", 
                       help="Text prompt for generation")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", 
                       help="Device to use for inference")
    parser.add_argument("--steps", type=int, default=35, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=8.5, help="Guidance scale")
    parser.add_argument("--model-variant", choices=["default", "quantized", "cpu"], default="auto",
                       help="Model variant to use")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offloading")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Determine device
    device = None if args.device == "auto" else args.device
    
    # Initialize processor
    print("Initializing BabyVis processor...")
    processor = OptimizedBabyVisProcessor(device=device, enable_cpu_offload=args.cpu_offload)
    
    # Determine model variant
    model_variant = args.model_variant
    if model_variant == "auto":
        if processor.device == "cuda":
            model_variant = "quantized"  # Use quantized for GPU
        else:
            model_variant = "cpu"
    
    # Load model
    print(f"Loading model (variant: {model_variant})...")
    if not processor.load_model(model_variant):
        print("Failed to load model!")
        sys.exit(1)
    
    # Show memory usage
    memory_info = processor.get_memory_usage()
    print(f"Memory usage: {memory_info}")
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        print(f"Processing single image: {args.input}")
        result = processor.process_ultrasound_to_baby(
            args.input,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance
        )
        
        if result is not None:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"baby_{input_path.stem}.png"
            result.save(output_file)
            print(f"Generated baby face saved to: {output_file}")
        else:
            print("Failed to generate baby face!")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Batch processing
        print(f"Batch processing directory: {args.input}")
        outputs = processor.batch_process(
            args.input,
            args.output,
            prompt=args.prompt
        )
        print(f"Successfully processed {len(outputs)} images")
        
        if outputs:
            print("Generated files:")
            for output in outputs:
                print(f"  - {output}")
    
    else:
        print(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Final memory usage
    final_memory = processor.get_memory_usage()
    print(f"Final memory usage: {final_memory}")
    
    # Cleanup
    processor.cleanup()
    print("Processing complete!")


if __name__ == "__main__":
    main()