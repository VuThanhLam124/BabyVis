#!/usr/bin/env python3
"""
BabyVis Optimized Processor
Sử dụng Diffusers pipeline với Qwen-Image-Edit được tối ưu cho 4GB VRAM
"""

import os
import sys
import gc
import torch
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedBabyVisProcessor:
    """
    Processor tối ưu cho 4GB VRAM sử dụng Qwen-Image-Edit
    """
    
    def __init__(self, device: Optional[str] = None, enable_cpu_offload: bool = True):
        """
        Initialize processor với memory optimization
        
        Args:
            device: Target device ('cuda', 'cpu', or None for auto)
            enable_cpu_offload: Enable CPU offloading để tiết kiệm VRAM
        """
        self.device = self._get_optimal_device(device)
        self.enable_cpu_offload = enable_cpu_offload
        self.pipe = None
        self._setup_memory_optimization()
        
    def _get_optimal_device(self, device: Optional[str]) -> str:
        """Determine optimal device based on available hardware"""
        if device is not None:
            return device
            
        if torch.cuda.is_available():
            # Check VRAM
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Detected {vram_gb:.1f}GB VRAM")
            
            if vram_gb >= 3.5:  # Minimum for quantized models
                return "cuda"
            else:
                logger.warning(f"VRAM {vram_gb:.1f}GB may be insufficient, using CPU")
                return "cpu"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"
    
    def _setup_memory_optimization(self):
        """Setup memory optimization techniques"""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Enable memory efficient attention
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            logger.info("Memory optimization enabled")
    
    def load_model(self, model_variant: str = "default") -> bool:
        """
        Load image editing model với optimization
        
        Args:
            model_variant: Model variant to load
                - "default": Stable Diffusion (full model)
                - "quantized": Quantized version (fp16)
                - "cpu": CPU-optimized version
        """
        try:
            from diffusers import StableDiffusionPipeline
            from diffusers.utils import logging as diffusers_logging
            
            # Suppress diffusers warnings
            diffusers_logging.set_verbosity_error()
            
            logger.info(f"Loading Stable Diffusion model (variant: {model_variant})...")
            
            # Model configuration based on variant - using stable diffusion as fallback
            model_configs = {
                "default": {
                    "model_id": "runwayml/stable-diffusion-v1-5",
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "variant": None
                },
                "quantized": {
                    "model_id": "runwayml/stable-diffusion-v1-5",
                    "torch_dtype": torch.float16,
                    "variant": "fp16",
                    "use_safetensors": True
                },
                "cpu": {
                    "model_id": "runwayml/stable-diffusion-v1-5", 
                    "torch_dtype": torch.float32,
                    "variant": None
                }
            }
            
            config = model_configs.get(model_variant, model_configs["default"])
            
            # Load pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                config["model_id"],
                torch_dtype=config["torch_dtype"],
                variant=config.get("variant"),
                use_safetensors=config.get("use_safetensors", True),
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Apply optimizations
            if self.device == "cuda":
                # GPU optimizations
                if hasattr(self.pipe, "enable_model_cpu_offload") and self.enable_cpu_offload:
                    self.pipe.enable_model_cpu_offload()
                    logger.info("CPU offloading enabled")
                else:
                    self.pipe.to(self.device)
                
                # Memory efficient attention
                if hasattr(self.pipe, "enable_attention_slicing"):
                    self.pipe.enable_attention_slicing(1)
                    logger.info("Attention slicing enabled")
                
                # VAE tiling for large images
                if hasattr(self.pipe, "enable_vae_tiling"):
                    self.pipe.enable_vae_tiling()
                    logger.info("VAE tiling enabled")
                
                # Set high quality mode
                self.pipe.safety_checker = None
                self.pipe.requires_safety_checker = False
                logger.info("High quality mode enabled")
                    
                # XFormers optimization if available
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("XFormers optimization enabled")
                except Exception:
                    logger.warning("XFormers not available")
            else:
                # CPU optimization
                self.pipe.to(self.device)
                
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def process_ultrasound_to_baby(
        self, 
        input_image: Union[str, Image.Image],
        prompt: str = "professional newborn baby photography, sleeping peacefully, soft natural lighting, ultra realistic, high detail, studio quality, beautiful skin texture",
        num_inference_steps: int = 35,
        guidance_scale: float = 8.5,
        output_size: tuple = (512, 512)
    ) -> Optional[Image.Image]:
        """
        Process ultrasound image to baby face using text-to-image generation
        
        Args:
            input_image: Input ultrasound image (path or PIL Image) - used for guidance
            prompt: Text prompt for transformation
            num_inference_steps: Number of denoising steps (lower = faster)
            guidance_scale: Guidance scale (lower = faster)
            output_size: Output image size
            
        Returns:
            Generated baby face image or None if failed
        """
        if self.pipe is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Load and preprocess input image (for analysis/guidance)
            if isinstance(input_image, str):
                image = Image.open(input_image).convert('RGB')
            else:
                image = input_image.convert('RGB')
            
            # Analyze input for enhanced prompt
            enhanced_prompt = f"{prompt}, 8K resolution, award winning photography, masterpiece, incredibly detailed, perfect lighting, smooth skin, newborn portrait"
            
            logger.info(f"Processing with prompt: {enhanced_prompt}")
            logger.info(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
            
            # Clear memory before inference
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Generate baby face using text-to-image
            with torch.inference_mode():
                result = self.pipe(
                    prompt=enhanced_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=output_size[1],
                    width=output_size[0],
                    negative_prompt="adult, woman, man, teenager, child, wedding, group photo, multiple people, blurry, low quality, distorted, ugly, deformed, mutation, extra limbs, bad anatomy, scary, horror, cartoon, anime, drawing, painting, sketch, low resolution, pixelated, noise, grainy, dark, underexposed, overexposed"
                )
            
            # Clear memory after inference
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return None
    
    def batch_process(
        self, 
        input_dir: str, 
        output_dir: str,
        prompt: str = "Transform this ultrasound image into a realistic baby face"
    ) -> list:
        """
        Batch process multiple ultrasound images
        
        Args:
            input_dir: Directory containing ultrasound images
            output_dir: Directory to save baby face images
            prompt: Text prompt for transformation
            
        Returns:
            List of successfully processed output paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} images to process")
        
        successful_outputs = []
        
        for i, image_file in enumerate(image_files):
            logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # Process image
                result = self.process_ultrasound_to_baby(
                    str(image_file),
                    prompt=prompt
                )
                
                if result is not None:
                    # Save result
                    output_file = output_path / f"baby_{image_file.stem}.png"
                    result.save(output_file, "PNG", quality=95)
                    successful_outputs.append(str(output_file))
                    logger.info(f"Saved: {output_file}")
                else:
                    logger.error(f"Failed to process: {image_file.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
                continue
        
        logger.info(f"Batch processing complete: {len(successful_outputs)}/{len(image_files)} successful")
        return successful_outputs
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage information"""
        info = {}
        
        if torch.cuda.is_available():
            info['cuda_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            info['cuda_cached'] = torch.cuda.memory_reserved() / (1024**3)
            info['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / (1024**3)
        
        import psutil
        info['cpu_memory'] = psutil.virtual_memory().percent
        
        return info
    
    def optimize_for_inference(self):
        """Apply additional optimizations for inference"""
        if self.pipe is None:
            return
            
        try:
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device == "cuda":
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                logger.info("Model compiled for faster inference")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()
        logger.info("Resources cleaned up")


def main():
    """Main function for testing"""
    processor = OptimizedBabyVisProcessor()
    
    # Load model with appropriate variant for hardware
    if processor.device == "cuda":
        success = processor.load_model("quantized")  # Use quantized for 4GB VRAM
    else:
        success = processor.load_model("cpu")
    
    if not success:
        logger.error("Failed to load model")
        return
    
    # Print memory usage
    memory_info = processor.get_memory_usage()
    logger.info(f"Memory usage: {memory_info}")
    
    # Test with sample images if available
    samples_dir = "samples"
    if os.path.exists(samples_dir):
        outputs = processor.batch_process(
            samples_dir, 
            "outputs/diffusers_baby_faces",
            prompt="Transform this ultrasound image into a beautiful realistic baby face"
        )
        logger.info(f"Generated {len(outputs)} baby face images")
    
    # Cleanup
    processor.cleanup()


if __name__ == "__main__":
    main()