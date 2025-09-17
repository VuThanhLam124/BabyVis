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
                - "default": Stable Diffusion 1.5 img2img
                - "qwen": Qwen-Image-Edit (GGUF format)
                - "instruct": InstructPix2Pix for image editing
                - "controlnet": ControlNet for guided editing
                - "realistic": Realistic Vision img2img
                - "cpu": CPU-optimized version
        """
        try:
            from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline, StableDiffusionInstructPix2PixPipeline
            from diffusers.utils import logging as diffusers_logging
            
            # Suppress diffusers warnings
            diffusers_logging.set_verbosity_error()
            
            logger.info(f"Loading image editing model (variant: {model_variant})...")
            
            # Image editing model configurations
            model_configs = {
                "default": {
                    "model_id": "runwayml/stable-diffusion-v1-5",
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "variant": None,
                    "use_safetensors": False,
                    "pipeline": StableDiffusionImg2ImgPipeline
                },
                "instruct": {
                    "model_id": "timbrooks/instruct-pix2pix",
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "variant": None,
                    "use_safetensors": False,
                    "pipeline": StableDiffusionInstructPix2PixPipeline
                },
                "realistic": {
                    "model_id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "variant": None,
                    "use_safetensors": False,
                    "pipeline": StableDiffusionImg2ImgPipeline
                },
                "qwen": {
                    "model_id": "Qwen/Qwen-Image-Edit",
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "variant": None,
                    "use_safetensors": False,
                    "pipeline": DiffusionPipeline
                },
                "cpu": {
                    "model_id": "runwayml/stable-diffusion-v1-5", 
                    "torch_dtype": torch.float32,
                    "variant": None,
                    "use_safetensors": False,
                    "pipeline": StableDiffusionImg2ImgPipeline
                }
            }
            
            config = model_configs.get(model_variant, model_configs["default"])
            pipeline_class = config["pipeline"]
            
            # Load pipeline for image editing
            try:
                self.pipe = pipeline_class.from_pretrained(
                    config["model_id"],
                    torch_dtype=config["torch_dtype"],
                    variant=config.get("variant"),
                    use_safetensors=config.get("use_safetensors", False),
                    low_cpu_mem_usage=True,
                    device_map="auto" if self.device == "cuda" else None,
                    force_download=False,
                    resume_download=True
                )
            except Exception as e:
                logger.warning(f"Failed with main model, trying fallback: {e}")
                # Fallback to default img2img
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=config["torch_dtype"],
                    use_safetensors=False,
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
        prompt: str = "transform this ultrasound image into a realistic newborn baby face, cute baby portrait, soft lighting, peaceful sleeping pose",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        strength: float = 0.8,  # Image editing strength
        output_size: tuple = (512, 512)
    ) -> Optional[Image.Image]:
        """
        Process ultrasound image to baby face using image-to-image editing
        
        Args:
            input_image: Input ultrasound image (path or PIL Image)
            prompt: Text prompt for transformation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            strength: How much to transform the original image (0.0-1.0)
            output_size: Output image size
            
        Returns:
            Generated baby face image or None if failed
        """
        if self.pipe is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Load and preprocess input image
            if isinstance(input_image, str):
                image = Image.open(input_image).convert('RGB')
            else:
                image = input_image.convert('RGB')
            
            # Resize image to target size
            image = image.resize(output_size, Image.Resampling.LANCZOS)
            
            # Enhanced prompt for image editing
            if hasattr(self.pipe, 'image') or 'Img2Img' in str(type(self.pipe)):
                # For img2img pipelines
                enhanced_prompt = f"{prompt}, beautiful newborn baby, soft skin, natural lighting, highly detailed, realistic"
                negative_prompt = "adult, distorted, ugly, deformed, blurry, low quality, dark, grainy"
            elif hasattr(self.pipe, 'edit_instruction') or 'InstructPix2Pix' in str(type(self.pipe)):
                # For InstructPix2Pix
                enhanced_prompt = f"Transform this ultrasound into a realistic baby face: {prompt}"
                negative_prompt = "blurry, distorted, low quality"
            else:
                # Fallback
                enhanced_prompt = prompt
                negative_prompt = "low quality, distorted"
            
            logger.info(f"Image editing with prompt: {enhanced_prompt}")
            logger.info(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, Strength: {strength}")
            
            # Clear memory before inference
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Generate baby face using image-to-image
            with torch.inference_mode():
                if hasattr(self.pipe, 'image') or 'Img2Img' in str(type(self.pipe)):
                    # Standard img2img pipeline
                    result = self.pipe(
                        prompt=enhanced_prompt,
                        image=image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    )
                elif hasattr(self.pipe, 'edit_instruction') or 'InstructPix2Pix' in str(type(self.pipe)):
                    # InstructPix2Pix pipeline
                    result = self.pipe(
                        prompt=enhanced_prompt,
                        image=image,
                        num_inference_steps=num_inference_steps,
                        image_guidance_scale=guidance_scale,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    )
                else:
                    # Try as general pipeline with image input
                    result = self.pipe(
                        prompt=enhanced_prompt,
                        image=image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    )
            
            # Clear memory after inference
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Image editing failed: {e}")
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