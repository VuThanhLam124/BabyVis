#!/usr/bin/env python3
"""
Qwen Image Edit Integration for BabyVis
Diffusers-first implementation optimized for 4GB VRAM (GPU or CPU offload).

This uses the Hugging Face diffusers img2img pipeline with the
Qwen/Qwen-Image-Edit model ID and applies VRAM optimizations such as
CPU offload, attention slicing, VAE tiling, and fp16 on CUDA.
"""

import os
import gc
import torch
import warnings
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import logging

# Import our professional GGUF loader
# NOTE: GGUF loaders (llama.cpp) target LLMs and are not applicable
# to diffusion image-editing models; we therefore use diffusers directly.

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenImageEditModel:
    """
    Qwen Image Edit via diffusers (img2img), optimized for 4GB VRAM.
    """

    def __init__(self, model_id: str = None, device: str = "auto"):
        # Prefer Qwen diffusers model; allow override via env QWEN_MODEL_ID
        self.model_id = model_id or os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-Edit")
        self.device = self._setup_device(device)
        self.pipeline = None
        self.model_loaded = False
        
        # Professional baby generation prompts for different quality levels
        self.baby_prompts = {
            "base": "beautiful newborn baby face, peaceful sleeping, soft baby skin, cute innocent features, realistic portrait photography, natural lighting, high definition",
            "enhanced": "adorable newborn baby portrait, perfect delicate skin, gentle peaceful expression, professional hospital photography, warm soft lighting, detailed facial features, ultra realistic, 8K quality",
            "premium": "stunning newborn baby photograph, crystal clear baby skin, perfect proportions, angelic expression, professional portrait studio lighting, hospital-grade photography, photorealistic, award-winning baby photography, ultra high definition"
        }
        
        # Negative prompts to avoid unwanted features
        self.negative_prompt = "adult face, child face, teenager, elderly, wrinkles, facial hair, beard, mustache, distorted features, blurry image, low quality, deformed face, scary expression, dark lighting, cartoon, anime, artificial, fake, unrealistic proportions"
        
        # Runtime settings
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
                logger.info(f"💾 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                device = "cpu"
                logger.info("💻 Using CPU (GPU not available)")
        
        return torch.device(device)
    
    def load_model(self) -> bool:
        """Load Qwen Image Edit diffusers pipeline with 4GB VRAM optimizations"""
        if self.model_loaded:
            logger.info("✅ Model already loaded")
            return True
        try:
            from diffusers import AutoPipelineForImage2Image
            logger.info(f"🚀 Loading diffusers model: {self.model_id}")

            # Load with low memory usage settings
            self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            )

            # Safety checker off for speed
            if hasattr(self.pipeline, "safety_checker"):
                self.pipeline.safety_checker = None
                self.pipeline.requires_safety_checker = False

            # VRAM optimization
            try:
                if self.device.type == "cuda":
                    # Use CPU offload to keep VRAM under ~4GB
                    if hasattr(self.pipeline, "enable_model_cpu_offload"):
                        self.pipeline.enable_model_cpu_offload()
                    else:
                        self.pipeline.to(self.device)
                else:
                    self.pipeline.to(self.device)

                if hasattr(self.pipeline, "enable_attention_slicing"):
                    self.pipeline.enable_attention_slicing("max")
                if hasattr(self.pipeline, "enable_vae_tiling"):
                    self.pipeline.enable_vae_tiling()
                if hasattr(self.pipeline, "set_progress_bar_config"):
                    self.pipeline.set_progress_bar_config(disable=True)
            except Exception as e:
                logger.warning(f"⚠️ Optimization setup issue: {e}")

            self.model_loaded = True
            logger.info("✅ Diffusers model loaded and optimized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load diffusers model: {e}")
            return False
    
    def _create_fallback_model(self) -> bool:
        """
        Create fallback using regular diffusers if GGUF fails
        IMPORTANT: This should never be used as per user requirements!
        """
        logger.warning("⚠️ CRITICAL: User requested ONLY strong models like Qwen!")
        logger.warning("⚠️ Attempting fallback to regular Qwen model...")
        
        try:
            from diffusers import AutoPipelineForImage2Image
            
            # Try the regular Qwen model as fallback
            logger.info("🔄 Loading regular Qwen/Qwen-Image-Edit as fallback...")
            
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            
            self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                "Qwen/Qwen-Image-Edit",
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if dtype == torch.float16 else None
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
            
            self.model_loaded = True
            self.model_id = "Qwen/Qwen-Image-Edit"
            logger.info("✅ Fallback Qwen model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Even fallback Qwen model failed: {e}")
            logger.error("❌ CRITICAL: Cannot load any strong model as requested!")
            return False
    
    def generate_baby_face(
        self,
        ultrasound_image: Image.Image,
        quality_level: str = "enhanced",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: Optional[int] = None
    ) -> Tuple[bool, Optional[Image.Image], str]:
        """
        Generate professional baby face from ultrasound using GGUF model
        
        Args:
            ultrasound_image: Input ultrasound PIL Image
            quality_level: "base", "enhanced", or "premium" 
            num_inference_steps: Number of denoising steps (15-50)
            guidance_scale: Prompt adherence strength (1.0-20.0)
            strength: Transformation strength (0.1-1.0)
            seed: Random seed for reproducible results
            
        Returns:
            Tuple of (success, generated_image, message)
        """
        if not self.model_loaded:
            logger.warning("🔄 Model not loaded, attempting to load...")
            if not self.load_model():
                return False, None, "❌ CRITICAL: Failed to load any strong model as requested!"
        
        try:
            # Set random seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                np.random.seed(seed)
            
            # Preprocess the ultrasound image professionally
            processed_image = self.preprocess_ultrasound_professional(ultrasound_image)
            
            # Get professional prompt for the quality level
            prompt = self.baby_prompts.get(quality_level, self.baby_prompts["enhanced"])
            
            logger.info(f"🎨 Generating baby face with diffusers model...")
            logger.info(f"   Quality: {quality_level} | Steps: {num_inference_steps}")
            logger.info(f"   Guidance: {guidance_scale} | Strength: {strength}")
            logger.info(f"   Model: {self.model_id}")

            if self.pipeline is None:
                return False, None, "❌ Model pipeline not available"

            generator = None
            if seed is not None:
                try:
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                except Exception:
                    generator = torch.Generator().manual_seed(seed)

            # Inference with autocast to reduce memory on CUDA
            autocast_device = self.device.type if self.device.type == "cuda" else "cpu"
            with torch.autocast(autocast_device):
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    image=processed_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator,
                    return_dict=True
                )

            baby_image = result.images[0]

            # Post-process for nicer presentation
            baby_image = self._postprocess_baby_face_professional(baby_image, quality_level)

            # Clean up GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            return True, baby_image, f"✅ Baby face generated with {self.model_id} ({quality_level} quality)"
            
        except Exception as e:
            error_msg = f"❌ Generation error: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def preprocess_ultrasound_professional(self, image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Professional preprocessing of ultrasound image for optimal GGUF generation
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to target size with high-quality resampling
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Professional ultrasound enhancement
        # 1. Enhance contrast for better feature definition
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        # 2. Optimize brightness for AI processing
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.15)
        
        # 3. Reduce ultrasound noise with advanced filtering
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # 4. Enhance sharpness for better feature extraction
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def _postprocess_baby_face_professional(self, image: Image.Image, quality_level: str) -> Image.Image:
        """
        Professional post-processing for generated baby faces
        """
        # Quality-based processing intensity
        intensity_map = {"base": 0.5, "enhanced": 0.7, "premium": 1.0}
        intensity = intensity_map.get(quality_level, 0.7)
        
        # Convert to numpy for advanced processing
        img_array = np.array(image)
        
        # Professional baby skin tone optimization
        # Warm, healthy baby skin enhancement
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + intensity * 0.08), 0, 255)  # Enhance red
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 + intensity * 0.05), 0, 255)  # Slightly enhance green  
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - intensity * 0.05), 0, 255)  # Reduce blue for warmth
        
        # Convert back to PIL for further processing
        processed_image = Image.fromarray(img_array.astype(np.uint8))
        
        # Professional baby skin smoothing (quality-dependent)
        blur_radius = 0.3 + (intensity * 0.4)
        processed_image = processed_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Professional lighting enhancement
        enhancer = ImageEnhance.Brightness(processed_image)
        processed_image = enhancer.enhance(1.0 + intensity * 0.08)
        
        # Color saturation for healthy baby appearance
        enhancer = ImageEnhance.Color(processed_image)
        processed_image = enhancer.enhance(1.0 + intensity * 0.15)
        
        # Premium quality additional enhancements
        if quality_level == "premium":
            # Ultra-fine detail enhancement
            sharpening_filter = ImageFilter.UnsharpMask(radius=0.8, percent=110, threshold=2)
            processed_image = processed_image.filter(sharpening_filter)
            
            # Professional contrast optimization
            enhancer = ImageEnhance.Contrast(processed_image)
            processed_image = enhancer.enhance(1.05)
        
        return processed_image
    
    def batch_generate_professional(
        self,
        ultrasound_image: Image.Image,
        num_variations: int = 3,
        quality_level: str = "enhanced",
        **generation_kwargs
    ) -> Tuple[bool, list, str]:
        """
        Generate multiple professional baby face variations using GGUF model
        """
        variations = []
        successful_generations = 0
        
        logger.info(f"🎨 Generating {num_variations} professional baby face variations...")
        
        for i in range(num_variations):
            # Use different seeds for variation
            current_seed = generation_kwargs.get('seed', 42) + i * 1337
            generation_kwargs['seed'] = current_seed
            
            logger.info(f"   Variation {i+1}/{num_variations} (seed: {current_seed})")
            
            success, image, message = self.generate_baby_face(
                ultrasound_image, 
                quality_level=quality_level,
                **generation_kwargs
            )
            
            if success and image is not None:
                variations.append(image)
                successful_generations += 1
                logger.info(f"   ✅ Variation {i+1} generated successfully")
            else:
                logger.error(f"   ❌ Variation {i+1} failed: {message}")
        
        if successful_generations > 0:
            return True, variations, f"✅ Generated {successful_generations}/{num_variations} professional variations using GGUF model"
        else:
            return False, [], "❌ All professional variations failed to generate"
    
    def unload_model(self):
        """Unload model and free memory"""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
        self.model_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("🗑️ Professional model resources cleaned up")


def main():
    """Simple test for Qwen Image Edit diffusers model"""
    model = QwenImageEditModel()
    logger.info("🚀 Testing diffusers model loading...")
    success = model.load_model()
    
    if success:
        logger.info("✅ Diffusers model loaded successfully!")
        
        # Test with sample image
        sample_path = "/home/ubuntu/DataScience/MyProject/BabyVis/samples/1.jpeg"
        
        if os.path.exists(sample_path):
            # Load test ultrasound image
            ultrasound_image = Image.open(sample_path)
            logger.info(f"📷 Loaded test image: {sample_path}")
            
            # Generate baby face
            success, baby_image, message = model.generate_baby_face(
                ultrasound_image,
                quality_level="enhanced",
                num_inference_steps=25,
                guidance_scale=7.5,
                strength=0.8,
                seed=42                   # Reproducible results
            )
            
            if success and baby_image:
                output_path = "/home/ubuntu/DataScience/MyProject/BabyVis/outputs/qwen_baby.png"
                baby_image.save(output_path)
                logger.info(f"✅ Professional baby face saved: {output_path}")
                logger.info(f"📄 Result: {message}")
                
                # Test batch generation
                logger.info("🎨 Testing batch generation...")
                batch_success, variations, batch_message = model.batch_generate_professional(
                    ultrasound_image,
                    num_variations=3,
                    quality_level="enhanced",
                    num_inference_steps=20,
                    strength=0.75
                )
                
                if batch_success:
                    for i, variation in enumerate(variations):
                        var_path = f"/home/ubuntu/DataScience/MyProject/BabyVis/outputs/qwen_variation_{i+1}.png"
                        variation.save(var_path)
                        logger.info(f"💎 Variation {i+1} saved: {var_path}")
                    
                    logger.info(f"✅ Batch generation: {batch_message}")
                else:
                    logger.error(f"❌ Batch generation failed: {batch_message}")
                
            else:
                logger.error(f"❌ Professional generation failed: {message}")
        else:
            logger.error(f"❌ Sample image not found: {sample_path}")
            
        # Cleanup
        model.unload_model()
        
    else:
        logger.error("❌ Failed to load diffusers model!")


if __name__ == "__main__":
    main()
