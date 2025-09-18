#!/usr/bin/env python3
"""
Qwen Image Edit Integration for BabyVis
Core model handler using GGUF and QuantStack/Qwen-Image-Edit-GGUF for professional baby face generation
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
from gguf_model_loader import GGUFModelLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenImageEditModel:
    """
    Professional Qwen Image Edit Model using GGUF quantized models
    Specialized for ultrasound to baby face transformation with maximum quality
    """
    
    def __init__(self, model_id: str = "QuantStack/Qwen-Image-Edit-GGUF", device: str = "auto"):
        self.model_id = model_id
        self.device = self._setup_device(device)
        
        # Initialize the professional GGUF loader
        self.gguf_loader = GGUFModelLoader(model_id=model_id, device=device)
        self.model_loaded = False
        
        # Professional baby generation prompts for different quality levels
        self.baby_prompts = {
            "base": "beautiful newborn baby face, peaceful sleeping, soft baby skin, cute innocent features, realistic portrait photography, natural lighting, high definition",
            "enhanced": "adorable newborn baby portrait, perfect delicate skin, gentle peaceful expression, professional hospital photography, warm soft lighting, detailed facial features, ultra realistic, 8K quality",
            "premium": "stunning newborn baby photograph, crystal clear baby skin, perfect proportions, angelic expression, professional portrait studio lighting, hospital-grade photography, photorealistic, award-winning baby photography, ultra high definition"
        }
        
        # Negative prompts to avoid unwanted features
        self.negative_prompt = "adult face, child face, teenager, elderly, wrinkles, facial hair, beard, mustache, distorted features, blurry image, low quality, deformed face, scary expression, dark lighting, cartoon, anime, artificial, fake, unrealistic proportions"
        
        # GGUF specific configurations for optimal performance
        self.gguf_config = {
            "quantization": "auto",  # Will auto-select best available (Q8_0, Q5_1, Q4_1, etc.)
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
                print(f"💾 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                device = "cpu"
                print("💻 Using CPU (GPU not available)")
        
        return torch.device(device)
    
    def load_model(self, quantization: str = "auto", use_safetensors: bool = True) -> bool:
        """
        Load the professional QuantStack/Qwen-Image-Edit-GGUF model
        
        Args:
            quantization: GGUF quantization level (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, F32, auto)
            use_safetensors: Use safetensors format (always True for GGUF)
            
        Returns:
            bool: True if model loaded successfully
        """
        if self.model_loaded:
            logger.info("✅ Professional GGUF model already loaded")
            return True
            
        try:
            logger.info(f"🚀 Loading professional GGUF model: {self.model_id}")
            logger.info(f"🎯 Quantization preference: {quantization}")
            
            # Load the GGUF model using our professional loader
            success = self.gguf_loader.load_model(quantization=quantization)
            
            if success:
                self.model_loaded = True
                logger.info("✅ Professional GGUF model loaded successfully!")
                logger.info(f"� Model: {self.gguf_loader.model_id}")
                logger.info(f"🔧 Device: {self.device}")
                return True
            else:
                logger.error("❌ Failed to load professional GGUF model")
                return self._create_fallback_model()
                
        except Exception as e:
            logger.error(f"❌ Error loading GGUF model: {e}")
            return self._create_fallback_model()
    
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
    
    def main():
    """Test the professional GGUF Qwen Image Edit model"""
    # Initialize professional GGUF model
    model = QwenImageEditModel(model_id="QuantStack/Qwen-Image-Edit-GGUF")
    
    logger.info("🚀 Testing professional GGUF model loading...")
    
    # Load the model with auto quantization
    success = model.load_model(quantization="auto")
    
    if success:
        logger.info("✅ Professional GGUF model loaded successfully!")
        
        # Test with sample image
        sample_path = "/home/ubuntu/DataScience/MyProject/BabyVis/samples/1.jpeg"
        
        if os.path.exists(sample_path):
            # Load test ultrasound image
            ultrasound_image = Image.open(sample_path)
            logger.info(f"📷 Loaded test image: {sample_path}")
            
            # Generate professional baby face
            success, baby_image, message = model.generate_baby_face(
                ultrasound_image,
                quality_level="premium",  # Use highest quality
                num_inference_steps=40,   # More steps for better quality
                guidance_scale=8.0,       # Strong prompt adherence
                strength=0.85,            # Strong transformation
                seed=42                   # Reproducible results
            )
            
            if success and baby_image:
                output_path = "/home/ubuntu/DataScience/MyProject/BabyVis/outputs/gguf_professional_baby.png"
                baby_image.save(output_path)
                logger.info(f"✅ Professional baby face saved: {output_path}")
                logger.info(f"📄 Result: {message}")
                
                # Test batch generation
                logger.info("🎨 Testing professional batch generation...")
                batch_success, variations, batch_message = model.batch_generate_professional(
                    ultrasound_image,
                    num_variations=3,
                    quality_level="enhanced",
                    num_inference_steps=30,
                    strength=0.8
                )
                
                if batch_success:
                    for i, variation in enumerate(variations):
                        var_path = f"/home/ubuntu/DataScience/MyProject/BabyVis/outputs/gguf_variation_{i+1}.png"
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
        logger.error("❌ Failed to load professional GGUF model!")
        logger.error("❌ CRITICAL: Cannot meet user requirement for strong models only!")


if __name__ == "__main__":
    main()
    
    def _create_dummy_model(self) -> bool:
        """Create a dummy model that applies image processing only"""
        print("🔧 Creating dummy processing model...")
        self.pipeline = None
        self.model_loaded = True
        self.model_id = "dummy-processing-model"
        print("✅ Dummy model created (will use image processing only)")
        return True
    
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
        if hasattr(self, 'gguf_loader'):
            self.gguf_loader.unload_model()
        
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
        self.model_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("🗑️ Professional model resources cleaned up")


def main():
    """Test the professional GGUF Qwen Image Edit model"""
    
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
            
            logger.info(f"🎨 Generating professional baby face with GGUF model...")
            logger.info(f"   Quality: {quality_level} | Steps: {num_inference_steps}")
            logger.info(f"   Guidance: {guidance_scale} | Strength: {strength}")
            logger.info(f"   Model: {self.gguf_loader.model_id}")
            
            # Check if we have GGUF model or fallback
            if hasattr(self.gguf_loader, 'model_loaded') and self.gguf_loader.model_loaded:
                # Use professional GGUF model
                success, baby_image, message = self.gguf_loader.generate_image(
                    prompt=prompt,
                    image=processed_image,
                    negative_prompt=self.negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    seed=seed
                )
                
                if success and baby_image:
                    # Apply professional post-processing
                    baby_image = self._postprocess_baby_face_professional(baby_image, quality_level)
                    
                    return True, baby_image, f"✅ Professional baby face generated using GGUF {self.gguf_loader.model_id} ({quality_level} quality)"
                else:
                    logger.error(f"❌ GGUF generation failed: {message}")
                    return False, None, f"❌ GGUF model generation failed: {message}"
            
            elif hasattr(self, 'pipeline') and self.pipeline is not None:
                # Use fallback diffusers pipeline (regular Qwen model)
                logger.info("🔄 Using fallback Qwen diffusers pipeline...")
                
                with torch.autocast(self.device.type if self.device.type == "cuda" else "cpu"):
                    result = self.pipeline(
                        prompt=prompt,
                        negative_prompt=self.negative_prompt,
                        image=processed_image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        return_dict=True
                    )
                
                baby_image = result.images[0]
                
                # Apply professional post-processing
                baby_image = self._postprocess_baby_face_professional(baby_image, quality_level)
                
                # Clean up GPU memory
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                
                return True, baby_image, f"✅ Baby face generated using fallback Qwen model ({quality_level} quality)"
            
            else:
                return False, None, "❌ CRITICAL: No strong model available (GGUF or Qwen)!"
            
        except Exception as e:
            error_msg = f"❌ Professional generation error: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def _generate_with_image_processing(
        self, 
        processed_image: Image.Image, 
        quality_level: str
    ) -> Tuple[bool, Image.Image, str]:
        """Generate baby face using advanced image processing only"""
        try:
            # Import image processing utilities
            from image_utils import BabyFaceProcessor
            
            # Apply advanced baby transformation
            baby_processor = BabyFaceProcessor()
            
            # Determine intensity based on quality level
            intensity_map = {"base": 0.3, "enhanced": 0.5, "premium": 0.7}
            intensity = intensity_map.get(quality_level, 0.5)
            
            # Apply baby-like transformations
            baby_image = baby_processor.enhance_baby_features(processed_image, intensity=intensity)
            baby_image = baby_processor.add_baby_glow(baby_image, glow_intensity=intensity * 0.4)
            
            # Additional post-processing
            baby_image = self._postprocess_baby_face(baby_image)
            
            return True, baby_image, f"Baby face created using advanced image processing ({quality_level} quality)"
            
        except Exception as e:
            return False, None, f"Image processing fallback failed: {str(e)}"
    
    def _postprocess_baby_face(self, image: Image.Image) -> Image.Image:
        """
        Apply post-processing to enhance baby-like features
        """
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Enhance skin tone for baby-like warmth
        # Slightly increase red and reduce blue for warmer skin
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.05, 0, 255)  # Red
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.95, 0, 255)  # Blue
        
        # Convert back to PIL
        processed_image = Image.fromarray(img_array.astype(np.uint8))
        
        # Apply gentle smoothing for baby-soft skin
        processed_image = processed_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Enhance brightness slightly for healthy baby glow
        enhancer = ImageEnhance.Brightness(processed_image)
        processed_image = enhancer.enhance(1.05)
        
        # Increase color saturation slightly
        enhancer = ImageEnhance.Color(processed_image)
        processed_image = enhancer.enhance(1.1)
        
        return processed_image
    
    def batch_generate(
        self,
        ultrasound_image: Image.Image,
        num_variations: int = 3,
        **generation_kwargs
    ) -> Tuple[bool, list, str]:
        """
        Generate multiple baby face variations
        """
        variations = []
        successful_generations = 0
        
        for i in range(num_variations):
            # Use different seeds for variation
            generation_kwargs['seed'] = generation_kwargs.get('seed', 42) + i
            
            success, image, message = self.generate_baby_face(ultrasound_image, **generation_kwargs)
            
            if success and image is not None:
                variations.append(image)
                successful_generations += 1
            else:
                print(f"⚠️ Variation {i+1} failed: {message}")
        
        if successful_generations > 0:
            return True, variations, f"Generated {successful_generations}/{num_variations} variations"
        else:
            return False, [], "All variations failed to generate"
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.model_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            print("🗑️ Model unloaded and memory cleared")


def main():
    """Test the Qwen Image Edit model"""
    # Initialize model
    model = QwenImageEditModel()
    
    # Test with sample image
    sample_path = "/home/ubuntu/DataScience/MyProject/BabyVis/samples/1.jpeg"
    
    if os.path.exists(sample_path):
        # Load test image
        ultrasound_image = Image.open(sample_path)
        
        # Generate baby face
        success, baby_image, message = model.generate_baby_face(
            ultrasound_image,
            quality_level="enhanced",
            num_inference_steps=25,
            strength=0.8
        )
        
        if success and baby_image:
            output_path = "/home/ubuntu/DataScience/MyProject/BabyVis/outputs/qwen_baby_test.png"
            baby_image.save(output_path)
            print(f"✅ Test successful! Baby face saved to: {output_path}")
            print(f"📄 Message: {message}")
        else:
            print(f"❌ Test failed: {message}")
    else:
        print(f"❌ Sample image not found: {sample_path}")


if __name__ == "__main__":
    main()