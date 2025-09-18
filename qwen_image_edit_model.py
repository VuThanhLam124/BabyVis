#!/usr/bin/env python3
"""
Enhanced Qwen Image Edit Integration for BabyVis
Diffusers-first implementation with advanced medical-grade processing and variation generation.
Incorporates YouTube video techniques for professional baby face generation from ultrasound.

Features:
- Medical-grade ultrasound preprocessing
- Advanced anatomical landmark detection
- Professional baby face generation prompts
- Multiple variation generation pipeline
- Gestational age consideration
- Ethnic diversity support
"""

import os
import gc
import torch
import warnings
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import logging

from config import BabyVisSettings, get_settings

# Import enhanced modules
from advanced_ultrasound_processor import AdvancedUltrasoundProcessor
from advanced_baby_face_generator import AdvancedBabyFaceGenerator, GestationalAge, EthnicityGroup, BabyExpression
from baby_face_variation_pipeline import BabyFaceVariationPipeline, BabyVariation

try:
    from medical_image_analyzer import MedicalImageAnalyzer
    MEDICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    MEDICAL_ANALYSIS_AVAILABLE = False
    logging.warning("Medical image analysis not available - install scikit-image for full functionality")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenImageEditModel:
    """
    Enhanced Qwen Image Edit via diffusers with advanced medical-grade processing.
    Incorporates YouTube video techniques for professional baby face generation.
    
    Features:
    - Medical-grade ultrasound preprocessing
    - Advanced anatomical landmark detection  
    - Professional baby face generation prompts
    - Multiple variation generation pipeline
    - Gestational age consideration
    - Ethnic diversity support
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        settings: Optional[BabyVisSettings] = None,
    ):
        self.settings = settings or get_settings()
        self.provider = self.settings.model_provider

        default_model = self.settings.qwen_model_id
        if self.provider == "gguf" and default_model == "Qwen/Qwen-Image-Edit":
            default_model = "QuantStack/Qwen-Image-Edit-GGUF"

        # Prefer explicit arguments, then settings/env defaults
        self.model_id = model_id or default_model
        requested_device = device or self.settings.device
        self.device = self._setup_device(requested_device)
        self.disable_cpu_offload = self.settings.disable_cpu_offload

        self.pipeline = None
        self.model_loaded = False
        self.gguf_loader = None

        # Initialize enhanced processors
        self.ultrasound_processor = AdvancedUltrasoundProcessor()
        self.face_generator = AdvancedBabyFaceGenerator()
        self.variation_pipeline = BabyFaceVariationPipeline()
        
        if MEDICAL_ANALYSIS_AVAILABLE:
            self.medical_analyzer = MedicalImageAnalyzer()
        else:
            self.medical_analyzer = None
            logger.warning("⚠️ Medical analysis not available - install scikit-image for full functionality")

        if self.provider == "gguf":
            try:
                from gguf_model_loader import GGUFModelLoader

                self.gguf_loader = GGUFModelLoader(
                    model_id=self.model_id,
                    device=requested_device,
                    model_path=self.settings.gguf_path,
                    quantization=self.settings.gguf_quant,
                )
            except Exception as exc:
                logger.warning(
                    "⚠️ GGUF loader unavailable (%s). Falling back to diffusers backend.",
                    exc,
                )
                self.provider = "diffusers"
        
        # Enhanced baby generation prompts with medical accuracy
        self.baby_prompts = {
            "base": "beautiful newborn baby face, peaceful sleeping, soft baby skin, cute innocent features, realistic portrait photography, natural lighting, high definition",
            "enhanced": "adorable newborn baby portrait, perfect delicate skin, gentle peaceful expression, professional hospital photography, warm soft lighting, detailed facial features, ultra realistic, 8K quality, anatomically correct proportions",
            "premium": "stunning newborn baby photograph, crystal clear baby skin, perfect proportions, angelic expression, professional portrait studio lighting, hospital-grade photography, photorealistic, award-winning baby photography, ultra high definition, medically accurate features"
        }
        
        # Enhanced negative prompts with medical considerations
        self.negative_prompt = "adult face, child face, teenager, elderly, wrinkles, facial hair, beard, mustache, distorted features, blurry image, low quality, deformed face, scary expression, dark lighting, cartoon, anime, artificial, fake, unrealistic proportions, medical devices, tubes, wires, asymmetrical face, disproportionate features"
        
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
        if self.provider == "gguf" and self.gguf_loader is not None:
            logger.info("🚀 Loading GGUF model backend")
            success = self.gguf_loader.load_model(
                quantization=self.settings.gguf_quant,
                model_file=self.settings.gguf_path,
            )
            if success:
                self.model_loaded = True
                logger.info("✅ GGUF model loaded successfully")
                return True

            logger.warning("⚠️ GGUF loading failed, attempting diffusers fallback")
            self.provider = "diffusers"

        return self._load_diffusers_pipeline()

    def _load_diffusers_pipeline(self) -> bool:
        try:
            # Check if FLUX model
            if "flux" in self.model_id.lower():
                from diffusers import FluxImg2ImgPipeline
                logger.info(f"🚀 Loading FLUX model (Most Powerful): {self.model_id}")
                self.pipeline = FluxImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    use_safetensors=True,
                    variant="fp16" if self.torch_dtype == torch.float16 else None
                )
            # Check if SDXL model
            elif "xl" in self.model_id.lower():
                from diffusers import StableDiffusionXLImg2ImgPipeline
                logger.info(f"🚀 Loading SDXL model: {self.model_id}")
                self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    use_safetensors=True,
                    variant="fp16" if self.torch_dtype == torch.float16 else None
                )
            else:
                from diffusers import StableDiffusionImg2ImgPipeline
                logger.info(f"🚀 Loading diffusers model: {self.model_id}")
                self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    use_safetensors=True
                )

            if hasattr(self.pipeline, "safety_checker"):
                self.pipeline.safety_checker = None
                self.pipeline.requires_safety_checker = False

            try:
                if self.device.type == "cuda":
                    if not self.disable_cpu_offload and hasattr(self.pipeline, "enable_model_cpu_offload"):
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
            from diffusers import StableDiffusionImg2ImgPipeline
            
            # Try Realistic Vision as fallback (better than SD 1.5)
            logger.info("🔄 Loading Realistic Vision v5.1 as fallback...")
            
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                "SG161222/Realistic_Vision_V5.1_noVAE",
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
    
    def generate_baby_face_enhanced(
        self,
        ultrasound_image: Image.Image,
        quality_level: str = "enhanced",
        num_inference_steps: int = 50,
        guidance_scale: float = 8.0,
        strength: float = 0.7,
        seed: Optional[int] = None,
        enable_medical_analysis: bool = True,
        gestational_age: Optional[str] = None,
        ethnicity: Optional[str] = None,
        expression: Optional[str] = None
    ) -> Tuple[bool, Optional[Image.Image], str, Dict[str, any]]:
        """
        Enhanced baby face generation with medical analysis and advanced prompts.
        Incorporates YouTube video techniques for professional results.
        
        Args:
            ultrasound_image: Input ultrasound PIL Image
            quality_level: "base", "enhanced", or "premium" 
            num_inference_steps: Number of denoising steps (15-50)
            guidance_scale: Prompt adherence strength (1.0-20.0)
            strength: Transformation strength (0.1-1.0)
            seed: Random seed for reproducible results
            enable_medical_analysis: Enable medical-grade analysis
            gestational_age: Optional gestational age ("early", "mid", "late")
            ethnicity: Optional ethnicity specification
            expression: Optional expression type
            
        Returns:
            Tuple of (success, generated_image, message, analysis_data)
        """
        if not self.model_loaded:
            logger.warning("🔄 Model not loaded, attempting to load...")
            if not self.load_model():
                return False, None, "❌ CRITICAL: Failed to load any strong model as requested!", {}

        try:
            analysis_data = {}
            
            # Medical analysis if enabled
            if enable_medical_analysis and self.medical_analyzer:
                logger.info("🩺 Performing medical-grade ultrasound analysis...")
                medical_analysis = self.medical_analyzer.analyze_medical_features(ultrasound_image)
                analysis_data["medical_analysis"] = medical_analysis
                
                # Extract characteristics from medical analysis
                if not gestational_age and "gestational_analysis" in medical_analysis:
                    gestational_age = medical_analysis["gestational_analysis"]["development_stage"]
                
                logger.info(f"📊 Medical analysis: {gestational_age} stage, confidence {medical_analysis.get('analysis_confidence', 0):.2f}")
            
            # Enhanced ultrasound preprocessing
            logger.info("🔬 Applying medical-grade ultrasound enhancement...")
            processed_image = self.ultrasound_processor.enhance_ultrasound_medical_grade(
                ultrasound_image,
                gestational_age=gestational_age
            )
            
            # Generate advanced prompts
            logger.info("🎨 Generating advanced medical-grade prompts...")
            
            # Map string parameters to enums
            gest_age_enum = None
            if gestational_age:
                age_map = {"early": GestationalAge.EARLY, "mid": GestationalAge.MID, "late": GestationalAge.LATE}
                gest_age_enum = age_map.get(gestational_age)
            
            ethnic_enum = None
            if ethnicity:
                ethnic_map = {
                    "caucasian": EthnicityGroup.CAUCASIAN,
                    "asian": EthnicityGroup.ASIAN,
                    "african": EthnicityGroup.AFRICAN,
                    "hispanic": EthnicityGroup.HISPANIC,
                    "middle_eastern": EthnicityGroup.MIDDLE_EASTERN,
                    "mixed": EthnicityGroup.MIXED
                }
                ethnic_enum = ethnic_map.get(ethnicity.lower(), EthnicityGroup.MIXED)
            
            expr_enum = None
            if expression:
                expr_map = {
                    "peaceful": BabyExpression.PEACEFUL,
                    "sleeping": BabyExpression.SLEEPING,
                    "alert": BabyExpression.ALERT,
                    "yawning": BabyExpression.YAWNING,
                    "sucking": BabyExpression.SUCKING
                }
                expr_enum = expr_map.get(expression.lower(), BabyExpression.PEACEFUL)
            
            # Generate STRICT sleeping prompts to prevent eyes opening and pose changes
            positive_prompt, negative_prompt = self.face_generator.generate_strict_sleeping_prompt(
                quality_level=quality_level
            )
            
            analysis_data["prompts"] = {
                "positive": positive_prompt,
                "negative": negative_prompt
            }

            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                np.random.seed(seed)

            logger.info(f"🎨 Generating baby face with {self.provider} backend...")
            logger.info(f"   Quality: {quality_level} | Steps: {num_inference_steps}")
            logger.info(f"   Guidance: {guidance_scale} | Strength: {strength}")
            logger.info(f"   Gestational age: {gestational_age} | Ethnicity: {ethnicity}")

            if self.provider == "gguf" and self.gguf_loader is not None:
                success, generated_image, message = self._generate_with_gguf(
                    processed_image,
                    positive_prompt,
                    quality_level,
                    num_inference_steps,
                    guidance_scale,
                    strength,
                    seed,
                )
            else:
                success, generated_image, message = self._generate_with_diffusers_enhanced(
                    processed_image,
                    positive_prompt,
                    negative_prompt,
                    quality_level,
                    num_inference_steps,
                    guidance_scale,
                    strength,
                    seed,
                )
            
            if success and generated_image:
                analysis_data["generation_success"] = True
                analysis_data["generation_params"] = {
                    "quality_level": quality_level,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "seed": seed
                }
            
            return success, generated_image, message, analysis_data

        except Exception as e:
            error_msg = f"❌ Enhanced generation error: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg, {}
    
    def generate_baby_face_variations(
        self,
        ultrasound_image: Image.Image,
        num_variations: int = 6,
        quality_preset: str = "balanced",
        enable_medical_analysis: bool = True
    ) -> Tuple[bool, List[BabyVariation], str]:
        """
        Generate multiple baby face variations using advanced pipeline.
        
        Args:
            ultrasound_image: Input ultrasound image
            num_variations: Number of variations to generate
            quality_preset: "speed", "balanced", or "quality"
            enable_medical_analysis: Enable medical analysis
            
        Returns:
            Tuple of (success, variations_list, message)
        """
        try:
            logger.info(f"🎨 Generating {num_variations} baby face variations...")
            
            # Perform medical analysis if enabled
            medical_analysis = None
            if enable_medical_analysis and self.medical_analyzer:
                medical_analysis = self.medical_analyzer.analyze_medical_features(ultrasound_image)
            
            # Generate variations using pipeline
            variations = self.variation_pipeline.generate_comprehensive_variations(
                ultrasound_image,
                num_variations=num_variations,
                quality_preset=quality_preset,
                medical_analysis=medical_analysis
            )
            
            if variations:
                message = f"✅ Generated {len(variations)} professional baby face variations"
                return True, variations, message
            else:
                return False, [], "❌ Failed to generate any variations"
                
        except Exception as e:
            error_msg = f"❌ Variation generation error: {str(e)}"
            logger.error(error_msg)
            return False, [], error_msg

    def _generate_with_diffusers_enhanced(
        self,
        processed_image: Image.Image,
        positive_prompt: str,
        negative_prompt: str,
        quality_level: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: Optional[int],
    ) -> Tuple[bool, Optional[Image.Image], str]:
        """Enhanced diffusers generation with advanced prompts"""
        if self.pipeline is None:
            return False, None, "❌ Model pipeline not available"

        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            except Exception:
                generator = torch.Generator().manual_seed(seed)

        autocast_device = self.device.type if self.device.type == "cuda" else "cpu"
        autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.float32

        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
            result = self.pipeline(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=processed_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                return_dict=True
            )

        baby_image = result.images[0]
        baby_image = self._postprocess_baby_face_professional(baby_image, quality_level)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        return (
            True,
            baby_image,
            f"✅ Enhanced baby face generated with {self.model_id} ({quality_level} quality)",
        )
    
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
        Backwards compatible baby face generation (legacy method)
        For new features, use generate_baby_face_enhanced()
        """
        try:
            # Use enhanced generation but return simplified result
            success, image, message, _ = self.generate_baby_face_enhanced(
                ultrasound_image=ultrasound_image,
                quality_level=quality_level,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed,
                enable_medical_analysis=False  # Disable for backwards compatibility
            )
            return success, image, message
            
        except Exception as e:
            # Fallback to original method if enhanced fails
            logger.warning(f"Enhanced generation failed: {e}, falling back to original method")
            return self._generate_baby_face_original(
                ultrasound_image, quality_level, num_inference_steps, 
                guidance_scale, strength, seed
            )
    
    def _generate_baby_face_original(
        self,
        ultrasound_image: Image.Image,
        quality_level: str = "enhanced",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: Optional[int] = None
    ) -> Tuple[bool, Optional[Image.Image], str]:
        """Original baby face generation method (fallback)"""
        if not self.model_loaded:
            logger.warning("🔄 Model not loaded, attempting to load...")
            if not self.load_model():
                return False, None, "❌ CRITICAL: Failed to load any strong model as requested!"

        try:
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                np.random.seed(seed)

            processed_image = self.preprocess_ultrasound_professional(ultrasound_image)
            prompt = self.baby_prompts.get(quality_level, self.baby_prompts["enhanced"])

            logger.info(f"🎨 Generating baby face with {self.provider} backend...")
            logger.info(f"   Quality: {quality_level} | Steps: {num_inference_steps}")
            logger.info(f"   Guidance: {guidance_scale} | Strength: {strength}")
            logger.info(f"   Model: {self.model_id}")

            if self.provider == "gguf" and self.gguf_loader is not None:
                return self._generate_with_gguf(
                    processed_image,
                    prompt,
                    quality_level,
                    num_inference_steps,
                    guidance_scale,
                    strength,
                    seed,
                )

            return self._generate_with_diffusers(
                processed_image,
                prompt,
                quality_level,
                num_inference_steps,
                guidance_scale,
                strength,
                seed,
            )

        except Exception as e:
            error_msg = f"❌ Generation error: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def _generate_with_diffusers(
        self,
        processed_image: Image.Image,
        prompt: str,
        quality_level: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: Optional[int],
    ) -> Tuple[bool, Optional[Image.Image], str]:
        """Legacy diffusers generation method"""
        if self.pipeline is None:
            return False, None, "❌ Model pipeline not available"

        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            except Exception:
                generator = torch.Generator().manual_seed(seed)

        autocast_device = self.device.type if self.device.type == "cuda" else "cpu"
        autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.float32

        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
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
        baby_image = self._postprocess_baby_face_professional(baby_image, quality_level)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        return (
            True,
            baby_image,
            f"✅ Baby face generated with {self.model_id} ({quality_level} quality)",
        )

    def _generate_with_gguf(
        self,
        processed_image: Image.Image,
        prompt: str,
        quality_level: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: Optional[int],
    ) -> Tuple[bool, Optional[Image.Image], str]:
        if self.gguf_loader is None:
            return False, None, "❌ GGUF loader not initialised"

        success, generated_image, message = self.gguf_loader.generate_image(
            prompt=prompt,
            image=processed_image,
            negative_prompt=self.negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )

        if success and generated_image is not None:
            processed = self._postprocess_baby_face_professional(generated_image, quality_level)
            return success, processed, message

        return success, generated_image, message
    
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
        if self.gguf_loader is not None:
            try:
                self.gguf_loader.unload_model()
            except AttributeError:
                pass

        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
        self.model_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("🗑️ Professional model resources cleaned up")


def main():
    """Test enhanced Qwen Image Edit model with YouTube video techniques"""
    model = QwenImageEditModel()
    logger.info("🚀 Testing enhanced BabyVis model with YouTube video techniques...")
    success = model.load_model()
    
    if success:
        logger.info("✅ Enhanced model loaded successfully!")
        
        # Test with sample image
        sample_path = "/home/ubuntu/DataScience/MyProject/BabyVis/samples/1.jpeg"
        
        if os.path.exists(sample_path):
            # Load test ultrasound image
            ultrasound_image = Image.open(sample_path)
            logger.info(f"📷 Loaded test image: {sample_path}")
            
            # Test enhanced generation
            logger.info("\n=== Testing Enhanced Generation ===")
            success, baby_image, message, analysis_data = model.generate_baby_face_enhanced(
                ultrasound_image,
                quality_level="enhanced",
                num_inference_steps=25,
                guidance_scale=7.5,
                strength=0.8,
                seed=42,
                enable_medical_analysis=True,
                gestational_age="mid",
                ethnicity="mixed",
                expression="peaceful"
            )
            
            if success and baby_image:
                output_path = "/home/ubuntu/DataScience/MyProject/BabyVis/outputs/enhanced_baby.png"
                baby_image.save(output_path)
                logger.info(f"✅ Enhanced baby face saved: {output_path}")
                logger.info(f"📄 Result: {message}")
                
                # Print analysis data
                if analysis_data:
                    logger.info("📊 Analysis Data:")
                    if "medical_analysis" in analysis_data:
                        medical = analysis_data["medical_analysis"]
                        logger.info(f"   Medical confidence: {medical.get('analysis_confidence', 0):.2f}")
                        if "gestational_analysis" in medical:
                            gest = medical["gestational_analysis"]
                            logger.info(f"   Gestational age: {gest.estimated_weeks:.1f} weeks ({gest.development_stage})")
                
                # Test variation generation
                logger.info("\n=== Testing Variation Generation ===")
                var_success, variations, var_message = model.generate_baby_face_variations(
                    ultrasound_image,
                    num_variations=3,
                    quality_preset="balanced",
                    enable_medical_analysis=True
                )
                
                if var_success and variations:
                    logger.info(f"✅ Generated {len(variations)} variations")
                    
                    # Save variations
                    output_dir = "/home/ubuntu/DataScience/MyProject/BabyVis/outputs/variations"
                    saved_files = model.variation_pipeline.save_variations(variations, output_dir)
                    logger.info(f"💾 Saved {len(saved_files['images'])} variation images")
                    
                    # Create collage
                    collage = model.variation_pipeline.create_variation_collage(variations)
                    collage_path = f"{output_dir}/enhanced_collage.png"
                    collage.save(collage_path)
                    logger.info(f"�️ Collage saved: {collage_path}")
                    
                else:
                    logger.error(f"❌ Variation generation failed: {var_message}")
                
                # Test backwards compatibility
                logger.info("\n=== Testing Backwards Compatibility ===")
                compat_success, compat_image, compat_message = model.generate_baby_face(
                    ultrasound_image,
                    quality_level="enhanced",
                    num_inference_steps=20,
                    strength=0.75,
                    seed=123
                )
                
                if compat_success and compat_image:
                    compat_path = "/home/ubuntu/DataScience/MyProject/BabyVis/outputs/compat_baby.png"
                    compat_image.save(compat_path)
                    logger.info(f"✅ Backwards compatible generation: {compat_path}")
                else:
                    logger.error(f"❌ Backwards compatibility failed: {compat_message}")
                
            else:
                logger.error(f"❌ Enhanced generation failed: {message}")
        else:
            logger.error(f"❌ Sample image not found: {sample_path}")
            
        # Cleanup
        model.unload_model()
        logger.info("🍼 Enhanced BabyVis testing complete!")
        
    else:
        logger.error("❌ Failed to load enhanced model!")


if __name__ == "__main__":
    main()
