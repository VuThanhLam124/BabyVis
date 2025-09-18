#!/usr/bin/env python3
"""
Advanced Baby Face Variation Pipeline for BabyVis
Generate multiple realistic baby face variations with different expressions, angles, and characteristics
Implements YouTube video techniques for comprehensive baby appearance prediction
"""

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path

from advanced_baby_face_generator import AdvancedBabyFaceGenerator, GestationalAge, EthnicityGroup, BabyExpression
from advanced_ultrasound_processor import AdvancedUltrasoundProcessor
from medical_image_analyzer import MedicalImageAnalyzer

logger = logging.getLogger(__name__)

class VariationType(Enum):
    EXPRESSION = "expression"
    LIGHTING = "lighting"
    ANGLE = "angle"
    GENETIC = "genetic"
    DEVELOPMENT = "development"

@dataclass
class BabyVariation:
    """Represents a generated baby face variation"""
    variation_id: str
    image: Image.Image
    variation_type: VariationType
    characteristics: Dict[str, any]
    generation_params: Dict[str, any]
    confidence_score: float
    description: str

class BabyFaceVariationPipeline:
    """
    Advanced pipeline for generating multiple realistic baby face variations
    Implements professional techniques for comprehensive baby appearance prediction
    """
    
    def __init__(self):
        self.face_generator = AdvancedBabyFaceGenerator()
        self.ultrasound_processor = AdvancedUltrasoundProcessor()
        self.medical_analyzer = MedicalImageAnalyzer()
        
        # Variation templates for different types
        self.variation_templates = {
            VariationType.EXPRESSION: {
                "peaceful": {
                    "expression": BabyExpression.PEACEFUL,
                    "prompt_modifiers": ["serene expression", "calm relaxed face"],
                    "strength_adjustment": 0.0
                },
                "sleeping": {
                    "expression": BabyExpression.SLEEPING,
                    "prompt_modifiers": ["closed eyes", "sleeping peacefully"],
                    "strength_adjustment": -0.1
                },
                "alert": {
                    "expression": BabyExpression.ALERT,
                    "prompt_modifiers": ["bright eyes", "curious expression"],
                    "strength_adjustment": 0.1
                },
                "yawning": {
                    "expression": BabyExpression.YAWNING,
                    "prompt_modifiers": ["tiny yawn", "sleepy expression"],
                    "strength_adjustment": 0.05
                }
            },
            VariationType.LIGHTING: {
                "soft_natural": {
                    "lighting_modifier": "soft natural hospital lighting, gentle illumination",
                    "brightness_adjust": 1.1,
                    "contrast_adjust": 1.0
                },
                "warm_studio": {
                    "lighting_modifier": "warm studio lighting, professional baby photography",
                    "brightness_adjust": 1.15,
                    "contrast_adjust": 1.05
                },
                "golden_hour": {
                    "lighting_modifier": "golden hour lighting, warm sunset glow",
                    "brightness_adjust": 1.2,
                    "contrast_adjust": 1.1
                },
                "clinical": {
                    "lighting_modifier": "clean clinical lighting, medical photography",
                    "brightness_adjust": 1.05,
                    "contrast_adjust": 0.95
                }
            },
            VariationType.ANGLE: {
                "front_view": {
                    "angle_modifier": "direct front view, centered composition",
                    "perspective_hint": "straight-on baby portrait"
                },
                "slight_left": {
                    "angle_modifier": "gentle left profile, natural pose",
                    "perspective_hint": "baby turned slightly left"
                },
                "slight_right": {
                    "angle_modifier": "gentle right profile, natural pose", 
                    "perspective_hint": "baby turned slightly right"
                },
                "three_quarter": {
                    "angle_modifier": "three-quarter view, artistic angle",
                    "perspective_hint": "professional portrait angle"
                }
            },
            VariationType.GENETIC: {
                "variation_1": {
                    "genetic_modifier": "first genetic variation, unique family traits",
                    "randomness_boost": 0.1
                },
                "variation_2": {
                    "genetic_modifier": "second genetic variation, alternative features",
                    "randomness_boost": 0.2
                },
                "variation_3": {
                    "genetic_modifier": "third genetic variation, diverse characteristics",
                    "randomness_boost": 0.15
                }
            },
            VariationType.DEVELOPMENT: {
                "early_features": {
                    "development_modifier": "early facial development, delicate features",
                    "feature_emphasis": 0.8
                },
                "mature_features": {
                    "development_modifier": "mature facial development, defined features",
                    "feature_emphasis": 1.2
                },
                "balanced": {
                    "development_modifier": "balanced facial development, natural proportions",
                    "feature_emphasis": 1.0
                }
            }
        }
        
        # Quality presets for different variation goals
        self.quality_presets = {
            "speed": {
                "steps": 20,
                "guidance_scale": 7.0,
                "quality_level": "base"
            },
            "balanced": {
                "steps": 30,
                "guidance_scale": 7.5,
                "quality_level": "enhanced"
            },
            "quality": {
                "steps": 40,
                "guidance_scale": 8.0,
                "quality_level": "premium"
            }
        }
    
    def generate_comprehensive_variations(
        self,
        ultrasound_image: Image.Image,
        num_variations: int = 6,
        variation_types: Optional[List[VariationType]] = None,
        quality_preset: str = "balanced",
        medical_analysis: Optional[Dict] = None
    ) -> List[BabyVariation]:
        """
        Generate comprehensive set of baby face variations
        
        Args:
            ultrasound_image: Input ultrasound image
            num_variations: Number of variations to generate
            variation_types: Types of variations to include
            quality_preset: Quality/speed tradeoff ("speed", "balanced", "quality")
            medical_analysis: Pre-computed medical analysis (optional)
            
        Returns:
            List of BabyVariation objects
        """
        
        logger.info(f"🎨 Starting comprehensive baby face variation generation...")
        logger.info(f"   Variations: {num_variations} | Quality: {quality_preset}")
        
        # Default variation types if not specified
        if variation_types is None:
            variation_types = [
                VariationType.EXPRESSION,
                VariationType.LIGHTING,
                VariationType.ANGLE,
                VariationType.GENETIC
            ]
        
        # Perform medical analysis if not provided
        if medical_analysis is None:
            logger.info("🩺 Performing medical analysis...")
            medical_analysis = self.medical_analyzer.analyze_medical_features(ultrasound_image)
        
        # Enhanced ultrasound preprocessing
        logger.info("🔬 Enhancing ultrasound image...")
        gestational_age = self._determine_gestational_age(medical_analysis)
        enhanced_ultrasound = self.ultrasound_processor.enhance_ultrasound_medical_grade(
            ultrasound_image,
            gestational_age=gestational_age
        )
        
        # Determine base characteristics from medical analysis
        base_characteristics = self._extract_base_characteristics(medical_analysis)
        
        # Generate variations
        variations = []
        quality_params = self.quality_presets[quality_preset]
        
        # Distribute variations across types
        variations_per_type = max(1, num_variations // len(variation_types))
        
        variation_count = 0
        for variation_type in variation_types:
            if variation_count >= num_variations:
                break
                
            type_variations = min(variations_per_type, num_variations - variation_count)
            
            for i in range(type_variations):
                try:
                    variation = self._generate_single_variation(
                        enhanced_ultrasound,
                        variation_type,
                        base_characteristics,
                        quality_params,
                        variation_index=i
                    )
                    
                    if variation:
                        variations.append(variation)
                        variation_count += 1
                        logger.info(f"   ✅ Generated variation {variation_count}/{num_variations}: {variation.description}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to generate variation: {e}")
                    continue
        
        # Fill remaining slots with genetic variations if needed
        while len(variations) < num_variations:
            try:
                variation = self._generate_single_variation(
                    enhanced_ultrasound,
                    VariationType.GENETIC,
                    base_characteristics,
                    quality_params,
                    variation_index=len(variations)
                )
                
                if variation:
                    variations.append(variation)
                else:
                    break
                    
            except Exception as e:
                logger.error(f"   ❌ Failed to generate additional variation: {e}")
                break
        
        logger.info(f"🎯 Successfully generated {len(variations)} baby face variations")
        
        # Rank variations by quality
        ranked_variations = self._rank_variations(variations)
        
        return ranked_variations
    
    def _determine_gestational_age(self, medical_analysis: Dict) -> str:
        """Determine gestational age from medical analysis"""
        if "gestational_analysis" in medical_analysis:
            return medical_analysis["gestational_analysis"]["development_stage"]
        return "mid"  # Default
    
    def _extract_base_characteristics(self, medical_analysis: Dict) -> Dict[str, any]:
        """Extract base characteristics from medical analysis"""
        characteristics = {
            "quality_level": "enhanced",
            "ethnicity": EthnicityGroup.MIXED,  # Default to inclusive
            "gestational_age": GestationalAge.MID
        }
        
        # Update based on medical analysis
        if "gestational_analysis" in medical_analysis:
            stage = medical_analysis["gestational_analysis"]["development_stage"]
            if stage == "early":
                characteristics["gestational_age"] = GestationalAge.EARLY
            elif stage == "late":
                characteristics["gestational_age"] = GestationalAge.LATE
        
        return characteristics
    
    def _generate_single_variation(
        self,
        enhanced_ultrasound: Image.Image,
        variation_type: VariationType,
        base_characteristics: Dict[str, any],
        quality_params: Dict[str, any],
        variation_index: int = 0
    ) -> Optional[BabyVariation]:
        """Generate a single baby face variation"""
        
        # Get variation template
        templates = self.variation_templates[variation_type]
        template_keys = list(templates.keys())
        template_key = template_keys[variation_index % len(template_keys)]
        template = templates[template_key]
        
        # Prepare generation parameters
        generation_params = {
            "num_inference_steps": quality_params["steps"],
            "guidance_scale": quality_params["guidance_scale"],
            "strength": 0.8,
            "seed": random.randint(1000, 99999)
        }
        
        # Modify parameters based on variation type
        if variation_type == VariationType.EXPRESSION:
            base_characteristics["expression"] = template["expression"]
            generation_params["strength"] += template["strength_adjustment"]
        
        elif variation_type == VariationType.LIGHTING:
            # Lighting variations handled in post-processing
            pass
        
        elif variation_type == VariationType.GENETIC:
            generation_params["strength"] += template.get("randomness_boost", 0)
        
        # Generate advanced prompt
        positive_prompt, negative_prompt = self.face_generator.generate_advanced_prompt(
            quality_level=quality_params["quality_level"],
            gestational_age=base_characteristics.get("gestational_age"),
            ethnicity=base_characteristics.get("ethnicity"),
            expression=base_characteristics.get("expression", BabyExpression.PEACEFUL),
            custom_attributes=self._get_variation_attributes(variation_type, template)
        )
        
        # Generate baby face (placeholder - would integrate with your model)
        success, baby_image, message = self._call_generation_model(
            enhanced_ultrasound,
            positive_prompt,
            negative_prompt,
            generation_params
        )
        
        if not success or baby_image is None:
            logger.error(f"Generation failed: {message}")
            return None
        
        # Apply variation-specific post-processing
        processed_image = self._apply_variation_postprocessing(
            baby_image,
            variation_type,
            template
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_variation_confidence(
            processed_image,
            variation_type,
            generation_params
        )
        
        # Create variation object
        variation = BabyVariation(
            variation_id=str(uuid.uuid4()),
            image=processed_image,
            variation_type=variation_type,
            characteristics=base_characteristics.copy(),
            generation_params=generation_params,
            confidence_score=confidence_score,
            description=f"{variation_type.value.title()} variation: {template_key}"
        )
        
        return variation
    
    def _get_variation_attributes(self, variation_type: VariationType, template: Dict) -> List[str]:
        """Get custom attributes for variation type"""
        attributes = []
        
        if variation_type == VariationType.EXPRESSION and "prompt_modifiers" in template:
            attributes.extend(template["prompt_modifiers"])
        
        elif variation_type == VariationType.LIGHTING and "lighting_modifier" in template:
            attributes.append(template["lighting_modifier"])
        
        elif variation_type == VariationType.ANGLE and "angle_modifier" in template:
            attributes.extend([template["angle_modifier"], template["perspective_hint"]])
        
        elif variation_type == VariationType.GENETIC and "genetic_modifier" in template:
            attributes.append(template["genetic_modifier"])
        
        elif variation_type == VariationType.DEVELOPMENT and "development_modifier" in template:
            attributes.append(template["development_modifier"])
        
        return attributes
    
    def _call_generation_model(
        self,
        ultrasound_image: Image.Image,
        positive_prompt: str,
        negative_prompt: str,
        generation_params: Dict[str, any]
    ) -> Tuple[bool, Optional[Image.Image], str]:
        """
        Call the actual generation model
        This is a placeholder - integrate with your QwenImageEditModel
        """
        try:
            # Import here to avoid circular dependency
            from qwen_image_edit_model import QwenImageEditModel
            
            # Create model instance (reuse existing if available)
            model = QwenImageEditModel()
            
            # Generate baby face
            success, baby_image, message = model.generate_baby_face(
                ultrasound_image,
                quality_level=generation_params.get("quality_level", "enhanced"),
                num_inference_steps=generation_params["num_inference_steps"],
                guidance_scale=generation_params["guidance_scale"],
                strength=generation_params["strength"],
                seed=generation_params["seed"]
            )
            
            return success, baby_image, message
            
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            return False, None, f"Generation failed: {str(e)}"
    
    def _apply_variation_postprocessing(
        self,
        baby_image: Image.Image,
        variation_type: VariationType,
        template: Dict
    ) -> Image.Image:
        """Apply variation-specific post-processing"""
        
        processed = baby_image.copy()
        
        if variation_type == VariationType.LIGHTING:
            # Apply lighting adjustments
            if "brightness_adjust" in template:
                enhancer = ImageEnhance.Brightness(processed)
                processed = enhancer.enhance(template["brightness_adjust"])
            
            if "contrast_adjust" in template:
                enhancer = ImageEnhance.Contrast(processed)
                processed = enhancer.enhance(template["contrast_adjust"])
        
        elif variation_type == VariationType.DEVELOPMENT:
            # Apply development-specific enhancements
            if "feature_emphasis" in template:
                emphasis = template["feature_emphasis"]
                if emphasis != 1.0:
                    # Subtle sharpening for mature features
                    if emphasis > 1.0:
                        sharpening_filter = ImageFilter.UnsharpMask(
                            radius=1.0, 
                            percent=int((emphasis - 1.0) * 100), 
                            threshold=2
                        )
                        processed = processed.filter(sharpening_filter)
                    # Subtle smoothing for early features
                    else:
                        blur_radius = (1.0 - emphasis) * 0.5
                        processed = processed.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return processed
    
    def _calculate_variation_confidence(
        self,
        baby_image: Image.Image,
        variation_type: VariationType,
        generation_params: Dict[str, any]
    ) -> float:
        """Calculate confidence score for variation quality"""
        
        # Base confidence from generation parameters
        base_confidence = 0.7
        
        # Adjust based on variation type
        type_bonuses = {
            VariationType.EXPRESSION: 0.1,
            VariationType.LIGHTING: 0.05,
            VariationType.ANGLE: 0.08,
            VariationType.GENETIC: 0.12,
            VariationType.DEVELOPMENT: 0.1
        }
        
        confidence = base_confidence + type_bonuses.get(variation_type, 0)
        
        # Adjust based on generation quality
        steps = generation_params.get("num_inference_steps", 30)
        if steps >= 40:
            confidence += 0.1
        elif steps >= 30:
            confidence += 0.05
        
        # Image quality assessment
        img_array = np.array(baby_image.convert("L"))
        image_contrast = np.std(img_array) / 255.0
        
        if image_contrast > 0.2:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _rank_variations(self, variations: List[BabyVariation]) -> List[BabyVariation]:
        """Rank variations by quality and diversity"""
        
        # Sort by confidence score (descending)
        ranked = sorted(variations, key=lambda v: v.confidence_score, reverse=True)
        
        # Ensure diversity by promoting different variation types
        final_ranking = []
        used_types = set()
        
        # First pass: include best of each type
        for variation in ranked:
            if variation.variation_type not in used_types:
                final_ranking.append(variation)
                used_types.add(variation.variation_type)
        
        # Second pass: add remaining variations
        for variation in ranked:
            if variation not in final_ranking:
                final_ranking.append(variation)
        
        return final_ranking
    
    def save_variations(
        self,
        variations: List[BabyVariation],
        output_dir: Union[str, Path],
        include_metadata: bool = True
    ) -> Dict[str, List[str]]:
        """Save variations to disk with optional metadata"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {
            "images": [],
            "metadata": []
        }
        
        for i, variation in enumerate(variations):
            # Save image
            image_filename = f"baby_variation_{i+1}_{variation.variation_type.value}.png"
            image_path = output_path / image_filename
            variation.image.save(image_path, "PNG", quality=95, optimize=True)
            saved_files["images"].append(str(image_path))
            
            # Save metadata if requested
            if include_metadata:
                metadata = {
                    "variation_id": variation.variation_id,
                    "variation_type": variation.variation_type.value,
                    "confidence_score": variation.confidence_score,
                    "description": variation.description,
                    "characteristics": variation.characteristics,
                    "generation_params": variation.generation_params
                }
                
                metadata_filename = f"baby_variation_{i+1}_metadata.json"
                metadata_path = output_path / metadata_filename
                
                with open(metadata_path, 'w') as f:
                    import json
                    json.dump(metadata, f, indent=2, default=str)
                
                saved_files["metadata"].append(str(metadata_path))
        
        logger.info(f"💾 Saved {len(variations)} variations to {output_path}")
        return saved_files
    
    def create_variation_collage(
        self,
        variations: List[BabyVariation],
        collage_size: Tuple[int, int] = (1024, 768),
        max_variations: int = 6
    ) -> Image.Image:
        """Create a collage showing multiple variations"""
        
        # Limit number of variations
        selected_variations = variations[:max_variations]
        num_variations = len(selected_variations)
        
        if num_variations == 0:
            return Image.new("RGB", collage_size, (255, 255, 255))
        
        # Calculate grid layout
        cols = min(3, num_variations)
        rows = (num_variations + cols - 1) // cols
        
        # Calculate cell size
        cell_width = collage_size[0] // cols
        cell_height = collage_size[1] // rows
        
        # Create collage canvas
        collage = Image.new("RGB", collage_size, (240, 240, 240))
        
        for i, variation in enumerate(selected_variations):
            # Calculate position
            col = i % cols
            row = i // cols
            
            x = col * cell_width
            y = row * cell_height
            
            # Resize variation image to fit cell
            var_image = variation.image.copy()
            var_image = var_image.resize(
                (cell_width - 10, cell_height - 30),
                Image.Resampling.LANCZOS
            )
            
            # Paste image
            collage.paste(var_image, (x + 5, y + 5))
            
            # Add label (would need PIL with text support)
            # For now, just paste the image
        
        return collage


def main():
    """Test baby face variation pipeline"""
    pipeline = BabyFaceVariationPipeline()
    
    # Test with sample ultrasound image
    sample_path = "/home/ubuntu/DataScience/MyProject/BabyVis/samples/1.jpeg"
    
    if Path(sample_path).exists():
        # Load test image
        ultrasound_image = Image.open(sample_path)
        logger.info(f"🍼 Testing variation pipeline with: {sample_path}")
        
        # Generate variations
        variations = pipeline.generate_comprehensive_variations(
            ultrasound_image,
            num_variations=4,
            quality_preset="balanced"
        )
        
        # Save variations
        output_dir = "/home/ubuntu/DataScience/MyProject/BabyVis/outputs/variations"
        saved_files = pipeline.save_variations(variations, output_dir)
        
        logger.info(f"✅ Generated and saved {len(variations)} variations")
        logger.info(f"📁 Images: {len(saved_files['images'])} files")
        logger.info(f"📋 Metadata: {len(saved_files['metadata'])} files")
        
        # Create collage
        collage = pipeline.create_variation_collage(variations)
        collage_path = Path(output_dir) / "baby_variations_collage.png"
        collage.save(collage_path)
        logger.info(f"🖼️ Collage saved: {collage_path}")
        
    else:
        logger.error(f"❌ Sample image not found: {sample_path}")


if __name__ == "__main__":
    main()