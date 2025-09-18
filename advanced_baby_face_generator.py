#!/usr/bin/env python3
"""
Advanced Baby Face Generation System for BabyVis
Specialized prompts and techniques for realistic baby face generation from ultrasound
Based on medical accuracy, genetic diversity, and YouTube tutorial best practices
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GestationalAge(Enum):
    EARLY = "early"    # 12-20 weeks
    MID = "mid"        # 20-28 weeks  
    LATE = "late"      # 28-40 weeks

class EthnicityGroup(Enum):
    CAUCASIAN = "caucasian"
    ASIAN = "asian"
    AFRICAN = "african"
    HISPANIC = "hispanic"
    MIDDLE_EASTERN = "middle_eastern"
    MIXED = "mixed"

class BabyExpression(Enum):
    PEACEFUL = "peaceful"
    SLEEPING = "sleeping"
    ALERT = "alert"
    YAWNING = "yawning"
    SUCKING = "sucking"

class AdvancedBabyFaceGenerator:
    """
    Advanced baby face generation system with medical accuracy and genetic diversity
    Implements YouTube video techniques for professional baby face prediction
    """
    
    def __init__(self):
        self.base_prompts = self._initialize_base_prompts()
        self.medical_descriptors = self._initialize_medical_descriptors()
        self.genetic_features = self._initialize_genetic_features()
        self.quality_enhancers = self._initialize_quality_enhancers()
        self.negative_prompts = self._initialize_negative_prompts()
        
        # Gestational age characteristics
        self.gestational_characteristics = {
            GestationalAge.EARLY: {
                "facial_development": "early facial development, delicate features forming",
                "skin_texture": "very soft translucent baby skin",
                "proportions": "small delicate proportions",
                "detail_level": "subtle gentle features"
            },
            GestationalAge.MID: {
                "facial_development": "developing facial features, clearer definition",
                "skin_texture": "soft developing baby skin",
                "proportions": "well-proportioned baby features",
                "detail_level": "clear defined features"
            },
            GestationalAge.LATE: {
                "facial_development": "fully developed baby facial features",
                "skin_texture": "healthy full-term baby skin",
                "proportions": "perfect newborn proportions",
                "detail_level": "detailed realistic features"
            }
        }
        
        # Expression-specific descriptors
        self.expression_descriptors = {
            BabyExpression.PEACEFUL: "peaceful serene expression, calm relaxed face",
            BabyExpression.SLEEPING: "sleeping baby, closed eyes, completely relaxed expression",
            BabyExpression.ALERT: "alert baby expression, bright curious eyes, attentive look",
            BabyExpression.YAWNING: "cute baby yawning, small open mouth, sleepy expression",
            BabyExpression.SUCKING: "baby with sucking reflex, pursed lips, content expression"
        }
    
    def _initialize_base_prompts(self) -> Dict[str, List[str]]:
        """Initialize base prompts for different quality levels"""
        return {
            "base": [
                "sleeping newborn baby face, minimal hair",
                "adorable sleeping infant portrait, gentle lighting", 
                "cute sleeping baby photograph, sparse hair",
                "precious sleeping newborn image, soft lighting",
                "sweet sleeping baby face, little hair"
            ],
            "enhanced": [
                "stunning newborn baby portrait, sleeping peacefully, maintaining original pose and angle, gentle soft lighting, minimal hair",
                "beautiful realistic baby face, sleeping baby, same position as original, soft lighting, sparse baby hair, peaceful expression",
                "adorable sleeping newborn photograph, preserving original angle and pose, gentle illumination, little hair, serene expression",
                "precious sleeping baby portrait, maintaining original positioning, soft natural lighting, minimal hair coverage",
                "gorgeous sleeping infant face, same angle as source image, gentle lighting, sparse hair, peaceful slumber"
            ],
            "premium": [
                "award-winning sleeping newborn baby photography, maintaining exact original pose and angle, ultra realistic, gentle lighting, minimal hair coverage",
                "professional sleeping baby portrait studio, preserving original positioning, soft lighting, sparse hair, photorealistic detail",
                "hospital-grade sleeping newborn photography, same angle as source, gentle illumination, little hair, artistic quality",
                "master photographer sleeping baby portrait, maintaining original pose, soft lighting, minimal hair, stunning detail",
                "world-class sleeping baby photography, preserving original angle, gentle lighting, sparse hair coverage, breathtaking realism"
            ]
        }
    
    def _initialize_medical_descriptors(self) -> List[str]:
        """Medical accuracy descriptors for realistic baby features"""
        return [
            "anatomically correct baby proportions",
            "realistic newborn facial structure",
            "medically accurate baby features",
            "healthy full-term baby appearance",
            "natural baby skin pigmentation",
            "proper newborn head-to-body ratio",
            "realistic baby eye positioning",
            "natural baby lip formation",
            "correct infant nose development",
            "authentic baby chin definition"
        ]
    
    def _initialize_genetic_features(self) -> Dict[EthnicityGroup, Dict[str, List[str]]]:
        """Genetic diversity features for different ethnicities"""
        return {
            EthnicityGroup.CAUCASIAN: {
                "skin_tone": ["fair baby skin", "light peachy skin tone", "pale baby complexion"],
                "features": ["European baby features", "caucasian infant characteristics"],
                "hair": ["light baby hair", "fine blonde hair", "soft brown hair"]
            },
            EthnicityGroup.ASIAN: {
                "skin_tone": ["warm asian baby skin", "golden baby complexion", "smooth asian skin tone"],
                "features": ["beautiful asian baby features", "east asian infant characteristics"],
                "hair": ["dark baby hair", "straight black hair", "fine asian hair"]
            },
            EthnicityGroup.AFRICAN: {
                "skin_tone": ["beautiful dark baby skin", "rich african complexion", "warm brown skin tone"],
                "features": ["african baby features", "beautiful ethnic characteristics"],
                "hair": ["dark curly baby hair", "natural african hair texture"]
            },
            EthnicityGroup.HISPANIC: {
                "skin_tone": ["warm olive baby skin", "hispanic complexion", "golden brown skin tone"],
                "features": ["hispanic baby features", "latino infant characteristics"],
                "hair": ["dark wavy baby hair", "soft brown hair", "natural hispanic hair"]
            },
            EthnicityGroup.MIDDLE_EASTERN: {
                "skin_tone": ["warm middle eastern complexion", "olive baby skin tone"],
                "features": ["middle eastern baby features", "mediterranean characteristics"],
                "hair": ["dark baby hair", "thick middle eastern hair"]
            },
            EthnicityGroup.MIXED: {
                "skin_tone": ["beautiful mixed heritage skin", "diverse ethnic complexion"],
                "features": ["mixed ethnicity baby features", "multicultural characteristics"],
                "hair": ["mixed heritage hair texture", "diverse hair characteristics"]
            }
        }
    
    def _initialize_quality_enhancers(self) -> Dict[str, List[str]]:
        """Quality enhancement terms for different levels"""
        return {
            "lighting": [
                "soft gentle lighting, maintaining original pose",
                "subtle natural lighting, preserving angle", 
                "gentle diffused lighting, same position as source",
                "soft warm lighting, minimal hair visible",
                "gentle hospital lighting, sleeping baby position"
            ],
            "technical": [
                "ultra high definition",
                "8K resolution",
                "photorealistic detail",
                "crystal clear image quality",
                "professional photography quality",
                "medical grade image clarity"
            ],
            "artistic": [
                "award-winning photography",
                "masterpiece portrait",
                "professional composition",
                "artistic excellence",
                "gallery-quality image"
            ]
        }
    
    def _initialize_negative_prompts(self) -> List[str]:
        """Comprehensive negative prompts to avoid unwanted features"""
        return [
            # Age-related negatives
            "adult face", "child face", "teenager", "elderly person", "grown-up features",
            
            # Unwanted features
            "facial hair", "beard", "mustache", "wrinkles", "age spots", "blemishes",
            "scars", "birth marks", "rashes", "skin imperfections",
            
            # Expression negatives  
            "crying", "screaming", "angry expression", "sad face", "distressed", 
            "scared expression", "frowning", "upset baby", "awake baby", "open eyes",
            
            # Position and angle changes
            "different angle", "changed position", "rotated pose", "different orientation",
            "altered pose", "modified position", "turned head", "different perspective",
            
            # Hair-related negatives
            "thick hair", "full hair", "abundant hair", "lots of hair", "long hair",
            "curly hair", "wavy hair", "styled hair", "hair accessories",
            
            # Lighting negatives
            "bright lighting", "harsh lighting", "strong light", "direct sunlight",
            "dark lighting", "harsh shadows", "overexposed", "underexposed",
            "poor lighting", "unnatural lighting", "dramatic lighting",
            
            # Technical quality negatives
            "blurry image", "low quality", "pixelated", "distorted features", 
            "deformed face", "unnatural proportions", "fake appearance",
            "artificial", "cartoon", "anime", "illustration", "painting",
            
            # Medical negatives
            "medical devices", "tubes", "wires", "hospital equipment visible",
            "surgical marks", "medical tape", "breathing apparatus",
            
            # Abnormal features
            "extra limbs", "missing features", "asymmetrical face", 
            "disproportionate features", "unrealistic anatomy"
        ]
    
    def generate_advanced_prompt(
        self,
        quality_level: str = "enhanced",
        gestational_age: Optional[GestationalAge] = None,
        ethnicity: Optional[EthnicityGroup] = None,
        expression: Optional[BabyExpression] = None,
        include_medical_accuracy: bool = True,
        custom_attributes: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Generate advanced prompt for baby face generation
        
        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        
        # Start with base prompt
        base_prompts = self.base_prompts.get(quality_level, self.base_prompts["enhanced"])
        base_prompt = random.choice(base_prompts)
        
        prompt_components = [base_prompt]
        
        # Add gestational age characteristics
        if gestational_age:
            age_chars = self.gestational_characteristics[gestational_age]
            prompt_components.extend([
                age_chars["facial_development"],
                age_chars["skin_texture"],
                age_chars["proportions"]
            ])
        
        # Add ethnic characteristics
        if ethnicity:
            ethnic_features = self.genetic_features[ethnicity]
            prompt_components.append(random.choice(ethnic_features["skin_tone"]))
            prompt_components.append(random.choice(ethnic_features["features"]))
            if "hair" in ethnic_features:
                prompt_components.append(random.choice(ethnic_features["hair"]))
        
        # Add expression
        if expression:
            prompt_components.append(self.expression_descriptors[expression])
        
        # Add medical accuracy
        if include_medical_accuracy:
            medical_desc = random.sample(self.medical_descriptors, 2)
            prompt_components.extend(medical_desc)
        
        # Add quality enhancers based on level
        if quality_level in ["enhanced", "premium"]:
            lighting = random.choice(self.quality_enhancers["lighting"])
            technical = random.choice(self.quality_enhancers["technical"])
            prompt_components.extend([lighting, technical])
        
        if quality_level == "premium":
            artistic = random.choice(self.quality_enhancers["artistic"])
            prompt_components.append(artistic)
        
        # Add custom attributes
        if custom_attributes:
            prompt_components.extend(custom_attributes)
        
        # Construct final positive prompt
        positive_prompt = ", ".join(prompt_components)
        
        # Construct negative prompt
        negative_prompt = ", ".join(self.negative_prompts)
        
        return positive_prompt, negative_prompt
    
    def generate_parent_similarity_prompt(
        self,
        parent_characteristics: Dict[str, str],
        quality_level: str = "enhanced"
    ) -> Tuple[str, str]:
        """
        Generate prompt that incorporates parent characteristics for genetic similarity
        
        Args:
            parent_characteristics: Dict with keys like 'eye_color', 'hair_color', 'skin_tone', etc.
        """
        
        base_prompts = self.base_prompts.get(quality_level, self.base_prompts["enhanced"])
        base_prompt = random.choice(base_prompts)
        
        prompt_components = [base_prompt]
        
        # Add parent-derived characteristics
        if "eye_color" in parent_characteristics:
            eye_color = parent_characteristics["eye_color"]
            prompt_components.append(f"baby with {eye_color} eyes, inherited eye color")
        
        if "hair_color" in parent_characteristics:
            hair_color = parent_characteristics["hair_color"]
            prompt_components.append(f"baby with {hair_color} hair, genetic hair color")
        
        if "skin_tone" in parent_characteristics:
            skin_tone = parent_characteristics["skin_tone"]
            prompt_components.append(f"baby with {skin_tone} complexion, inherited skin tone")
        
        if "facial_structure" in parent_characteristics:
            structure = parent_characteristics["facial_structure"]
            prompt_components.append(f"baby with {structure} facial structure, family resemblance")
        
        # Add genetic realism
        prompt_components.extend([
            "genetic family resemblance", 
            "inherited parental features",
            "realistic family genetics",
            "natural hereditary characteristics"
        ])
        
        # Add medical accuracy
        medical_desc = random.sample(self.medical_descriptors, 2)
        prompt_components.extend(medical_desc)
        
        # Add quality enhancers
        if quality_level in ["enhanced", "premium"]:
            lighting = random.choice(self.quality_enhancers["lighting"])
            technical = random.choice(self.quality_enhancers["technical"])
            prompt_components.extend([lighting, technical])
        
        positive_prompt = ", ".join(prompt_components)
        negative_prompt = ", ".join(self.negative_prompts)
        
        return positive_prompt, negative_prompt
    
    def generate_variation_prompts(
        self,
        base_characteristics: Dict[str, any],
        num_variations: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Generate multiple prompt variations for diverse baby face generation
        """
        variations = []
        
        expressions = list(BabyExpression)
        
        for i in range(num_variations):
            # Vary expression for each generation
            expression = expressions[i % len(expressions)]
            
            # Slightly vary quality descriptors
            variation_attributes = []
            
            if i == 0:
                variation_attributes.append("perfect baby proportions")
            elif i == 1:
                variation_attributes.append("gentle innocent expression")
            else:
                variation_attributes.append("angelic baby features")
            
            # Generate prompt with variations
            positive, negative = self.generate_advanced_prompt(
                quality_level=base_characteristics.get("quality_level", "enhanced"),
                gestational_age=base_characteristics.get("gestational_age"),
                ethnicity=base_characteristics.get("ethnicity"),
                expression=expression,
                custom_attributes=variation_attributes
            )
            
            variations.append((positive, negative))
        
        return variations
    
    def analyze_ultrasound_for_characteristics(self, facial_regions: Dict) -> Dict[str, any]:
        """
        Analyze ultrasound facial regions to determine optimal generation characteristics
        """
        characteristics = {}
        
        # Estimate gestational age from facial development
        if facial_regions.get("total_regions", 0) > 5:
            characteristics["gestational_age"] = GestationalAge.LATE
        elif facial_regions.get("total_regions", 0) > 2:
            characteristics["gestational_age"] = GestationalAge.MID
        else:
            characteristics["gestational_age"] = GestationalAge.EARLY
        
        # Default to mixed ethnicity for inclusive generation
        characteristics["ethnicity"] = EthnicityGroup.MIXED
        
        # Default to peaceful expression
        characteristics["expression"] = BabyExpression.PEACEFUL
        
        return characteristics
    
    def get_prompt_explanation(self, positive_prompt: str, negative_prompt: str) -> Dict[str, str]:
        """
        Provide explanation of prompt components for user understanding
        """
        return {
            "positive_focus": "Emphasizes medical accuracy, realistic baby features, and professional quality",
            "negative_prevention": "Prevents adult features, poor quality, and unrealistic characteristics", 
            "medical_accuracy": "Includes anatomically correct proportions and healthy baby appearance",
            "quality_level": "Optimized for professional hospital-grade photography standards",
            "genetic_realism": "Incorporates realistic genetic diversity and family characteristics"
        }


def main():
    """Test advanced baby face generation prompts"""
    generator = AdvancedBabyFaceGenerator()
    
    logger.info("🍼 Testing Advanced Baby Face Generation Prompts")
    
    # Test basic prompt generation
    logger.info("\n=== Basic Enhanced Prompt ===")
    positive, negative = generator.generate_advanced_prompt(
        quality_level="enhanced",
        gestational_age=GestationalAge.MID,
        ethnicity=EthnicityGroup.MIXED,
        expression=BabyExpression.PEACEFUL
    )
    
    logger.info(f"Positive: {positive}")
    logger.info(f"Negative: {negative}")
    
    # Test parent similarity prompt
    logger.info("\n=== Parent Similarity Prompt ===")
    parent_chars = {
        "eye_color": "warm brown",
        "hair_color": "dark brown", 
        "skin_tone": "olive",
        "facial_structure": "oval"
    }
    
    positive, negative = generator.generate_parent_similarity_prompt(parent_chars, "premium")
    logger.info(f"Positive: {positive}")
    logger.info(f"Negative: {negative}")
    
    # Test variation prompts
    logger.info("\n=== Variation Prompts ===")
    base_chars = {
        "quality_level": "enhanced",
        "gestational_age": GestationalAge.LATE,
        "ethnicity": EthnicityGroup.ASIAN
    }
    
    variations = generator.generate_variation_prompts(base_chars, 3)
    for i, (pos, neg) in enumerate(variations):
        logger.info(f"\nVariation {i+1}:")
        logger.info(f"Positive: {pos}")
    
    logger.info("\n✅ Advanced prompt generation test completed!")


if __name__ == "__main__":
    main()