#!/usr/bin/env python3
"""
Advanced Baby Face Generation System for BabyVis
Specia            "base": [
                "peaceful sleeping baby portrait",
                "adorable sleeping infant", 
                "cute sleeping baby photograph",
                "precious sleeping newborn image",
                "sweet sleeping baby face"
            ],rompts and techniques for realistic baby face generation from ultrasound
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
            BabyExpression.PEACEFUL: "peaceful serene expression, calm relaxed face, gently closed eyes, maintaining exact pose",
            BabyExpression.SLEEPING: "sleeping baby, eyes completely closed, peaceful sleeping expression, same pose and angle, preserving original direction and orientation",
            BabyExpression.ALERT: "alert baby expression, bright wide open eyes, attentive curious look, awake and engaged",
            BabyExpression.YAWNING: "cute baby yawning, small open mouth, sleepy eyes, maintaining original position",
            BabyExpression.SUCKING: "baby with sucking reflex, pursed lips, peaceful closed eyes, content sleeping expression, same orientation"
        }
    
    def _initialize_base_prompts(self) -> Dict[str, List[str]]:
        """Initialize base prompts for different quality levels"""
        return {
            "base": [
                "beautiful newborn baby face",
                "adorable infant portrait", 
                "cute baby photograph",
                "precious newborn image",
                "sweet baby face"
            ],
            "enhanced": [
                "beautiful healthy newborn baby portrait, soft natural skin, perfect proportions, sleeping peacefully with eyes completely closed, exact same pose and direction",
                "adorable baby face, realistic features, deep peaceful sleeping expression, eyes fully closed, maintaining identical pose and orientation as source",
                "cute newborn photograph, healthy baby skin, natural baby features, soft lighting, sleeping soundly with closed eyes, preserving original angle and direction",
                "precious baby portrait, perfect baby proportions, peaceful sleeping expression with closed eyes, maintaining original head position and orientation",
                "gorgeous infant face, healthy appearance, natural baby beauty, soft hospital lighting, sleeping baby with eyes closed, same pose as original image"
            ],
            "premium": [
                "masterpiece award-winning newborn baby photography, ultra realistic 8K, medical grade perfection, sleeping peacefully with eyes completely closed, preserving exact original pose and direction",
                "professional studio baby portrait, flawless lighting, photorealistic precision, artistic excellence, deep peaceful sleep with closed eyes, maintaining identical composition and orientation",
                "world-class newborn photography, crystal clear baby features, museum quality detail, peaceful sleeping expression with eyes closed, exact same angle and direction as source",
                "master photographer baby portrait, perfect skin texture, breathtaking realism, artistic mastery, sleeping soundly with closed eyes, preserving original head position and body orientation",
                "legendary baby photography, ultra high definition, cinematic quality, professional perfection, peaceful sleeping baby with eyes closed, maintaining source image pose and direction"
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
                "soft natural hospital lighting",
                "professional studio lighting", 
                "warm medical photography lighting",
                "gentle diffused lighting",
                "optimal newborn photography lighting"
            ],
            "technical": [
                "ultra high definition",
                "8K resolution", 
                "photorealistic detail",
                "crystal clear image quality",
                "professional photography quality",
                "medical grade image clarity",
                "maintaining exact original pose and angle",
                "preserving identical source image composition",
                "same head position and orientation as input",
                "keeping original baby position and direction",
                "exact pose replication",
                "identical facial orientation",
                "preserving source image geometry",
                "matching original head angle and body position"
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
            "scared expression", "frowning", "upset baby",
            
            # Eyes and expression negatives - STRONG
            "open eyes", "awake eyes", "bright eyes", "looking eyes", "alert eyes",
            "wide eyes", "staring", "gazing", "eye contact", "eyes open",
            "looking at camera", "looking at viewer", "direct gaze", "eye focus",
            
            # Technical quality negatives
            "blurry image", "low quality", "pixelated", "distorted features", 
            "deformed face", "unnatural proportions", "fake appearance",
            "artificial", "cartoon", "anime", "illustration", "painting",
            "weird", "strange", "abnormal", "creepy", "scary", "disturbing",
            "malformed", "disfigured", "mutated", "distorted", "warped",
            
            # Pose and composition negatives - STRONGER
            "different angle", "changed position", "rotated head", "tilted face",
            "different orientation", "modified pose", "altered composition",
            "wrong angle", "shifted position", "turned head", "facing different direction",
            "mirror image", "flipped horizontally", "reversed pose", "opposite direction",
            
            # Lighting negatives
            "dark lighting", "harsh shadows", "overexposed", "underexposed",
            "poor lighting", "unnatural lighting",
            
            # Medical negatives
            "medical devices", "tubes", "wires", "hospital equipment visible",
            "surgical marks", "medical tape", "breathing apparatus",
            
            # Abnormal features
            "extra limbs", "missing features", "asymmetrical face", 
            "disproportionate features", "unrealistic anatomy",
            "deformity", "birth defect", "abnormal development",
            "strange proportions", "weird anatomy", "unnatural body"
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
    
    def generate_safe_baby_prompt(
        self,
        quality_level: str = "enhanced"
    ) -> Tuple[str, str]:
        """
        Generate very safe prompts focused on healthy, normal baby appearance
        """
        
        safe_prompts = [
            "beautiful healthy newborn baby, perfect normal proportions, soft natural skin",
            "adorable baby portrait, healthy appearance, gentle peaceful expression",
            "cute newborn photograph, normal baby features, soft lighting",
            "precious healthy baby, perfect proportions, natural baby beauty"
        ]
        
        base_prompt = random.choice(safe_prompts)
        
        # Safe enhancement terms
        safe_enhancements = [
            "healthy normal baby",
            "perfect baby proportions", 
            "natural baby features",
            "soft baby skin",
            "gentle peaceful expression",
            "beautiful baby portrait",
            "professional baby photography"
        ]
        
        # Very strong negative prompts to prevent deformity
        strong_negatives = [
            "deformed", "malformed", "disfigured", "mutated", "distorted", "warped",
            "abnormal", "weird", "strange", "creepy", "scary", "disturbing",
            "extra limbs", "missing limbs", "wrong proportions", "unnatural",
            "cartoon", "anime", "illustration", "fake", "artificial",
            "blurry", "low quality", "poor quality", "bad anatomy",
            "birth defect", "deformity", "abnormal development"
        ]
        
        prompt_components = [base_prompt] + random.sample(safe_enhancements, 3)
        positive_prompt = ", ".join(prompt_components)
        negative_prompt = ", ".join(strong_negatives)
        
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
        
        # Default to sleeping expression for peaceful babies
        characteristics["expression"] = BabyExpression.SLEEPING
        
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
    
    def generate_anatomy_focused_prompt(
        self,
        quality_level: str = "enhanced",
        ethnicity: Optional[EthnicityGroup] = None,
        custom_bed_description: str = "wide comfortable bed"
    ) -> Tuple[str, str]:
        """
        Generate prompt with strong focus on correct anatomy and bed setting
        
        Args:
            quality_level: Quality preset level
            ethnicity: Optional ethnicity specification
            custom_bed_description: Description of bed setting
            
        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        
        # Anatomy-focused base prompts
        anatomy_prompts = [
            f"sleeping newborn baby on {custom_bed_description}, maintaining original angle and pose",
            f"peaceful baby lying on {custom_bed_description}, preserving source position", 
            f"adorable sleeping infant on {custom_bed_description}, same angle as original image"
        ]
        
        base_prompt = random.choice(anatomy_prompts)
        
        # Core anatomy requirements
        anatomy_requirements = [
            "two hands with exactly five fingers each",
            "anatomically correct baby proportions",
            "normal baby limb structure",
            "proper hand positioning",
            "realistic finger anatomy",
            "correct newborn body proportions"
        ]
        
        # Lighting and quality
        quality_terms = [
            "gentle soft lighting",
            "minimal hair coverage", 
            "peaceful sleeping expression",
            "soft natural illumination"
        ]
        
        prompt_components = [base_prompt] + anatomy_requirements + quality_terms
        
        # Add ethnicity if specified
        if ethnicity and ethnicity in self.genetic_features:
            ethnic_features = self.genetic_features[ethnicity]
            prompt_components.append(random.choice(ethnic_features["skin_tone"]))
        
        # Enhanced negative prompts for anatomy
        enhanced_negatives = self.negative_prompts + [
            "three hands", "extra hands", "missing hands", "deformed hands",
            "six fingers", "four fingers", "wrong finger count", "extra fingers",
            "missing fingers", "malformed anatomy", "incorrect hand structure",
            "floating limbs", "disconnected hands", "merged hands",
            "narrow bed", "cramped space", "different angle", "changed position"
        ]
        
        positive_prompt = ", ".join(prompt_components)
        negative_prompt = ", ".join(enhanced_negatives)
        
        return positive_prompt, negative_prompt

    def generate_strict_sleeping_prompt(self, quality_level: str = "enhanced") -> Tuple[str, str]:
        """
        Generate STRICT sleeping baby prompts with pose preservation
        Specifically for cases where open eyes and pose changes are major issues
        """
        
        # Super strict sleeping terms
        strict_sleeping_terms = [
            "sleeping baby with eyes completely closed",
            "peaceful deep sleep with closed eyelids", 
            "baby sleeping soundly, eyes fully shut",
            "serene sleeping expression, eyes closed tight",
            "baby in deep peaceful slumber, eyelids closed"
        ]
        
        # Strict pose preservation terms
        strict_pose_terms = [
            "maintaining exact original pose and orientation",
            "preserving identical head position and body angle", 
            "keeping same facial direction as source image",
            "matching original composition and positioning exactly",
            "replicating source image pose and orientation perfectly"
        ]
        
        # Build strict prompt
        base_prompt = random.choice(self.base_prompts[quality_level])
        sleeping_term = random.choice(strict_sleeping_terms)
        pose_term = random.choice(strict_pose_terms)
        
        # Additional quality terms
        quality_terms = random.sample(self.quality_enhancers.get("lighting", []), 2)
        technical_terms = random.sample(self.quality_enhancers.get("technical", []), 3)
        
        prompt_parts = [
            base_prompt,
            sleeping_term, 
            pose_term,
        ] + quality_terms + technical_terms
        
        final_prompt = ", ".join(prompt_parts)
        
        # SUPER STRICT negative prompts
        strict_negatives = self.negative_prompts + [
            # Eyes - MULTIPLE VARIATIONS
            "open eyes", "awake eyes", "bright eyes", "alert eyes", "staring eyes",
            "wide eyes", "looking eyes", "gazing eyes", "eye contact", "eyes open",
            "eyelids open", "visible pupils", "eye focus", "alert gaze", "awake look",
            "bright gaze", "conscious eyes", "attentive eyes", "watchful eyes",
            
            # Pose changes - MULTIPLE VARIATIONS  
            "different pose", "changed position", "altered orientation", "rotated head",
            "tilted face", "different angle", "modified composition", "shifted position", 
            "turned head", "facing different direction", "mirror image", "flipped pose",
            "reversed orientation", "opposite direction", "different head position",
            "changed facial direction", "altered body position", "rotated composition"
        ]
        
        return final_prompt, ", ".join(strict_negatives)


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