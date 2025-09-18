#!/usr/bin/env python3
"""
Model Options for BabyVis
Configure different AI models for baby face generation
"""

from typing import Dict, List, NamedTuple

class ModelInfo(NamedTuple):
    model_id: str
    description: str
    quality: str
    speed: str
    vram_requirement: str
    pros: List[str]
    cons: List[str]

# Available models for baby face generation
AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    "sdxl": ModelInfo(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        description="Stable Diffusion XL - Highest quality",
        quality="Excellent (9/10)",
        speed="Slow (3/10)", 
        vram_requirement="8GB+",
        pros=[
            "Best image quality and detail",
            "Superior facial features",
            "Better prompt understanding",
            "More realistic skin textures"
        ],
        cons=[
            "Requires more VRAM",
            "Slower generation time",
            "Larger model download"
        ]
    ),
    
    "realistic_vision": ModelInfo(
        model_id="SG161222/Realistic_Vision_V5.1_noVAE",
        description="Realistic Vision v5.1 - Specialized for realistic photos",
        quality="Very Good (8/10)",
        speed="Medium (6/10)",
        vram_requirement="4GB+",
        pros=[
            "Excellent for realistic faces",
            "Good balance of quality/speed",
            "Works well with baby features",
            "Less VRAM required than SDXL"
        ],
        cons=[
            "Not as detailed as SDXL",
            "Limited style variations"
        ]
    ),
    
    "dreamshaper": ModelInfo(
        model_id="Lykon/DreamShaper",
        description="DreamShaper - Artistic and creative",
        quality="Good (7/10)",
        speed="Fast (7/10)",
        vram_requirement="4GB+",
        pros=[
            "Fast generation",
            "Creative interpretations",
            "Good with varied prompts",
            "Stable performance"
        ],
        cons=[
            "Sometimes too artistic",
            "Less realistic than others"
        ]
    ),
    
    "sd15": ModelInfo(
        model_id="runwayml/stable-diffusion-v1-5",
        description="Stable Diffusion 1.5 - Lightweight and fast",
        quality="Good (6/10)",
        speed="Very Fast (9/10)",
        vram_requirement="2GB+",
        pros=[
            "Very fast generation",
            "Low VRAM requirements",
            "Stable and reliable",
            "Large community support"
        ],
        cons=[
            "Lower image quality",
            "Less detailed faces",
            "Older technology"
        ]
    ),
    
    "sdxl_turbo": ModelInfo(
        model_id="stabilityai/sdxl-turbo",
        description="SDXL Turbo - Fast high-quality generation",
        quality="Very Good (8/10)",
        speed="Very Fast (8/10)",
        vram_requirement="6GB+",
        pros=[
            "SDXL quality with speed",
            "Optimized for quick generation",
            "Good detail retention",
            "Better than SD 1.5"
        ],
        cons=[
            "Slightly less quality than full SDXL",
            "Still requires decent VRAM"
        ]
    )
}

def get_recommended_model(vram_gb: float) -> str:
    """Get recommended model based on available VRAM"""
    if vram_gb >= 8:
        return "sdxl"
    elif vram_gb >= 6:
        return "sdxl_turbo"
    elif vram_gb >= 4:
        return "realistic_vision"
    else:
        return "sd15"

def get_model_id(model_key: str) -> str:
    """Get model ID from key"""
    return AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["realistic_vision"]).model_id

def list_models() -> None:
    """Print available models with descriptions"""
    print("🤖 Available Models for BabyVis:")
    print("=" * 50)
    
    for key, model in AVAILABLE_MODELS.items():
        print(f"\n📋 {key.upper()}")
        print(f"   Model: {model.model_id}")
        print(f"   Description: {model.description}")
        print(f"   Quality: {model.quality}")
        print(f"   Speed: {model.speed}")
        print(f"   VRAM: {model.vram_requirement}")
        
        print("   ✅ Pros:")
        for pro in model.pros:
            print(f"      • {pro}")
        
        print("   ❌ Cons:")
        for con in model.cons:
            print(f"      • {con}")

if __name__ == "__main__":
    list_models()