#!/usr/bin/env python3
"""
Model Options for BabyVis
Configure different AI models for baby face generation
"""

# Available models with web API format
AVAILABLE_MODELS = [
    {
        "id": "black-forest-labs/FLUX.1-dev",
        "name": "FLUX.1-dev (Most Powerful)",
        "quality": 10,
        "speed": 2,
        "vram_gb": 12,
        "is_default": False,
        "description": "State-of-the-art model with unmatched quality"
    },
    {
        "id": "stabilityai/stable-diffusion-xl-base-1.0", 
        "name": "SDXL Base (High Quality)",
        "quality": 9,
        "speed": 3,
        "vram_gb": 8,
        "is_default": False,
        "description": "Excellent quality and detail"
    },
    {
        "id": "SG161222/Realistic_Vision_V5.1_noVAE",
        "name": "Realistic Vision v5.1",
        "quality": 8,
        "speed": 6, 
        "vram_gb": 4,
        "is_default": True,
        "description": "Specialized for realistic photos"
    },
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "name": "SD 1.5 (Lightweight)",
        "quality": 6,
        "speed": 9,
        "vram_gb": 2,
        "is_default": False,
        "description": "Lightweight and fast"
    },
    {
        "id": "stabilityai/sdxl-turbo",
        "name": "SDXL Turbo (Fast)",
        "quality": 8,
        "speed": 8,
        "vram_gb": 6,
        "is_default": False,
        "description": "Fast high-quality generation"
    },
    {
        "id": "dreamlike-art/dreamlike-photoreal-2.0",
        "name": "Dreamlike Photoreal 2.0",
        "quality": 7,
        "speed": 7,
        "vram_gb": 4,
        "is_default": False,
        "description": "Photorealistic image generation"
    }
]

def get_recommended_model(vram_gb: float) -> str:
    """Get recommended model based on available VRAM"""
    if vram_gb >= 12:
        return "black-forest-labs/FLUX.1-dev"
    elif vram_gb >= 8:
        return "stabilityai/stable-diffusion-xl-base-1.0"
    elif vram_gb >= 6:
        return "stabilityai/sdxl-turbo"
    elif vram_gb >= 4:
        return "SG161222/Realistic_Vision_V5.1_noVAE"
    else:
        return "runwayml/stable-diffusion-v1-5"