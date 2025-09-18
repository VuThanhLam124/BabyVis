#!/usr/bin/env python3
"""
Test script for model selection functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_options import AVAILABLE_MODELS

def test_model_options():
    """Test the available models"""
    print("=== Available Models ===")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"{i}. {model['name']}")
        print(f"   ID: {model['id']}")
        print(f"   Quality: {model['quality']}/10")
        print(f"   Speed: {model['speed']}")
        print(f"   VRAM: {model['vram_gb']}GB")
        print(f"   Default: {model['is_default']}")
        print(f"   Description: {model['description']}")
        print()

def test_model_selection():
    """Test model selection logic"""
    from config import BabyVisSettings
    
    # Test default model
    settings = BabyVisSettings()
    print(f"Default model: {settings.qwen_model_id}")
    
    # Test model selection for different VRAM
    vram_levels = [4, 8, 12, 24]
    for vram in vram_levels:
        print(f"\nRecommended models for {vram}GB VRAM:")
        suitable = [m for m in AVAILABLE_MODELS if m['vram_gb'] <= vram]
        suitable.sort(key=lambda x: x['quality'], reverse=True)
        for model in suitable[:3]:  # Top 3
            print(f"  - {model['name']} (Quality: {model['quality']}/10)")

if __name__ == "__main__":
    test_model_options()
    test_model_selection()