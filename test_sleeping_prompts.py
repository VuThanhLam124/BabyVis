#!/usr/bin/env python3
"""
Test the improved sleeping baby prompts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_baby_face_generator import AdvancedBabyFaceGenerator
from enum import Enum

def test_sleeping_prompts():
    """Test new sleeping baby prompts"""
    generator = AdvancedBabyFaceGenerator()
    
    print("=== Testing Sleeping Baby Prompts ===\n")
    
    # Test different quality levels
    for quality in ["base", "enhanced", "premium"]:
        print(f"--- {quality.upper()} Quality ---")
        prompt, negative = generator.generate_advanced_prompt(quality_level=quality)
        print(f"Prompt: {prompt}")
        print(f"Negative: {negative}")
        print()
    
    print("=== Testing Safe Baby Prompts ===\n")
    safe_prompt, safe_negative = generator.safe_baby_prompt()
    print(f"Safe Prompt: {safe_prompt}")
    print(f"Safe Negative: {safe_negative}")

if __name__ == "__main__":
    test_sleeping_prompts()