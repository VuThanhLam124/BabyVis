#!/usr/bin/env python3
"""
BabyVis ComfyUI Workflow
Automated workflow for ultrasound to baby face conversion
"""

import json
import os
from pathlib import Path

def create_baby_face_workflow():
    """Create ComfyUI workflow for ultrasound to baby face conversion"""
    
    workflow = {
        "3": {
            "inputs": {
                "seed": 42,
                "steps": 25,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.8,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "4": {
            "inputs": {
                "ckpt_name": "Qwen_Image_Edit-Q4_K_M.gguf"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"}
        },
        "6": {
            "inputs": {
                "text": "beautiful newborn baby face, realistic, soft lighting, peaceful sleeping, high quality portrait",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"}
        },
        "7": {
            "inputs": {
                "text": "adult, teenager, child, woman, man, distorted, blurry, low quality, deformed, ugly",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative)"}
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "9": {
            "inputs": {
                "filename_prefix": "BabyVis",
                "images": ["8", 0]
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        },
        "10": {
            "inputs": {
                "image": "ultrasound_input.jpg",
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Input Image"}
        },
        "11": {
            "inputs": {
                "pixels": ["10", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode Input"}
        },
        "12": {
            "inputs": {
                "samples": ["11", 0],
                "denoise": 0.8,
                "seed": 42,
                "steps": 25,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0]
            },
            "class_type": "KSamplerAdvanced",
            "_meta": {"title": "KSampler Advanced (img2img)"}
        }
    }
    
    return workflow

def save_workflow(workflow_path="ComfyUI/workflows/"):
    """Save workflow to JSON file"""
    workflow = create_baby_face_workflow()
    
    # Create workflows directory
    os.makedirs(workflow_path, exist_ok=True)
    
    # Save workflow
    workflow_file = os.path.join(workflow_path, "baby_face_workflow.json")
    with open(workflow_file, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"Workflow saved to: {workflow_file}")
    return workflow_file

if __name__ == "__main__":
    save_workflow()