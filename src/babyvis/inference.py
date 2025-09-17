#!/usr/bin/env python3
"""
BabyVis Inference - Simplified for GGUF only
"""

import os
import base64
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from .model_utils import create_model_4gb, create_model_12gb, create_model_cpu, detect_vram

# Global model cache để tránh load lại model nhiều lần
_cached_model = None
_cached_model_type = None


def get_model(model_type: str = "auto"):
    """Lấy model từ cache hoặc load mới"""
    global _cached_model, _cached_model_type
    
    # Nếu đã có model trong cache và cùng loại
    if _cached_model and _cached_model_type == model_type:
        return _cached_model
    
    # Load model mới
    print(f"🤖 Loading model with type: {model_type}")
    
    try:
        if model_type == "auto":
            vram = detect_vram()
            if vram >= 12:
                _cached_model = create_model_12gb()
                _cached_model_type = "12gb"
            elif vram >= 4:
                _cached_model = create_model_4gb()
                _cached_model_type = "4gb"
            else:
                _cached_model = create_model_cpu()
                _cached_model_type = "cpu"
        elif model_type == "4gb":
            _cached_model = create_model_4gb()
            _cached_model_type = "4gb"
        elif model_type == "12gb":
            _cached_model = create_model_12gb()
            _cached_model_type = "12gb"
        elif model_type == "cpu":
            _cached_model = create_model_cpu()
            _cached_model_type = "cpu"
        else:
            print(f"❌ Unknown model type: {model_type}")
            return None
        
        if _cached_model:
            print(f"✅ Model loaded successfully: {_cached_model_type}")
        return _cached_model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
from typing import Optional, List
import time

from .model_utils import create_model_auto, create_model_4gb, create_model_12gb, create_model_cpu


def encode_image_to_base64(image_path: str) -> str:
    """Mã hóa ảnh thành base64 data URI"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Xác định định dạng file
    ext = Path(image_path).suffix.lower()
    if ext == '.jpg' or ext == '.jpeg':
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    elif ext == '.webp':
        mime_type = 'image/webp'
    else:
        mime_type = 'image/jpeg'  # default
    
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"


def create_baby_instruction(ethnicity: str = "mixed") -> str:
    """Tạo instruction chuyên nghiệp cho việc tạo ảnh em bé"""
    return f"""Transform this ultrasound image into a high-quality, photorealistic newborn baby portrait.

Requirements:
- Generate a peaceful sleeping {ethnicity} newborn infant
- Professional medical photography style
- Soft, natural lighting with gentle shadows
- Clean, neutral background
- Focus on delicate facial features: small button nose, tiny lips, soft skin
- Healthy, natural skin tone
- Serene, calm expression
- High resolution, crisp details
- No hands visible in the image

Avoid:
- Adult features, unrealistic proportions
- Poor lighting, harsh shadows
- Artificial or cartoon-like appearance
- Any medical equipment or devices
- Multiple babies or people
- Open eyes or crying expression"""


def process_ultrasound_image(
    model,
    input_path: str,
    output_path: str,
    ethnicity: str = "mixed",
    max_tokens: int = 1024,
    temperature: float = 0.3
) -> bool:
    """
    Xử lý ảnh siêu âm thành ảnh em bé
    
    Args:
        model: Loaded Qwen model
        input_path: Đường dẫn ảnh siêu âm đầu vào
        output_path: Đường dẫn lưu ảnh kết quả
        ethnicity: Dân tộc của em bé
        max_tokens: Max tokens cho response
        temperature: Temperature cho generation
        
    Returns:
        bool: True nếu thành công
    """
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        return False
    
    print(f"🖼️ Processing: {Path(input_path).name}")
    print(f"   Ethnicity: {ethnicity}")
    print(f"   Output: {output_path}")
    
    try:
        # Encode image to base64
        image_data_uri = encode_image_to_base64(input_path)
        
        # Create instruction
        instruction = create_baby_instruction(ethnicity)
        
        # Prepare messages for Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]
            }
        ]
        
        print("🤖 Generating baby image...")
        start_time = time.time()
        
        # Generate with model
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        generation_time = time.time() - start_time
        print(f"   Generation time: {generation_time:.1f}s")
        
        # Extract response
        if response and "choices" in response and response["choices"]:
            content = response["choices"][0]["message"]["content"]
            
            # Try to decode as image
            if decode_response_as_image(content, output_path):
                print(f"✅ Image saved: {output_path}")
                return True
            else:
                # Save text response for debugging
                text_path = output_path.replace('.png', '_response.txt')
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(f"Generation time: {generation_time:.1f}s\n\n")
                    f.write(f"Instruction:\n{instruction}\n\n")
                    f.write(f"Response:\n{content}")
                
                print(f"⚠️ Model returned text instead of image")
                print(f"   Text saved to: {text_path}")
                return False
        else:
            print("❌ Empty response from model")
            return False
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False


def decode_response_as_image(content: str, output_path: str) -> bool:
    """
    Thử decode response thành ảnh
    
    Args:
        content: Response content từ model
        output_path: Đường dẫn lưu ảnh
        
    Returns:
        bool: True nếu decode thành công
    """
    try:
        # Method 1: Direct base64 data URI
        if content.startswith("data:image/"):
            base64_data = content.split(",", 1)[1]
            image_data = base64.b64decode(base64_data)
            
            with open(output_path, 'wb') as f:
                f.write(image_data)
            return True
        
        # Method 2: Raw base64 string
        elif len(content) > 1000 and content.replace('\n', '').replace(' ', '').isalnum():
            try:
                image_data = base64.b64decode(content.replace('\n', '').replace(' ', ''))
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                return True
            except Exception:
                pass
        
        # Method 3: File path
        elif os.path.exists(content.strip()):
            import shutil
            shutil.copy2(content.strip(), output_path)
            return True
        
        return False
        
    except Exception as e:
        print(f"   Decode error: {e}")
        return False


def process_single_image(input_path: str, output_path: str = None, model_type: str = "auto", ethnicity: str = "mixed") -> bool:
    """Xử lý một ảnh đơn lẻ"""
    try:
        # Get model from cache or load new one
        model = get_model(model_type)
        if not model:
            return False
        
        # Determine output path
        if not output_path:
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"outputs/predicted_{input_name}.png"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process image
        result = process_ultrasound_image(input_path, model, os.path.dirname(output_path), ethnicity)
        return result is not None
        
    except Exception as e:
        print(f"❌ Error processing single image: {e}")
        return False


def batch_process_images(image_paths: list, output_dir: str = "outputs", model_type: str = "auto", ethnicity: str = "mixed") -> list:
    """Xử lý nhiều ảnh cùng một lúc"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model from cache or load new one
    model = get_model(model_type)
    if not model:
        return []
    
    results = []
    for image_path in image_paths:
        try:
            result = process_ultrasound_image(image_path, model, output_dir, ethnicity)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue
    
    return results