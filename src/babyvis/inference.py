# enhanced_inference.py
"""
Enhanced inference module for baby image generation from ultrasound images.
Optimized for better quality and performance with professional prompts.
"""

import os, cv2
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from matplotlib import pyplot as plt
from .model_utils import (
    load_qwen_image_edit,
    load_qwen_image_edit_gguf,
)
from typing import List, Tuple, Optional, Dict, Any
import torch
import io, base64, re


# Configuration constants for optimal settings
OPTIMAL_CFG_SCALE = 8.5
OPTIMAL_INFERENCE_STEPS = 50
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 150


def adaptive_canny_thresholds(image: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    """
    Calculate adaptive Canny thresholds based on image statistics.

    Args:
        image: Grayscale image as numpy array
        sigma: Sigma parameter for threshold calculation

    Returns:
        Tuple of (lower_threshold, upper_threshold)
    """
    # Calculate median of pixel intensities
    median = np.median(image)

    # Calculate adaptive thresholds
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    # Ensure proper threshold relationship
    if upper - lower < 50:
        lower = max(0, int(median * 0.5))
        upper = min(255, int(median * 1.5))

    return lower, upper


def enhanced_canny_detection(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Enhanced Canny edge detection with multiple methods.

    Args:
        image: Input image as numpy array
        method: Detection method ("adaptive", "fixed", "enhanced")

    Returns:
        Canny edge detected image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    if method == "adaptive":
        low, high = adaptive_canny_thresholds(blurred)
    elif method == "enhanced":
        # Use enhanced method with gradient analysis
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)

        # Calculate thresholds based on gradient statistics
        grad_mean = np.mean(gradient)
        grad_std = np.std(gradient)
        low = max(10, int(grad_mean - 0.5 * grad_std))
        high = min(200, int(grad_mean + 1.5 * grad_std))
    else:  # fixed
        low, high = DEFAULT_CANNY_LOW, DEFAULT_CANNY_HIGH

    # Apply Canny edge detection
    canny = cv2.Canny(blurred, low, high, apertureSize=3, L2gradient=True)

    # Post-process to enhance edge connectivity
    kernel = np.ones((3, 3), np.uint8)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=1)

    return canny


# Removed ControlNet-specific functions: 
# - convert_to_canny()
# - apply_freeu_enhancement()
# These are no longer needed since we focus exclusively on Qwen models.


def get_professional_baby_prompt(ethnicity: str = "Asian", 
                                style: str = "professional") -> Dict[str, str]:
    """
    Generate professional prompts for baby image generation.

    Args:
        ethnicity: Ethnicity specification
        style: Style type

    Returns:
        Dictionary with positive and negative prompts
    """
    base_positive = f"""
    best quality, photorealistic, hyperrealistic, color,
    professional baby portrait photograph, {ethnicity}, newborn infant,
    peaceful sleeping expression, delicate facial features, soft baby skin texture,
    natural skin tone, sleeping in bed, small button nose, tiny lips,
    studio lighting, soft diffused light, perfect anatomy,
    high resolution, clean white background, medical photography,
    gentle pose, serene expression, healthy baby, no hand in image.
    """.strip()

    # Comprehensive negative prompts to avoid common issues
    negative_prompt = """
    worst quality, low quality, normal quality, jpeg artifacts, blurry, 
    out of focus, noise, grainy, pixelated, compressed,
    deformed, mutated, disfigured, malformed, bad anatomy, 
    extra limbs, missing limbs, extra fingers, missing fingers,
    extra arms, missing arms, extra legs, missing legs,
    fused fingers, too many fingers, bad hands, malformed hands,
    bad eyes, deformed eyes, extra eyes, missing eyes, cross-eyed,
    bad face, ugly face, deformed face, asymmetrical face,
    extra heads, multiple heads, two heads, conjoined,
    bad proportions, elongated body, stretched, distorted,
    watermark, signature, text, logo, username, artist name,
    frame, border, cropped, cut off, amputee,
    cartoon, anime, 3d render, painting, drawing, sketch,
    artificial, fake, plastic, doll-like, mannequin,
    adult, elderly, teenager, child, mature,
    scary, frightening, disturbing, nightmare, horror,
    dark, shadow, poor lighting, harsh lighting,
    alien, monster, creature, non-human, fantasy,
    multiple babies, twins, group, crowd,
    clothed, dressed, wearing clothes, hat, cap,
    open eyes, awake, crying, upset, distressed,
    medical equipment, tubes, wires, machinery,
    dirty, messy, unkempt, sick, ill, unhealthy,
    unrealistic skin, weird skin, scales, fur, hair,
    oversaturated, undersaturated, wrong colors,
    low contrast, high contrast, overexposed, underexposed, palm, full hand.
    """.strip()

    return {
        "positive": base_positive,
        "negative": negative_prompt
    }


# Removed apply_freeu_enhancement() - only needed for diffusion pipelines


def post_process_image(image: Image.Image, enhance_factor: float = 1.2) -> Image.Image:
    """
    Apply post-processing to enhance image quality.

    Args:
        image: PIL Image to enhance
        enhance_factor: Enhancement strength

    Returns:
        Enhanced PIL Image
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply sharpening
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(enhance_factor)

    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)

    # Apply subtle unsharp mask
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))

    return image


def generate_predict_qwen_only(
    input_path: str,
    output_path: str,
    backend: str = "qwen_gguf",  # Default to GGUF for better compatibility
    ethnicity: str = "mixed ethnicity",
    instruction: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs,
):
    """
    Generate prediction using only Qwen models (Image Edit or GGUF).
    No ControlNet fallback - pure Qwen approach.

    Args:
        input_path: Path to input ultrasound image
        output_path: Path to save generated image
        backend: "qwen" for transformers, "qwen_gguf" for llama.cpp
        ethnicity: Ethnicity for baby generation
        instruction: Custom instruction (uses default if None)
        seed: Random seed
        **kwargs: Additional parameters
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    print(f"🚀 Generating with Qwen backend: {backend}")

    if backend.lower() in {"qwen_gguf", "gguf"}:
        return generate_predict_qwen_edit_gguf(
            input_path=input_path,
            output_path=output_path,
            ethnicity=ethnicity,
            seed=seed,
            **kwargs
        )
    elif backend.lower() in {"qwen", "qwen-image-edit", "qwen_image_edit"}:
        return generate_predict_qwen_edit(
            input_path=input_path,
            output_path=output_path,
            ethnicity=ethnicity,
            instruction=instruction,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'qwen' or 'qwen_gguf'")


def generate_predict_enhanced(input_path: str, 
                            output_path: str,
                            ethnicity: str = "mixed ethnicity",
                            num_inference_steps: int = OPTIMAL_INFERENCE_STEPS,
                            guidance_scale: float = OPTIMAL_CFG_SCALE,
                            canny_method: str = "adaptive",
                            control_strength: float = 0.8,
                            base_model_id: Optional[str] = None,
                            controlnet_id: Optional[str] = None,
                            prefer_sdxl: Optional[bool] = None,
                            seed: int = 42,
                            width: int = 512,
                            height: int = 512,
                            use_freeu: bool = True,
                            enhance_output: bool = True,
                            prompt_text: Optional[str] = None,
                            negative_text: Optional[str] = None,
                            ):
    """
    DEPRECATED: This function has been removed as we focus exclusively on Qwen.
    Use generate_predict_qwen_only instead.
    """
    raise NotImplementedError(
        "ControlNet support has been removed. Use generate_predict_qwen_only() with backend='qwen' or 'qwen_gguf'"
    )


# --- Qwen Image Edit integration ---
def _default_qwen_instruction(ethnicity: str = "mixed ethnicity") -> str:
    prompts = get_professional_baby_prompt(ethnicity)
    # Fold the negative prompt into a natural instruction to avoid undesired artifacts
    negative = re.sub(r"\s+", " ", prompts["negative"]).strip()
    positive = re.sub(r"\s+", " ", prompts["positive"]).strip()
    instr = (
        f"Edit this ultrasound image into a photorealistic newborn baby portrait. "
        f"Follow these quality requirements: {positive}. "
        f"Avoid these issues: {negative}."
    )
    return instr


def _decode_qwen_image(response: Any) -> Image.Image:
    """
    Try to decode an image returned by Qwen Image Edit remote code.
    Supports PIL.Image, data URI, bytes/base64, or a file path.
    """
    # 1) Direct PIL image
    if isinstance(response, Image.Image):
        return response

    # 2) Dict payloads
    if isinstance(response, dict):
        # Common keys to check
        for key in ("image", "images", "output", "result"):
            if key in response:
                val = response[key]
                if isinstance(val, Image.Image):
                    return val
                if isinstance(val, (bytes, bytearray)):
                    return Image.open(io.BytesIO(val)).convert("RGB")
                if isinstance(val, str):
                    # try data URI or path
                    if val.startswith("data:image/"):
                        b64 = val.split(",", 1)[-1]
                        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                    if os.path.exists(val):
                        return Image.open(val).convert("RGB")
        # Some remote codes may include a list
        for key in ("images", "outputs"):
            if key in response and isinstance(response[key], list) and response[key]:
                return _decode_qwen_image(response[key][0])

    # 3) String payloads
    if isinstance(response, str):
        if response.startswith("data:image/"):
            b64 = response.split(",", 1)[-1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        if os.path.exists(response):
            return Image.open(response).convert("RGB")

    # 4) Bytes
    if isinstance(response, (bytes, bytearray)):
        return Image.open(io.BytesIO(response)).convert("RGB")

    raise ValueError("Unable to decode edited image from Qwen response.")


def generate_predict_qwen_edit(
    input_path: str,
    output_path: str,
    ethnicity: str = "mixed ethnicity",
    instruction: Optional[str] = None,
    seed: Optional[int] = None,
    use_cpu_fallback: bool = True,
):
    """
    Generate an edited image using Qwen/Qwen-Image-Edit.

    Args:
        input_path: Path to input ultrasound image
        output_path: Path to save edited image
        ethnicity: Used to craft a high-quality instruction
        instruction: Optional explicit instruction; if None, a default is created
        seed: Optional RNG seed if supported by remote code
        use_cpu_fallback: If loading fails on GPU, retry on CPU
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    instruction = instruction or _default_qwen_instruction(ethnicity)

    # Load model/tokenizer
    try:
        tokenizer, model, device = load_qwen_image_edit()
    except Exception as e:
        if use_cpu_fallback and ("cuda" in str(e).lower() or "gpu" in str(e).lower()):
            os.environ["FORCE_CPU"] = "1"
            tokenizer, model, device = load_qwen_image_edit()
        else:
            raise

    # Build Qwen-VL style query using trust_remote_code utilities
    try:
        query = tokenizer.from_list_format([
            {"image": input_path},
            {"text": instruction},
        ])
    except Exception:
        # Fallback to raw instruction if tokenizer lacks helper
        query = instruction

    # Some Qwen remote codes accept a seed argument; handle gracefully if available
    chat_kwargs = {}
    if seed is not None:
        chat_kwargs["seed"] = seed

    # Perform edit
    try:
        response, _ = model.chat(tokenizer, query=query, history=None, **chat_kwargs)
    except TypeError:
        # If chat signature differs, call without extra kwargs
        response, _ = model.chat(tokenizer, query=query, history=None)

    # Decode result to PIL
    edited = _decode_qwen_image(response)
    edited = edited.convert("RGB")

    # Save result
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    edited.save(output_path, quality=95, optimize=True)
    print(f"✅ Qwen Image Edit saved: {output_path}")


def _encode_image_to_data_uri(path: str) -> str:
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    ext = os.path.splitext(path)[1].lower().lstrip('.') or 'png'
    if ext == 'jpg':
        ext = 'jpeg'
    return f"data:image/{ext};base64,{b64}"


def generate_predict_qwen_edit_gguf(
    input_path: str,
    output_path: str,
    ethnicity: str = "mixed ethnicity",
    seed: Optional[int] = None,
    n_predict: int = 512,
):
    """
    Use GGUF (llama.cpp) Qwen Image Edit for direct image generation.
    No ControlNet fallback - pure Qwen approach with enhanced retry logic.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    try:
        llm = load_qwen_image_edit_gguf()
    except Exception as e:
        raise RuntimeError(f"Failed to load Qwen GGUF backend: {e}")

    # Compose a concise instruction focused on direct image generation
    instruction = _default_qwen_instruction(ethnicity)
    data_uri = _encode_image_to_data_uri(input_path)

    # llama-cpp chat messages format with image content
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]

    # Try multiple approaches for better compatibility
    response_text = None
    methods = [
        ("create_chat_completion", lambda: llm.create_chat_completion(
            messages=messages, temperature=0.2, max_tokens=n_predict
        )),
        ("chat_completions.create", lambda: llm.chat_completions.create(
            messages=messages, temperature=0.2, max_tokens=n_predict
        )),
        ("direct_completion", lambda: llm(
            prompt=instruction, max_tokens=n_predict, temperature=0.2
        ))
    ]

    for method_name, method_func in methods:
        try:
            print(f"🔍 Trying {method_name}...")
            out = method_func()
            
            if hasattr(out, 'choices') and out.choices:
                response_text = out.choices[0].message.content
                break
            elif isinstance(out, dict):
                if "choices" in out and out["choices"]:
                    response_text = out["choices"][0].get("message", {}).get("content") or out["choices"][0].get("text")
                    break
                elif "text" in out:
                    response_text = out["text"]
                    break
        except Exception as e:
            print(f"⚠️ {method_name} failed: {e}")
            continue

    if not response_text:
        raise RuntimeError("Qwen GGUF returned empty response after trying all methods")

    # Enhanced image decoding with multiple fallback approaches
    decoded_successfully = False
    
    # Try to decode as image data
    try:
        response_text = response_text.strip()
        
        # Method 1: Direct data URI
        if response_text.startswith("data:image/"):
            img = _decode_qwen_image(response_text)
            img.save(output_path, quality=95, optimize=True)
            print(f"✅ Qwen GGUF Image (data URI) saved: {output_path}")
            decoded_successfully = True
        
        # Method 2: JSON payload
        elif response_text.startswith("{") and response_text.endswith("}"):
            import json
            payload = json.loads(response_text)
            img = _decode_qwen_image(payload)
            img.save(output_path, quality=95, optimize=True)
            print(f"✅ Qwen GGUF Image (JSON) saved: {output_path}")
            decoded_successfully = True
            
        # Method 3: File path
        elif os.path.exists(response_text):
            img = _decode_qwen_image(response_text)
            img.save(output_path, quality=95, optimize=True)
            print(f"✅ Qwen GGUF Image (file path) saved: {output_path}")
            decoded_successfully = True
            
    except Exception as decode_error:
        print(f"⚠️ Image decoding failed: {decode_error}")

    # If we couldn't decode an image, the response might be text description
    if not decoded_successfully:
        # Use the response as an enhanced description to generate with Qwen Image Edit (transformers)
        print("🔄 GGUF returned text description, trying Qwen Image Edit (transformers)...")
        try:
            enhanced_instruction = f"{response_text}\n\nAdditional requirements: {instruction}"
            return generate_predict_qwen_edit(
                input_path=input_path,
                output_path=output_path,
                ethnicity=ethnicity,
                instruction=enhanced_instruction,
                seed=seed,
            )
        except Exception as fallback_error:
            # If both approaches fail, save the text response for debugging
            text_output = output_path.replace('.png', '_description.txt')
            with open(text_output, 'w', encoding='utf-8') as f:
                f.write(f"Qwen GGUF Response:\n{response_text}\n\nOriginal instruction:\n{instruction}")
            
            raise RuntimeError(
                f"Qwen GGUF could not generate image. Text response saved to {text_output}. "
                f"GGUF error: Could not decode image. Fallback error: {fallback_error}"
            )


def generate_predict_auto(
    input_path: str,
    output_path: str,
    backend: Optional[str] = None,
    ethnicity: str = "mixed ethnicity",
    **kwargs,
):
    """
    Auto-dispatch to the best available Qwen backend.
    Priority: Qwen Image Edit → Qwen GGUF (no ControlNet fallback).
    """
    # Determine backend priority: explicit → env → auto-detect
    if backend:
        bl = backend.lower()
        if bl in {"qwen", "qwen-image-edit", "qwen_image_edit"}:
            use_backend = "qwen"
        elif bl in {"qwen-gguf", "qwen_image_edit_gguf", "gguf"}:
            use_backend = "qwen_gguf"
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'qwen' or 'qwen_gguf'")
    else:
        # Check environment variables
        if os.getenv("USE_QWEN_IMAGE_EDIT", "").lower() in {"1", "true", "yes"}:
            use_backend = "qwen"
        elif os.getenv("USE_QWEN_GGUF", "").lower() in {"1", "true", "yes"}:
            use_backend = "qwen_gguf"
        else:
            # Auto-detect: Try GGUF first (more compatible), then transformers
            print("🔍 Auto-detecting best Qwen backend...")
            try:
                # Test GGUF availability
                load_qwen_image_edit_gguf()
                use_backend = "qwen_gguf"
                print("✅ Using Qwen GGUF (better compatibility)")
            except Exception as e1:
                print(f"⚠️ Qwen GGUF unavailable: {e1}")
                try:
                    # Test transformers availability
                    load_qwen_image_edit()
                    use_backend = "qwen"
                    print("✅ Using Qwen Image Edit (transformers)")
                except Exception as e2:
                    print(f"❌ Both Qwen backends failed: GGUF={e1}, Transformers={e2}")
                    raise RuntimeError("No Qwen backend available. Install llama-cpp-python or transformers.")

    return generate_predict_qwen_only(
        input_path=input_path,
        output_path=output_path,
        backend=use_backend,
        ethnicity=ethnicity,
        **kwargs
    )


def show_7_channels_auto(image_path: str, delay_sec: float = 2):
    """
    Display 7 color channels automatically with delay (keeping original interface).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    channels = []
    names = []

    # 1. Original
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    channels.append(img_rgb)
    names.append("Original")

    # 2. Enhanced Canny Edge
    canny = enhanced_canny_detection(img, method="adaptive")
    channels.append(canny)
    names.append("Enhanced Canny Edge")

    # 3-5. R, G, B channels
    for idx, color in enumerate(['Red', 'Green', 'Blue']):
        ch = np.zeros_like(img)
        ch[:,:,idx] = img[:,:,idx]
        ch_rgb = cv2.cvtColor(ch, cv2.COLOR_BGR2RGB)
        channels.append(ch_rgb)
        names.append(f"{color} Channel")

    # 6-7. Hue and Saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0]
    sat = hsv[:,:,1]
    channels.append(hue)
    names.append("Hue Channel")
    channels.append(sat)
    names.append("Saturation Channel")

    # Display each channel with delay
    for i, (im, name) in enumerate(zip(channels, names)):
        print(f"  [{i+1}/7] {name}...")
        plt.figure(figsize=(8, 6))
        plt.imshow(im, cmap='gray' if len(im.shape) == 2 else None)
        plt.title(f"{name} - {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(delay_sec)
        plt.close()


def batch_process_with_display(image_paths: List[str], 
                             output_dir: str = "outputs/batch", 
                             delay_sec: float = 2, 
                             generate_predictions: bool = True,
                             **kwargs) -> List[str]:
    """
    Enhanced batch processing with improved quality and settings.
    Maintains compatibility with existing batch_processor.py

    Args:
        image_paths: List of image paths to process
        output_dir: Output directory for generated images
        delay_sec: Delay between channel displays
        generate_predictions: Whether to generate predictions
        **kwargs: Additional parameters for generation

    Returns:
        List of output file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Enhanced settings with defaults
    enhanced_settings = {
        'num_inference_steps': kwargs.get('num_inference_steps', OPTIMAL_INFERENCE_STEPS),
        'guidance_scale': kwargs.get('guidance_scale', OPTIMAL_CFG_SCALE),
        'canny_method': kwargs.get('canny_method', 'adaptive'),
        'use_freeu': kwargs.get('use_freeu', True),
        'enhance_output': kwargs.get('enhance_output', True),
        'ethnicity': kwargs.get('ethnicity', 'mixed ethnicity')
    }

    print(f"🔧 Enhanced settings: {enhanced_settings}")

    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue

        print(f"\n🖼️ Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

        # Display 7 color channels
        try:
            show_7_channels_auto(image_path, delay_sec)
        except Exception as e:
            print(f"⚠️ Error displaying channels: {e}")
            continue

        # Generate prediction (Qwen Image Edit if enabled, else ControlNet)
        if generate_predictions:
            try:
                filename = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"enhanced_baby_{filename}.png")

                generate_predict_auto(
                    input_path=image_path,
                    output_path=output_path,
                    ethnicity=enhanced_settings['ethnicity'],
                    num_inference_steps=enhanced_settings['num_inference_steps'],
                    guidance_scale=enhanced_settings['guidance_scale'],
                    canny_method=enhanced_settings['canny_method'],
                    use_freeu=enhanced_settings['use_freeu'],
                    enhance_output=enhanced_settings['enhance_output'],
                )

                results.append(output_path)
                print(f"✅ Generated: {output_path}")

            except Exception as e:
                print(f"❌ Error generating prediction: {e}")
                continue

        # Pause between images
        if i < len(image_paths) - 1:
            print(f"⏸️ Pausing {delay_sec*2} seconds before next image...")
            time.sleep(delay_sec * 2)

    print(f"\n🎉 Batch processing completed. Generated {len(results)} images.")
    return results


# Module cleaned up: presets and CLI removed; keep only functions used by
# app_gradio.py and batch_processor.py.
