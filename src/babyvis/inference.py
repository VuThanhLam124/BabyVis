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
from .model_utils import load_canny_pipeline
from typing import List, Tuple, Optional, Dict, Any
import torch


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


def convert_to_canny(input_path: str, canny_path: str, method: str = "adaptive"):
    """
    Convert input image to Canny edge map with enhanced detection.

    Args:
        input_path: Path to input image
        canny_path: Path to save Canny edge map
        method: Edge detection method
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("OpenCV không đọc được ảnh – kiểm tra định dạng/quyền truy cập.")

    canny = enhanced_canny_detection(img, method)
    cv2.imwrite(canny_path, canny)


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


def apply_freeu_enhancement(pipe, b1: float = 1.2, b2: float = 1.4, 
                           s1: float = 0.8, s2: float = 0.8):
    """
    Apply FreeU enhancement to improve image quality without additional cost.

    Args:
        pipe: Diffusion pipeline
        b1, b2: Backbone enhancement factors
        s1, s2: Skip connection factors
    """
    try:
        # Enable FreeU if available
        if hasattr(pipe, 'enable_freeu'):
            pipe.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)
            print("✅ FreeU enhancement applied")
        else:
            print("⚠️ FreeU not available for this pipeline")
    except Exception as e:
        print(f"⚠️ FreeU enhancement failed: {e}")


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
                            enhance_output: bool = True):
    """
    Enhanced image generation with optimal settings and post-processing.

    Args:
        input_path: Path to input image
        output_path: Path to save generated image
        ethnicity: Ethnicity for baby generation
        num_inference_steps: Number of inference steps
        guidance_scale: CFG scale value
        canny_method: Canny edge detection method
        use_freeu: Whether to apply FreeU enhancement
        enhance_output: Whether to apply post-processing
    """
    print(f"🚀 Generating enhanced prediction for: {os.path.basename(input_path)}")

    # Load pipeline (allows SDXL/custom via env/args)
    pipe = load_canny_pipeline(
        base_model_id=base_model_id,
        controlnet_id=controlnet_id,
        prefer_sdxl=prefer_sdxl,
    )

    # Apply FreeU enhancement if requested
    if use_freeu:
        apply_freeu_enhancement(pipe)

    # Generate Canny edge map (store in outputs/tmp)
    tmp_dir = os.path.join("outputs", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    canny_tmp = os.path.join(tmp_dir, "tmp_canny_enhanced.png")
    convert_to_canny(input_path, canny_tmp, method=canny_method)
    canny_pil = Image.open(canny_tmp).convert("RGB")

    # Get professional prompts
    prompts = get_professional_baby_prompt(ethnicity)

    print(f"📝 Using {num_inference_steps} steps, CFG scale {guidance_scale}")

    # Generate image with enhanced settings
    call_kwargs = dict(
        prompt=prompts["positive"],
        negative_prompt=prompts["negative"],
        image=canny_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=torch.Generator(device=str(pipe.device)).manual_seed(seed),
    )
    # Control strength if supported
    try:
        call_kwargs["controlnet_conditioning_scale"] = control_strength
    except Exception:
        pass

    result = pipe(**call_kwargs)

    result_img = result.images[0]

    # Apply post-processing if requested
    if enhance_output:
        result_img = post_process_image(result_img)
        print("✨ Post-processing applied")

    # Save with high quality
    result_img.save(output_path, quality=95, optimize=True)

    # Clean up temporary file
    if os.path.exists(canny_tmp):
        os.remove(canny_tmp)

    print(f"✅ Enhanced image saved: {output_path}")


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

        # Generate enhanced prediction
        if generate_predictions:
            try:
                filename = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"enhanced_baby_{filename}.png")

                generate_predict_enhanced(
                    input_path=image_path,
                    output_path=output_path,
                    **enhanced_settings
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
