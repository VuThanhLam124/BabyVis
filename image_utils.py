#!/usr/bin/env python3
"""
Image Processing Utilities for BabyVis
Advanced image processing functions for ultrasound and baby face generation
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltrasoundProcessor:
    """Specialized processor for ultrasound images"""
    
    @staticmethod
    def enhance_ultrasound(image: Image.Image, enhance_factor: float = 1.5) -> Image.Image:
        """
        Enhance ultrasound image for better AI processing
        
        Args:
            image: Input ultrasound PIL Image
            enhance_factor: Enhancement strength (1.0-3.0)
            
        Returns:
            Enhanced PIL Image
        """
        # Convert to grayscale if needed, then back to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to OpenCV format for advanced processing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        enhanced_cv = cv2.merge([l_channel, a_channel, b_channel])
        enhanced_cv = cv2.cvtColor(enhanced_cv, cv2.COLOR_LAB2BGR)
        
        # Convert back to PIL
        enhanced_image = Image.fromarray(cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB))
        
        # Additional PIL enhancements
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(enhance_factor)
        
        enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = enhancer.enhance(1.2)
        
        return enhanced_image
    
    @staticmethod
    def detect_fetal_features(image: Image.Image) -> Dict[str, Any]:
        """
        Detect potential fetal features in ultrasound image
        
        Args:
            image: Input ultrasound PIL Image
            
        Returns:
            Dictionary with detected features and confidence scores
        """
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours (potential anatomical structures)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours for head-like shapes
        head_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                # Check if contour is roughly circular (head-like)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.3 < circularity < 1.2:  # Reasonable range for head
                        x, y, w, h = cv2.boundingRect(contour)
                        head_candidates.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'circularity': circularity,
                            'confidence': min(circularity, 1.0)
                        })
        
        # Sort by confidence and area
        head_candidates.sort(key=lambda x: x['confidence'] * x['area'], reverse=True)
        
        return {
            'head_candidates': head_candidates[:3],  # Top 3 candidates
            'total_features': len(contours),
            'image_quality': UltrasoundProcessor._assess_image_quality(cv_image)
        }
    
    @staticmethod
    def _assess_image_quality(cv_image: np.ndarray) -> float:
        """Assess ultrasound image quality based on contrast and clarity"""
        # Calculate image variance (measure of contrast)
        variance = cv2.Laplacian(cv_image, cv2.CV_64F).var()
        
        # Normalize to 0-1 scale
        quality_score = min(variance / 1000.0, 1.0)
        
        return quality_score

class BabyFaceProcessor:
    """Specialized processor for baby face images"""
    
    @staticmethod
    def enhance_baby_features(image: Image.Image, intensity: float = 0.3) -> Image.Image:
        """
        Enhance baby-like features in generated images
        
        Args:
            image: Input baby face PIL Image
            intensity: Enhancement intensity (0.0-1.0)
            
        Returns:
            Enhanced PIL Image with baby-like features
        """
        enhanced_image = image.copy()
        
        # Convert to numpy for processing
        img_array = np.array(enhanced_image)
        
        # Baby skin tone enhancement
        # Warm, peachy skin tone adjustments
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1.0 + intensity * 0.1), 0, 255)  # Slight red boost
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1.0 + intensity * 0.05), 0, 255)  # Slight green boost
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1.0 - intensity * 0.05), 0, 255)  # Slight blue reduction
        
        # Convert back to PIL
        enhanced_image = Image.fromarray(img_array.astype(np.uint8))
        
        # Apply baby-soft skin filter
        enhanced_image = enhanced_image.filter(ImageFilter.GaussianBlur(radius=intensity * 1.5))
        
        # Enhance brightness for healthy baby glow
        enhancer = ImageEnhance.Brightness(enhanced_image)
        enhanced_image = enhancer.enhance(1.0 + intensity * 0.2)
        
        # Increase color saturation slightly
        enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = enhancer.enhance(1.0 + intensity * 0.3)
        
        return enhanced_image
    
    @staticmethod
    def add_baby_glow(image: Image.Image, glow_intensity: float = 0.2) -> Image.Image:
        """
        Add a soft, healthy glow effect typical of baby portraits
        
        Args:
            image: Input PIL Image
            glow_intensity: Glow effect intensity (0.0-1.0)
            
        Returns:
            PIL Image with glow effect
        """
        # Create a blurred version for the glow effect
        glow = image.filter(ImageFilter.GaussianBlur(radius=20))
        
        # Blend the original with the glow
        glowing_image = Image.blend(image, glow, glow_intensity)
        
        # Enhance the center area (face) more than edges
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Create radial gradient mask
        width, height = image.size
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 2
        
        for radius in range(max_radius, 0, -5):
            alpha = int(255 * (1.0 - radius / max_radius) * glow_intensity)
            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], fill=alpha)
        
        # Apply the glow mask
        result = Image.composite(glowing_image, image, mask)
        
        return result

class ImageComposer:
    """Utility class for image composition and comparison"""
    
    @staticmethod
    def create_before_after(
        ultrasound_image: Image.Image,
        baby_image: Image.Image,
        output_size: Tuple[int, int] = (1024, 512)
    ) -> Image.Image:
        """
        Create a before/after comparison image
        
        Args:
            ultrasound_image: Original ultrasound PIL Image
            baby_image: Generated baby PIL Image
            output_size: Final comparison image size
            
        Returns:
            PIL Image showing before/after comparison
        """
        width, height = output_size
        half_width = width // 2
        
        # Create comparison canvas
        comparison = Image.new("RGB", output_size, "white")
        
        # Resize images to fit
        ultrasound_resized = ultrasound_image.resize((half_width - 10, height - 60))
        baby_resized = baby_image.resize((half_width - 10, height - 60))
        
        # Paste images
        comparison.paste(ultrasound_resized, (5, 50))
        comparison.paste(baby_resized, (half_width + 5, 50))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            # Try to use a nice font
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        # Add "Before" and "After" labels
        draw.text((half_width // 2 - 30, 10), "Ultrasound", fill="black", font=font)
        draw.text((half_width + half_width // 2 - 20, 10), "Baby Face", fill="black", font=font)
        
        # Add divider line
        draw.line([(half_width, 0), (half_width, height)], fill="gray", width=2)
        
        return comparison
    
    @staticmethod
    def create_gallery(
        images: List[Image.Image],
        titles: Optional[List[str]] = None,
        grid_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Create a gallery of images in a grid layout
        
        Args:
            images: List of PIL Images
            titles: Optional titles for each image
            grid_size: Optional (cols, rows) - auto-calculated if None
            
        Returns:
            PIL Image containing the gallery
        """
        if not images:
            return Image.new("RGB", (400, 300), "white")
        
        # Calculate grid size if not provided
        if grid_size is None:
            num_images = len(images)
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
        else:
            cols, rows = grid_size
        
        # Determine cell size
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        cell_width = max_width + 20  # padding
        cell_height = max_height + 50 if titles else max_height + 20
        
        # Create gallery canvas
        gallery_width = cols * cell_width
        gallery_height = rows * cell_height
        gallery = Image.new("RGB", (gallery_width, gallery_height), "white")
        
        # Place images
        for idx, img in enumerate(images):
            if idx >= cols * rows:
                break
                
            row = idx // cols
            col = idx % cols
            
            x = col * cell_width + 10
            y = row * cell_height + 10
            
            # Resize image to fit cell if needed
            if img.width > max_width or img.height > max_height:
                img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
            
            gallery.paste(img, (x, y))
            
            # Add title if provided
            if titles and idx < len(titles):
                draw = ImageDraw.Draw(gallery)
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                title_y = y + img.height + 5
                draw.text((x, title_y), titles[idx], fill="black", font=font)
        
        return gallery

class ImageValidator:
    """Utility class for validating images"""
    
    @staticmethod
    def validate_ultrasound(image: Image.Image) -> Tuple[bool, str]:
        """
        Validate if an image looks like an ultrasound
        
        Args:
            image: PIL Image to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Basic format checks
            if image.mode not in ["RGB", "RGBA", "L"]:
                return False, "Invalid image format"
            
            width, height = image.size
            if width < 100 or height < 100:
                return False, "Image too small (minimum 100x100)"
            
            if width > 4096 or height > 4096:
                return False, "Image too large (maximum 4096x4096)"
            
            # Convert to grayscale for analysis
            if image.mode != "L":
                gray_image = image.convert("L")
            else:
                gray_image = image
            
            # Check if image has reasonable contrast (ultrasounds typically do)
            img_array = np.array(gray_image)
            contrast = img_array.std()
            
            if contrast < 10:
                return False, "Image appears to have very low contrast"
            
            # Check for typical ultrasound characteristics
            # Ultrasounds often have distinctive noise patterns and bright/dark regions
            histogram = np.histogram(img_array, bins=50, range=(0, 255))[0]
            
            # Check if histogram is reasonable (not just solid color)
            if np.max(histogram) > len(img_array.flatten()) * 0.9:
                return False, "Image appears to be mostly uniform color"
            
            return True, "Image appears to be a valid ultrasound"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def check_image_quality(image: Image.Image) -> Dict[str, Any]:
        """
        Assess image quality metrics
        
        Args:
            image: PIL Image to assess
            
        Returns:
            Dictionary with quality metrics
        """
        # Convert to grayscale for analysis
        if image.mode != "L":
            gray_image = image.convert("L")
        else:
            gray_image = image
        
        img_array = np.array(gray_image)
        
        # Calculate quality metrics
        metrics = {
            'width': image.width,
            'height': image.height,
            'aspect_ratio': image.width / image.height,
            'mean_brightness': float(np.mean(img_array)),
            'contrast': float(np.std(img_array)),
            'sharpness': ImageValidator._calculate_sharpness(img_array),
            'format': image.mode,
            'size_mb': ImageValidator._estimate_size_mb(image)
        }
        
        # Quality assessment
        metrics['quality_score'] = ImageValidator._calculate_quality_score(metrics)
        
        return metrics
    
    @staticmethod
    def _calculate_sharpness(img_array: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
        return float(laplacian_var)
    
    @staticmethod
    def _estimate_size_mb(image: Image.Image) -> float:
        """Estimate image size in MB"""
        # Rough estimation based on dimensions and mode
        pixels = image.width * image.height
        
        if image.mode == "RGB":
            bytes_per_pixel = 3
        elif image.mode == "RGBA":
            bytes_per_pixel = 4
        else:
            bytes_per_pixel = 1
        
        size_bytes = pixels * bytes_per_pixel
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb
    
    @staticmethod
    def _calculate_quality_score(metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-1)"""
        score = 0.0
        
        # Resolution score
        if metrics['width'] >= 512 and metrics['height'] >= 512:
            score += 0.3
        elif metrics['width'] >= 256 and metrics['height'] >= 256:
            score += 0.2
        else:
            score += 0.1
        
        # Contrast score
        if metrics['contrast'] > 50:
            score += 0.3
        elif metrics['contrast'] > 30:
            score += 0.2
        else:
            score += 0.1
        
        # Sharpness score
        if metrics['sharpness'] > 100:
            score += 0.2
        elif metrics['sharpness'] > 50:
            score += 0.15
        else:
            score += 0.05
        
        # Brightness score (not too dark, not too bright)
        brightness = metrics['mean_brightness']
        if 50 <= brightness <= 200:
            score += 0.2
        elif 30 <= brightness <= 230:
            score += 0.1
        else:
            score += 0.05
        
        return min(score, 1.0)


def main():
    """Test the image processing utilities"""
    sample_path = "/home/ubuntu/DataScience/MyProject/BabyVis/samples/1.jpeg"
    
    if os.path.exists(sample_path):
        print("🧪 Testing image processing utilities...")
        
        # Load test image
        test_image = Image.open(sample_path)
        
        # Test ultrasound processing
        processor = UltrasoundProcessor()
        enhanced = processor.enhance_ultrasound(test_image)
        features = processor.detect_fetal_features(test_image)
        
        print(f"✅ Enhanced ultrasound image")
        print(f"📊 Detected features: {features}")
        
        # Test validation
        validator = ImageValidator()
        is_valid, message = validator.validate_ultrasound(test_image)
        quality = validator.check_image_quality(test_image)
        
        print(f"✅ Validation: {is_valid} - {message}")
        print(f"📈 Quality score: {quality['quality_score']:.2f}")
        
        # Save test results
        output_dir = "/home/ubuntu/DataScience/MyProject/BabyVis/outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        enhanced.save(os.path.join(output_dir, "enhanced_ultrasound.png"))
        print(f"💾 Test results saved to {output_dir}")
        
    else:
        print(f"❌ Test image not found: {sample_path}")


if __name__ == "__main__":
    main()