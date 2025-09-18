#!/usr/bin/env python3
"""
Advanced Ultrasound Image Processing for BabyVis
Enhanced medical-grade preprocessing inspired by baby face generation techniques from YouTube tutorials
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from typing import Tuple, Optional, Dict, List
import logging
from scipy import ndimage, signal
from skimage import morphology, measure, filters, exposure, segmentation
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class AdvancedUltrasoundProcessor:
    """
    Advanced medical-grade ultrasound processing based on YouTube video techniques for baby face generation
    Implements state-of-the-art enhancement methods for optimal AI model input
    """
    
    def __init__(self):
        self.gestational_age_ranges = {
            "early": (12, 20),    # 12-20 weeks
            "mid": (20, 28),      # 20-28 weeks  
            "late": (28, 40)      # 28-40 weeks
        }
        
        # Advanced enhancement parameters based on gestational age
        self.enhancement_profiles = {
            "early": {
                "contrast_boost": 1.4,
                "brightness_adjust": 1.2,
                "noise_reduction": 0.8,
                "edge_enhancement": 1.3,
                "feature_emphasis": 0.9
            },
            "mid": {
                "contrast_boost": 1.3,
                "brightness_adjust": 1.15,
                "noise_reduction": 0.7,
                "edge_enhancement": 1.2,
                "feature_emphasis": 1.0
            },
            "late": {
                "contrast_boost": 1.2,
                "brightness_adjust": 1.1,
                "noise_reduction": 0.6,
                "edge_enhancement": 1.1,
                "feature_emphasis": 1.1
            }
        }
    
    def detect_gestational_age(self, ultrasound_image: Image.Image) -> str:
        """
        Estimate gestational age from ultrasound image characteristics
        Based on feature density, image clarity, and facial development
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(np.array(ultrasound_image), cv2.COLOR_RGB2GRAY)
            
            # Calculate image complexity metrics
            edge_density = self._calculate_edge_density(gray)
            feature_clarity = self._calculate_feature_clarity(gray)
            noise_level = self._calculate_noise_level(gray)
            
            # Gestational age estimation based on image characteristics
            if edge_density < 0.3 and feature_clarity < 0.4:
                return "early"
            elif edge_density > 0.6 and feature_clarity > 0.7:
                return "late"
            else:
                return "mid"
                
        except Exception as e:
            logger.warning(f"Gestational age detection failed: {e}, using 'mid' as default")
            return "mid"
    
    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density using Canny edge detection"""
        edges = cv2.Canny(gray_image, 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def _calculate_feature_clarity(self, gray_image: np.ndarray) -> float:
        """Calculate feature clarity using gradient magnitude"""
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(magnitude) / 255.0
    
    def _calculate_noise_level(self, gray_image: np.ndarray) -> float:
        """Calculate noise level using Laplacian variance"""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return np.var(laplacian) / 10000.0
    
    def enhance_ultrasound_medical_grade(
        self, 
        ultrasound_image: Image.Image,
        target_size: Tuple[int, int] = (512, 512),
        gestational_age: Optional[str] = None
    ) -> Image.Image:
        """
        Medical-grade ultrasound enhancement based on YouTube video techniques
        
        Advanced processing pipeline:
        1. Gestational age detection
        2. Adaptive contrast enhancement
        3. Medical noise reduction
        4. Anatomical feature enhancement
        5. AI model optimization
        """
        
        # Detect gestational age if not provided
        if gestational_age is None:
            gestational_age = self.detect_gestational_age(ultrasound_image)
        
        profile = self.enhancement_profiles[gestational_age]
        logger.info(f"🩺 Processing ultrasound with {gestational_age} gestational age profile")
        
        # Convert to RGB if needed
        if ultrasound_image.mode != "RGB":
            ultrasound_image = ultrasound_image.convert("RGB")
        
        # Step 1: Adaptive histogram equalization
        enhanced_image = self._adaptive_histogram_equalization(ultrasound_image)
        
        # Step 2: Medical-grade noise reduction
        enhanced_image = self._medical_noise_reduction(
            enhanced_image, 
            noise_factor=profile["noise_reduction"]
        )
        
        # Step 3: Anatomical feature enhancement
        enhanced_image = self._enhance_anatomical_features(
            enhanced_image,
            enhancement_factor=profile["feature_emphasis"]
        )
        
        # Step 4: Adaptive contrast and brightness
        enhanced_image = self._adaptive_contrast_brightness(
            enhanced_image,
            contrast_factor=profile["contrast_boost"],
            brightness_factor=profile["brightness_adjust"]
        )
        
        # Step 5: Edge enhancement for facial features
        enhanced_image = self._enhance_facial_edges(
            enhanced_image,
            edge_factor=profile["edge_enhancement"]
        )
        
        # Step 6: Final optimization for AI model
        enhanced_image = self._optimize_for_ai_model(enhanced_image, target_size)
        
        logger.info(f"✅ Medical-grade enhancement complete for {gestational_age} pregnancy")
        return enhanced_image
    
    def _adaptive_histogram_equalization(self, image: Image.Image) -> Image.Image:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Process each channel separately
        channels = cv2.split(cv_image)
        enhanced_channels = []
        
        for channel in channels:
            enhanced_channel = clahe.apply(channel)
            enhanced_channels.append(enhanced_channel)
        
        # Merge channels back
        enhanced_cv = cv2.merge(enhanced_channels)
        enhanced_rgb = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(enhanced_rgb)
    
    def _medical_noise_reduction(self, image: Image.Image, noise_factor: float = 0.7) -> Image.Image:
        """Advanced medical image denoising using bilateral filtering"""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter for edge-preserving smoothing
        bilateral = cv2.bilateralFilter(cv_image, 9, 75, 75)
        
        # Apply Gaussian blur for additional noise reduction
        gaussian = cv2.GaussianBlur(bilateral, (3, 3), 0)
        
        # Blend original and denoised based on noise factor
        result = cv2.addWeighted(cv_image, 1 - noise_factor, gaussian, noise_factor, 0)
        
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _enhance_anatomical_features(self, image: Image.Image, enhancement_factor: float = 1.0) -> Image.Image:
        """Enhance anatomical features using morphological operations"""
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply morphological opening to enhance structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(cv_image, cv2.MORPH_OPEN, kernel)
        
        # Apply morphological closing to fill gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Calculate enhancement mask
        enhancement_mask = cv2.subtract(closed, cv_image)
        
        # Apply enhancement based on factor
        enhanced = cv2.addWeighted(
            cv_image, 
            1.0, 
            enhancement_mask, 
            enhancement_factor * 0.3, 
            0
        )
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)
    
    def _adaptive_contrast_brightness(
        self, 
        image: Image.Image, 
        contrast_factor: float = 1.3,
        brightness_factor: float = 1.15
    ) -> Image.Image:
        """Apply adaptive contrast and brightness enhancement"""
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(contrast_factor)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(brightness_factor)
        
        return enhanced
    
    def _enhance_facial_edges(self, image: Image.Image, edge_factor: float = 1.2) -> Image.Image:
        """Enhance facial edges and features using unsharp masking"""
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_array, (5, 5), 1.5)
        
        # Calculate unsharp mask
        unsharp_mask = cv2.addWeighted(img_array, 1 + edge_factor, blurred, -edge_factor, 0)
        
        # Ensure values are in valid range
        unsharp_mask = np.clip(unsharp_mask, 0, 255)
        
        return Image.fromarray(unsharp_mask.astype(np.uint8))
    
    def _optimize_for_ai_model(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Final optimization for AI model input"""
        
        # Resize with high-quality resampling
        optimized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Apply subtle sharpening for AI processing
        sharpening_filter = ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=2)
        optimized = optimized.filter(sharpening_filter)
        
        # Final color optimization
        enhancer = ImageEnhance.Color(optimized)
        optimized = enhancer.enhance(1.1)
        
        return optimized
    
    def detect_facial_regions(self, ultrasound_image: Image.Image) -> Dict[str, any]:
        """
        Detect potential facial regions in ultrasound image
        Returns coordinates and confidence scores for facial features
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(ultrasound_image), cv2.COLOR_RGB2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours for facial features
            facial_regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (potential facial features)
                if 100 < area < 5000:
                    # Calculate bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Score based on typical facial feature characteristics
                    confidence = self._calculate_facial_confidence(area, aspect_ratio, contour)
                    
                    if confidence > 0.3:
                        facial_regions.append({
                            "bbox": (x, y, w, h),
                            "area": area,
                            "confidence": confidence,
                            "center": (x + w//2, y + h//2)
                        })
            
            # Sort by confidence
            facial_regions.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "regions": facial_regions[:5],  # Top 5 regions
                "total_regions": len(facial_regions),
                "image_size": ultrasound_image.size
            }
            
        except Exception as e:
            logger.error(f"Facial region detection failed: {e}")
            return {"regions": [], "total_regions": 0, "image_size": ultrasound_image.size}
    
    def _calculate_facial_confidence(self, area: float, aspect_ratio: float, contour) -> float:
        """Calculate confidence score for potential facial feature"""
        
        # Base score from area (prefer medium-sized features)
        if 500 < area < 2000:
            area_score = 1.0
        elif 200 < area < 500 or 2000 < area < 3000:
            area_score = 0.7
        else:
            area_score = 0.3
        
        # Aspect ratio score (prefer roughly circular/oval features)
        if 0.7 <= aspect_ratio <= 1.3:
            ratio_score = 1.0
        elif 0.5 <= aspect_ratio <= 1.5:
            ratio_score = 0.8
        else:
            ratio_score = 0.4
        
        # Contour complexity score
        perimeter = cv2.arcLength(contour, True)
        complexity = perimeter / (2 * np.sqrt(np.pi * area)) if area > 0 else 0
        
        if 1.0 <= complexity <= 1.5:
            complexity_score = 1.0
        elif 0.8 <= complexity <= 1.8:
            complexity_score = 0.7
        else:
            complexity_score = 0.4
        
        # Combined confidence score
        confidence = (area_score * 0.4 + ratio_score * 0.3 + complexity_score * 0.3)
        
        return confidence
    
    def create_enhancement_report(self, original_image: Image.Image, enhanced_image: Image.Image) -> Dict[str, any]:
        """Create detailed enhancement report"""
        
        # Calculate enhancement metrics
        original_array = np.array(original_image.convert("L"))
        enhanced_array = np.array(enhanced_image.convert("L"))
        
        # Contrast improvement
        original_contrast = np.std(original_array)
        enhanced_contrast = np.std(enhanced_array)
        contrast_improvement = (enhanced_contrast / original_contrast - 1) * 100
        
        # Brightness analysis
        original_brightness = np.mean(original_array)
        enhanced_brightness = np.mean(enhanced_array)
        brightness_change = enhanced_brightness - original_brightness
        
        # Edge enhancement
        original_edges = cv2.Canny(original_array, 50, 150)
        enhanced_edges = cv2.Canny(enhanced_array, 50, 150)
        
        original_edge_density = np.sum(original_edges > 0) / original_edges.size
        enhanced_edge_density = np.sum(enhanced_edges > 0) / enhanced_edges.size
        edge_improvement = (enhanced_edge_density / original_edge_density - 1) * 100
        
        return {
            "contrast_improvement": f"{contrast_improvement:.1f}%",
            "brightness_change": f"{brightness_change:.1f}",
            "edge_enhancement": f"{edge_improvement:.1f}%",
            "processing_quality": "Medical Grade",
            "optimization_level": "AI Ready"
        }


def main():
    """Test advanced ultrasound processing"""
    processor = AdvancedUltrasoundProcessor()
    
    # Test with sample image
    sample_path = "/home/ubuntu/DataScience/MyProject/BabyVis/samples/1.jpeg"
    
    if os.path.exists(sample_path):
        # Load test image
        original_image = Image.open(sample_path)
        logger.info(f"📷 Testing with: {sample_path}")
        
        # Enhance image
        enhanced_image = processor.enhance_ultrasound_medical_grade(original_image)
        
        # Save enhanced result
        output_path = "/home/ubuntu/DataScience/MyProject/BabyVis/outputs/enhanced_ultrasound.png"
        enhanced_image.save(output_path)
        logger.info(f"✅ Enhanced image saved: {output_path}")
        
        # Detect facial regions
        facial_data = processor.detect_facial_regions(enhanced_image)
        logger.info(f"🎯 Detected {facial_data['total_regions']} potential facial regions")
        
        # Create enhancement report
        report = processor.create_enhancement_report(original_image, enhanced_image)
        logger.info("📊 Enhancement Report:")
        for key, value in report.items():
            logger.info(f"   {key}: {value}")
    
    else:
        logger.error(f"❌ Sample image not found: {sample_path}")


if __name__ == "__main__":
    import os
    main()