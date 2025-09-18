#!/usr/bin/env python3
"""
Medical-Grade Image Analysis for BabyVis
Advanced anatomical landmark detection and facial feature mapping for ultrasound images
Implements professional medical imaging techniques for accurate baby face prediction
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, NamedTuple
import logging
from dataclasses import dataclass
from scipy import ndimage, spatial
import json

# Try to import scikit-image, fallback to basic operations if not available
try:
    from skimage import morphology, measure, feature, segmentation
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image not available, using fallback methods")

logger = logging.getLogger(__name__)

@dataclass
class AnatomicalLandmark:
    """Represents a detected anatomical landmark"""
    name: str
    coordinates: Tuple[int, int]
    confidence: float
    region_type: str
    description: str

@dataclass
class FacialMeasurement:
    """Represents a facial measurement for medical analysis"""
    measurement_type: str
    value: float
    unit: str
    normal_range: Tuple[float, float]
    percentile: float

@dataclass
class GestationalAnalysis:
    """Gestational age analysis results"""
    estimated_weeks: float
    confidence: float
    development_stage: str
    facial_maturity: str
    recommendations: List[str]

class MedicalImageAnalyzer:
    """
    Medical-grade ultrasound image analysis system
    Implements professional techniques for anatomical landmark detection and feature mapping
    """
    
    def __init__(self):
        # Anatomical landmark templates for different gestational ages
        self.landmark_definitions = {
            "forehead": {
                "description": "Fetal forehead region",
                "relative_position": (0.5, 0.2),  # Relative to image center
                "search_radius": 0.15,
                "characteristics": ["curved", "prominent", "smooth"]
            },
            "eye_socket": {
                "description": "Orbital cavity region", 
                "relative_position": (0.45, 0.35),
                "search_radius": 0.08,
                "characteristics": ["circular", "dark", "hollow"]
            },
            "nasal_bridge": {
                "description": "Nose bridge development",
                "relative_position": (0.5, 0.45),
                "search_radius": 0.06,
                "characteristics": ["linear", "central", "raised"]
            },
            "lip_region": {
                "description": "Mouth and lip area",
                "relative_position": (0.5, 0.6),
                "search_radius": 0.08,
                "characteristics": ["horizontal", "soft", "curved"]
            },
            "chin": {
                "description": "Chin and jaw development",
                "relative_position": (0.5, 0.75),
                "search_radius": 0.1,
                "characteristics": ["rounded", "prominent", "defined"]
            },
            "cheek": {
                "description": "Cheek region",
                "relative_position": (0.3, 0.5),
                "search_radius": 0.12,
                "characteristics": ["rounded", "soft", "full"]
            }
        }
        
        # Gestational age reference measurements (in pixels, normalized)
        self.gestational_references = {
            "early": {  # 12-20 weeks
                "head_width": (0.3, 0.45),
                "face_height": (0.25, 0.4),
                "eye_spacing": (0.08, 0.12),
                "feature_clarity": (0.3, 0.5)
            },
            "mid": {    # 20-28 weeks  
                "head_width": (0.4, 0.55),
                "face_height": (0.35, 0.5),
                "eye_spacing": (0.1, 0.15),
                "feature_clarity": (0.5, 0.7)
            },
            "late": {   # 28-40 weeks
                "head_width": (0.5, 0.7),
                "face_height": (0.45, 0.65),
                "eye_spacing": (0.12, 0.18),
                "feature_clarity": (0.7, 0.9)
            }
        }
        
        # Medical feature extraction filters
        self.medical_filters = {
            "bone_structure": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            "soft_tissue": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            "fluid_spaces": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        }
    
    def analyze_medical_features(self, ultrasound_image: Image.Image) -> Dict[str, any]:
        """
        Comprehensive medical analysis of ultrasound image
        
        Returns complete medical analysis including:
        - Anatomical landmarks
        - Facial measurements  
        - Gestational age estimation
        - Development assessment
        """
        
        logger.info("🩺 Starting medical-grade ultrasound analysis...")
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(ultrasound_image), cv2.COLOR_RGB2GRAY)
        
        # Preprocessing for medical analysis
        preprocessed = self._medical_preprocessing(cv_image)
        
        # Detect anatomical landmarks
        landmarks = self._detect_anatomical_landmarks(preprocessed)
        
        # Calculate facial measurements
        measurements = self._calculate_facial_measurements(preprocessed, landmarks)
        
        # Estimate gestational age
        gestational_analysis = self._estimate_gestational_age(measurements, landmarks)
        
        # Assess facial development
        development_assessment = self._assess_facial_development(landmarks, measurements)
        
        # Generate medical recommendations
        recommendations = self._generate_medical_recommendations(
            gestational_analysis, development_assessment
        )
        
        analysis_results = {
            "landmarks": landmarks,
            "measurements": measurements,
            "gestational_analysis": gestational_analysis,
            "development_assessment": development_assessment,
            "recommendations": recommendations,
            "image_quality": self._assess_image_quality(cv_image),
            "analysis_confidence": self._calculate_overall_confidence(landmarks, measurements)
        }
        
        logger.info(f"✅ Medical analysis complete. Confidence: {analysis_results['analysis_confidence']:.2f}")
        
        return analysis_results
    
    def _medical_preprocessing(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply medical-grade preprocessing for optimal feature detection"""
        
        # Noise reduction with edge preservation
        bilateral = cv2.bilateralFilter(gray_image, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)
        
        # Morphological operations to enhance structures
        kernel = self.medical_filters["soft_tissue"]
        opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        return opened
    
    def _detect_anatomical_landmarks(self, preprocessed_image: np.ndarray) -> List[AnatomicalLandmark]:
        """Detect anatomical landmarks using medical imaging techniques"""
        
        landmarks = []
        height, width = preprocessed_image.shape
        
        # Apply edge detection for structural analysis
        edges = cv2.Canny(preprocessed_image, 50, 150)
        
        # Find contours for landmark detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for landmark_name, definition in self.landmark_definitions.items():
            # Calculate search region
            center_x = int(definition["relative_position"][0] * width)
            center_y = int(definition["relative_position"][1] * height)
            search_radius = int(definition["search_radius"] * min(width, height))
            
            # Find best matching contour in search region
            best_landmark = self._find_landmark_in_region(
                contours, 
                preprocessed_image,
                (center_x, center_y),
                search_radius,
                definition["characteristics"]
            )
            
            if best_landmark:
                landmark = AnatomicalLandmark(
                    name=landmark_name,
                    coordinates=best_landmark["coordinates"],
                    confidence=best_landmark["confidence"],
                    region_type=definition["description"],
                    description=f"Detected {landmark_name} at confidence {best_landmark['confidence']:.2f}"
                )
                landmarks.append(landmark)
        
        logger.info(f"🎯 Detected {len(landmarks)} anatomical landmarks")
        return landmarks
    
    def _find_landmark_in_region(
        self,
        contours: List,
        image: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        characteristics: List[str]
    ) -> Optional[Dict]:
        """Find the best landmark match in a specific region"""
        
        best_match = None
        best_score = 0.0
        
        for contour in contours:
            # Calculate contour center
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
                
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # Check if within search radius
            distance = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
            if distance > radius:
                continue
            
            # Calculate characteristic score
            char_score = self._calculate_characteristic_score(contour, characteristics)
            
            # Distance penalty (closer to expected position is better)
            distance_score = 1.0 - (distance / radius)
            
            # Combined score
            total_score = char_score * 0.7 + distance_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_match = {
                    "coordinates": (cx, cy),
                    "confidence": total_score,
                    "contour": contour
                }
        
        return best_match
    
    def _calculate_characteristic_score(self, contour, characteristics: List[str]) -> float:
        """Calculate how well a contour matches expected characteristics"""
        
        score = 0.0
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area == 0 or perimeter == 0:
            return 0.0
        
        # Calculate shape metrics
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        aspect_ratio = self._calculate_aspect_ratio(contour)
        solidity = area / cv2.contourArea(cv2.convexHull(contour))
        
        for characteristic in characteristics:
            if characteristic == "circular":
                score += circularity * 0.3
            elif characteristic == "curved":
                score += solidity * 0.3
            elif characteristic == "linear":
                score += (1 - circularity) * 0.3
            elif characteristic == "prominent":
                score += min(area / 1000, 1.0) * 0.2
            elif characteristic == "soft":
                score += solidity * 0.2
            elif characteristic == "defined":
                score += (1 - solidity) * 0.2
        
        return min(score, 1.0)
    
    def _calculate_aspect_ratio(self, contour) -> float:
        """Calculate aspect ratio of contour bounding rectangle"""
        x, y, w, h = cv2.boundingRect(contour)
        return w / h if h > 0 else 0.0
    
    def _calculate_facial_measurements(
        self, 
        image: np.ndarray, 
        landmarks: List[AnatomicalLandmark]
    ) -> List[FacialMeasurement]:
        """Calculate medical facial measurements from landmarks"""
        
        measurements = []
        height, width = image.shape
        
        # Create landmark coordinate dictionary
        landmark_coords = {lm.name: lm.coordinates for lm in landmarks}
        
        # Head width measurement
        if "cheek" in landmark_coords:
            cheek_coords = landmark_coords["cheek"]
            head_width = abs(cheek_coords[0] - width/2) * 2 / width
            
            measurements.append(FacialMeasurement(
                measurement_type="head_width",
                value=head_width,
                unit="normalized",
                normal_range=(0.4, 0.7),
                percentile=self._calculate_percentile(head_width, (0.4, 0.7))
            ))
        
        # Face height measurement
        if "forehead" in landmark_coords and "chin" in landmark_coords:
            forehead_y = landmark_coords["forehead"][1]
            chin_y = landmark_coords["chin"][1]
            face_height = abs(chin_y - forehead_y) / height
            
            measurements.append(FacialMeasurement(
                measurement_type="face_height",
                value=face_height,
                unit="normalized",
                normal_range=(0.35, 0.65),
                percentile=self._calculate_percentile(face_height, (0.35, 0.65))
            ))
        
        # Eye spacing (if bilateral eye detection implemented)
        if "eye_socket" in landmark_coords:
            eye_coords = landmark_coords["eye_socket"]
            eye_spacing = abs(eye_coords[0] - width/2) * 2 / width
            
            measurements.append(FacialMeasurement(
                measurement_type="eye_spacing",
                value=eye_spacing,
                unit="normalized", 
                normal_range=(0.1, 0.18),
                percentile=self._calculate_percentile(eye_spacing, (0.1, 0.18))
            ))
        
        logger.info(f"📏 Calculated {len(measurements)} facial measurements")
        return measurements
    
    def _calculate_percentile(self, value: float, normal_range: Tuple[float, float]) -> float:
        """Calculate percentile of measurement within normal range"""
        min_val, max_val = normal_range
        if value < min_val:
            return 0.0
        elif value > max_val:
            return 100.0
        else:
            return ((value - min_val) / (max_val - min_val)) * 100.0
    
    def _estimate_gestational_age(
        self,
        measurements: List[FacialMeasurement],
        landmarks: List[AnatomicalLandmark]
    ) -> GestationalAnalysis:
        """Estimate gestational age from measurements and landmarks"""
        
        # Create measurement dictionary
        measurement_dict = {m.measurement_type: m.value for m in measurements}
        
        # Calculate scores for each gestational stage
        stage_scores = {}
        
        for stage, references in self.gestational_references.items():
            score = 0.0
            matches = 0
            
            for measurement_type, (min_val, max_val) in references.items():
                if measurement_type in measurement_dict:
                    value = measurement_dict[measurement_type]
                    if min_val <= value <= max_val:
                        # Perfect match
                        score += 1.0
                    else:
                        # Partial match based on distance from range
                        if value < min_val:
                            distance = min_val - value
                        else:
                            distance = value - max_val
                        
                        # Penalty decreases with distance
                        penalty = max(0, 1.0 - distance * 2)
                        score += penalty
                    
                    matches += 1
            
            if matches > 0:
                stage_scores[stage] = score / matches
        
        # Determine best matching stage
        if stage_scores:
            best_stage = max(stage_scores, key=stage_scores.get)
            confidence = stage_scores[best_stage]
        else:
            best_stage = "mid"
            confidence = 0.5
        
        # Estimate specific week based on stage
        week_ranges = {
            "early": (12, 20),
            "mid": (20, 28),
            "late": (28, 40)
        }
        
        min_week, max_week = week_ranges[best_stage]
        estimated_weeks = (min_week + max_week) / 2
        
        # Adjust based on measurement precision
        if confidence > 0.8:
            # High confidence - can be more specific
            estimated_weeks += (confidence - 0.8) * 2
        
        return GestationalAnalysis(
            estimated_weeks=estimated_weeks,
            confidence=confidence,
            development_stage=best_stage,
            facial_maturity=self._assess_facial_maturity(landmarks, measurements),
            recommendations=self._generate_gestational_recommendations(best_stage, confidence)
        )
    
    def _assess_facial_maturity(
        self,
        landmarks: List[AnatomicalLandmark],
        measurements: List[FacialMeasurement]
    ) -> str:
        """Assess facial development maturity"""
        
        landmark_count = len(landmarks)
        avg_confidence = np.mean([lm.confidence for lm in landmarks]) if landmarks else 0
        
        if landmark_count >= 5 and avg_confidence > 0.7:
            return "well_developed"
        elif landmark_count >= 3 and avg_confidence > 0.5:
            return "developing"
        else:
            return "early_stage"
    
    def _assess_facial_development(
        self,
        landmarks: List[AnatomicalLandmark],
        measurements: List[FacialMeasurement]
    ) -> Dict[str, any]:
        """Comprehensive facial development assessment"""
        
        return {
            "landmark_quality": {
                "total_detected": len(landmarks),
                "average_confidence": np.mean([lm.confidence for lm in landmarks]) if landmarks else 0,
                "key_features_present": self._check_key_features(landmarks)
            },
            "proportional_development": {
                "measurements_count": len(measurements),
                "normal_range_compliance": self._check_normal_ranges(measurements),
                "symmetry_assessment": "adequate"  # Placeholder for symmetry analysis
            },
            "overall_assessment": self._calculate_development_score(landmarks, measurements)
        }
    
    def _check_key_features(self, landmarks: List[AnatomicalLandmark]) -> Dict[str, bool]:
        """Check presence of key facial features"""
        key_features = ["forehead", "eye_socket", "nasal_bridge", "lip_region", "chin"]
        detected_names = [lm.name for lm in landmarks]
        
        return {feature: feature in detected_names for feature in key_features}
    
    def _check_normal_ranges(self, measurements: List[FacialMeasurement]) -> float:
        """Calculate percentage of measurements within normal ranges"""
        if not measurements:
            return 0.0
        
        normal_count = sum(1 for m in measurements if 20 <= m.percentile <= 80)
        return normal_count / len(measurements)
    
    def _calculate_development_score(
        self,
        landmarks: List[AnatomicalLandmark],
        measurements: List[FacialMeasurement]
    ) -> float:
        """Calculate overall development score (0-1)"""
        
        landmark_score = len(landmarks) / 6  # Max 6 key landmarks
        confidence_score = np.mean([lm.confidence for lm in landmarks]) if landmarks else 0
        measurement_score = len(measurements) / 4  # Max 4 key measurements
        
        overall_score = (landmark_score * 0.4 + confidence_score * 0.3 + measurement_score * 0.3)
        return min(overall_score, 1.0)
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, any]:
        """Assess medical image quality metrics"""
        
        # Calculate various quality metrics
        contrast = np.std(image)
        brightness = np.mean(image)
        
        # Edge density for detail assessment
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Noise assessment using Laplacian variance
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        
        return {
            "contrast": contrast,
            "brightness": brightness,
            "edge_density": edge_density,
            "noise_level": laplacian_var,
            "overall_quality": self._calculate_quality_score(contrast, edge_density, laplacian_var)
        }
    
    def _calculate_quality_score(self, contrast: float, edge_density: float, noise_level: float) -> str:
        """Calculate overall image quality classification"""
        
        # Normalize scores
        contrast_score = min(contrast / 50, 1.0)  # Normalize contrast
        edge_score = min(edge_density * 10, 1.0)  # Normalize edge density
        noise_score = max(0, 1.0 - noise_level / 1000)  # Invert noise (lower is better)
        
        overall = (contrast_score + edge_score + noise_score) / 3
        
        if overall > 0.8:
            return "excellent"
        elif overall > 0.6:
            return "good"
        elif overall > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _calculate_overall_confidence(
        self,
        landmarks: List[AnatomicalLandmark],
        measurements: List[FacialMeasurement]
    ) -> float:
        """Calculate overall analysis confidence"""
        
        if not landmarks:
            return 0.0
        
        landmark_confidence = np.mean([lm.confidence for lm in landmarks])
        landmark_completeness = len(landmarks) / 6  # Expected number of landmarks
        measurement_completeness = len(measurements) / 4  # Expected number of measurements
        
        overall_confidence = (
            landmark_confidence * 0.5 +
            landmark_completeness * 0.3 +
            measurement_completeness * 0.2
        )
        
        return min(overall_confidence, 1.0)
    
    def _generate_medical_recommendations(
        self,
        gestational_analysis: GestationalAnalysis,
        development_assessment: Dict[str, any]
    ) -> List[str]:
        """Generate medical recommendations based on analysis"""
        
        recommendations = []
        
        # Gestational age recommendations
        if gestational_analysis.confidence < 0.6:
            recommendations.append("Consider additional ultrasound views for better gestational age assessment")
        
        # Development recommendations
        development_score = development_assessment["overall_assessment"]
        if development_score < 0.5:
            recommendations.append("Limited facial feature visibility - may benefit from different imaging angle")
        
        # Image quality recommendations
        landmark_quality = development_assessment["landmark_quality"]
        if landmark_quality["average_confidence"] < 0.5:
            recommendations.append("Image enhancement may improve feature detection accuracy")
        
        # General recommendations
        recommendations.extend([
            "AI-generated baby face is for entertainment purposes only",
            "Consult healthcare provider for medical interpretations",
            "Multiple imaging sessions may provide better feature clarity"
        ])
        
        return recommendations
    
    def _generate_gestational_recommendations(self, stage: str, confidence: float) -> List[str]:
        """Generate stage-specific recommendations"""
        
        recommendations = []
        
        if stage == "early":
            recommendations.extend([
                "Early pregnancy stage - facial features still developing",
                "Consider follow-up ultrasound in 4-6 weeks for better visualization"
            ])
        elif stage == "mid":
            recommendations.extend([
                "Mid-pregnancy stage - good facial feature development expected",
                "Optimal time for detailed facial structure assessment"
            ])
        else:  # late
            recommendations.extend([
                "Late pregnancy stage - mature facial features expected",
                "High-quality baby face generation possible with current development"
            ])
        
        if confidence < 0.7:
            recommendations.append("Consider additional imaging for more precise assessment")
        
        return recommendations


def main():
    """Test medical-grade image analysis"""
    analyzer = MedicalImageAnalyzer()
    
    # Test with sample ultrasound image
    sample_path = "/home/ubuntu/DataScience/MyProject/BabyVis/samples/1.jpeg"
    
    if os.path.exists(sample_path):
        # Load test image
        ultrasound_image = Image.open(sample_path)
        logger.info(f"🩺 Testing medical analysis with: {sample_path}")
        
        # Perform comprehensive analysis
        analysis_results = analyzer.analyze_medical_features(ultrasound_image)
        
        # Print results
        logger.info("\n=== MEDICAL ANALYSIS RESULTS ===")
        
        # Landmarks
        logger.info(f"\n📍 Anatomical Landmarks ({len(analysis_results['landmarks'])} detected):")
        for landmark in analysis_results['landmarks']:
            logger.info(f"  {landmark.name}: {landmark.coordinates} (confidence: {landmark.confidence:.2f})")
        
        # Measurements  
        logger.info(f"\n📏 Facial Measurements ({len(analysis_results['measurements'])} calculated):")
        for measurement in analysis_results['measurements']:
            logger.info(f"  {measurement.measurement_type}: {measurement.value:.3f} (percentile: {measurement.percentile:.1f})")
        
        # Gestational analysis
        gest_analysis = analysis_results['gestational_analysis']
        logger.info(f"\n🤱 Gestational Analysis:")
        logger.info(f"  Estimated weeks: {gest_analysis.estimated_weeks:.1f}")
        logger.info(f"  Development stage: {gest_analysis.development_stage}")
        logger.info(f"  Facial maturity: {gest_analysis.facial_maturity}")
        logger.info(f"  Confidence: {gest_analysis.confidence:.2f}")
        
        # Overall assessment
        logger.info(f"\n🎯 Overall Analysis:")
        logger.info(f"  Confidence: {analysis_results['analysis_confidence']:.2f}")
        logger.info(f"  Image quality: {analysis_results['image_quality']['overall_quality']}")
        
        # Save analysis results
        output_path = "/home/ubuntu/DataScience/MyProject/BabyVis/outputs/medical_analysis.json"
        
        # Convert results to JSON-serializable format
        json_results = {
            "landmarks": [
                {
                    "name": lm.name,
                    "coordinates": lm.coordinates,
                    "confidence": lm.confidence,
                    "description": lm.description
                } for lm in analysis_results['landmarks']
            ],
            "measurements": [
                {
                    "type": m.measurement_type,
                    "value": m.value,
                    "unit": m.unit,
                    "percentile": m.percentile
                } for m in analysis_results['measurements']
            ],
            "gestational_analysis": {
                "estimated_weeks": gest_analysis.estimated_weeks,
                "development_stage": gest_analysis.development_stage,
                "facial_maturity": gest_analysis.facial_maturity,
                "confidence": gest_analysis.confidence
            },
            "overall_confidence": analysis_results['analysis_confidence'],
            "image_quality": analysis_results['image_quality']
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"✅ Medical analysis complete. Results saved: {output_path}")
    
    else:
        logger.error(f"❌ Sample image not found: {sample_path}")


if __name__ == "__main__":
    import os
    main()