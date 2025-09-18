#!/usr/bin/env python3
"""
BabyVis Web Application
Modern FastAPI web interface for ultrasound to baby face conversion using Diffusers & Qwen-Image-Edit
"""

import os
import uuid
import asyncio
import shutil
import logging
import io
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from PIL import Image
from config import get_settings
from qwen_image_edit_model import QwenImageEditModel
from image_utils import UltrasoundProcessor, BabyFaceProcessor, ImageValidator, ImageComposer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Load settings once for the application lifetime
SETTINGS = get_settings()

# Initialize components
app = FastAPI(
    title="BabyVis",
    description="AI-Powered Ultrasound to Baby Face Generator using Qwen-Image-Edit",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded on demand)
model: Optional[QwenImageEditModel] = None

def get_model(model_name: Optional[str] = None) -> QwenImageEditModel:
    """Get or initialize the model instance"""
    global model
    if model is None or model_name:
        # If model_name is provided, use it; otherwise use default
        effective_model = model_name or SETTINGS.qwen_model_id
        logger.info(f"Initializing model: {effective_model}")
        
        # Create new settings with the selected model
        from config import build_settings
        temp_settings = build_settings(qwen_model_id=effective_model)
        
        model = QwenImageEditModel(settings=temp_settings)
    return model

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🍼 BabyVis starting up...")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    logger.info(f"⚙️ Model provider: {SETTINGS.model_provider}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    global model
    if model is not None:
        model.unload_model()
    logger.info("🍼 BabyVis shutting down...")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve main page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BabyVis - AI Baby Face Generator</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            .gradient-bg {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .card-custom {
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                border: none;
            }
            .drag-drop-area {
                border: 3px dashed #dee2e6;
                border-radius: 15px;
                min-height: 200px;
                text-align: center;
                padding: 40px 20px;
                margin: 20px 0;
                cursor: pointer;
                transition: all 0.3s ease;
                background: #f8f9fa;
            }
            .drag-drop-area:hover {
                background: #e9ecef;
                border-color: #667eea;
                transform: translateY(-2px);
            }
            .drag-drop-area.dragover {
                background: #e3f2fd;
                border-color: #2196f3;
                transform: scale(1.02);
            }
            .preview-image {
                max-width: 100%;
                max-height: 300px;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }
            .btn-generate {
                background: linear-gradient(45deg, #667eea, #764ba2);
                border: none;
                padding: 15px 30px;
                font-size: 1.1rem;
                font-weight: 600;
                border-radius: 50px;
                transition: all 0.3s ease;
            }
            .btn-generate:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            }
            .result-container {
                margin-top: 30px;
                animation: fadeInUp 0.8s ease;
            }
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            .loading-spinner {
                display: none;
                color: #667eea;
            }
            .progress-bar {
                height: 8px;
                border-radius: 4px;
                background: linear-gradient(45deg, #667eea, #764ba2);
            }
            .quality-badge {
                font-size: 0.9rem;
                padding: 8px 16px;
                border-radius: 20px;
                background: linear-gradient(45deg, #28a745, #20c997);
                color: white;
                border: none;
            }
            .feature-icon {
                font-size: 3rem;
                color: #667eea;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="gradient-bg py-5">
            <div class="container text-center">
                <h1 class="display-4 fw-bold mb-3">🍼 BabyVis</h1>
                <p class="lead mb-4">Transform ultrasound images into beautiful baby faces using AI</p>
                <div class="row justify-content-center">
                    <div class="col-md-3">
                        <i class="fas fa-baby feature-icon"></i>
                        <h5>AI-Powered</h5>
                        <p class="small">Advanced Qwen-Image-Edit model</p>
                    </div>
                    <div class="col-md-3">
                        <i class="fas fa-heart-pulse feature-icon"></i>
                        <h5>Medical Grade</h5>
                        <p class="small">Optimized for ultrasound images</p>
                    </div>
                    <div class="col-md-3">
                        <i class="fas fa-magic feature-icon"></i>
                        <h5>Realistic</h5>
                        <p class="small">Photorealistic baby portraits</p>
                    </div>
                    <div class="col-md-3">
                        <i class="fas fa-download feature-icon"></i>
                        <h5>High Quality</h5>
                        <p class="small">Professional results</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="container my-5">
            <div class="row justify-content-center">
                <div class="col-lg-8">
                    <div class="card card-custom">
                        <div class="card-body p-4">
                            <form id="uploadForm" enctype="multipart/form-data">
                                <div class="drag-drop-area" id="dragDropArea">
                                    <div class="d-flex flex-column align-items-center justify-content-center h-100">
                                        <i class="fas fa-cloud-upload-alt fa-4x text-muted mb-3"></i>
                                        <h5 class="text-dark">Drop your ultrasound image here</h5>
                                        <p class="text-muted">or <strong>click to browse</strong></p>
                                        <small class="text-muted">Supports JPG, PNG, WebP • Max 10MB</small>
                                    </div>
                                    <input type="file" id="fileInput" name="file" accept="image/*" style="display: none;">
                                </div>
                                
                                <div id="preview" class="text-center" style="display: none;">
                                    <img id="previewImage" class="preview-image mb-3">
                                    <div id="validationInfo" class="mb-3"></div>
                                </div>
                                
                                <div class="row g-3 mb-4">
                                    <div class="col-md-3">
                                        <label for="modelSelect" class="form-label fw-semibold">AI Model</label>
                                        <select class="form-select" id="modelSelect">
                                        </select>
                                        <small class="text-muted" id="modelInfo">Loading models...</small>
                                    </div>
                                    <div class="col-md-3">
                                        <label for="qualityLevel" class="form-label fw-semibold">Quality Level</label>
                                        <select class="form-select" id="qualityLevel">
                                            <option value="base">Base (Fast)</option>
                                            <option value="enhanced" selected>Enhanced (Recommended)</option>
                                            <option value="premium">Premium (Best Quality)</option>
                                        </select>
                                    </div>
                                    <div class="col-md-3">
                                        <label for="steps" class="form-label fw-semibold">Steps: <span id="stepsValue">30</span></label>
                                        <input type="range" class="form-range" id="steps" min="15" max="50" value="30">
                                        <small class="text-muted">Higher = better quality but slower</small>
                                    </div>
                                    <div class="col-md-3">
                                        <label for="strength" class="form-label fw-semibold">Strength: <span id="strengthValue">0.8</span></label>
                                        <input type="range" class="form-range" id="strength" min="0.3" max="1.0" step="0.1" value="0.8">
                                        <small class="text-muted">Higher = more transformation</small>
                                    </div>
                                </div>
                                
                                <button type="submit" class="btn btn-generate btn-lg w-100 text-white" id="generateBtn" disabled>
                                    <i class="fas fa-magic me-2"></i>Generate Beautiful Baby Face
                                </button>
                            </form>
                            
                            <div class="loading-spinner text-center mt-4" id="loading">
                                <div class="spinner-border spinner-border-lg mb-3" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                                <h5>Creating your beautiful baby face...</h5>
                                <div class="progress mb-3">
                                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                         role="progressbar" style="width: 0%" id="progressBar"></div>
                                </div>
                                <p class="text-muted">This may take 30-90 seconds depending on quality settings</p>
                                <div id="statusText" class="small text-muted"></div>
                            </div>
                            
                            <div class="result-container" id="result" style="display: none;">
                                <div class="text-center mb-4">
                                    <span class="quality-badge">
                                        <i class="fas fa-check-circle me-1"></i>Generation Complete!
                                    </span>
                                </div>
                                <img id="resultImage" class="preview-image mb-4 mx-auto d-block">
                                <div class="row g-2">
                                    <div class="col-md-6">
                                        <a id="downloadBtn" class="btn btn-success btn-lg w-100" download>
                                            <i class="fas fa-download me-2"></i>Download Image
                                        </a>
                                    </div>
                                    <div class="col-md-3">
                                        <button class="btn btn-outline-primary w-100" onclick="generateAnother()">
                                            <i class="fas fa-redo me-1"></i>Another
                                        </button>
                                    </div>
                                    <div class="col-md-3">
                                        <button class="btn btn-outline-secondary w-100" onclick="location.reload()">
                                            <i class="fas fa-refresh me-1"></i>Reset
                                        </button>
                                    </div>
                                </div>
                                <div id="generationInfo" class="mt-3 small text-muted"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="text-center mt-4">
                        <p class="text-muted">
                            <i class="fas fa-shield-alt me-1"></i>
                            Your images are processed securely and automatically deleted after download
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <footer class="bg-light py-4 mt-5">
            <div class="container text-center">
                <p class="text-muted mb-2">
                    <strong>BabyVis v2.0</strong> - Powered by Qwen-Image-Edit & Diffusers
                </p>
                <p class="small text-muted">
                    <a href="https://github.com/VuThanhLam124/BabyVis" class="text-decoration-none">
                        <i class="fab fa-github me-1"></i>View on GitHub
                    </a>
                </p>
            </div>
        </footer>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            const dragDropArea = document.getElementById('dragDropArea');
            const fileInput = document.getElementById('fileInput');
            const preview = document.getElementById('preview');
            const previewImage = document.getElementById('previewImage');
            const generateBtn = document.getElementById('generateBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const resultImage = document.getElementById('resultImage');
            const downloadBtn = document.getElementById('downloadBtn');
            const progressBar = document.getElementById('progressBar');
            const statusText = document.getElementById('statusText');
            const validationInfo = document.getElementById('validationInfo');
            const generationInfo = document.getElementById('generationInfo');
            const modelSelect = document.getElementById('modelSelect');
            const modelInfo = document.getElementById('modelInfo');
            
            let availableModels = [];
            
            // Load available models on page load
            async function loadModels() {
                try {
                    const response = await fetch('/models');
                    const data = await response.json();
                    availableModels = data.models || [];
                    
                    modelSelect.innerHTML = '';
                    availableModels.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.id;
                        option.textContent = `${model.name} (Quality: ${model.quality}/10)`;
                        // Set default model
                        if (model.is_default) {
                            option.selected = true;
                        }
                        modelSelect.appendChild(option);
                    });
                    
                    updateModelInfo();
                    modelInfo.textContent = 'Models loaded successfully';
                } catch (error) {
                    console.error('Error loading models:', error);
                    modelInfo.textContent = 'Error loading models: ' + error.message;
                }
            }
            
            // Update model info when selection changes
            function updateModelInfo() {
                const selectedValue = modelSelect.value;
                if (selectedValue && availableModels.length > 0) {
                    const selectedModel = availableModels.find(m => m.id === selectedValue);
                    if (selectedModel) {
                        modelInfo.textContent = `Quality: ${selectedModel.quality}/10 • Speed: ${selectedModel.speed}/10 • VRAM: ${selectedModel.vram_gb}GB`;
                    }
                } else {
                    modelInfo.textContent = 'Select a model';
                }
            }
            
            modelSelect.addEventListener('change', updateModelInfo);
            
            // Load models when page loads
            loadModels();
            
            // Update slider values
            document.getElementById('steps').addEventListener('input', (e) => {
                document.getElementById('stepsValue').textContent = e.target.value;
            });
            
            document.getElementById('strength').addEventListener('input', (e) => {
                document.getElementById('strengthValue').textContent = e.target.value;
            });
            
            // Drag and drop functionality
            dragDropArea.addEventListener('click', () => fileInput.click());
            
            dragDropArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                dragDropArea.classList.add('dragover');
            });
            
            dragDropArea.addEventListener('dragleave', () => {
                dragDropArea.classList.remove('dragover');
            });
            
            dragDropArea.addEventListener('drop', (e) => {
                e.preventDefault();
                dragDropArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFile(e.target.files[0]);
                }
            });
            
            function handleFile(file) {
                if (!file.type.startsWith('image/')) {
                    alert('Please select an image file');
                    return;
                }
                
                if (file.size > 10 * 1024 * 1024) {
                    alert('File size must be less than 10MB');
                    return;
                }
                
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    preview.style.display = 'block';
                    generateBtn.disabled = false;
                    
                    // Validate image
                    validateImage(file);
                };
                reader.readAsDataURL(file);
            }
            
            async function validateImage(file) {
                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const response = await fetch('/validate', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    let html = '<div class="d-flex justify-content-center gap-3">';
                    
                    if (data.is_valid) {
                        html += '<span class="badge bg-success"><i class="fas fa-check me-1"></i>Valid Ultrasound</span>';
                    } else {
                        html += '<span class="badge bg-warning"><i class="fas fa-exclamation-triangle me-1"></i>Image Quality Warning</span>';
                    }
                    
                    html += `<span class="badge bg-info">Quality: ${(data.quality_score * 100).toFixed(0)}%</span>`;
                    html += `<span class="badge bg-secondary">${data.width}x${data.height}</span>`;
                    html += '</div>';
                    
                    validationInfo.innerHTML = html;
                } catch (error) {
                    console.error('Validation error:', error);
                }
            }
            
            // Form submission
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('model_name', modelSelect.value);
                formData.append('quality_level', document.getElementById('qualityLevel').value);
                formData.append('steps', document.getElementById('steps').value);
                formData.append('strength', document.getElementById('strength').value);
                
                // Show loading
                loading.style.display = 'block';
                generateBtn.disabled = true;
                result.style.display = 'none';
                
                // Simulate progress with status updates
                let progress = 0;
                const statusMessages = [
                    'Loading AI model...',
                    'Analyzing ultrasound image...',
                    'Preprocessing image...',
                    'Generating baby face...',
                    'Enhancing baby features...',
                    'Finalizing image...'
                ];
                
                let statusIndex = 0;
                const progressInterval = setInterval(() => {
                    if (progress < 90) {
                        progress += Math.random() * 10;
                        progressBar.style.width = Math.min(progress, 90) + '%';
                        
                        if (statusIndex < statusMessages.length && progress > statusIndex * 15) {
                            statusText.textContent = statusMessages[statusIndex];
                            statusIndex++;
                        }
                    }
                }, 1500);
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        body: formData
                    });
                    
                    clearInterval(progressInterval);
                    progressBar.style.width = '100%';
                    statusText.textContent = 'Complete!';
                    
                    if (response.ok) {
                        const data = await response.json();
                        
                        resultImage.src = '/download/' + data.filename;
                        downloadBtn.href = '/download/' + data.filename;
                        downloadBtn.download = data.filename;
                        
                        generationInfo.innerHTML = `
                            <div class="text-center">
                                <strong>Generation Details:</strong><br>
                                Model: ${modelSelect.selectedOptions[0].textContent}<br>
                                Quality: ${formData.get('quality_level')}<br>
                                Processing time: ${data.processing_time || 'N/A'}<br>
                                <small class="text-muted">${data.message}</small>
                            </div>
                        `;
                        
                        setTimeout(() => {
                            loading.style.display = 'none';
                            result.style.display = 'block';
                        }, 1000);
                    } else {
                        const errorData = await response.json();
                        throw new Error(errorData.message || 'Generation failed');
                    }
                } catch (error) {
                    clearInterval(progressInterval);
                    loading.style.display = 'none';
                    alert('Error generating image: ' + error.message);
                    generateBtn.disabled = false;
                }
            });
            
            function generateAnother() {
                // Keep the same image but regenerate
                if (fileInput.files.length > 0) {
                    document.getElementById('uploadForm').dispatchEvent(new Event('submit'));
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/validate")
async def validate_image(file: UploadFile = File(...)):
    """Validate uploaded ultrasound image"""
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Validate the image
        validator = ImageValidator()
        is_valid, message = validator.validate_ultrasound(image)
        quality_metrics = validator.check_image_quality(image)
        
        return JSONResponse({
            "is_valid": is_valid,
            "message": message,
            "quality_score": quality_metrics["quality_score"],
            "width": quality_metrics["width"],
            "height": quality_metrics["height"],
            "format": quality_metrics["format"],
            "size_mb": quality_metrics["size_mb"]
        })
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return JSONResponse({
            "is_valid": False,
            "message": f"Validation failed: {str(e)}",
            "quality_score": 0.0,
            "width": 0,
            "height": 0,
            "format": "unknown",
            "size_mb": 0.0
        })

@app.get("/models")
async def get_available_models():
    """Get list of available AI models"""
    try:
        from model_options import AVAILABLE_MODELS
        
        models_list = []
        for model in AVAILABLE_MODELS:
            models_list.append({
                "id": model["id"],
                "name": model["name"],
                "quality": model["quality"],
                "speed": model["speed"],
                "vram_gb": model["vram_gb"],
                "is_default": model["is_default"],
                "description": model["description"]
            })
        
        return JSONResponse({"models": models_list})
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return JSONResponse({"models": []})

@app.post("/generate")
async def generate_baby_face(
    file: UploadFile = File(...),
    quality_level: str = Form("enhanced"),
    steps: int = Form(30),
    strength: float = Form(0.8),
    model_name: str = Form("SG161222/Realistic_Vision_V5.1_noVAE")
):
    """Generate baby face from ultrasound image"""
    start_time = datetime.now()
    
    # Validate and get proper model ID
    from model_options import AVAILABLE_MODELS
    valid_model_ids = [model["id"] for model in AVAILABLE_MODELS]
    
    # If model_name not in valid list, use default
    if model_name not in valid_model_ids:
        model_name = "SG161222/Realistic_Vision_V5.1_noVAE"
        logger.warning(f"Invalid model name provided, using default: {model_name}")
    
    # Save uploaded file
    input_filename = f"ultrasound_{uuid.uuid4()}.{file.filename.split('.')[-1] if '.' in file.filename else 'jpg'}"
    input_path = UPLOAD_DIR / input_filename
    
    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and validate image
        ultrasound_image = Image.open(input_path)
        
        validator = ImageValidator()
        is_valid, validation_message = validator.validate_ultrasound(ultrasound_image)
        
        if not is_valid:
            logger.warning(f"Image validation warning: {validation_message}")
        
        # Get model with selected model_name and generate baby face
        model = get_model(model_name)
        
        logger.info(f"🤖 Using model: {model_name}")
        
        success, baby_image, message = model.generate_baby_face(
            ultrasound_image,
            quality_level=quality_level,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=10.0
        )
        
        if not success or baby_image is None:
            raise HTTPException(status_code=500, detail=f"Generation failed: {message}")
        
        # Apply additional baby enhancement
        baby_processor = BabyFaceProcessor()
        enhanced_baby = baby_processor.enhance_baby_features(baby_image, intensity=0.3)
        enhanced_baby = baby_processor.add_baby_glow(enhanced_baby, glow_intensity=0.2)
        
        # Save result
        output_filename = f"baby_face_{uuid.uuid4()}.png"
        output_path = OUTPUT_DIR / output_filename
        enhanced_baby.save(output_path, "PNG", quality=95, optimize=True)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Successfully generated baby face: {output_filename} in {processing_time:.1f}s")
        
        return JSONResponse({
            "success": True,
            "filename": output_filename,
            "message": message,
            "model_used": model.model_id,
            "processing_time": f"{processing_time:.1f}s",
            "quality_level": quality_level,
            "steps": steps,
            "strength": strength
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    finally:
        # Clean up uploaded file
        if input_path.exists():
            input_path.unlink()

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated image"""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type='application/octet-stream',
        filename=filename,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "application/octet-stream"
        }
    )

@app.get("/status")
async def get_status():
    """Get system status"""
    global model
    
    return {
        "status": "online",
        "model_loaded": model is not None and model.model_loaded,
        "model_id": model.model_id if model else None,
        "uploads_count": len(list(UPLOAD_DIR.glob("*"))),
        "outputs_count": len(list(OUTPUT_DIR.glob("*"))),
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Serve favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

if __name__ == "__main__":
    import io
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
