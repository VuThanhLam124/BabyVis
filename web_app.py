#!/usr/bin/env python3
"""
BabyVis Web Application
FastAPI web interface for ultrasound to baby face conversion using ComfyUI
"""

import os
import json
import uuid
import asyncio
import shutil
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image, ImageFilter, ImageEnhance

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import requests
import websocket
import threading
import time

from real_ai_generator import RealAIBabyGenerator
from contextlib import asynccontextmanager

# Backend selection via env vars
BACKEND = os.getenv("BABYVIS_BACKEND", "diffusers").lower()
COMFY_CKPT = os.getenv("COMFYUI_CKPT", "Qwen_Image_Edit-Q4_K_M.gguf")

# Initialize AI processors only when needed
real_ai_generator = None
if BACKEND == "diffusers":
    real_ai_generator = RealAIBabyGenerator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Startup
    if BACKEND == "comfyui":
        print("Starting ComfyUI server (backend=comfyui)...")
        threading.Thread(target=start_comfyui, daemon=True).start()
        # Wait a bit for ComfyUI to start
        await asyncio.sleep(10)
    
    yield
    
    # Shutdown
    global comfyui_process
    if BACKEND == "comfyui":
        if comfyui_process:
            comfyui_process.terminate()
            comfyui_process.wait()

app = FastAPI(
    title="BabyVis", 
    description="Ultrasound to Baby Face Converter",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
COMFYUI_URL = "http://localhost:8188"
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
COMFYUI_DIR = Path("ComfyUI")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ComfyUI process
comfyui_process = None

class ComfyUIClient:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
    
    def queue_prompt(self, prompt):
        """Queue a prompt for processing"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = requests.post(f"http://{self.server_address}/prompt", data=data)
        return req.json()

    def upload_image(self, image_path):
        """Upload an image to ComfyUI input folder and return filename"""
        url = f"http://{self.server_address}/upload/image"
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/png")}
            resp = requests.post(url, files=files, timeout=30)
            resp.raise_for_status()
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
            # API returns {'name': 'filename'} normally
            if isinstance(data, dict) and "name" in data:
                return data["name"]
            # Fallback: return basename
            return os.path.basename(image_path)
    
    def get_image(self, filename, subfolder, folder_type):
        """Get generated image"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = "&".join([f"{k}={v}" for k, v in data.items()])
        url = f"http://{self.server_address}/view?{url_values}"
        return requests.get(url)
    
    def get_history(self, prompt_id):
        """Get processing history"""
        response = requests.get(f"http://{self.server_address}/history/{prompt_id}")
        return response.json()
    
    def wait_for_completion(self, prompt_id, timeout=300):
        """Wait for prompt completion"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                history = self.get_history(prompt_id)
                if prompt_id in history:
                    return history[prompt_id]
                time.sleep(2)
            except Exception as e:
                print(f"Error checking status: {e}")
                time.sleep(2)
        return None

def load_workflow():
    """Load ComfyUI workflow"""
    preferred = COMFYUI_DIR / "workflows" / "qwen_baby_workflow.json"
    workflow_path = preferred if preferred.exists() else (COMFYUI_DIR / "workflows" / "baby_face_workflow.json")
    with open(workflow_path, 'r') as f:
        return json.load(f)

def start_comfyui():
    """Start ComfyUI server"""
    global comfyui_process
    import subprocess
    
    comfyui_path = COMFYUI_DIR / "main.py"
    cmd = ["python3", str(comfyui_path), "--listen", "0.0.0.0", "--port", "8188"]
    
    comfyui_process = subprocess.Popen(
        cmd,
        cwd=str(COMFYUI_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    
    # Wait for server to start
    for _ in range(30):
        try:
            response = requests.get(f"{COMFYUI_URL}/system_stats")
            if response.status_code == 200:
                print("ComfyUI server started successfully")
                return True
        except:
            time.sleep(2)
    
    print("Failed to start ComfyUI server")
    return False



@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve main page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BabyVis - Ultrasound to Baby Face</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .drag-drop-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                width: 100%;
                height: 200px;
                text-align: center;
                padding: 20px;
                margin: 20px 0;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .drag-drop-area:hover {
                background-color: #f8f9fa;
            }
            .drag-drop-area.dragover {
                background-color: #e3f2fd;
                border-color: #2196f3;
            }
            .preview-image {
                max-width: 100%;
                max-height: 300px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .result-container {
                margin-top: 20px;
                text-align: center;
            }
            .loading {
                display: none;
            }
            .progress-bar {
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-md-8">
                    <h1 class="text-center mt-5 mb-4">🍼 BabyVis</h1>
                    <div class="alert alert-warning text-center">
                        <strong>🤖 Real AI Processing!</strong> - Advanced baby face transformation (Creating visible differences)
                    </div>
                    <p class="text-center text-muted">Convert ultrasound images to beautiful baby faces using AI</p>
                    
                    <div class="card">
                        <div class="card-body">
                            <form id="uploadForm" enctype="multipart/form-data">
                                <div class="drag-drop-area" id="dragDropArea">
                                    <div class="d-flex flex-column align-items-center justify-content-center h-100">
                                        <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                                        <p>Drag & drop your ultrasound image here or <strong>click to browse</strong></p>
                                        <small class="text-muted">Supports JPG, PNG, WebP</small>
                                    </div>
                                    <input type="file" id="fileInput" name="file" accept="image/*" style="display: none;">
                                </div>
                                
                                <div id="preview" class="text-center" style="display: none;">
                                    <img id="previewImage" class="preview-image">
                                    <p class="mt-2"><strong>Ready to transform!</strong></p>
                                </div>
                                
                                <div class="row">
                                    <div class="col-md-6">
                                        <label for="steps" class="form-label">Quality Steps (15-30)</label>
                                        <input type="range" class="form-range" id="steps" min="15" max="30" value="25">
                                        <small class="text-muted">Higher = better quality but slower</small>
                                    </div>
                                    <div class="col-md-6">
                                        <label for="strength" class="form-label">Transformation Strength (0.5-1.0)</label>
                                        <input type="range" class="form-range" id="strength" min="0.5" max="1.0" step="0.1" value="0.8">
                                        <small class="text-muted">Higher = more transformation</small>
                                    </div>
                                </div>
                                
                                <button type="submit" class="btn btn-primary btn-lg w-100 mt-3" id="generateBtn" disabled>
                                    ✨ Generate Baby Face
                                </button>
                            </form>
                            
                            <div class="loading text-center" id="loading">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                                <p class="mt-2">Generating your beautiful baby face...</p>
                                <div class="progress">
                                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                         role="progressbar" style="width: 0%" id="progressBar"></div>
                                </div>
                                <small class="text-muted">This may take 30-60 seconds</small>
                            </div>
                            
                            <div class="result-container" id="result" style="display: none;">
                                <h4>Your Beautiful Baby! 👶</h4>
                                <img id="resultImage" class="preview-image">
                                <div class="mt-3">
                                    <a id="downloadBtn" class="btn btn-success" download>
                                        📥 Download Image
                                    </a>
                                    <button class="btn btn-secondary" onclick="location.reload()">
                                        🔄 Try Another
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <footer class="text-center mt-5 mb-3">
                        <small class="text-muted">
                            Powered by ComfyUI + Qwen-Image-Edit |
                            <a href="https://github.com/VuThanhLam124/BabyVis">GitHub</a>
                        </small>
                    </footer>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://kit.fontawesome.com/a076d05399.js"></script>
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
                
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    preview.style.display = 'block';
                    generateBtn.disabled = false;
                };
                reader.readAsDataURL(file);
            }
            
            // Form submission
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('steps', document.getElementById('steps').value);
                formData.append('strength', document.getElementById('strength').value);
                
                // Show loading
                loading.style.display = 'block';
                generateBtn.disabled = true;
                result.style.display = 'none';
                
                // Simulate progress
                let progress = 0;
                const progressInterval = setInterval(() => {
                    progress += Math.random() * 15;
                    if (progress > 90) progress = 90;
                    progressBar.style.width = progress + '%';
                }, 1000);
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        body: formData
                    });
                    
                    clearInterval(progressInterval);
                    progressBar.style.width = '100%';
                    
                    if (response.ok) {
                        const data = await response.json();
                        resultImage.src = '/download/' + data.filename;
                        downloadBtn.href = '/download/' + data.filename;
                        downloadBtn.download = data.filename;
                        
                        loading.style.display = 'none';
                        result.style.display = 'block';
                    } else {
                        throw new Error('Generation failed');
                    }
                } catch (error) {
                    clearInterval(progressInterval);
                    loading.style.display = 'none';
                    alert('Error generating image: ' + error.message);
                    generateBtn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Avoid noisy 404s from browser favicon requests
    return Response(status_code=204)


@app.post("/generate")
async def generate_baby_face(file: UploadFile = File(...), steps: int = 25, strength: float = 0.8):
    """Generate baby face from ultrasound image using Real AI"""
    
    # Save uploaded file
    input_filename = f"ultrasound_{uuid.uuid4()}.{file.filename.split('.')[-1]}"
    input_path = UPLOAD_DIR / input_filename
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Generate output filename
        output_filename = f"baby_real_ai_{uuid.uuid4()}.png"
        output_path = OUTPUT_DIR / output_filename
        
        # Configure processing
        config = {
            "steps": steps,
            "strength": strength,
            "prompt": f"Transform this ultrasound image into a beautiful realistic newborn baby face. Steps: {steps}, Strength: {strength}"
        }

        if BACKEND == "comfyui":
            print(f"🧠 Backend: ComfyUI | Steps: {steps}, Strength: {strength}")
            success, message = await process_with_comfyui(str(input_path), str(output_path), config)
        else:
            # Default: Diffusers/SD pipeline
            global real_ai_generator
            if real_ai_generator is None:
                real_ai_generator = RealAIBabyGenerator()
            print(f"🧠 Backend: Diffusers | Steps: {steps}, Strength: {strength}")
            success, message = real_ai_generator.generate_real_baby_face(
                str(input_path),
                str(output_path),
                config
            )
        
        if success:
            return JSONResponse({
                "success": True,
                "filename": output_filename,
                "message": f"Real AI baby face generated! {message}"
            })
        else:
            raise Exception(message)
        
    except Exception as e:
        print(f"Generation error: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Generation failed: {str(e)}"
        }, status_code=500)
    
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
        media_type='image/png',
        filename=filename
    )

async def process_with_comfyui(input_path: str, output_path: str, config: dict):
    """Run generation via ComfyUI workflow and save to output_path"""
    try:
        client = ComfyUIClient()
        # Upload to ComfyUI 'input' folder
        uploaded_name = client.upload_image(input_path)

        # Load and patch workflow
        wf = load_workflow()

        # Try to locate nodes by class_type
        # Update checkpoint/diffusers path
        diffusers_model = os.getenv("COMFYUI_DIFFUSERS", "Qwen-Image-Edit")
        for node_id, node in wf.items():
            ct = node.get("class_type")
            if ct == "CheckpointLoaderSimple":
                if "inputs" in node and isinstance(node["inputs"], dict):
                    node["inputs"]["ckpt_name"] = COMFY_CKPT
            elif ct == "DiffusersLoader":
                if "inputs" in node and isinstance(node["inputs"], dict):
                    node["inputs"]["model_path"] = diffusers_model
            elif ct == "UnetLoaderGGUF":
                # If present in workflow, keep name in sync
                if "inputs" in node and isinstance(node["inputs"], dict):
                    node["inputs"]["unet_name"] = COMFY_CKPT
        # Update prompts
        for node_id, node in wf.items():
            if node.get("class_type") == "CLIPTextEncode" and "inputs" in node:
                if node["_meta"]["title"].lower().find("prompt") != -1:
                    node["inputs"]["text"] = config.get("prompt", node["inputs"].get("text", ""))
        # Update steps/denoise on KSampler/KSamplerAdvanced
        for node_id, node in wf.items():
            if node.get("class_type") in ("KSampler", "KSamplerAdvanced") and "inputs" in node:
                node["inputs"]["steps"] = int(config.get("steps", node["inputs"].get("steps", 25)))
                # Use strength as denoise
                if "denoise" in node["inputs"]:
                    node["inputs"]["denoise"] = float(config.get("strength", node["inputs"].get("denoise", 0.8)))
        # Update LoadImage node to point to uploaded file
        for node_id, node in wf.items():
            if node.get("class_type") == "LoadImage" and "inputs" in node:
                node["inputs"]["image"] = uploaded_name

        # Queue prompt
        result = client.queue_prompt(wf)
        prompt_id = result.get("prompt_id") or result.get("prompt") or result
        history = client.wait_for_completion(prompt_id)
        if not history:
            return False, "ComfyUI processing timed out"

        # Extract image info (use SaveImage or last decoder output)
        image_info = None
        for out in history.get("outputs", {}).values():
            if isinstance(out, dict) and "images" in out:
                images = out["images"]
                if images:
                    image_info = images[0]
                    break
        if not image_info:
            return False, "No image produced by ComfyUI"

        # Download the image
        resp = client.get_image(
            filename=image_info.get("filename"),
            subfolder=image_info.get("subfolder", ""),
            folder_type=image_info.get("type", "output")
        )
        if resp.status_code != 200:
            return False, f"Failed to retrieve image: {resp.status_code}"
        with open(output_path, "wb") as f:
            f.write(resp.content)

        return True, "ComfyUI baby face generated!"
    except Exception as e:
        print(f"ComfyUI error: {e}")
        return False, str(e)


@app.get("/status")
async def status():
    """Check system status"""
    try:
        comfyui_status = False
        if BACKEND == "comfyui":
            # Check ComfyUI only if used
            comfyui_response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            comfyui_status = comfyui_response.status_code == 200
    except:
        comfyui_status = False
    
    return {
        "comfyui": comfyui_status,
        "uploads": len(list(UPLOAD_DIR.glob("*"))),
        "outputs": len(list(OUTPUT_DIR.glob("*")))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
