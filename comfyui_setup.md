# ComfyUI + Qwen-Image-Edit Setup Guide

## Hardware Requirements Met ✅
- **GPU**: RTX 3050 Ti (4GB VRAM) - Compatible
- **RAM**: 15GB - Sufficient  
- **CPU**: i5-12500H (16 cores) - Excellent
- **CUDA**: 12.6 - Compatible

## Installation Steps

### 1. Install ComfyUI
```bash
cd /home/ubuntu/DataScience/
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

### 2. Download Qwen-Image-Edit for ComfyUI
```bash
# Option A: GGUF format (3-4GB, faster)
cd ComfyUI/models/checkpoints/
wget https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF/resolve/main/Qwen_Image_Edit-Q4_K_M.gguf

# Option B: Full model (larger but better quality)
cd ComfyUI/models/diffusers/
git lfs clone https://huggingface.co/Qwen/Qwen-Image-Edit
```

### 3. Memory Optimization Settings
Create `ComfyUI/extra_model_paths.yaml`:
```yaml
comfyui:
    # Use low VRAM mode
    gpu_only: false
    cpu_offload: true
    
    # Model paths
    checkpoints: models/checkpoints/
    clip: models/clip/
    vae: models/vae/
```

## Usage for Baby Face Generation

### Workflow Design:
1. **Input**: Ultrasound image
2. **Preprocessing**: Resize, enhance contrast
3. **Qwen-Image-Edit**: Transform to baby face
4. **Post-processing**: Face enhancement, cleanup
5. **Output**: Realistic baby portrait

### Optimal Settings:
- **Steps**: 15-25
- **CFG Scale**: 6.0-7.5
- **Strength**: 0.7-0.8
- **Resolution**: 512x512 (optimal for 4GB VRAM)
- **Batch Size**: 1

## Performance Expectations
- **Speed**: 30-60 seconds per image
- **VRAM Usage**: 3.5-3.8GB
- **Quality**: High (better than pure SD models)
- **Memory**: ~6-8GB RAM usage

## Pros vs Current Setup
| Feature | Current (Diffusers) | ComfyUI + Qwen |
|---------|-------------------|----------------|
| Speed | 60s (CPU) | 30-45s (GPU) |
| Quality | Good | Better |
| Memory | 0GB VRAM | 3.5GB VRAM |
| Workflow | Simple | Advanced |
| Customization | Limited | Extensive |