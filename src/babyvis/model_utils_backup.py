import os
import torch


def _select_device():
    # Environment overrides
    override = os.getenv("DEVICE")
    if override:
        return override
    if (os.getenv("FORCE_CPU", "").lower() in {"1", "true", "ye    device = _select_device()
    adaptive_config = _get_adaptive_config()
    
    # Override with environment variables if set
    use_4bit = os.getenv("QWEN_4BIT", str(adaptive_config['use_4bit'])).lower() == "true"
    use_8bit = os.getenv("QWEN_8BIT", str(adaptive_config['use_8bit'])).lower() == "true"
    flash_attention = os.getenv("QWEN_FLASH_ATTN", str(adaptive_config['flash_attention'])).lower() == "true"
    
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Optimized device mapping for performance
    device_map = "auto" if device == "cuda" else "cpu"
    if os.getenv("QWEN_DEVICE_MAP"):
        device_map = os.getenv("QWEN_DEVICE_MAP")

    print(f"🚀 Loading Qwen Image Edit (Profile: {adaptive_config['profile'].upper()})")
    print(f"   📊 VRAM: {_detect_vram_gb():.1f}GB, Device: {device} ({torch_dtype})")
    print(f"   ⚙️ Config: 4bit={use_4bit}, 8bit={use_8bit}, FlashAttn={flash_attention}")     os.getenv("CUDA_VISIBLE_DEVICES", None) in {"-1", ""}):
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_vram_gb():
    """Detect available VRAM in GB for adaptive configuration"""
    try:
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            return gpu_props.total_memory / (1024**3)
        return 0.0
    except Exception:
        return 0.0


def _get_adaptive_config():
    """Get adaptive configuration based on available VRAM"""
    vram_gb = _detect_vram_gb()
    
    if vram_gb >= 9:  # Server environment (10GB+)
        return {
            'use_4bit': False,
            'use_8bit': False,
            'flash_attention': True,
            'gpu_layers': 35,
            'batch_size': 4,
            'max_length': 2048,
            'profile': 'server'
        }
    elif vram_gb >= 3:  # Personal computer (4GB)
        return {
            'use_4bit': True,
            'use_8bit': False,
            'flash_attention': False,
            'gpu_layers': 20,
            'batch_size': 1,
            'max_length': 1024,
            'profile': 'personal'
        }
    elif vram_gb >= 1:  # Minimal VRAM
        return {
            'use_4bit': True,
            'use_8bit': True,
            'flash_attention': False,
            'gpu_layers': 8,
            'batch_size': 1,
            'max_length': 512,
            'profile': 'minimal'
        }
    else:  # CPU only
        return {
            'use_4bit': False,
            'use_8bit': False,
            'flash_attention': False,
            'gpu_layers': 0,
            'batch_size': 1,
            'max_length': 512,
            'profile': 'cpu'
        }


from typing import Optional


def load_canny_pipeline(
    base_model_id: Optional[str] = None,
    controlnet_id: Optional[str] = None,
    prefer_sdxl: Optional[bool] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """
    Load a ControlNet+Canny pipeline with sensible defaults and fallbacks.

    - Defaults to SD v1.5 + Canny ControlNet on CPU.
    - If prefer_sdxl=True (or base_model_id contains 'xl'), tries SDXL ControlNet.
    - You can override via env:
        BASE_MODEL_ID, CONTROLNET_ID, PREFER_SDXL, TORCH_DTYPE=float16|float32
    """
    # Resolve config from args/env/defaults
    env_base = os.getenv("BASE_MODEL_ID")
    env_ctrl = os.getenv("CONTROLNET_ID")
    env_sdxl = os.getenv("PREFER_SDXL")
    env_dtype = os.getenv("TORCH_DTYPE")

    base_model_id = base_model_id or env_base or "runwayml/stable-diffusion-v1-5"
    controlnet_id = controlnet_id or env_ctrl or "lllyasviel/sd-controlnet-canny"
    if prefer_sdxl is None:
        prefer_sdxl = (env_sdxl or "").lower() in {"1", "true", "yes"} or ("xl" in base_model_id.lower())

    device = _select_device()

    # dtype selection
    if torch_dtype is None:
        if env_dtype:
            torch_dtype = torch.float16 if env_dtype.lower() in {"fp16", "float16", "half"} else torch.float32
        else:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Try SDXL pipeline first if requested
    if prefer_sdxl:
        try:
            from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

            # Allow default SDXL IDs if user didn't override
            if base_model_id == "runwayml/stable-diffusion-v1-5":
                base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            if controlnet_id == "lllyasviel/sd-controlnet-canny":
                # Note: You can override this via env if unavailable
                controlnet_id = os.getenv("CONTROLNET_ID", "diffusers/controlnet-canny-sdxl-1.0")

            controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch_dtype)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_model_id,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
            pipe = pipe.to(device)

            # Tweaks
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass
            if hasattr(pipe, "safety_checker"):
                pipe.safety_checker = None

            # Use DPMSolver scheduler for quality/speed
            try:
                from diffusers import DPMSolverMultistepScheduler
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            except Exception:
                pass

            print(f"✅ Loaded SDXL pipeline: {base_model_id} + {controlnet_id} on {device} ({torch_dtype})")
            return pipe
        except Exception as e:
            print(f"⚠️ SDXL pipeline unavailable ({e}). Falling back to SD v1.5.")

    # Fallback to SD v1.5 ControlNet
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch_dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    # Tweaks
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    try:
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pass

    print(f"✅ Loaded SD v1.5 pipeline: {base_model_id} + {controlnet_id} on {device} ({torch_dtype})")
    return pipe


# --- Qwen Image Edit loader ---
def load_qwen_image_edit(
    model_id: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """
    Load the Qwen Image Edit model via transformers with trust_remote_code.
    Optimized for high performance with GPU acceleration and memory management.

    - Default model: "Qwen/Qwen-Image-Edit"
    - Respects device selection logic in this module
    - Chooses FP16 on CUDA, FP32 otherwise
    - Uses optimized loading for better performance
    """
    model_id = model_id or os.getenv("QWEN_IMAGE_EDIT_ID", "Qwen/Qwen-Image-Edit")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import gc
    except Exception as e:
        raise RuntimeError(
            "transformers is required for Qwen Image Edit. Please `pip install transformers accelerate`"
        ) from e

    device = _select_device()
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Optimized device mapping for performance
    device_map = "auto" if device == "cuda" else "cpu"
    if os.getenv("QWEN_DEVICE_MAP"):
        device_map = os.getenv("QWEN_DEVICE_MAP")

    print(f"� Loading Qwen Image Edit (High Performance): {model_id} on {device} ({torch_dtype})")
    
    # Clear GPU memory first
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load tokenizer with optimizations
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        use_fast=True,  # Use fast tokenizer for speed
        cache_dir=os.getenv("HF_CACHE_DIR")  # Allow custom cache
    )
    
    # Load model with performance optimizations
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "use_cache": True,  # Enable KV cache for faster inference
    }
    
    # Add GPU-specific optimizations
    if device == "cuda":
        load_kwargs.update({
            "load_in_8bit": os.getenv("QWEN_8BIT", "false").lower() == "true",
            "load_in_4bit": os.getenv("QWEN_4BIT", "false").lower() == "true",
            "attn_implementation": "flash_attention_2" if os.getenv("QWEN_FLASH_ATTN", "false").lower() == "true" else None,
        })
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
    
    # Optimize model for inference
    if hasattr(model, 'generation_config'):
        model.generation_config.use_cache = True
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7
    
    print(f"✅ Qwen Image Edit loaded successfully with optimizations!")
    print(f"   💾 Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f}GB" if device == "cuda" else "   💾 CPU mode")
    
    return tokenizer, model, device


# --- Qwen Image Edit GGUF (llama.cpp) loader ---
def load_qwen_image_edit_gguf(
    gguf_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    n_ctx: int = 4096,  # Increased context for better quality
    n_threads: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
):
    """
    Load a GGUF-quantized Qwen Image Edit using llama-cpp-python.
    Optimized for high performance with aggressive GPU acceleration.

    - Default HF repo: QuantStack/Qwen-Image-Edit-GGUF
    - Performance-focused defaults: more GPU layers, larger context
    - Environment overrides for fine-tuning
    """
    try:
        from llama_cpp import Llama
        import psutil
    except Exception as e:
        raise RuntimeError(
            "llama-cpp-python is required for GGUF backend. Please `pip install llama-cpp-python`"
        ) from e

    gguf_path = gguf_path or os.getenv("QWEN_GGUF_PATH")
    repo_id = repo_id or os.getenv("QWEN_GGUF_REPO", "QuantStack/Qwen-Image-Edit-GGUF")
    filename = filename or os.getenv("QWEN_GGUF_FILENAME", None)

    # Performance-focused configuration
    if os.getenv("QWEN_N_CTX"):
        try:
            n_ctx = int(os.getenv("QWEN_N_CTX"))
        except Exception:
            pass
    
    # Optimize thread count for performance
    if n_threads is None:
        cpu_count = os.cpu_count() or 4
        n_threads = max(1, cpu_count - 1)  # Leave one core free
    if os.getenv("QWEN_N_THREADS"):
        try:
            n_threads = int(os.getenv("QWEN_N_THREADS"))
        except Exception:
            pass
    
    # Aggressive GPU layer offloading for performance
    if n_gpu_layers is None:
        device = _select_device()
        if device == "cuda":
            try:
                # Try to use most GPU layers for performance
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_mem_gb >= 8:
                    n_gpu_layers = 35  # Most layers on 8GB+ GPU
                elif gpu_mem_gb >= 6:
                    n_gpu_layers = 28  # Balanced for 6GB GPU
                else:
                    n_gpu_layers = 20  # Conservative for 4GB GPU
                print(f"🚀 Auto-detected {gpu_mem_gb:.1f}GB GPU, using {n_gpu_layers} GPU layers")
            except Exception:
                n_gpu_layers = 25  # Safe default for performance
        else:
            n_gpu_layers = 0  # CPU only
    
    if os.getenv("QWEN_N_GPU_LAYERS"):
        try:
            n_gpu_layers = int(os.getenv("QWEN_N_GPU_LAYERS"))
        except Exception:
            pass

    print(f"� Loading Qwen Image Edit (GGUF, High Performance)...")
    print(f"   ⚙️ Config: {n_threads} threads, {n_gpu_layers} GPU layers, {n_ctx} context")

    # Try local path first
    if gguf_path and os.path.exists(gguf_path):
        llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            chat_format=os.getenv("QWEN_CHAT_FORMAT", "qwen2_vl"),
            logits_all=False,
            use_mmap=True,  # Memory mapping for efficiency
            use_mlock=False,  # Don't lock memory for flexibility
            verbose=False,  # Reduce verbosity for cleaner output
        )
        print(f"✅ Loaded local GGUF: {gguf_path}")
        return llm

    # Performance-focused filename selection (prioritize speed/quality balance)
    if filename is None:
        # Prioritize Q5/Q4 models for best performance/quality trade-off
        performance_filenames = [
            "Qwen_Image_Edit-Q5_K_M.gguf",  # Best quality for performance
            "Qwen_Image_Edit-Q4_K_M.gguf",  # Good balance
            "Qwen_Image_Edit-Q5_K_S.gguf",  # Alternative Q5
            "Qwen_Image_Edit-Q6_K.gguf",    # Higher quality
            "Qwen_Image_Edit-Q4_K_S.gguf",  # Faster alternative
            "Qwen_Image_Edit-Q8_0.gguf",    # Highest quality (slow)
        ]
        
        for pref_filename in performance_filenames:
            try:
                print(f"🔍 Trying high-performance model: {pref_filename}")
                llm = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=pref_filename,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    chat_format=os.getenv("QWEN_CHAT_FORMAT", "qwen2_vl"),
                    logits_all=False,
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                )
                print(f"🚀 Successfully loaded high-performance model: {pref_filename}")
                return llm
            except Exception as e:
                print(f"⚠️ Failed to load {pref_filename}: {e}")
                continue
        
        raise RuntimeError(
            f"Failed to load any high-performance Qwen GGUF models from {repo_id}. "
            f"Tried: {performance_filenames}. Check your internet connection or specify a local gguf_path."
        )
    
    # User specified custom filename
    try:
        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            chat_format=os.getenv("QWEN_CHAT_FORMAT", "qwen2_vl"),
            logits_all=False,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        print(f"🚀 Successfully loaded custom model: {filename}")
        return llm
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Qwen GGUF backend: {filename} not found in {repo_id}. Error: {e}"
        ) from e
