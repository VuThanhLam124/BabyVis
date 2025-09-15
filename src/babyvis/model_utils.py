import os
import torch


def _select_device():
    # Environment overrides
    override = os.getenv("DEVICE")
    if override:
        return override
    if (os.getenv("FORCE_CPU", "").lower() in {"1", "true", "yes"} or
        os.getenv("CUDA_VISIBLE_DEVICES", None) in {"-1", ""}):
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


# --- Qwen Image Edit loader ---
def load_qwen_image_edit(
    model_id: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """
    Load the Qwen Image Edit model via transformers with trust_remote_code.
    Enhanced with multiple fallback strategies and local file support.
    Automatically adapts to available VRAM for optimal performance.

    - Primary model: "Qwen/Qwen-Image-Edit"  
    - Fallback models: Multiple alternative repos and versions
    - Local file support: Load from local directory if available
    - Auto-detects VRAM and configures accordingly:
      * 10GB+: Full precision, flash attention
      * 4GB: 4-bit quantization 
      * <3GB: 8-bit + 4-bit quantization
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
    print(f"   ⚙️ Config: 4bit={use_4bit}, 8bit={use_8bit}, FlashAttn={flash_attention}")
    
    # Clear GPU memory first
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Enhanced model loading with multiple fallback strategies
    model_fallbacks = [
        model_id,  # Primary specified model
        "Qwen/Qwen-Image-Edit",  # Official repo
        "bartowski/Qwen-Image-Edit",  # Alternative repo 1
        "mradermacher/Qwen-Image-Edit",  # Alternative repo 2
        "unsloth/Qwen-Image-Edit",  # Alternative repo 3
    ]
    
    # Add local path fallback if specified
    local_model_path = os.getenv("QWEN_LOCAL_PATH")
    if local_model_path and os.path.exists(local_model_path):
        model_fallbacks.insert(0, local_model_path)
        print(f"🔍 Will try local model first: {local_model_path}")

    # Load tokenizer with optimizations (try all fallbacks)
    tokenizer = None
    for model_candidate in model_fallbacks:
        try:
            print(f"🔍 Trying tokenizer from: {model_candidate}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_candidate, 
                trust_remote_code=True, 
                use_fast=True,  # Use fast tokenizer for speed
                cache_dir=os.getenv("HF_CACHE_DIR")  # Allow custom cache
            )
            print(f"✅ Tokenizer loaded from: {model_candidate}")
            break
        except Exception as e:
            print(f"⚠️ Tokenizer failed for {model_candidate}: {e}")
            continue
    
    if tokenizer is None:
        raise RuntimeError("Failed to load tokenizer from any fallback repository")
    
    # Load model with adaptive optimizations and fallbacks
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "use_cache": True,  # Enable KV cache for faster inference
    }
    
    # Add GPU-specific optimizations based on VRAM
    if device == "cuda":
        load_kwargs.update({
            "load_in_8bit": use_8bit,
            "load_in_4bit": use_4bit,
        })
        
        # Add flash attention only if supported and beneficial
        if flash_attention:
            try:
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                print("⚠️ Flash Attention 2 not available, using default attention")

    # Try loading model from fallbacks with different configurations
    model = None
    successful_model_id = None
    
    for model_candidate in model_fallbacks:
        # Try optimal settings first
        for attempt, (attempt_name, attempt_kwargs) in enumerate([
            ("optimal", load_kwargs.copy()),
            ("conservative", {**load_kwargs, "load_in_8bit": True, "load_in_4bit": False, "attn_implementation": None}),
            ("minimal", {**load_kwargs, "load_in_8bit": False, "load_in_4bit": True, "attn_implementation": None}),
            ("cpu_fallback", {**load_kwargs, "device_map": "cpu", "load_in_8bit": False, "load_in_4bit": False})
        ]):
            try:
                print(f"🔍 Trying {model_candidate} with {attempt_name} settings...")
                model = AutoModelForCausalLM.from_pretrained(model_candidate, **attempt_kwargs).eval()
                successful_model_id = model_candidate
                print(f"✅ Model loaded: {model_candidate} ({attempt_name})")
                break
            except Exception as e:
                print(f"⚠️ {model_candidate} failed with {attempt_name}: {e}")
                continue
        
        if model is not None:
            break
    
    if model is None:
        raise RuntimeError(
            f"Failed to load Qwen Image Edit from any fallback repository.\n"
            f"Tried: {model_fallbacks}\n"
            f"Consider:\n"
            f"1. Check internet connection\n"
            f"2. Update transformers: pip install -U transformers\n"
            f"3. Set QWEN_LOCAL_PATH to local model directory\n"
            f"4. Install missing dependencies: pip install accelerate bitsandbytes"
        )
    
    # Optimize model for inference
    if hasattr(model, 'generation_config'):
        model.generation_config.use_cache = True
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7
        model.generation_config.max_length = adaptive_config['max_length']
    
    # Report memory usage and configuration
    if device == "cuda":
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"✅ Qwen Image Edit loaded successfully!")
        print(f"   💾 GPU Memory: {allocated_gb:.2f}GB allocated")
        print(f"   🎯 Profile: {adaptive_config['profile']} optimized")
        print(f"   📂 Source: {successful_model_id}")
    else:
        print(f"✅ Qwen Image Edit loaded on CPU from: {successful_model_id}")
    
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
    Load a GGUF-quantized Qwen Image Edit using llama-cpp-python with enhanced fallback.
    Optimized for high performance with adaptive GPU acceleration and multiple load strategies.

    - Default HF repo: QuantStack/Qwen-Image-Edit-GGUF
    - Adapts GPU layers based on available VRAM
    - Performance-focused defaults: more GPU layers, larger context
    - Enhanced error handling with multiple repo fallbacks
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

    adaptive_config = _get_adaptive_config()
    
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
    
    # Adaptive GPU layer offloading based on VRAM
    if n_gpu_layers is None:
        n_gpu_layers = adaptive_config['gpu_layers']
    
    if os.getenv("QWEN_N_GPU_LAYERS"):
        try:
            n_gpu_layers = int(os.getenv("QWEN_N_GPU_LAYERS"))
        except Exception:
            pass

    vram_gb = _detect_vram_gb()
    print(f"🚀 Loading Qwen Image Edit (GGUF, Profile: {adaptive_config['profile'].upper()})")
    print(f"   📊 VRAM: {vram_gb:.1f}GB")
    print(f"   ⚙️ Config: {n_threads} threads, {n_gpu_layers} GPU layers, {n_ctx} context")

    # Enhanced model loading configuration
    load_kwargs = {
        "n_ctx": n_ctx,
        "n_threads": n_threads,
        "n_gpu_layers": n_gpu_layers,
        "chat_format": os.getenv("QWEN_CHAT_FORMAT", "qwen2_vl"),
        "logits_all": False,
        "use_mmap": True,  # Memory mapping for efficiency
        "use_mlock": False,  # Don't lock memory for flexibility
        "verbose": False,  # Reduce verbosity for cleaner output
    }

    # Try local path first
    if gguf_path and os.path.exists(gguf_path):
        try:
            llm = Llama(model_path=gguf_path, **load_kwargs)
            print(f"✅ Loaded local GGUF: {gguf_path}")
            return llm
        except Exception as e:
            print(f"⚠️ Failed to load local GGUF {gguf_path}: {e}")
            # Continue to online fallback

    # Enhanced online loading with multiple repo fallbacks
    repo_fallbacks = [
        repo_id,  # Primary repo
        "bartowski/Qwen-Image-Edit-GGUF",  # Alternative repo 1
        "mradermacher/Qwen-Image-Edit-GGUF",  # Alternative repo 2
        "unsloth/Qwen-Image-Edit-GGUF",  # Alternative repo 3
    ]

    # Performance-focused filename selection (prioritize based on VRAM)
    if filename is None:
        if vram_gb >= 9:  # Server environment - prioritize quality
            performance_filenames = [
                "Qwen_Image_Edit-Q6_K.gguf",    # High quality for server
                "Qwen_Image_Edit-Q5_K_M.gguf",  # Good balance
                "Qwen_Image_Edit-Q8_0.gguf",    # Highest quality
                "qwen-image-edit-q6-k.gguf",    # Alternative naming
                "qwen-image-edit-q5-k-m.gguf",  # Alternative naming
            ]
        elif vram_gb >= 3:  # Personal computer - balance quality/speed
            performance_filenames = [
                "Qwen_Image_Edit-Q5_K_M.gguf",  # Best quality/speed ratio
                "Qwen_Image_Edit-Q4_K_M.gguf",  # Good balance
                "Qwen_Image_Edit-Q5_K_S.gguf",  # Alternative Q5
                "qwen-image-edit-q5-k-m.gguf",  # Alternative naming
                "qwen-image-edit-q4-k-m.gguf",  # Alternative naming
            ]
        else:  # Minimal VRAM - prioritize efficiency
            performance_filenames = [
                "Qwen_Image_Edit-Q4_K_M.gguf",  # Best for limited VRAM
                "Qwen_Image_Edit-Q4_K_S.gguf",  # Faster alternative
                "Qwen_Image_Edit-Q3_K_M.gguf",  # Most efficient
                "qwen-image-edit-q4-k-m.gguf",  # Alternative naming
                "qwen-image-edit-q3-k-m.gguf",  # Alternative naming
            ]
        
        # Try all combinations of repos and filenames
        for repo in repo_fallbacks:
            for pref_filename in performance_filenames:
                try:
                    print(f"🔍 Trying {repo}/{pref_filename}")
                    llm = Llama.from_pretrained(
                        repo_id=repo,
                        filename=pref_filename,
                        **load_kwargs
                    )
                    print(f"🚀 Successfully loaded: {repo}/{pref_filename}")
                    return llm
                except Exception as e:
                    print(f"⚠️ Failed {repo}/{pref_filename}: {e}")
                    continue
        
        # If all combinations failed, provide helpful error
        raise RuntimeError(
            f"Failed to load any suitable Qwen GGUF models for {adaptive_config['profile']} profile.\n"
            f"Tried repos: {repo_fallbacks}\n"
            f"Tried files: {performance_filenames}\n"
            f"Check your internet connection or specify a local gguf_path.\n"
            f"Alternative: Download manually from https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF"
        )
    
    # User specified custom filename - try all repos
    for repo in repo_fallbacks:
        try:
            print(f"🔍 Trying custom model: {repo}/{filename}")
            llm = Llama.from_pretrained(
                repo_id=repo,
                filename=filename,
                **load_kwargs
            )
            print(f"🚀 Successfully loaded custom model: {repo}/{filename}")
            return llm
        except Exception as e:
            print(f"⚠️ Failed {repo}/{filename}: {e}")
            continue
    
    raise RuntimeError(
        f"Failed to load custom Qwen GGUF model: {filename}\n"
        f"Tried repos: {repo_fallbacks}\n"
        f"Check if the filename exists in any of these repositories."
    )