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

    - Default model: "Qwen/Qwen-Image-Edit"
    - Respects device selection logic in this module
    - Chooses FP16 on CUDA, FP32 otherwise
    """
    model_id = model_id or os.getenv("QWEN_IMAGE_EDIT_ID", "Qwen/Qwen-Image-Edit")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError(
            "transformers is required for Qwen Image Edit. Please `pip install transformers accelerate`"
        ) from e

    device = _select_device()
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Use device_map="auto" when available; otherwise map to a single device
    device_map = "auto"
    if os.getenv("QWEN_DEVICE_MAP", "auto").lower() not in {"auto", ""}:
        device_map = os.getenv("QWEN_DEVICE_MAP")

    print(f"🔌 Loading Qwen Image Edit: {model_id} on {device} ({torch_dtype})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    ).eval()

    return tokenizer, model, device


# --- Qwen Image Edit GGUF (llama.cpp) loader ---
def load_qwen_image_edit_gguf(
    gguf_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    n_ctx: int = 2048,
    n_threads: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
):
    """
    Load a GGUF-quantized Qwen Image Edit using llama-cpp-python.

    - Default HF repo: QuantStack/Qwen-Image-Edit-GGUF
    - Provide either local `gguf_path` or (`repo_id` + `filename`) for from_pretrained
    - Environment overrides:
        QWEN_GGUF_PATH, QWEN_GGUF_REPO, QWEN_GGUF_FILENAME,
        QWEN_N_CTX, QWEN_N_THREADS, QWEN_N_GPU_LAYERS
    - On 4GB VRAM, a small `n_gpu_layers` (e.g. 8–16) is recommended; 0 uses CPU only.
    """
    try:
        from llama_cpp import Llama
    except Exception as e:
        raise RuntimeError(
            "llama-cpp-python is required for GGUF backend. Please `pip install llama-cpp-python`"
        ) from e

    gguf_path = gguf_path or os.getenv("QWEN_GGUF_PATH")
    repo_id = repo_id or os.getenv("QWEN_GGUF_REPO", "QuantStack/Qwen-Image-Edit-GGUF")
    filename = filename or os.getenv("QWEN_GGUF_FILENAME", None)

    # Threads/GPU layers
    if os.getenv("QWEN_N_CTX"):
        try:
            n_ctx = int(os.getenv("QWEN_N_CTX"))
        except Exception:
            pass
    if n_threads is None:
        n_threads = max(1, (os.cpu_count() or 4) - 0)
    if os.getenv("QWEN_N_THREADS"):
        try:
            n_threads = int(os.getenv("QWEN_N_THREADS"))
        except Exception:
            pass
    if os.getenv("QWEN_N_GPU_LAYERS"):
        try:
            n_gpu_layers = int(os.getenv("QWEN_N_GPU_LAYERS"))
        except Exception:
            pass
    # Heuristic for 4GB VRAM if user hasn't set it
    if n_gpu_layers is None:
        # Small offload by default for constrained GPUs; set 0 for CPU-only
        n_gpu_layers = 12

    print("🔌 Loading Qwen Image Edit (GGUF, llama.cpp)...")

    # Try local path first
    if gguf_path and os.path.exists(gguf_path):
        llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            # chat format for Qwen2-VL style; fallback to 'llava-1-5' if unsupported
            chat_format=os.getenv("QWEN_CHAT_FORMAT", "qwen2_vl"),
            logits_all=False,
        )
        return llm

    # Else attempt from_pretrained (requires network at runtime)
    if filename is None:
        # Provide a likely default filename; user can override via env
        filename = "qwen-image-edit-q4_k_m.gguf"
    llm = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        chat_format=os.getenv("QWEN_CHAT_FORMAT", "qwen2_vl"),
        logits_all=False,
    )
    return llm
