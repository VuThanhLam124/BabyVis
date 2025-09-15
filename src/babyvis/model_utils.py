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
