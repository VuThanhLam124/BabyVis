"""Application configuration for BabyVis."""
import os
from functools import lru_cache
from typing import Literal, Optional

class BabyVisSettings:
    """Centralised settings loaded from environment variables or defaults."""

    def __init__(self):
        self.model_provider: Literal["diffusers", "gguf"] = os.getenv("BABYVIS_MODEL_PROVIDER", "diffusers")
        self.qwen_model_id: str = os.getenv("QWEN_MODEL_ID", "SG161222/Realistic_Vision_V5.1_noVAE")
        self.device: str = os.getenv("BABYVIS_DEVICE", "auto")
        self.gguf_path: Optional[str] = os.getenv("BABYVIS_GGUF_PATH")
        self.gguf_quant: str = os.getenv("BABYVIS_GGUF_QUANT", "auto")
        self.huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
        self.disable_cpu_offload: bool = os.getenv("BABYVIS_DISABLE_CPU_OFFLOAD", "").lower() in ("1", "true", "yes")
        self.download_to: str = os.getenv("BABYVIS_DOWNLOAD_DIR", "~/.cache/babyvis")


@lru_cache(maxsize=1)
def get_settings() -> BabyVisSettings:
    """Return cached settings instance."""
    return BabyVisSettings()


def build_settings(**overrides: object) -> BabyVisSettings:
    """Create a settings instance with optional overrides."""
    settings = BabyVisSettings()
    for key, value in overrides.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def configure_settings(**overrides: object) -> BabyVisSettings:
    """Persist overrides to environment and rebuild cached settings."""
    if overrides:
        env_mapping = {
            "model_provider": "BABYVIS_MODEL_PROVIDER",
            "qwen_model_id": "QWEN_MODEL_ID", 
            "device": "BABYVIS_DEVICE",
            "gguf_path": "BABYVIS_GGUF_PATH",
            "gguf_quant": "BABYVIS_GGUF_QUANT",
            "huggingface_token": "HUGGINGFACE_TOKEN",
            "disable_cpu_offload": "BABYVIS_DISABLE_CPU_OFFLOAD",
            "download_to": "BABYVIS_DOWNLOAD_DIR"
        }
        
        for field_name, value in overrides.items():
            env_var = env_mapping.get(field_name, field_name.upper())
            if isinstance(value, bool):
                os.environ[env_var] = "1" if value else "0"
            elif value is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = str(value)

    get_settings.cache_clear()
    return get_settings()
