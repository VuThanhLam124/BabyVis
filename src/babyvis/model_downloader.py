#!/usr/bin/env python3
"""
Auto-downloader cho Qwen Image Edit GGUF model
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
import hashlib
import time

# Model configuration
MODEL_URL = "https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF/resolve/main/Qwen_Image_Edit-Q4_K_M.gguf?download=true"
MODEL_FILENAME = "Qwen_Image_Edit-Q4_K_M.gguf"
MODELS_DIR = "ComfyUI/models/checkpoints"
EXPECTED_SIZE_MB = 15000  # Approximately 15GB


def ensure_models_dir():
    """Tạo thư mục models nếu chưa có"""
    models_path = Path(MODELS_DIR)
    models_path.mkdir(exist_ok=True)
    return models_path


def get_model_path():
    """Lấy đường dẫn đầy đủ của model"""
    return Path(MODELS_DIR) / MODEL_FILENAME


def check_model_exists():
    """Kiểm tra model đã tồn tại và hợp lệ"""
    model_path = get_model_path()
    if not model_path.exists():
        return False
    
    # Kiểm tra size file (tối thiểu 6GB để đảm bảo download hoàn chính)
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    if file_size_mb < 6000:  # Tối thiểu 6GB
        print(f"⚠️ Model file too small ({file_size_mb:.1f}MB), re-downloading...")
        model_path.unlink()  # Xóa file corrupt
        return False
    
    print(f"✅ Model found: {model_path} ({file_size_mb:.1f}MB)")
    return True


def download_model():
    """Download model từ HuggingFace"""
    models_path = ensure_models_dir()
    model_path = get_model_path()
    
    print(f"📥 Downloading Qwen Image Edit GGUF model...")
    print(f"   Source: {MODEL_URL}")
    print(f"   Target: {model_path}")
    print(f"   Expected size: ~{EXPECTED_SIZE_MB}MB")
    print(f"   This may take 10-30 minutes depending on your connection...")
    
    try:
        # Create a temporary file for download
        temp_path = model_path.with_suffix('.tmp')
        
        def progress_hook(block_num, block_size, total_size):
            """Progress callback"""
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f}MB)", end='', flush=True)
            else:
                mb_downloaded = downloaded / (1024 * 1024)
                print(f"\r   Downloaded: {mb_downloaded:.1f}MB", end='', flush=True)
        
        # Download with progress
        urllib.request.urlretrieve(MODEL_URL, temp_path, progress_hook)
        print()  # New line after progress
        
        # Verify download
        if temp_path.exists():
            file_size_mb = temp_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 6000:
                print(f"❌ Download failed or incomplete ({file_size_mb:.1f}MB)")
                temp_path.unlink()
                return False
            
            # Move to final location
            temp_path.rename(model_path)
            print(f"✅ Download completed: {model_path} ({file_size_mb:.1f}MB)")
            return True
        else:
            print("❌ Download failed - temporary file not found")
            return False
            
    except urllib.error.URLError as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False


def ensure_model():
    """Đảm bảo model có sẵn (download nếu cần)"""
    if check_model_exists():
        return True
    
    print("🤖 Qwen Image Edit GGUF model not found")
    print("   Will download from HuggingFace...")
    
    return download_model()


def main():
    """Main function for standalone usage"""
    print("🤖 BabyVis - Qwen Model Downloader")
    print("=" * 50)
    
    if ensure_model():
        model_path = get_model_path()
        print(f"\n🎉 Model ready: {model_path}")
        print("You can now run BabyVis!")
    else:
        print("\n❌ Failed to ensure model availability")
        print("Please check your internet connection and try again")
        exit(1)


if __name__ == "__main__":
    main()
