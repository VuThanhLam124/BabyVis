# 🍼 BabyVis - Hướng Dẫn Sử Dụng

## 🚀 Cách Sử Dụng Dự Án BabyVis

### 1. 🌐 Sử Dụng Web Interface (Khuyến nghị)

#### Khởi động web server:
```bash
cd /home/ubuntu/DataScience/MyProject/BabyVis
python main.py --mode web
```

#### Truy cập qua trình duyệt:
```
http://localhost:8000
```

#### Tính năng web interface:
- 📷 **Drag & Drop**: Kéo thả ảnh siêu âm vào vùng upload
- 🎨 **Chọn Chất Lượng**: Base, Enhanced, Premium
- ⚙️ **Điều Chỉnh Settings**: Steps, Guidance Scale, Strength
- 💾 **Download Kết Quả**: Tải về ảnh baby face đã generate

### 2. 🖥️ Sử Dụng Command Line

#### Test model loading:
```bash
python main.py --mode test
```

#### Generate baby face từ ảnh cụ thể:
```bash
python qwen_image_edit_model.py
```

### 3. 📝 Sử Dụng Python API

```python
from qwen_image_edit_model import QwenImageEditModel
from PIL import Image

# Khởi tạo model (Diffusers)
model = QwenImageEditModel()  # hoặc đặt env QWEN_MODEL_ID=Qwen/Qwen-Image-Edit

# Load model (tự tối ưu hóa 4GB VRAM)
success = model.load_model()

if success:
    ultrasound_image = Image.open("path/to/ultrasound.jpg")

    success, baby_image, message = model.generate_baby_face(
        ultrasound_image,
        quality_level="enhanced",
        num_inference_steps=25,
        guidance_scale=7.5,
        strength=0.8,
        seed=42
    )

    if success:
        baby_image.save("output/baby_face.png")
        print(f"✅ Success: {message}")
    else:
        print(f"❌ Error: {message}")

    model.unload_model()
```

### 4. 🔧 Batch Generation (Multiple Variations)

```python
# Generate multiple variations
batch_success, variations, batch_message = model.batch_generate_professional(
    ultrasound_image,
    num_variations=3,
    quality_level="enhanced",
    num_inference_steps=30,
    strength=0.8
)

if batch_success:
    for i, variation in enumerate(variations):
        variation.save(f"output/baby_variation_{i+1}.png")
```

## ⚙️ Configuration Options

### Quality Levels:
- **🥉 Base**: Chất lượng cơ bản, xử lý nhanh
- **🥈 Enhanced**: Chất lượng cao, cân bằng tốc độ/chất lượng  
- **🥇 Premium**: Chất lượng tối đa, xử lý chậm nhưng kết quả tốt nhất

### Parameters:
- **num_inference_steps**: 15-50 (nhiều steps = chất lượng cao hơn)
- **guidance_scale**: 1.0-20.0 (cao = tuân theo prompt chặt chẽ hơn)
- **strength**: 0.1-1.0 (cao = biến đổi nhiều hơn)
- **seed**: Number (để tạo kết quả reproducible)

## 📁 File Structure

```
BabyVis/
├── main.py                    # Entry point chính
├── app.py                     # FastAPI web application  
├── qwen_image_edit_model.py   # Diffusers model handler (Qwen/Qwen-Image-Edit)
├── gguf_model_loader.py       # (Tùy chọn) GGUF placeholder – không cần dùng
├── image_utils.py             # Image processing utilities
├── requirements.txt           # Dependencies
├── samples/                   # Sample ultrasound images
├── outputs/                   # Generated baby faces
└── uploads/                   # Uploaded files (web)
```

## 🔍 Troubleshooting

### Vấn đề thường gặp:

1. **Model loading failed**: 
   - Kiểm tra internet connection (Hugging Face download)
   - Đảm bảo đủ disk space (5–10GB cache)

2. **CUDA out of memory**:
   - Giảm `steps` (vd. 20–25), tăng `strength` vừa phải (0.7–0.85)
   - Đảm bảo độ phân giải 512x512
   - Giữ bật CPU offload/attention slicing (đã cấu hình sẵn)

3. **Dependencies conflicts**:
   ```bash
   pip install --upgrade diffusers transformers huggingface-hub
   ```

### Log Files:
- Model loading logs sẽ hiển thị trong terminal
- Web server logs cho debugging

## 🎯 Best Practices

1. **💡 Chọn ảnh siêu âm chất lượng tốt**: Rõ nét, đủ sáng
2. **⚙️ Bắt đầu với Enhanced quality**: Cân bằng tốc độ/chất lượng
3. **🔄 Thử nhiều seeds khác nhau**: Để có variations đa dạng
4. **📊 Monitor GPU memory**: Với RTX 3050 Ti 3.7GB
5. **💾 Backup kết quả tốt**: Save các baby faces đẹp

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra terminal logs
2. Đảm bảo dependencies đúng version
3. Verify model files đã download hoàn toàn
4. Check GPU memory usage

---

**🎉 Chúc bạn sử dụng BabyVis thành công để tạo ra những baby faces đáng yêu từ ảnh siêu âm!**
