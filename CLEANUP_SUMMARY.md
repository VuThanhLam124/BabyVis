# BabyVis - Cấu trúc đã được rút gọn

## 📁 Cấu trúc repo tối giản:

```
BabyVis/
├── batch_processor.py      # Script xử lý chính (đã rút gọn)
├── model_downloader.py     # Auto-download GGUF model  
├── run_4gb.sh             # GPU 4GB configuration
├── run_12gb.sh            # GPU 12GB configuration
├── run_cpu.sh             # CPU configuration
├── requirements.txt       # Dependencies tối thiểu
├── README.md              # Hướng dẫn đơn giản
├── LICENSE                # License file
├── samples/               # Ảnh input mẫu
└── src/babyvis/           # Core code
    ├── __init__.py
    ├── inference.py       # Image processing (đã rút gọn)
    └── model_utils.py     # Model loading (đã rút gọn)
```

## ✅ Đã xóa:

### Test files và Performance scripts:
- adaptive_test.py
- auto_config.py  
- max_performance_test.py
- performance_test.py
- quick_test_old.py
- quick_test.py
- test_qwen_fix.py
- test_qwen.py

### Backup files:
- README_old.md
- RESTRUCTURE_SUMMARY.md
- inference_old.py
- model_utils_backup.py
- model_utils_old.py

### Unused scripts:
- high_performance_setup.sh
- run_qwen_auto.sh
- run_qwen_cpu.sh
- run_qwen_gguf.sh
- run_qwen_transformers.sh

### Complex directories:
- apps/ (Gradio web interface)
- web/ (Web components)
- config/ (Complex configuration)
- data/ (Data files)
- flagged/ (Gradio flagged files)
- logs/ (Log files)

## 🎯 Kết quả:

- **18 files** thay vì 50+ files
- **3 directories** thay vì 10+ directories
- **Chỉ 3 core Python files** trong src/
- **3 run scripts** đơn giản
- **Requirements tối thiểu** (5 packages)
- **README ngắn gọn** và dễ hiểu

## 🚀 Cách sử dụng:

```bash
# GPU 4GB
./run_4gb.sh

# GPU 12GB  
./run_12gb.sh

# CPU only
./run_cpu.sh
```

Repo giờ đây cực kỳ đơn giản và tập trung vào mục đích chính!