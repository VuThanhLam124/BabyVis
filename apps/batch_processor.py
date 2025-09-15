# apps/batch_processor.py
import os, sys
import json
from datetime import datetime

# Ensure src/ is on path for local runs
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from babyvis.inference import batch_process_with_display

class BatchImageProcessor:
    def __init__(self, config_file="config/batch_config.json"):
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Tải cấu hình từ file JSON"""
        default_config = {
            "delay_seconds": 0.25,
            "output_directory": "outputs/batch",
            "generate_predictions": True,
            "save_channels": False,
            "log_file": "logs/batch.log"
        }
        
        # Ensure parent folder exists
        os.makedirs(os.path.dirname(self.config_file) or ".", exist_ok=True)
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Lưu cấu hình ra file JSON"""
        os.makedirs(os.path.dirname(self.config_file) or ".", exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def log_message(self, message):
        """Ghi log vào file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        log_path = self.config["log_file"]
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(log_entry)
        
        print(message)
    
    def process_image_list(self, image_paths):
        """Xử lý danh sách ảnh với log đầy đủ"""
        
        # Kiểm tra file tồn tại
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                self.log_message(f"File không tồn tại: {path}")
        
        if not valid_paths:
            self.log_message("Không có file hợp lệ để xử lý")
            return []
        
        # Xử lý batch
        results = batch_process_with_display(
            valid_paths,
            output_dir=self.config["output_directory"],
            delay_sec=self.config["delay_seconds"],
            generate_predictions=self.config["generate_predictions"]
        )
        
        self.log_message(f"Hoàn thành xử lý. Tạo được {len(results)} ảnh dự đoán")
        return results
    
    def process_from_text_file(self, file_path):
        """Xử lý từ file text chứa danh sách đường dẫn"""
        if not os.path.exists(file_path):
            self.log_message(f"File danh sách không tồn tại: {file_path}")
            return []
        
        with open(file_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        return self.process_image_list(image_paths)

# Sử dụng
if __name__ == "__main__":
    processor = BatchImageProcessor()

    # Ví dụ 1: Xử lý từ list (đọc ảnh mẫu trong thư mục samples/)
    image_list = [
        "samples/1.jpeg",
        "samples/1B.png",
        "samples/1C.webp",
        "samples/3.png",
        "samples/4.pjeg",  # intentionally invalid for logging demo
        "samples/5.jpeg",
        "samples/6.jpeg",
    ]

    results = processor.process_image_list(image_list)

    # Ví dụ 2: Xử lý từ file text
    # results = processor.process_from_text_file("data/image_list.txt")
