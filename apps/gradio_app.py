# apps/gradio_app.py
import os, sys, tempfile

# Ensure src/ is on path for local runs
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr
from PIL import Image
from babyvis.inference import (
    generate_predict_enhanced,
    OPTIMAL_INFERENCE_STEPS,
    OPTIMAL_CFG_SCALE,
    enhanced_canny_detection,
)

# ---- Thư mục lưu kết quả ----
OUTPUT_DIR = os.path.join("outputs", "gradio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_ultrasound(image):
    """
    Hàm chính cho Gradio:
    1. Lưu ảnh tạm
    2. Hiển thị 7 kênh màu (trả về dạng list PIL để Gradio gallery)
    3. Gọi mô hình sinh ảnh em bé
    4. Trả về list ảnh + link tải
    """
    if image is None:
        return [], None

    # Bước 1: Lưu ảnh gốc để inference
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(tmp_fd)
    image.save(tmp_path)

    # Bước 2: Tạo gallery 7 kênh
    gallery_imgs = []
    # show_7_channels_auto chỉ hiển thị bằng matplotlib; ta viết nhanh hàm lấy ndarray
    try:
        import cv2, numpy as np
        img_cv = cv2.imread(tmp_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        gallery_imgs.append(Image.fromarray(img_rgb))

        canny = enhanced_canny_detection(img_cv, method="adaptive")
        gallery_imgs.append(Image.fromarray(canny))

        # R-G-B
        for idx in range(3):
            ch = img_cv.copy()*0
            ch[:,:,idx] = img_cv[:,:,idx]
            gallery_imgs.append(Image.fromarray(cv2.cvtColor(ch, cv2.COLOR_BGR2RGB)))

        # Hue + Saturation
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        gallery_imgs.append(Image.fromarray(hsv[:,:,0]))
        gallery_imgs.append(Image.fromarray(hsv[:,:,1]))
    except Exception as e:
        print("⚠️ Lỗi hiển thị kênh màu:", e)

    # Bước 3: Sinh ảnh em bé
    filename = os.path.splitext(os.path.basename(tmp_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"baby_{filename}.png")
    generate_predict_enhanced(
        input_path=tmp_path,
        output_path=out_path,
        num_inference_steps=OPTIMAL_INFERENCE_STEPS,
        guidance_scale=OPTIMAL_CFG_SCALE,
        canny_method="adaptive",
        ethnicity="mixed ethnicity"
    )
    baby_img = Image.open(out_path)
    gallery_imgs.append(baby_img)

    # Bước 4: Kết quả trả về
    download_link = out_path  # Gradio sẽ render link tải
    return gallery_imgs, download_link

# ---- Giao diện Gradio ----
with gr.Blocks(title="Baby Image Generator") as demo:
    gr.Markdown("## Tạo ảnh em bé từ ảnh **siêu âm** • Demo Gradio")

    with gr.Row():
        with gr.Column(scale=1):
            infile = gr.Image(type="pil", label="Tải ảnh siêu âm")

            run_btn = gr.Button("Bắt đầu xử lý")

        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Quy trình & Kết quả",
                                 show_label=True, columns=3, height=400)

            download = gr.File(label="Tải ảnh em bé")

    run_btn.click(process_ultrasound,
                  inputs=[infile],
                  outputs=[gallery, download],
                  api_name="generate")

# server host là localhost
demo.launch(server_name="localhost", share=True, server_port=7860)
