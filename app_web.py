#!/usr/bin/env python3
"""
BabyVis Web App - Gradio Interface
"""

import os
import sys
from pathlib import Path
import gradio as gr
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from babyvis.inference import process_single_image, get_model
from model_downloader import ensure_model


def process_ultrasound(image, model_type="auto", ethnicity="mixed"):
    """Process ultrasound image and return baby face prediction"""
    if image is None:
        return None, "❌ Please upload an ultrasound image"
    
    try:
        # Ensure model is available
        if not ensure_model():
            return None, "❌ Failed to load model. Check your internet connection."
        
        # Save uploaded image temporarily
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        input_path = temp_dir / "input.png"
        output_path = temp_dir / "output.png"
        
        # Save input image
        image.save(input_path)
        
        # Process image
        success = process_single_image(
            input_path=str(input_path),
            output_path=str(output_path),
            model_type=model_type,
            ethnicity=ethnicity
        )
        
        if success and output_path.exists():
            result_image = Image.open(output_path)
            return result_image, "✅ Baby face generated successfully!"
        else:
            return None, "❌ Failed to generate baby face. Please try again."
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="BabyVis - AI Baby Face Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 BabyVis - AI Baby Face Generator
        
        Upload an ultrasound image and get an AI-generated prediction of your baby's face!
        """)
        
        with gr.Row():
            with gr.Column():
                # Input section
                gr.Markdown("### 📤 Upload Ultrasound Image")
                input_image = gr.Image(
                    label="Ultrasound Image",
                    type="pil",
                    height=400
                )
                
                with gr.Row():
                    model_type = gr.Dropdown(
                        label="Hardware Configuration",
                        choices=["auto", "4gb", "12gb", "cpu"],
                        value="auto",
                        info="Auto-detect or choose manually"
                    )
                    
                    ethnicity = gr.Dropdown(
                        label="Ethnicity Preference",
                        choices=["mixed", "asian", "caucasian", "african", "hispanic"],
                        value="mixed",
                        info="Ethnicity for baby face generation"
                    )
                
                process_btn = gr.Button("🚀 Generate Baby Face", variant="primary", size="lg")
                
            with gr.Column():
                # Output section
                gr.Markdown("### 📥 Generated Baby Face")
                output_image = gr.Image(
                    label="Predicted Baby Face",
                    height=400
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to process. Upload an image and click Generate!",
                    interactive=False
                )
        
        # Examples section
        gr.Markdown("### 📋 How to use:")
        gr.Markdown("""
        1. **Upload** an ultrasound image (JPG, PNG, WebP supported)
        2. **Choose** your hardware configuration (auto-detect recommended)
        3. **Select** ethnicity preference (optional)
        4. **Click** "Generate Baby Face" button
        5. **Wait** for processing (30 seconds - 5 minutes depending on hardware)
        6. **Download** your generated baby face image
        """)
        
        # Event handlers
        process_btn.click(
            fn=process_ultrasound,
            inputs=[input_image, model_type, ethnicity],
            outputs=[output_image, status_text],
            show_progress=True
        )
    
    return interface


def main():
    """Launch the web app"""
    print("🤖 Starting BabyVis Web App...")
    
    # Ensure model is downloaded
    print("📥 Checking model availability...")
    if not ensure_model():
        print("❌ Failed to ensure model. Please check your internet connection.")
        return
    
    # Create and launch interface
    interface = create_interface()
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True if you want public link
        show_error=True,
        show_tips=False,
        enable_queue=True,
        max_threads=1           # Limit concurrent processing
    )


if __name__ == "__main__":
    main()