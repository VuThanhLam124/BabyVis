#!/usr/bin/env python3
"""
BabyVis Desktop App - Tkinter Interface
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk
import threading

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from babyvis.inference import process_single_image
from model_downloader import ensure_model


class BabyVisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BabyVis - AI Baby Face Generator")
        self.root.geometry("800x600")
        
        # Variables
        self.input_image_path = tk.StringVar()
        self.output_image_path = tk.StringVar()
        self.model_type = tk.StringVar(value="auto")
        self.ethnicity = tk.StringVar(value="mixed")
        self.status = tk.StringVar(value="Ready")
        
        # Images
        self.input_image = None
        self.output_image = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create UI widgets"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 BabyVis - AI Baby Face Generator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="📤 Input Ultrasound Image", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Input image display
        self.input_canvas = tk.Canvas(input_frame, width=300, height=300, bg="lightgray")
        self.input_canvas.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Browse button
        browse_btn = ttk.Button(input_frame, text="📁 Browse Image", command=self.browse_image)
        browse_btn.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Configuration
        ttk.Label(input_frame, text="Hardware:").grid(row=2, column=0, sticky=tk.W)
        model_combo = ttk.Combobox(input_frame, textvariable=self.model_type, 
                                  values=["auto", "4gb", "12gb", "cpu"], state="readonly")
        model_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(input_frame, text="Ethnicity:").grid(row=3, column=0, sticky=tk.W)
        ethnicity_combo = ttk.Combobox(input_frame, textvariable=self.ethnicity,
                                      values=["mixed", "asian", "caucasian", "african", "hispanic"], 
                                      state="readonly")
        ethnicity_combo.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Process button
        self.process_btn = ttk.Button(input_frame, text="🚀 Generate Baby Face", 
                                     command=self.process_image)
        self.process_btn.grid(row=4, column=0, columnspan=2, pady=(0, 10))
        
        # Output section
        output_frame = ttk.LabelFrame(main_frame, text="📥 Generated Baby Face", padding="10")
        output_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Output image display
        self.output_canvas = tk.Canvas(output_frame, width=300, height=300, bg="lightgray")
        self.output_canvas.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Save button
        self.save_btn = ttk.Button(output_frame, text="💾 Save Image", 
                                  command=self.save_image, state="disabled")
        self.save_btn.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(output_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
        status_label = ttk.Label(status_frame, textvariable=self.status)
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def browse_image(self):
        """Browse and select input image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Ultrasound Image",
            filetypes=file_types
        )
        
        if filename:
            self.input_image_path.set(filename)
            self.display_input_image(filename)
            self.status.set(f"Image loaded: {Path(filename).name}")
    
    def display_input_image(self, image_path):
        """Display input image on canvas"""
        try:
            # Open and resize image
            image = Image.open(image_path)
            image.thumbnail((280, 280), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Clear canvas and display image
            self.input_canvas.delete("all")
            self.input_canvas.create_image(150, 150, image=photo)
            
            # Keep reference to prevent garbage collection
            self.input_image = photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def display_output_image(self, image_path):
        """Display output image on canvas"""
        try:
            # Open and resize image
            image = Image.open(image_path)
            image.thumbnail((280, 280), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Clear canvas and display image
            self.output_canvas.delete("all")
            self.output_canvas.create_image(150, 150, image=photo)
            
            # Keep reference
            self.output_image = photo
            
            # Enable save button
            self.save_btn.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display output: {e}")
    
    def process_image(self):
        """Process ultrasound image in background thread"""
        if not self.input_image_path.get():
            messagebox.showwarning("Warning", "Please select an ultrasound image first")
            return
        
        # Start processing in background
        thread = threading.Thread(target=self._process_worker)
        thread.daemon = True
        thread.start()
    
    def _process_worker(self):
        """Background worker for image processing"""
        try:
            # Update UI
            self.root.after(0, self._processing_started)
            
            # Ensure model is available
            if not ensure_model():
                self.root.after(0, lambda: self._processing_finished(False, "Failed to load model"))
                return
            
            # Create temp directory
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            output_path = temp_dir / "output.png"
            
            # Process image
            success = process_single_image(
                input_path=self.input_image_path.get(),
                output_path=str(output_path),
                model_type=self.model_type.get(),
                ethnicity=self.ethnicity.get()
            )
            
            if success and output_path.exists():
                self.output_image_path.set(str(output_path))
                self.root.after(0, lambda: self._processing_finished(True, "Baby face generated successfully!"))
            else:
                self.root.after(0, lambda: self._processing_finished(False, "Failed to generate baby face"))
                
        except Exception as e:
            self.root.after(0, lambda: self._processing_finished(False, f"Error: {e}"))
    
    def _processing_started(self):
        """Update UI when processing starts"""
        self.status.set("Processing... This may take a few minutes")
        self.process_btn.config(state="disabled")
        self.progress.start()
    
    def _processing_finished(self, success, message):
        """Update UI when processing finishes"""
        self.status.set(message)
        self.process_btn.config(state="normal")
        self.progress.stop()
        
        if success:
            self.display_output_image(self.output_image_path.get())
        else:
            messagebox.showerror("Error", message)
    
    def save_image(self):
        """Save generated image"""
        if not self.output_image_path.get():
            return
        
        file_types = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save Baby Face Image",
            defaultextension=".png",
            filetypes=file_types
        )
        
        if filename:
            try:
                # Copy output image to selected location
                import shutil
                shutil.copy2(self.output_image_path.get(), filename)
                self.status.set(f"Image saved: {Path(filename).name}")
                messagebox.showinfo("Success", f"Image saved successfully to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")


def main():
    """Launch desktop app"""
    print("🤖 Starting BabyVis Desktop App...")
    
    root = tk.Tk()
    app = BabyVisApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 App closed by user")


if __name__ == "__main__":
    main()