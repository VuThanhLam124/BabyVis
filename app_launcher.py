#!/usr/bin/env python3
"""
BabyVis App Launcher - Simple GUI Selector
"""

import sys
import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
from pathlib import Path


class AppLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("BabyVis - Choose Interface")
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create launcher widgets"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 BabyVis", 
                               font=("Arial", 20, "bold"))
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="AI Baby Face Generator", 
                                  font=("Arial", 12))
        subtitle_label.pack(pady=(0, 30))
        
        # Description
        desc_label = ttk.Label(main_frame, 
                              text="Choose your preferred interface:",
                              font=("Arial", 11))
        desc_label.pack(pady=(0, 20))
        
        # Web App button
        web_frame = ttk.Frame(main_frame)
        web_frame.pack(fill=tk.X, pady=(0, 15))
        
        web_btn = ttk.Button(web_frame, text="🌐 Web Interface", 
                            command=self.launch_web, width=25)
        web_btn.pack()
        
        web_desc = ttk.Label(web_frame, 
                            text="Browser-based interface (requires gradio)",
                            font=("Arial", 9), foreground="gray")
        web_desc.pack(pady=(5, 0))
        
        # Desktop App button
        desktop_frame = ttk.Frame(main_frame)
        desktop_frame.pack(fill=tk.X, pady=(0, 15))
        
        desktop_btn = ttk.Button(desktop_frame, text="🖥️ Desktop App", 
                                command=self.launch_desktop, width=25)
        desktop_btn.pack()
        
        desktop_desc = ttk.Label(desktop_frame, 
                                text="Native desktop interface (built-in)",
                                font=("Arial", 9), foreground="gray")
        desktop_desc.pack(pady=(5, 0))
        
        # Batch Processing button
        batch_frame = ttk.Frame(main_frame)
        batch_frame.pack(fill=tk.X, pady=(0, 20))
        
        batch_btn = ttk.Button(batch_frame, text="📦 Batch Processing", 
                              command=self.launch_batch, width=25)
        batch_btn.pack()
        
        batch_desc = ttk.Label(batch_frame, 
                              text="Process multiple images from samples/ folder",
                              font=("Arial", 9), foreground="gray")
        batch_desc.pack(pady=(5, 0))
        
        # Exit button
        exit_btn = ttk.Button(main_frame, text="❌ Exit", command=self.root.quit)
        exit_btn.pack(pady=(10, 0))
        
    def launch_web(self):
        """Launch web interface"""
        try:
            print("🌐 Launching Web Interface...")
            subprocess.Popen([sys.executable, "app_web.py"])
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch web interface:\n{e}")
    
    def launch_desktop(self):
        """Launch desktop interface"""
        try:
            print("🖥️ Launching Desktop Interface...")
            subprocess.Popen([sys.executable, "app_desktop.py"])
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch desktop interface:\n{e}")
    
    def launch_batch(self):
        """Launch batch processing"""
        try:
            print("📦 Launching Batch Processing...")
            subprocess.Popen([sys.executable, "batch_processor.py"])
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch batch processing:\n{e}")


def main():
    """Main launcher"""
    print("🚀 BabyVis App Launcher")
    
    root = tk.Tk()
    app = AppLauncher(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Launcher closed")


if __name__ == "__main__":
    main()