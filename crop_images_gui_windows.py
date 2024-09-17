#!/usr/bin/env python3

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import sys
import os
from ultralytics import YOLO

# Ensure the current directory is in the system path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Class labels for YOLOv8 (COCO dataset)
CLASS_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # Add more classes as needed
]

def run_detection(source, output, class_id):
    try:
        # Load the YOLO model
        model = YOLO('yolov8n.pt')

        # Run object detection and cropping
        model.predict(
            source=source,
            save=True,
            save_crop=True,
            classes=[class_id],
            project=output
        )

        messagebox.showinfo("Success", f"Processing complete.\nCropped images are saved in:\n{output}\\predict\\crops\\")
    except Exception as e:
        messagebox.showerror("Error", str(e))

class CropImagesGUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Cropper using YOLOv8")

        # Source folder selection
        self.label_source = tk.Label(master, text="Source Image Folder:")
        self.label_source.grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.entry_source = tk.Entry(master, width=50)
        self.entry_source.grid(row=0, column=1, padx=5, pady=5)
        self.button_source = tk.Button(master, text="Browse...", command=self.browse_source)
        self.button_source.grid(row=0, column=2, padx=5, pady=5)

        # Output folder selection
        self.label_output = tk.Label(master, text="Output Folder:")
        self.label_output.grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.entry_output = tk.Entry(master, width=50)
        self.entry_output.grid(row=1, column=1, padx=5, pady=5)
        self.button_output = tk.Button(master, text="Browse...", command=self.browse_output)
        self.button_output.grid(row=1, column=2, padx=5, pady=5)

        # Class identifier selection
        self.label_class = tk.Label(master, text="Class Identifier:")
        self.label_class.grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.class_var = tk.StringVar(master)
        self.class_var.set(CLASS_LABELS[0])  # Default value
        self.dropdown_class = tk.OptionMenu(master, self.class_var, *CLASS_LABELS)
        self.dropdown_class.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        # Start button
        self.button_start = tk.Button(master, text="Start Cropping", command=self.start_cropping)
        self.button_start.grid(row=3, column=1, padx=5, pady=20)

    def browse_source(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.entry_source.delete(0, tk.END)
            self.entry_source.insert(0, folder_selected)

    def browse_output(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, folder_selected)

    def start_cropping(self):
        source = self.entry_source.get()
        output = self.entry_output.get()
        class_name = self.class_var.get()

        if not source or not output or not class_name:
            messagebox.showwarning("Input Error", "Please select all fields.")
            return

        # Get class ID from class name
        class_id = CLASS_LABELS.index(class_name)

        # Run detection in a separate thread to keep GUI responsive
        threading.Thread(target=run_detection, args=(source, output, class_id)).start()

def main():
    root = tk.Tk()
    gui = CropImagesGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
