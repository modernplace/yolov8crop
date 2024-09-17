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
    "person",           # 0
    "bicycle",          # 1
    "car",              # 2
    "motorcycle",       # 3
    "airplane",         # 4
    "bus",              # 5
    "train",            # 6
    "truck",            # 7
    "boat",             # 8
    "traffic light",    # 9
    "fire hydrant",     # 10
    "stop sign",        # 11
    "parking meter",    # 12
    "bench",            # 13
    "bird",             # 14
    "cat",              # 15
    "dog",              # 16
    "horse",            # 17
    "sheep",            # 18
    "cow",              # 19
    "elephant",         # 20
    "bear",             # 21
    "zebra",            # 22
    "giraffe",          # 23
    "backpack",         # 24
    "umbrella",         # 25
    "handbag",          # 26
    "tie",              # 27
    "suitcase",         # 28
    "frisbee",          # 29
    "skis",             # 30
    "snowboard",        # 31
    "sports ball",      # 32
    "kite",             # 33
    "baseball bat",     # 34
    "baseball glove",   # 35
    "skateboard",       # 36
    "surfboard",        # 37
    "tennis racket",    # 38
    "bottle",           # 39
    "wine glass",       # 40
    "cup",              # 41
    "fork",             # 42
    "knife",            # 43
    "spoon",            # 44
    "bowl",             # 45
    "banana",           # 46
    "apple",            # 47
    "sandwich",         # 48
    "orange",           # 49
    "broccoli",         # 50
    "carrot",           # 51
    "hot dog",          # 52
    "pizza",            # 53
    "donut",            # 54
    "cake",             # 55
    "chair",            # 56
    "couch",            # 57
    "potted plant",     # 58
    "bed",              # 59
    "dining table",     # 60
    "toilet",           # 61
    "tv",               # 62
    "laptop",           # 63
    "mouse",            # 64
    "remote",           # 65
    "keyboard",         # 66
    "cell phone",       # 67
    "microwave",        # 68
    "oven",             # 69
    "toaster",          # 70
    "sink",             # 71
    "refrigerator",     # 72
    "book",             # 73
    "clock",            # 74
    "vase",             # 75
    "scissors",         # 76
    "teddy bear",       # 77
    "hair drier",       # 78
    "toothbrush"        # 79
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
