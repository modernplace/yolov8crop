#!/usr/bin/env python3

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import sys
import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO
from PIL import Image

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

def run_detection(source, output, class_id, class_name, resize_option, target_size, padding_option, padding):
    try:
        # Load the YOLO model
        model = YOLO('yolov8n.pt')

        # Get list of image files
        image_files = glob.glob(os.path.join(source, '*.*'))

        for image_file in image_files:
            # Read the image
            orig_img = cv2.imread(image_file)
            if orig_img is None:
                print(f"Error reading image {image_file}")
                continue
            orig_height, orig_width = orig_img.shape[:2]

            # Run object detection on the image without resizing
            results = model.predict(
                source=image_file,
                save=True,  # Enable saving of detection previews
                classes=[class_id],
                imgsz=[orig_width, orig_height],  # Set imgsz to the original image size
                save_txt=False,
                project=output,
                name='detect',
                exist_ok=True,
                conf=0.25  # Adjust the confidence threshold if needed
            )

            result = results[0]  # There is only one result

            # Get boxes
            boxes = result.boxes  # Boxes object
            for idx, box in enumerate(boxes):
                # box.xyxy[0]: tensor([x1, y1, x2, y2])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Convert to int
                x_min = int(x1)
                y_min = int(y1)
                x_max = int(x2)
                y_max = int(y2)

                # Ensure coordinates are within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(orig_width, x_max)
                y_max = min(orig_height, y_max)

                # Apply padding if enabled
                if padding_option and padding:
                    padding_left, padding_top, padding_right, padding_bottom = padding
                    x_min_padded = x_min - padding_left
                    y_min_padded = y_min - padding_top
                    x_max_padded = x_max + padding_right
                    y_max_padded = y_max + padding_bottom

                    # Ensure padded coordinates are within image bounds
                    x_min_padded = max(0, x_min_padded)
                    y_min_padded = max(0, y_min_padded)
                    x_max_padded = min(orig_width, x_max_padded)
                    y_max_padded = min(orig_height, y_max_padded)
                else:
                    x_min_padded = x_min
                    y_min_padded = y_min
                    x_max_padded = x_max
                    y_max_padded = y_max

                # Crop the image with padding
                crop_img = orig_img[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

                # Resize if needed
                if resize_option and target_size:
                    # Resize while maintaining aspect ratio
                    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    pil_img.thumbnail(target_size, Image.LANCZOS)
                    crop_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                # Save the cropped image
                crop_dir = os.path.join(output, 'crops', class_name)
                os.makedirs(crop_dir, exist_ok=True)
                # Create a unique filename
                image_name = os.path.splitext(os.path.basename(image_file))[0]
                crop_filename = f'{image_name}_{idx}.jpg'
                crop_path = os.path.join(crop_dir, crop_filename)
                cv2.imwrite(crop_path, crop_img)

        messagebox.showinfo("Success", f"Processing complete.\nCropped images and detection previews are saved in:\n{output}")
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

        # Resize option
        self.resize_var = tk.BooleanVar()
        self.checkbox_resize = tk.Checkbutton(master, text="Resize Cropped Images", variable=self.resize_var, command=self.toggle_resize_fields)
        self.checkbox_resize.grid(row=3, column=1, padx=5, pady=5, sticky='w')

        # Width and height fields (hidden by default)
        self.label_width = tk.Label(master, text="Max Width:")
        self.entry_width = tk.Entry(master, width=10)
        self.label_height = tk.Label(master, text="Max Height:")
        self.entry_height = tk.Entry(master, width=10)

        # Padding option
        self.padding_var = tk.BooleanVar()
        self.checkbox_padding = tk.Checkbutton(master, text="Add Padding to Crops", variable=self.padding_var, command=self.toggle_padding_fields)
        self.checkbox_padding.grid(row=4, column=1, padx=5, pady=5, sticky='w')

        # Padding fields (hidden by default)
        self.label_padding_top = tk.Label(master, text="Top Padding:")
        self.entry_padding_top = tk.Entry(master, width=10)
        self.label_padding_bottom = tk.Label(master, text="Bottom Padding:")
        self.entry_padding_bottom = tk.Entry(master, width=10)
        self.label_padding_left = tk.Label(master, text="Left Padding:")
        self.entry_padding_left = tk.Entry(master, width=10)
        self.label_padding_right = tk.Label(master, text="Right Padding:")
        self.entry_padding_right = tk.Entry(master, width=10)

        # Start button
        self.button_start = tk.Button(master, text="Start Cropping", command=self.start_cropping)
        self.button_start.grid(row=10, column=1, padx=5, pady=20)

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

    def toggle_resize_fields(self):
        if self.resize_var.get():
            self.label_width.grid(row=5, column=0, padx=5, pady=5, sticky='e')
            self.entry_width.grid(row=5, column=1, padx=5, pady=5, sticky='w')
            self.label_height.grid(row=6, column=0, padx=5, pady=5, sticky='e')
            self.entry_height.grid(row=6, column=1, padx=5, pady=5, sticky='w')
        else:
            self.label_width.grid_remove()
            self.entry_width.grid_remove()
            self.label_height.grid_remove()
            self.entry_height.grid_remove()

    def toggle_padding_fields(self):
        if self.padding_var.get():
            self.label_padding_top.grid(row=7, column=0, padx=5, pady=5, sticky='e')
            self.entry_padding_top.grid(row=7, column=1, padx=5, pady=5, sticky='w')
            self.label_padding_bottom.grid(row=8, column=0, padx=5, pady=5, sticky='e')
            self.entry_padding_bottom.grid(row=8, column=1, padx=5, pady=5, sticky='w')
            self.label_padding_left.grid(row=9, column=0, padx=5, pady=5, sticky='e')
            self.entry_padding_left.grid(row=9, column=1, padx=5, pady=5, sticky='w')
            self.label_padding_right.grid(row=10, column=0, padx=5, pady=5, sticky='e')
            self.entry_padding_right.grid(row=10, column=1, padx=5, pady=5, sticky='w')
        else:
            self.label_padding_top.grid_remove()
            self.entry_padding_top.grid_remove()
            self.label_padding_bottom.grid_remove()
            self.entry_padding_bottom.grid_remove()
            self.label_padding_left.grid_remove()
            self.entry_padding_left.grid_remove()
            self.label_padding_right.grid_remove()
            self.entry_padding_right.grid_remove()

    def start_cropping(self):
        source = self.entry_source.get()
        output = self.entry_output.get()
        class_name = self.class_var.get()
        resize_option = self.resize_var.get()
        padding_option = self.padding_var.get()

        if not source or not output or not class_name:
            messagebox.showwarning("Input Error", "Please select all fields.")
            return

        if resize_option:
            try:
                width = int(self.entry_width.get())
                height = int(self.entry_height.get())
                if width <= 0 or height <= 0:
                    raise ValueError
                target_size = (width, height)
            except ValueError:
                messagebox.showwarning("Input Error", "Please enter valid positive integers for max width and max height.")
                return
        else:
            target_size = None

        if padding_option:
            try:
                padding_top = int(self.entry_padding_top.get())
                padding_bottom = int(self.entry_padding_bottom.get())
                padding_left = int(self.entry_padding_left.get())
                padding_right = int(self.entry_padding_right.get())
                if any(p < 0 for p in [padding_top, padding_bottom, padding_left, padding_right]):
                    raise ValueError
                padding = (padding_left, padding_top, padding_right, padding_bottom)
            except ValueError:
                messagebox.showwarning("Input Error", "Please enter valid non-negative integers for padding.")
                return
        else:
            padding = (0, 0, 0, 0)

        # Get class ID from class name
        class_id = CLASS_LABELS.index(class_name)

        # Run detection in a separate thread to keep GUI responsive
        threading.Thread(target=run_detection, args=(source, output, class_id, class_name, resize_option, target_size, padding_option, padding)).start()

def main():
    root = tk.Tk()
    gui = CropImagesGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
