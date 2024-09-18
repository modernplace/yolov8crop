This simple tool was made for dataset preparation for LORA training although it could be used for a variety of applications.

This GUI was meant for Windows, but if you're on Linux it should run the same. 


### How to Use:

1. **Save the Script:**

   Save the script above as `crop_images_gui.py` on your Windows machine.

2. **Install Python:**

   * Download Python 3.7 or newer from the [official website](https://www.python.org/downloads/windows/).
   * During installation, make sure to check the box that says **"Add Python to PATH"**.
   * The script is compatible with Python 3.x

3. **Install Required Packages:**

   Open Command Prompt (you can search for `cmd` in the Start menu) and run:

   ```
   pip install ultralytics opencv-python Pillow numpy
   ```

   * If `pip` is not recognized, you may need to restart Command Prompt or add Python to your system PATH manually.

4. **Run the Script:**

   In Command Prompt, navigate to the directory where you saved `crop_images_gui.py`:

   ```
   cd path\to\your\script
   ```

   Then run the script:

   ```
   python crop_images_gui.py
   ```

5. **Using the GUI:**

FYI: I'm no picasso. 

   * **Source Image Folder:** Click the "Browse..." button next to "Source Image Folder" to select the folder containing your input images.
   * **Output Folder:** Click the "Browse..." button next to "Output Folder" to select where you want the cropped images to be saved.
   * **Class Identifier:** Use the dropdown menu to select the object class you want to detect and crop.
   * **Max Dimensions:** Set the maximum dimensions to save the cropped images in.
   * **Padding:** Include the padding (in pixels) for the crop around the subject.
   * **Start Cropping:** Click the "Start Cropping" button to begin the process.

6. **Processing:**

   * The script will process the images and save the cropped objects in the specified output directory under `predict\crops\`.
   * A message box will appear upon completion.

***

### Dependencies:

* **Ultralytics YOLOv8:** Install via pip:

  ```
  pip install ultralytics
  ```

* **Tkinter:** Tkinter is included with Python on Windows. If you encounter any issues, reinstall Python and ensure that the "tcl/tk and IDLE" feature is selected during installation.

***

### Class Labels:

The `CLASS_LABELS` list contains object classes from the COCO dataset. You can modify or extend this list based on your needs.

**Common Class Identifiers:**

| ID  | Class         |
| :-- | :------------ |
| 0   | person        |
| 1   | bicycle       |
| 2   | car           |
| 3   | motorcycle    |
| 4   | airplane      |
| 5   | bus           |
| 6   | train         |
| 7   | truck         |
| 8   | boat          |
| 9   | traffic light |
| 10  | fire hydrant  |
| 11  | stop sign     |
| 12  | parking meter |
| 13  | bench         |
| 14  | bird          |
| 15  | cat           |
| 16  | dog           |
| 17  | horse         |
| 18  | sheep         |
| 19  | cow           |
| 20  | elephant      |
| 21  | bear          |
| 22  | zebra         |
| 23  | giraffe       |
| ... | ...           |

***

YOLO is based on the COCO Dataset: <https://cocodataset.org/#explore>



All 80 classes are in the script.

### Notes:

* **Threading:** The script uses threading to keep the GUI responsive during image processing.
* **Error Handling:** Basic error handling is included to catch exceptions and display error messages.
* **Extensibility:** You can add more classes to the `CLASS_LABELS` list as needed.

***

### Troubleshooting:

* **Module Not Found:**

  If you encounter `ModuleNotFoundError: No module named 'ultralytics'`, ensure you're using the correct Python environment where Ultralytics is installed. You can check installed packages with:

  ```
  pip list
  ```

* **Tkinter Not Found:**

  If Tkinter is not working, ensure that you have the "tcl/tk and IDLE" option selected during Python installation.

* **Permission Issues:**

  Ensure you have read permissions for the source folder and write permissions for the output folder.

* **Long File Paths:**

  Windows has a maximum path length limitation. If you encounter issues related to file paths, try using directories with shorter paths.

***

### Customization:

* **Adding More Classes:**

  Modify the `CLASS_LABELS` list in the script to include more classes. For example:

  ```
  CLASS_LABELS = [
      "person", "bicycle", "car", "motorcycle", "airplane", "bus",
      "train", "truck", "boat", "traffic light", "fire hydrant",
      "stop sign", "parking meter", "bench", "bird", "cat", "dog",
      "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
      "backpack", "umbrella", "handbag", "tie", "suitcase",
      # Add more classes as needed
  ]
  ```

* **Model Selection:**

  If you wish to use a different YOLOv8 model, modify the line:

  ```
  model = YOLO('yolov8n.pt')
  ```

  to use other model weights like `yolov8s.pt`, `yolov8m.pt`, etc.

* **Advanced Settings:**

  For additional settings like confidence thresholds or image sizes, adjust the `model.predict()` parameters. For example:

  ```
  model.predict( source=source, save=True, save_crop=True, classes=[class_id], project=output, conf=0.5, # Confidence threshold imgsz=640 # Image size )
  ```

***

### Additional Tips:

* **Running as an Executable:**

  If you prefer to run the script without invoking Python explicitly, you can create a batch file (`.bat`) or use tools like PyInstaller to create an executable.

* **Environment Isolation:**

  Consider using a virtual environment to manage dependencies and avoid conflicts:

  ```
  python -m venv venv venv\Scripts\activate pip install ultralytics
  ```

* **Check GPU Availability:**

  If you have a compatible GPU, you can leverage it for faster processing. Ensure you have the appropriate CUDA drivers and install torch with CUDA support:

  ```
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
  ```

  *(Replace* *`cu117`* *with the appropriate CUDA version for your system.)*

***

### Conclusion:

This Windows-compatible GUI simplifies the process of cropping objects from images using YOLOv8 by providing an intuitive interface. The script should run smoothly on Windows systems with Python and the necessary packages installed.
