# CNN Character Box Detection and Auto-Typing Tool

## Overview

This project is an end-to-end solution for real-time recognition and automatic input of letter/number sequences that appear as white boxes during live applications. The system uses a custom-trained Convolutional Neural Network (CNN) to classify characters inside detected boxes, then automatically types the recognized string into the app window within a strict time constraint.

---

## Task

- **Goal:** When a sequence of character boxes pops up on the screen (for example, as a challenge or captcha in a game), the tool should:
  1. Capture the screen,
  2. Detect and classify the sequence of characters inside the white boxes,
  3. Construct the string in left-to-right order,
  4. Automatically type the recognized string into the input field,
  5. Complete the entire process in under 5 seconds.

---

## Project Plan

1. **Data Collection & Labeling**
   - Collect screenshots containing the character boxes.
   - Manually crop and label each box with its correct character.

2. **Model Training**
   - Train a CNN classifier (`charbox_cnn_lightaug.h5`) using the labeled box images.
   - Classes: `0-9` and `A-Z` (36 total).

3. **Inference Pipeline Development**
   - Implement image preprocessing to crop the relevant region and detect individual boxes using OpenCV.
   - Classify each box using the trained CNN.
   - Sort boxes from left to right to reconstruct the string.

4. **Automation**
   - Script the workflow to:
     - Trigger on a hotkey press,
     - Capture and process the screenshot,
     - Run the model and reconstruct the string,
     - Automatically type the string into the application.

5. **User Interface**
   - Provide a simple UI and/or hotkey-based command-line tool for ease of use.

---

## Model Pipeline

**Input:** Live screenshot or pre-captured screenshot containing the character boxes.

**Steps:**
1. **Screenshot & Crop:**
   - Capture the screen using `pyautogui`.
   - Crop to the region where boxes appear (to improve speed and accuracy).

2. **Box Detection:**
   - Use thresholding and contour detection (OpenCV) to find white boxes within the cropped region.

3. **Box Classification:**
   - Each cropped box image is resized and preprocessed.
   - The CNN model predicts the character inside each box.

4. **String Construction:**
   - Detected boxes are sorted left-to-right.
   - Characters are concatenated to form the final string.

5. **Auto-Typing:**
   - The recognized string is automatically typed into the game window using pynput.

---

## Automation & Usage

- **Hotkey Trigger:** Press a designated hotkey (e.g., F8, or a custom key combination) to start the process.
- **End-to-End Flow:** Screenshot → Crop & Detect → Classify → String → Auto-Type
- **Runs entirely within couple of seconds**, suitable for fast-paced live gaming.

### Example Usage

# clone the repo: https://github.com/qubitsculptor/text-detection-cnn

# Install dependencies first
pip install tensorflow opencv-python pyautogui pynput

# Run the main automation script
python autotypeGUI.py


- The script will listen for the hotkey and perform Object detection + auto-typing when triggered.


---

## Notes

- **OS Compatibility:** Some automation libraries (e.g., `pyautogui`, `pynput`) may require accessibility permissions on macOS. Hotkey setup may differ by OS.
- **Model Retraining:** To improve accuracy, simply add new labeled box images to your dataset and retrain the model.
- **Extensibility:** The pipeline is modular—can be upgraded to use object detectors (e.g., YOLO) for more complex layouts in the future.

---
