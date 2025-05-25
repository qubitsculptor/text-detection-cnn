import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import time
from pynput import keyboard
import tkinter as tk
from tkinter import messagebox
import threading

MODEL_PATH = "charbox_cnn_lightaug.h5"
IMG_SIZE = (32, 32)
SCREENSHOT_SAVE_PATH = "latest_screenshot.png"

class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

def capture_screenshot():
    print("Taking screenshot...")
    screenshot = pyautogui.screenshot()
    screenshot.save(SCREENSHOT_SAVE_PATH)
    print(f"Screenshot saved at {SCREENSHOT_SAVE_PATH}")
    return SCREENSHOT_SAVE_PATH

def crop_middle_region(img):
    h, w = img.shape[:2]
    x1 = int(w * 0.28)
    x2 = int(w * 0.75)
    y1 = int(h * 0.45)
    y2 = int(h * 0.60)
    return img[y1:y2, x1:x2]

def detect_boxes(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:
            boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes

def preprocess_box(box_img):
    img = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY) if len(box_img.shape) == 3 else box_img
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def classify_boxes_from_screenshot(screenshot_path):
    print("Reading screenshot...")
    img = cv2.imread(screenshot_path)
    if img is None:
        print("Failed to load screenshot!")
        return ""
    cropped = crop_middle_region(img)
    boxes = detect_boxes(cropped)
    print(f"Detected {len(boxes)} boxes.")
    detected_chars = []
    for i, (x, y, w, h) in enumerate(boxes):
        box_img = cropped[y:y+h, x:x+w]
        input_img = preprocess_box(box_img)
        preds = model.predict(input_img, verbose=0)
        pred_idx = np.argmax(preds[0])
        detected_chars.append(class_names[pred_idx])
        print(f"Box {i+1}: predicted '{class_names[pred_idx]}'")
    detected_string = ''.join(detected_chars)
    print("Detected string:", detected_string)
    return detected_string

def classify_and_type():
    path = capture_screenshot()
    result = classify_boxes_from_screenshot(path)
    pyautogui.typewrite(result)

class AutotypeApp:
    def __init__(self, root):
        self.root = root
        self.listener = None
        self.hotkey_combo = []

        root.title("AutoTyper Config")
        root.geometry("300x250")

        tk.Label(root, text="Set your hotkey combo (e.g. ctrl+shift+a or f5):").pack(pady=10)
        self.entry = tk.Entry(root)
        self.entry.pack(pady=5)

        tk.Button(root, text="Start Listener", command=self.start_listener).pack(pady=20)

    def start_listener(self):
        combo = self.entry.get().strip().lower().split('+')
        if not combo:
            messagebox.showerror("Error", "Please enter a hotkey")
            return
        self.hotkey_combo = set(combo)
        self.entry.config(state='disabled')
        threading.Thread(target=self.listen_keys, daemon=True).start()
        messagebox.showinfo("Info", f"Hotkey set to: {' + '.join(self.hotkey_combo).upper()}")

    def listen_keys(self):
        current_keys = set()

        def on_press(key):
            name = self._get_key_name(key)
            current_keys.add(name)
            if self.hotkey_combo.issubset(current_keys):
                print("Hotkey combo pressed.")
                classify_and_type()

        def on_release(key):
            name = self._get_key_name(key)
            current_keys.discard(name)

        with keyboard.Listener(on_press=on_press, on_release=on_release) as self.listener:
            self.listener.join()

    def _get_key_name(self, key):
        try:
            return key.char.lower()
        except AttributeError:
            return str(key).replace('Key.', '').lower()

if __name__ == "__main__":
    root = tk.Tk()
    app = AutotypeApp(root)
    root.mainloop()
