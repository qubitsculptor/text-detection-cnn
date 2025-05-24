import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "charbox_cnn_lightaug.h5"
IMG_SIZE = (32, 32)
SCREENSHOT_PATH = "4A21FFB6-1EA0-4113-B97F-2027875C3E0F.png"

class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

OUTPUT_ROOT = "predicted_dataset"

model = tf.keras.models.load_model(MODEL_PATH)

def crop_middle_region(img):
    h, w = img.shape[:2]
    x1 = int(w * 0.20)
    x2 = int(w * 0.78)
    y1 = int(h * 0.30)
    y2 = int(h * 0.70)
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
    img = np.expand_dims(img, axis=-1)   # (32, 32, 1)
    img = np.expand_dims(img, axis=0)    # (1, 32, 32, 1)
    return img

def save_predicted_crops(screenshot_path):
    img = cv2.imread(screenshot_path)
    cropped = crop_middle_region(img)
    boxes = detect_boxes(cropped)
    print(f"Detected {len(boxes)} boxes.")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for i, (x, y, w, h) in enumerate(boxes):
        box_img = cropped[y:y+h, x:x+w]
        input_img = preprocess_box(box_img)
        preds = model.predict(input_img, verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_label = class_names[pred_idx]

        # Create class folder 
        class_folder = os.path.join(OUTPUT_ROOT, pred_label)
        os.makedirs(class_folder, exist_ok=True)
        # Save the image in the correct subfolder
        filename = f"{pred_label}_{i+1}.png"
        filepath = os.path.join(class_folder, filename)
        cv2.imwrite(filepath, box_img)
        print(f"Saved {filepath}")

    print(f"All predicted crops saved in '{OUTPUT_ROOT}' folder.")

if __name__ == "__main__":
    save_predicted_crops(SCREENSHOT_PATH)