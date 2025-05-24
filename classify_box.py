import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "charbox_cnn_lightaug.h5"
IMG_SIZE = (32, 32)
SCREENSHOT_PATH = "Screenshot 2025-05-24 at 5.01.59 AM.png"  

# Paste your class_names list here, in the correct order from your training set
class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# ==== LOAD MODEL ====
model = tf.keras.models.load_model(MODEL_PATH)

# ==== UTILS ====
import cv2

def crop_middle_region(img):
    h, w = img.shape[:2]
    x1 = int(w * 0.28)
    x2 = int(w * 0.75)
    y1 = int(h * 0.45)
    y2 = int(h * 0.60)
    return img[y1:y2, x1:x2]

img = cv2.imread("Screenshot 2025-05-24 at 5.01.59 AM.png")
cropped = crop_middle_region(img)
cv2.imwrite("debug_cropped.png", cropped)

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
    # Model rescales, so we can leave as uint8 (0-255)
    return img

# ==== MAIN PIPELINE ====
def classify_boxes_from_screenshot(screenshot_path):
    img = cv2.imread(screenshot_path)
    cropped = crop_middle_region(img)
    boxes = detect_boxes(cropped)
    detected_chars = []
    print(f"Detected {len(boxes)} boxes.")
    for i, (x, y, w, h) in enumerate(boxes):
        box_img = cropped[y:y+h, x:x+w]
        # For debugging: save each box crop
        cv2.imwrite(f"box_{i+1}.png", box_img)
        input_img = preprocess_box(box_img)
        preds = model.predict(input_img, verbose=0)
        pred_idx = np.argmax(preds[0])
        detected_chars.append(class_names[pred_idx])
        print(f"Box {i+1}: Predicted '{class_names[pred_idx]}'")
    detected_string = ''.join(detected_chars)
    print("Detected string:", detected_string)
    return detected_string

if __name__ == "__main__":
    classify_boxes_from_screenshot(SCREENSHOT_PATH)