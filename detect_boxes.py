import cv2
import numpy as np

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

img = cv2.imread("4A21FFB6-1EA0-4113-B97F-2027875C3E0F.png")
cropped = crop_middle_region(img)
boxes = detect_boxes(cropped)

print(f"Detected {len(boxes)} boxes:")
for i, (x, y, w, h) in enumerate(boxes):
    print(f"Box {i+1}: x={x}, y={y}, w={w}, h={h}")
    cv2.rectangle(cropped, (x, y), (x+w, y+h), (0,255,0), 2)
cv2.imwrite("cropped_with_boxes.png", cropped)
cv2.imshow("Detected Boxes", cropped)

cv2.imwrite("debug_cropped.png", cropped)



'''   x1 = int(w * 0.28)
    x2 = int(w * 0.75)
    y1 = int(h * 0.40)
    y2 = int(h * 0.55)'''