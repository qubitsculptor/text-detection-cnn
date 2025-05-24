import os
import cv2
import numpy as np

src_folder = "labeled_boxes"
dst_folder = "light_augmented_dataset"
n_variations = 8   # Number of augmentations per original

os.makedirs(dst_folder, exist_ok=True)

def light_augment(img):
    augmented = []
    for _ in range(n_variations):
        aug = img.astype(np.float32)
        # Brightness: ±10%
        alpha = np.random.uniform(0.90, 1.10)
        aug *= alpha
        # Exposure: add/subtract up to 10
        beta = np.random.uniform(-10, 10)
        aug += beta
        aug = np.clip(aug, 0, 255)
        # Blur: very slight (1px) sometimes
        if np.random.rand() < 0.5:
            aug = cv2.GaussianBlur(aug, (3, 3), 0)
        # Light noise
        if np.random.rand() < 0.7:
            noise = np.random.normal(0, 2, aug.shape)
            aug += noise
            aug = np.clip(aug, 0, 255)
        augmented.append(aug.astype(np.uint8))
    return augmented

for fname in os.listdir(src_folder):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    class_name = os.path.splitext(fname)[0]
    dst_class_path = os.path.join(dst_folder, class_name)
    os.makedirs(dst_class_path, exist_ok=True)
    src_img_path = os.path.join(src_folder, fname)
    img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read image {src_img_path}")
        continue
    # Save original
    cv2.imwrite(os.path.join(dst_class_path, f"{class_name}_orig.png"), img)
    # Save augmentations
    for i, aug_img in enumerate(light_augment(img)):
        cv2.imwrite(os.path.join(dst_class_path, f"{class_name}_aug{i+1}.png"), aug_img)

print(f"Augmented images are saved in '{dst_folder}' with class subfolders.")