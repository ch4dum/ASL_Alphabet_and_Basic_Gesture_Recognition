import os
import cv2
import shutil

def yolo_to_box(yolo_line, img_w, img_h):
    class_id, x_c, y_c, w, h = map(float, yolo_line.strip().split())
    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h
    x1 = int(x_c - w / 2)
    y1 = int(y_c - h / 2)
    x2 = int(x_c + w / 2)
    y2 = int(y_c + h / 2)
    return int(class_id), max(x1, 0), max(y1, 0), min(x2, img_w), min(y2, img_h)

# ----- Alphabet dataset path -----
# image_dir = r'alphabet_dataset\test\images'
# label_dir = r'alphabet_dataset\test\labels'
# output_dir = r'cnn_alphabet_dataset\test'
# image_dir = r'alphabet_dataset\train\images'
# label_dir = r'alphabet_dataset\train\labels'
# output_dir = r'cnn_alphabet_dataset\train'
# image_dir = r'alphabet_dataset\valid\images'
# label_dir = r'alphabet_dataset\valid\labels'
# output_dir = r'cnn_alphabet_dataset\valid'

# ----- Basic dataset path -----
# image_dir = r'basic_dataset\test\images'
# label_dir = r'basic_dataset\test\labels'
# output_dir = r'cnn_basic_dataset\test'
image_dir = r'basic_dataset\train\images'
label_dir = r'basic_dataset\train\labels'
output_dir = r'cnn_basic_dataset\train'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    if not filename.endswith('.jpg'):
        continue

    img_path = os.path.join(image_dir, filename)
    txt_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

    img = cv2.imread(img_path)
    h_img, w_img = img.shape[:2]

    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            continue  # label

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) == 5:
                # YOLO object detection format (with bbox)
                class_id, x1, y1, x2, y2 = yolo_to_box(line, w_img, h_img)
                cropped = img[y1:y2, x1:x2]
            elif len(parts) == 1:
                # classification-only format (no bbox)
                class_id = int(parts[0])
                cropped = img
            else:
                continue

            class_folder = os.path.join(output_dir, str(class_id))
            os.makedirs(class_folder, exist_ok=True)

            out_filename = f"{filename[:-4]}_{i}.jpg" if len(lines) > 1 else filename
            out_path = os.path.join(class_folder, out_filename)
            cv2.imwrite(out_path, cropped)
    else:
        continue
