import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from galaxy_datasets import *
import pandas as pd
import cv2
import numpy as np
from glob import glob
import random
from tqdm import tqdm

# Load the catalog (this downloads ~210k galaxy images and their labels from Galaxyzoo)
catalog, label_cols = gz2(
    root='../gz2',  # local path to store data
    train=True,                       # or False to load test split
    download=True                    # downloads if not already present
)

# # Load the small debugging dataset (~1000 images and their labels)
# catalog, label_cols = demo_rings(
#     root='../demo_rings',  # local path to store data
#     train=True,                       # or False to load test split
#     download=True                    # downloads if not already present
# )


# Get idenfitifers and file paths from data


# print(catalog.head()) # Display the first few rows


# print(catalog.columns) # See available columns


# subset = catalog[['file_loc'] + label_cols]  # file_loc = absolute path to image, Get identifiers and file paths
# print(subset.head())


# Converting imported data to YOLO format and adhering to the categories in the data.yaml file

root_dir_demo = '../demo_yolo_format'  # Where to write new dataset
root_dir = '../gz2_yolo_format'  # Where to write new dataset

os.makedirs(root_dir, exist_ok=True)

# Extracting unique classes from summary column of catalog
unique_classes = []
for idx, row in catalog.iterrows():
    for col in catalog.columns:
        if col == 'summary':
            unique_classes.append(row[col])
unique_classes = set(unique_classes)
class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}


# printing classes to another file to verify
with open(os.path.join(root_dir, 'classes.txt'), 'w') as f:
    for cls in unique_classes:
        f.write(f"{cls}\n")

# Converting single image to yolo format
def convert_bbox_to_yolo(image_width, image_height):
    return (0.5, 0.5, 1.0, 1.0)

# making the train/test split and creating txt files containing labels corresponding to each image

train_df, val_df = train_test_split(catalog, test_size=0.2, random_state=42)

# Folder structure
for split in ['train', 'val']:
    os.makedirs(os.path.join(root_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, split, 'labels'), exist_ok=True)


def export_to_yolo(df, split):
    for idx, row in df.iterrows():
        src_image_path = row['file_loc']
        image = Image.open(src_image_path)
        width, height = image.size
        
        filename = os.path.basename(src_image_path)
        dst_image_path = os.path.join(root_dir, split, 'images', filename)
        dst_label_path = os.path.join(root_dir, split, 'labels', filename.replace('.jpg', '.txt'))
        
        shutil.copyfile(src_image_path, dst_image_path)
         
        class_id = class_to_id[row['summary']]
        x_center, y_center, bbox_width, bbox_height = convert_bbox_to_yolo(width, height)
        
        with open(dst_label_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")



# Data augmentation segment

SPLIT = 'train'  # or 'val'
INPUT_IMAGE_DIR = f"../gz2_yolo_format/{SPLIT}/images"
INPUT_LABEL_DIR = f"../gz2_yolo_format/{SPLIT}/labels"

# Output augmented structure
OUTPUT_IMAGE_DIR = f"dataset/augmented/{SPLIT}/images"
OUTPUT_LABEL_DIR = f"dataset/augmented/{SPLIT}/labels"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def load_yolo_labels(label_path):
    """Load YOLO label file and return a list of [class, cx, cy, w, h]."""
    with open(label_path, "r") as f:
        return [list(map(float, line.strip().split())) for line in f.readlines()]

def save_yolo_labels(label_path, labels):
    """Save YOLO labels (list of lists) to a .txt file."""
    with open(label_path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")

# --- 1. Stitching 10x10 grid ---
# def create_stitched_image_grid(image_paths, grid_size=10):
#     """
#     Create a stitched image from a grid of images and adjust bounding boxes.
    
#     Parameters:
#         image_paths (List[str]): List of image paths to stitch.
#         grid_size (int): Grid size (e.g. 10 for 10x10).
    
#     Returns:
#         (stitched_img, adjusted_labels): Tuple of image and adjusted YOLO labels.
#     """
#     imgs = [cv2.imread(p) for p in image_paths]
#     h, w = imgs[0].shape[:2]
#     canvas = np.zeros((h * grid_size, w * grid_size, 3), dtype=np.uint8)
#     labels_out = []
    
#     for idx, img in enumerate(imgs):
#         gx, gy = idx % grid_size, idx // grid_size
#         x_offset, y_offset = gx * w, gy * h
#         canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img

#         # Load and adjust labels
#         img_name = os.path.basename(image_paths[idx]).replace(".jpg", ".txt")
#         label_path = os.path.join(INPUT_LABEL_DIR, img_name)
#         for label in load_yolo_labels(label_path):
#             cls, cx, cy, bw, bh = label
#             cx = (cx + gx) / grid_size
#             cy = (cy + gy) / grid_size
#             bw /= grid_size
#             bh /= grid_size
#             labels_out.append([cls, cx, cy, bw, bh])
    
#     return canvas, labels_out

# --- 2. Background color replacement ---
def replace_black_background(img, color=(255, 0, 0)):
    """
    Replace black pixels in image with given color.
    
    Parameters:
        img (ndarray): Original image.
        color (tuple): RGB color to use.
    
    Returns:
        img (ndarray): Modified image.
    """
    mask = np.all(img == [0, 0, 0], axis=-1)
    img[mask] = color
    return img

# --- 3. Padding image into a larger black canvas by a random factor between 1.1 and 3 ---
def pad_image_center(img):
    """
    Pad image onto a larger black canvas with randomized padding scale.
    
    Returns:
        Tuple of padded image, x/y offset fraction, and width/height scale.
    """
    h, w = img.shape[:2]
    pad_scale = random.uniform(1.1, 3.0)
    new_h, new_w = int(h * pad_scale), int(w * pad_scale)
    canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    y_off = (new_h - h) // 2
    x_off = (new_w - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = img
    return canvas, x_off / new_w, y_off / new_h, w / new_w, h / new_h

# -- 4. Randomly stitching images of galaxy together with blank space
def create_stitched_image_grid_with_blanks_and_scaling(image_paths, grid_size=10, blank_prob=0.7):
    """
    Create a grid with galaxy images, optionally scaled (1x, 2x, or 3x),
    and optionally replaced by black cells based on blank_prob.

    Parameters:
        image_paths (List[str]): Image paths to choose from.
        grid_size (int): Width/height of the grid (e.g., 10 for 10x10).
        blank_prob (float): Probability of leaving a grid cell blank.

    Returns:
        Tuple of stitched image (np.ndarray) and adjusted YOLO labels (List[List]).
    """
    h, w = cv2.imread(image_paths[0]).shape[:2]
    canvas = np.zeros((h * grid_size, w * grid_size, 3), dtype=np.uint8)
    labels_out = []
    occupied = np.zeros((grid_size, grid_size), dtype=bool)

    def fits(x, y, scale):
        """Check if a scaled patch fits starting at grid cell (x, y)."""
        if x + scale > grid_size or y + scale > grid_size:
            return False
        return not occupied[y:y+scale, x:x+scale].any()

    def mark_occupied(x, y, scale):
        """Mark grid cells as occupied for the inserted patch."""
        occupied[y:y+scale, x:x+scale] = True

    for y in range(grid_size):
        for x in range(grid_size):
            if occupied[y, x]:
                continue
            if random.random() < blank_prob:
                continue

            random.shuffle(image_paths)
            for img_path in image_paths:
                scale = random.choice([1, 2, 3])  # Try smaller first
                if not fits(x, y, scale):
                    continue

                # Load and resize image
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_AREA)

                # Paste into canvas
                px, py = x * w, y * h
                canvas[py:py + h * scale, px:px + w * scale] = img_resized
                mark_occupied(x, y, scale)

                # Adjust labels
                label_path = os.path.join(INPUT_LABEL_DIR, os.path.basename(img_path).replace(".jpg", ".txt"))
                for label in load_yolo_labels(label_path):
                    cls, cx, cy, bw, bh = label

                    # New relative coordinates after scaling and placement
                    cx = (cx * scale + x) / grid_size
                    cy = (cy * scale + y) / grid_size
                    bw = (bw * scale) / grid_size
                    bh = (bh * scale) / grid_size
                    labels_out.append([cls, cx, cy, bw, bh])
                break  # Move to next grid cell

    return canvas, labels_out
# --- Run All Four Types of Augmentation ---

def generate_augmented_data(n=10):
    """
    Generate n augmented images of each type:
    - Stitched grid
    - Stitched grid with blanks
    - Background recolored
    - Image padded with random scale
    
    Parameters:
        n (int): Number of images to generate for each type
    """
    all_images = glob(os.path.join(INPUT_IMAGE_DIR, "*.jpg"))
    if not all_images:
        print("No images found.")
        return

    for i in range(n):
        # === 1. Stitched full grid ===
        # if len(all_images) >= 100:
        #     selected = random.sample(all_images, 100)
        #     img, labels = create_stitched_image_grid(selected)
        #     fname = f"stitched_{i}.jpg"
        #     cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, fname), img)
        #     save_yolo_labels(os.path.join(OUTPUT_LABEL_DIR, fname.replace('.jpg', '.txt')), labels)

        # === 2. Stitched grid with black cells ===
        if len(all_images) >= 10:
            img, labels = create_stitched_image_grid_with_blanks_and_scaling(all_images, blank_prob=0.8)
            fname = f"stitched_blank_{i}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, fname), img)
            save_yolo_labels(os.path.join(OUTPUT_LABEL_DIR, fname.replace('.jpg', '.txt')), labels)

        # === 3. Background recoloring ===
        path = random.choice(all_images)
        img = cv2.imread(path)
        color = tuple(np.random.randint(0, 255, size=3))
        modified = replace_black_background(img.copy(), color)
        fname = f"recolored_{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, fname), modified)
        label_path = os.path.join(INPUT_LABEL_DIR, os.path.basename(path).replace(".jpg", ".txt"))
        shutil.copy(label_path, os.path.join(OUTPUT_LABEL_DIR, fname.replace('.jpg', '.txt')))

        # === 4. Random padding ===
        path = random.choice(all_images)
        img = cv2.imread(path)
        padded_img, x_off, y_off, w_scale, h_scale = pad_image_center(img)
        label_path = os.path.join(INPUT_LABEL_DIR, os.path.basename(path).replace(".jpg", ".txt"))
        padded_labels = []
        for label in load_yolo_labels(label_path):
            cls, cx, cy, bw, bh = label
            cx = cx * w_scale + x_off
            cy = cy * h_scale + y_off
            bw *= w_scale
            bh *= h_scale
            padded_labels.append([cls, cx, cy, bw, bh])
        fname = f"padded_{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, fname), padded_img)
        save_yolo_labels(os.path.join(OUTPUT_LABEL_DIR, fname.replace('.jpg', '.txt')), padded_labels)


# Export both splits

if __name__ == "__main__":
    
    # export_to_yolo(train_df, 'train') # uncomment on first run
    # export_to_yolo(val_df, 'val')     # uncomment on first run
    # print("Train:", len(train_df))
    # print("Val:", len(val_df))
    #generate_augmented_data(n=10000)
    pass
    