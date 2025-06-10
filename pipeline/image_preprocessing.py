import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from galaxy_datasets import gz2
import pandas as pd

# Load the catalog (this downloads ~210k galaxy images and their labels from Galaxyzoo)
catalog, label_cols = gz2(
    root='../gz2',  # local path to store data
    train=True,                       # or False to load test split
    download=True                    # downloads if not already present
)


# Converting imported data to YOLO format and adhering to the categories in the data.yaml file

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

# Export both splits

if __name__ == "__main__":
    
    export_to_yolo(train_df, 'train')
    export_to_yolo(val_df, 'val')
    