from ultralytics import YOLO
import time
import os

data_used = "augmented.yaml"
train_epoch = 2
image_size = 640
train_optimizer = 'SGD'
batch_size = 32
save_epoch = 1
plot = True
name = "augmented_data"

resume_path = "runs/train/control/weights/last.pt"
model = YOLO(resume_path if os.path.exists(resume_path) else "yolo11n.yaml")

if __name__ == "__main__":
    results = model.train(data=data_used, epochs=train_epoch, imgsz=image_size, save=True, batch=batch_size, optimizer=train_optimizer, save_period = save_epoch, plots=True, name=name)
    pass