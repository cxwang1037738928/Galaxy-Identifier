from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_path):
        results = self.model(image_path)
        return results[0]  # assume one image
