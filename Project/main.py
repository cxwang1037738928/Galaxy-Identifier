from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from tkinter import filedialog
import cv2
import os
import uuid
from kivy.lang import Builder
import PIL.Image
from io import BytesIO
from kivy.core.image import Image as CoreImage
from detector import ObjectDetector


class MainWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = ObjectDetector()
        self.image_path = None
        self.result_image = None  # Holds the NumPy array of result

    def load_image(self):

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image_path = file_path
            self.ids.display_image.source = file_path
            self.ids.display_image.reload()
            self.ids.status_label.text = f"Loaded image: {os.path.basename(file_path)}"

    def detect_objects(self):
        if not self.image_path:
            self.ids.status_label.text = "No image loaded."
            return

        result = self.detector.detect(self.image_path)
        self.result_image = result.plot()  # This is a NumPy array (BGR) resulting from the YOLO model. 

        # Convert the result image to PNG in-memory

        # Generates the result image dynamically so it does not get saved automatically

        # Converts the BGR image from YOLO into RGB so kivy can display it
        img_rgb = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(img_rgb) # Converts the numpy array to a pil image

        # Saves the image in Buffer to display, but does not save it in directory
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        # Rewinds the system to beginning for kivy to read
        buffer.seek(0)

        # Display it in Kivy without saving to disk

        # Loads the image into a kivy compatible image
        core_image = CoreImage(buffer, ext='png')
        self.ids.display_image.texture = core_image.texture
        self.ids.status_label.text = "Detection complete."

        # Keep the image in memory if user wants to save later
        self._last_result_image = self.result_image  # Store for future saving

    def save_image(self):
        if hasattr(self, "_last_result_image") and self._last_result_image is not None:
            # file_path = filedialog.asksaveasfile(mode='w', defaultextension='.jpg')
            # if file_path:
            #     self._last_result_image.save(file_path)
            save_path = f"result_{uuid.uuid4().hex}.jpg" # Generates random string trailing name of image
            cv2.imwrite(save_path, self._last_result_image) # saves image in the same directory as main.py
            self.ids.status_label.text = f"Image saved to {save_path}"
        else:
            self.ids.status_label.text = "No result to save."


class ObjectDetectionApp(App):
    def build(self):
        Builder.load_file("main.kv")
        return MainWidget()

if __name__ == '__main__':
    ObjectDetectionApp().run()
