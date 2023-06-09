import cv2
import gi
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import os
import numpy as np
import threading
import time

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def perform_object_detection(model, device, frame):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    frame = frame.to(device)

    with torch.no_grad():
        predictions = model([frame])

    predictions = [{k: v.to(torch.device('cpu')) for k, v in pred.items()} for pred in predictions]

    frame_with_objects = frame.permute(1, 2, 0).mul(255).byte().numpy()
    num_objects = 0
    for pred in predictions:
        labels = pred['labels'].numpy()
        scores = pred['scores'].numpy()
        boxes = pred['boxes'].numpy().astype(int)
        for label, score, box in zip(labels, scores, boxes):
            if score > 0.5:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame_with_objects, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_name = COCO_CLASSES[label]
                cv2.putText(frame_with_objects, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                num_objects += 1
    print(f"Detected {num_objects} objects")
    return frame_with_objects, predictions
def object_detection_setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91, pretrained_backbone=True)
    weights = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).state_dict()
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    return model, device

class FrameCaptureThread(threading.Thread):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def stop(self):
        self.running = False

class UpdateFrameThread(threading.Thread):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.running = True

    def run(self):
        while self.running:
            self.window.update_frame()
            time.sleep(0.01)  # Sleep for 10 milliseconds

    def stop(self):
        self.running = False

class ObjectDetectionWindow(Gtk.ApplicationWindow):
    def __init__(self, application):
        super().__init__(application=application)
        self.set_title("Road Object Detection")
        self.set_default_size(640, 480)

        self.model, self.device = object_detection_setup()

        self.image = Gtk.Image()
        self.set_child(self.image)

        self.update_frame_thread = UpdateFrameThread(self)
        self.update_frame_thread.start()

    def update_frame(self):
        frame = self.capture_thread.frame
        if frame is None:
            return True

        print(f"Frame shape: {frame.shape}")

        result = perform_object_detection(self.model, self.device, frame)
        if result is not None:
            frame_with_objects, _ = result
            pixbuf = self.numpy_to_pixbuf(frame_with_objects)
            self.image.set_from_pixbuf(pixbuf)

        return True

    def numpy_to_pixbuf(self, frame):
        height, width, channels = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return GdkPixbuf.Pixbuf.new_from_data(
            frame.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            width,
            height,
            width * channels,
        )

    def do_destroy(self):
        self.update_frame_thread.stop()
        self.update_frame_thread.join()

class ObjectDetectionApp(Gtk.Application):
    def __init__(self):
        super().__init__()

    def do_startup(self):
        Gtk.Application.do_startup(self)

    def do_activate(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Unable to open the camera.")
            return

        self.capture_thread = FrameCaptureThread(self.cap)
        self.capture_thread.start()

        win = ObjectDetectionWindow(self)
        win.capture_thread = self.capture_thread
        win.connect("destroy", self.on_destroy)
        win.present()

    def on_destroy(self, window):
        window.capture_thread.stop()
        window.capture_thread.join()
        window.cap.release()
        self.quit()

app = ObjectDetectionApp()
app.run(None)
