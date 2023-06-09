import cv2
import gi
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import os
import numpy as np
import threading

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

def perform_object_detection(model, device, frame):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() /255.0
    frame = frame.to(device)

    with torch.no_grad():
        predictions = model([frame])

    predictions = [{k: v.to(torch.device('cpu')) for k, v in pred.items()} for pred in predictions]

    frame_with_objects = frame.permute(1, 2, 0).mul(255).byte().numpy()
    for pred in predictions:
        labels = pred['labels'].numpy()
        scores = pred['scores'].numpy()
        boxes = pred['boxes'].numpy().astype(int)
        for label, score, box in zip(labels, scores, boxes):
            if score > 0.5:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame_with_objects, (1, y1), (x2, y2), (0, 255,0), 2)
    return frame_with_objects, predictions

def object_detection():
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

class ObjectDetectionWindow(Gtk.ApplicationWindow):
    def __init__(self, application):
        super().__init__(application=application)
        self.set_title("Satellite-Oriented Object Detection")
        self.set_default_size(640, 480)

        self.model, self.device = object_detection_setup()

        self.image = Gtk.Image()
        self.set_child(self.image)

        self.timeout_id = GLib.timeout_add(30, self.update_frame)

    def update_frame(self):
        frame = self.capture_thread.frame
        if frame is None:
            return True

        frame_with_objects, _ = perform_object_detection(self.model, self.device, frame)
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
        GLib.source_remove(self.timeout_id)

    def do_show(self):
        Gtk.Widget.do_show_all(self)

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
        win.show()

    def on_destroy(self, window):
        window.capture_thread.stop()
        window.capture_thread.join()
        window.cap.release()
        self.quit()

app = ObjectDetectionApp()
app.run(None)
