import cv2
import gi
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import os
import numpy as np
import threading
import time
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib,Gio
import sys
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
#video capture
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

def detect_objects_in_video(input_video_path, output_video_path):
    model, device = object_detection_setup()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open the input video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = perform_object_detection(model, device, frame)
        if result is not None:
            frame_with_objects, _ = result
            out.write(frame_with_objects)

    cap.release()
    out.release()
    print(f"Output video saved to {output_video_path}")
#image capture 
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

def detect_objects_in_image(input_image_path, output_image_path):
    model, device = object_detection_setup()

    frame = cv2.imread(input_image_path)
    if frame is None:
        print("Error: Unable to read the input image.")
        return

    result = perform_object_detection(model, device, frame)
    if result is not None:
        frame_with_objects, _ = result
        cv2.imwrite(output_image_path, frame_with_objects)
        print(f"Output image saved to {output_image_path}")

#live capture 
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

    def stop(self):
        self.running = False
class ObjectDetectionExperimental(Gtk.ApplicationWindow):
    def __init__(self, app):
        Gtk.ApplicationWindow.__init__(self, application=app)
        self.set_default_size(800, 600)

        header = Gtk.HeaderBar()
        title_label = Gtk.Label.new("Deep Action Experimental")
        header.set_title_widget(title_label)
        header.set_show_title_buttons(True)
        self.set_titlebar(header)

        menu = Gio.Menu.new()
        menu.append("Home", "app.Home")
        menu.append("Image Capture", "app.imagecapture")
        menu.append("Live Capture", "app.livecapture")
        menu.append("Video Capture", "app.videocapture")
        menu.append("Quit", "app.quit")

        popover = Gtk.PopoverMenu.new_from_model(menu)
        hamburger = Gtk.MenuButton.new()
        hamburger.set_popover(popover)
        hamburger.set_icon_name("open-menu-symbolic")
        header.pack_start(hamburger)

        home_action = Gio.SimpleAction.new("Home", None)
        home_action.connect("activate", self.on_Home)
        app.add_action(home_action)

        imagecapture_action = Gio.SimpleAction.new("imagecapture", None)
        imagecapture_action.connect("activate", self.on_imagecapture)
        app.add_action(imagecapture_action)

        livecapture_action = Gio.SimpleAction.new("livecapture", None)
        livecapture_action.connect("activate", self.on_livecapture)
        app.add_action(livecapture_action)

        videocapture_action = Gio.SimpleAction.new("videocapture", None)
        videocapture_action.connect("activate", self.on_videocapture)
        app.add_action(videocapture_action)

        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", self.on_quit)
        app.add_action(quit_action)


    def on_Home(self, action, parameter):
        print("Home")
    def on_imagecapture(self, action, parameter):
        filechooser = Gtk.FileChooserDialog(title="Open image", parent=self, action=Gtk.FileChooserAction.OPEN)
        filechooser.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        filechooser.add_button("_Open", Gtk.ResponseType.ACCEPT)

        filechooser.show()

        def on_response(dialog, response_id):
            if response_id == Gtk.ResponseType.ACCEPT:
                input_image_path = filechooser.get_file().get_path()
                output_image_path = os.path.splitext(input_image_path)[0] + "_output.png"
                print(f"Processing image: {input_image_path}")
                detect_objects_in_image(input_image_path, output_image_path)
                print(f"Output image saved to {output_image_path}")

            filechooser.hide()

        filechooser.connect("response", on_response)
    def on_livecapture(self, action, parameter):
        capture_thread = FrameCaptureThread
        self.capture_thread = capture_thread

        self.model, self.device = object_detection_setup()

        self.image = Gtk.Image()
        self.set_child(self.image)

        self.update_frame_thread = UpdateFrameThread(self)
        self.update_frame_thread.start()

        # Make the window resizable and add maximize, minimize, and close buttons
        self.set_resizable(True)
        self.set_decorated(True)
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

    def on_videocapture(self, action, parameter):
        filechooser = Gtk.FileChooserDialog(title="Open Video", parent=self, action=Gtk.FileChooserAction.OPEN)
        filechooser.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        filechooser.add_button("_Open", Gtk.ResponseType.ACCEPT)

        filechooser.show()

        def on_response(dialog, response_id):
            if response_id == Gtk.ResponseType.ACCEPT:
                input_video_path = filechooser.get_file().get_path()
                output_video_path = os.path.splitext(input_video_path)[0] + "_output.mp4"
                print(f"Processing video: {input_video_path}")
                detect_objects_in_video(input_video_path, output_video_path)
                print(f"Output video saved to {output_video_path}")

            filechooser.hide()

        filechooser.connect("response", on_response)


    def on_quit(self, action, parameter):
        self.get_application().quit()


class ObjectDetection(Gtk.Application):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect('activate', self.on_activate)

    def on_activate(self, app):
        self.win = ObjectDetectionExperimental(app)
        self.win.present()


if __name__ == "__main__":
    app = ObjectDetection(application_id='org.DeepActionExperimental.GtkApplication')
    app.run(sys.argv)
