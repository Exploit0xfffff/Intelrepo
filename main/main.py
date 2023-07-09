import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import threading
import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gio, GdkPixbuf, Gdk
import sys
import os
import time
import webbrowser
# Define your classes here...
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
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'road', 'pedestrian', 'cyclist', 'traffic sign', 'bicycle lane', 'crosswalk',
    'pavement', 'sidewalk', 'guardrail', 'median', 'bridge', 'tunnel',
    'construction cone', 'construction barrier', 'pothole'
]

# Function to perform object detection goes here...
def perform_object_detection(model, device, frame):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    frame = frame.to(device)

    with torch.no_grad():
        predictions = model([frame])

    predictions = [{k: v.to(torch.device('cpu')) for k, v in pred.items()} for pred in predictions]

    frame_with_objects = frame.permute(1, 2, 0).mul(255).byte().cpu().numpy()
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
                cv2.putText(
                    frame_with_objects,
                    class_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                num_objects += 1
    print(f"Detected {num_objects} objects")
    return frame_with_objects, predictions
# Function to setup object detection goes here...
def object_detection_setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model = model.to(device)
    model.eval()

    return model, device

# The threading classes go here...
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
        self.cap.release()# Release the camera
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
#imagecapture
def detect_objects_in_image(inumpyut_image_path, output_image_path):
    model, device = object_detection_setup()

    frame = cv2.imread(inumpyut_image_path)
    if frame is None:
        print("Error: Unable to read the inumpyut image.")
        return

    result = perform_object_detection(model, device, frame)
    if result is not None:
        frame_with_objects, _ = result
        cv2.imwrite(output_image_path, frame_with_objects)
        print(f"Output image saved to {output_image_path}")
# The LiveCapture class goes here...
class livecapture(Gtk.ApplicationWindow):
    def __init__(self, application, capture_thread):
        super().__init__(application=application)
        self.set_title("DeepActionsExperimental")
        self.set_default_size(640, 480)

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

    def do_delete_event(self, event):
        self.update_frame_thread.stop()
        self.update_frame_thread.join()
        self.capture_thread.stop()  # Stop the capture thread
        self.capture_thread.join()
        self.get_application().quit()  # End the GTK application
        return False

#videocapture
def detect_objects_in_video(input_video_path, output_video_path):
    model, device = object_detection_setup()

    cap = cv2.VideoCapture(input_video_path)

    # Get video properties for output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or replace 'mp4v' with 'MP4V'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not cap.isOpened():
        print("Error: Unable to open the input video.")
        return

    print("Started capturing...")
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            result = perform_object_detection(model, device, frame)
            if result is not None:
                frame_with_objects, _ = result
                out.write(frame_with_objects)
        else:
            break
    end_time = time.time()
    print("Ended capturing.")

    cap.release()
    out.release()

    print(f"Output video saved to {output_video_path}")
    print(f"Time taken for capturing: {end_time - start_time} seconds")
    print(f"Frames per second: {fps}")

# Define your classes here...
class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stack = Gtk.Stack()
        self.set_child(self.stack)

        # Main Page
        self.grid = Gtk.Grid()
        self.stack.add_named(self.grid, "main_page")

        # Buttons and pages
        buttons_methods = {
            "Image Capture": self.on_imagecapture_clicked, 
            "Live Capture": self.on_livecapture_clicked, 
            "Live Capture Stop": self.on_LiveCaptureStop_clicked, 
            "Video Capture": self.on_videocapture_clicked, 
            "Exit": self.on_exit_clicked,
            "Contact Us": self.on_contactus_clicked
        }

        button_images = {
            "Image Capture": "img/imagecapture.jpg",
            "Live Capture": "img/Live.png",
            "Live Capture Stop": "img/LiveStop.png",
            "Video Capture": "img/videocapture.png",
            "Exit": "img/exit.png",
            "Contact Us": "img/contact-us.jpg"
        }

        # Button size
        button_size = 300

        for i, button in enumerate(buttons_methods.keys()):
            # Create a button with no label
            btn = Gtk.Button()

            # Set cursor
            btn.set_cursor_from_name('pointer')

            # Load image with Pixbuf and scale it
            img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), button_images[button])
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(img_path)
            scaled_pixbuf = pixbuf.scale_simple(button_size, button_size, GdkPixbuf.InterpType.BILINEAR)

            # Create an image widget and set the scaled pixbuf to it
            img = Gtk.Image.new_from_pixbuf(scaled_pixbuf)

            # Set image as the child of the button
            btn.set_child(img)

            btn.get_style_context().add_class("custom-button")
            btn.get_style_context().add_class("custom-square-button")
            btn.set_name(button.replace(" ", "-").lower())  # Set button id for CSS styling
            btn.connect("clicked", buttons_methods[button])

            self.grid.attach(btn, i % 3, i // 3, 1, 1)

            btn.set_size_request(button_size, button_size)
            btn.set_hexpand(True)
            btn.set_vexpand(True)
            btn.set_halign(Gtk.Align.CENTER)
            btn.set_valign(Gtk.Align.CENTER)
    def on_LiveCaptureStop_clicked(self, widget):
        if self.live_capture_window is not None:
            # Stop the capture thread and join it
            self.live_capture_window.capture_thread.stop()
            self.live_capture_window.capture_thread.join()
            
            # Stop the update frame thread and join it
            self.live_capture_window.update_frame_thread.stop()
            self.live_capture_window.update_frame_thread.join()
            
            # Destroy the window
            self.live_capture_window.destroy()
            
            # Reset the window reference
            self.live_capture_window = None

    def on_imagecapture_clicked(self, widget):
        filechooser = Gtk.FileChooserDialog(title="Open image", parent=self, action=Gtk.FileChooserAction.OPEN)
        filechooser.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        filechooser.add_button("_Open", Gtk.ResponseType.ACCEPT)
        filechooser.show()
        def on_response(dialog, response_id):
            if response_id == Gtk.ResponseType.ACCEPT:
                inumpyut_image_path = filechooser.get_file().get_path()
                output_image_path = os.path.splitext(inumpyut_image_path)[0] + "_output.png"
                print(f"Processing image: {inumpyut_image_path}")
                detect_objects_in_image(inumpyut_image_path, output_image_path)
                print(f"Output image saved to {output_image_path}")
            filechooser.hide()
        filechooser.connect("response", on_response)
    def on_livecapture_clicked(self, widget):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open the camera.")
            return

        capture_thread = FrameCaptureThread(cap)
        capture_thread.start()

        app = Gtk.Application.new("org.DeepActionExperimental.GtkApplication", Gio.ApplicationFlags.FLAGS_NONE)
        win = livecapture(app, capture_thread)
        self.live_capture_window = win  # Save the reference to the live capture window
        win.present()
        app.run()

    def on_videocapture_clicked(self, widget):
        filechooser = Gtk.FileChooserDialog(title="Open Video", parent=self, action=Gtk.FileChooserAction.OPEN)
        filechooser.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        filechooser.add_button("_Open", Gtk.ResponseType.ACCEPT)

        def on_response(dialog, response_id):
            if response_id == Gtk.ResponseType.ACCEPT:
                input_video_path = filechooser.get_file().get_path()
                output_video_path = os.path.splitext(input_video_path)[0] + "_output.mp4"
                print(f"Processing video: {input_video_path}")
                detect_objects_in_video(input_video_path, output_video_path)
                print(f"Output video saved to {output_video_path}")

            filechooser.hide()

        filechooser.connect("response", on_response)
        filechooser.show()
    def on_exit_clicked(self, widget):
        self.get_application().quit()

    def on_contactus_clicked(self, widget):
        webbrowser.open('mailto:exploit0xffff@gmail.com')

class ObjectDetection(Gtk.Application):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect('activate', self.on_activate)

    def on_activate(self, app):
        self.win = MainWindow(application=app, title="Deep Actions Experimental")
        self.win.present()

        # Add the following code to load the CSS file:
        style_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(style_css_path)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

if __name__ == "__main__":
    app = ObjectDetection(application_id='org.PenetrationApp.GtkApplication')
    app.run(sys.argv)
