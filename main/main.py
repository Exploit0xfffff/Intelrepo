"""
Deep Actions Experimental - Road Object Detection System
A comprehensive application for real-time object detection using state-of-the-art deep learning models.

Key Features:
- Multiple detection models (YOLO and Faster R-CNN)
- Real-time object detection
- Image, video, and live camera processing
- Hardware-optimized performance

This system demonstrates practical applications of computer vision in road safety,
utilizing both traditional (Faster R-CNN) and modern (YOLO) object detection approaches.
"""

# Standard library imports
import os
import sys
import time
import threading
import webbrowser

# Deep learning and computer vision imports
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

# GUI framework imports
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gio, GdkPixbuf, Gdk

# Local imports
from model_factory import ModelFactory

def object_detection_setup(model_type='fasterrcnn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'fasterrcnn':
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    else:
        model = ModelFactory.create_model('yolo')
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model, device

# COCO dataset classes with additional road-specific objects
COCO_CLASSES = [
    # Standard COCO classes
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
    # Additional road-specific classes
    'road', 'pedestrian', 'cyclist', 'traffic sign', 'bicycle lane', 'crosswalk',
    'pavement', 'sidewalk', 'guardrail', 'median', 'bridge', 'tunnel',
    'construction cone', 'construction barrier', 'pothole'
]

def perform_object_detection(model, device, frame, model_type='fasterrcnn'):
    """
    Perform object detection on a single frame using the specified model.

    This function demonstrates the core object detection pipeline:
    1. Image preprocessing - Converting the image to the format required by the model
    2. Model inference - Running the detection model on the preprocessed image
    3. Post-processing - Drawing bounding boxes and labels on detected objects

    Args:
        model: The neural network model (YOLO or Faster R-CNN)
        device: Processing device (CPU or GPU) for model inference
        frame: Input image frame as a NumPy array (BGR format)
        model_type: Type of model being used ('fasterrcnn' or 'yolo')

    Returns:
        tuple: (processed frame with detections drawn, model predictions)

    Example:
        >>> model, device = object_detection_setup('fasterrcnn')
        >>> frame = cv2.imread('road_scene.jpg')
        >>> result_frame, predictions = perform_object_detection(model, device, frame)
    """
    try:
        # Step 1: Prepare the frame for model input
        if model_type == 'fasterrcnn':
            # Convert BGR to RGB and normalize pixel values to [0,1]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.to(device)

            # Run detection with no gradient calculation for efficiency
            with torch.no_grad():
                predictions = model([frame_tensor])
        else:  # YOLO model
            predictions = model.detect(frame)

        # Step 2: Move predictions to CPU for visualization
        predictions = [{k: v.to(torch.device('cpu')) for k, v in pred.items()} for pred in predictions]

        # Step 3: Draw detection boxes and labels
        frame_with_objects = frame.copy()
        num_objects = 0

        for pred in predictions:
            labels = pred['labels'].numpy()
            scores = pred['scores'].numpy()
            boxes = pred['boxes'].numpy().astype(int)

            # Draw each detected object with confidence > 0.5
            for label, score, box in zip(labels, scores, boxes):
                if score > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box
                    # Draw green rectangle around object
                    cv2.rectangle(frame_with_objects, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Get class name and ensure label is within bounds
                    class_name = COCO_CLASSES[int(label) % len(COCO_CLASSES)]
                    # Add text label with class name and confidence score
                    cv2.putText(frame_with_objects, f"{class_name} {score:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    num_objects += 1

        print(f"Detected {num_objects} objects using {model_type}")
        return frame_with_objects, predictions

    except Exception as e:
        print(f"Error in object detection: {str(e)}")
        return frame.copy(), []

"""
Thread Management and Media Processing Components
This section contains the core classes and functions for handling:
1. Multi-threaded video capture
2. Frame processing
3. GUI updates
4. Image and video file processing
"""

class FrameCaptureThread(threading.Thread):
    """
    Thread dedicated to continuous frame capture from camera or video input.

    This class demonstrates multi-threading best practices:
    - Separate I/O operations (frame capture) from processing
    - Safe thread termination with flag checking
    - Resource cleanup on completion

    Usage Example:
        cap = cv2.VideoCapture(0)  # Open default camera
        capture_thread = FrameCaptureThread(cap)
        capture_thread.start()
        # ... use capture_thread.frame to access latest frame ...
        capture_thread.stop()
        capture_thread.join()
    """
    def __init__(self, cap):
        """Initialize the capture thread with a video capture object"""
        super().__init__()
        self.cap = cap  # VideoCapture object
        self.frame = None  # Latest captured frame
        self.running = True  # Thread control flag

    def run(self):
        """Main thread loop for continuous frame capture"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def stop(self):
        """Safely stop the capture thread and release resources"""
        self.running = False
        self.cap.release()

class UpdateFrameThread(threading.Thread):
    """
    Thread for updating the GUI with processed frames.

    This class demonstrates:
    - GUI update threading (keeping UI responsive)
    - Safe thread termination
    - Separation of concerns (display vs processing)
    """
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.running = True

    def run(self):
        while self.running:
            self.window.update_frame()

    def stop(self):
        self.running = False

def detect_objects_in_image(input_image_path, output_image_path, model_type='fasterrcnn'):
    """
    Process a single image file for object detection.

    This function demonstrates the complete image processing pipeline:
    1. Model initialization
    2. Image loading and preprocessing
    3. Object detection
    4. Result visualization and saving

    Args:
        input_image_path: Path to the input image file
        output_image_path: Path where the processed image will be saved
        model_type: Detection model to use ('fasterrcnn' or 'yolo')

    Example:
        >>> detect_objects_in_image('road.jpg', 'road_detected.jpg', 'yolo')
    """
    try:
        print(f"Initializing {model_type} model for image detection...")
        model, device = object_detection_setup(model_type)

        frame = cv2.imread(input_image_path)
        if frame is None:
            print("Error: Unable to read the input image.")
            return

        print(f"Processing image with {model_type} model...")
        result = perform_object_detection(model, device, frame, model_type)
        if result is not None:
            frame_with_objects, predictions = result
            cv2.imwrite(output_image_path, frame_with_objects)

            # Print detection summary
            num_objects = sum(1 for pred in predictions for score in pred['scores'] if score > 0.5)
            print(f"\nDetection complete:")
            print(f"- Model used: {model_type}")
            print(f"- Objects detected: {num_objects}")
            print(f"- Output saved to: {output_image_path}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")

class livecapture(Gtk.ApplicationWindow):
    """
    Window for real-time object detection from camera feed.

    This class demonstrates:
    1. Real-time video processing
    2. FPS monitoring and display
    3. GTK window management
    4. Multi-threaded GUI application design

    The window shows:
    - Live camera feed with object detection
    - FPS counter
    - Detection boxes and labels
    """
    def __init__(self, application, capture_thread, model_type='fasterrcnn'):
        super().__init__(application=application)
        self.set_title(f"DeepActionsExperimental - {model_type.upper()}")
        self.set_default_size(640, 480)

        self.capture_thread = capture_thread
        self.model_type = model_type
        print(f"Initializing live capture with {model_type} model...")

        try:
            self.model, self.device = object_detection_setup(model_type)
            print(f"Model initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

        self.image = Gtk.Image()
        self.set_child(self.image)

        # Add FPS counter
        self.last_frame_time = time.time()
        self.fps_count = 0
        self.fps = 0

        self.update_frame_thread = UpdateFrameThread(self)
        self.update_frame_thread.start()

        self.set_resizable(True)
        self.set_decorated(True)

    def update_frame(self):
        frame = self.capture_thread.frame
        if frame is None:
            return True

        # Calculate FPS
        current_time = time.time()
        self.fps_count += 1
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.fps_count
            self.fps_count = 0
            self.last_frame_time = current_time
            print(f"FPS: {self.fps}")

        try:
            result = perform_object_detection(self.model, self.device, frame, self.model_type)
            if result is not None:
                frame_with_objects, predictions = result

                # Add FPS counter to frame
                cv2.putText(
                    frame_with_objects,
                    f"FPS: {self.fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                pixbuf = self.numpy_to_pixbuf(frame_with_objects)
                self.image.set_from_pixbuf(pixbuf)
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")

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

def detect_objects_in_video(input_video_path, output_video_path, model_type='fasterrcnn'):
    """Process a video file for object detection"""
    try:
        print(f"Initializing {model_type} model for video detection...")
        model, device = object_detection_setup(model_type)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("Error: Unable to open the input video.")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        print(f"Processing video with {model_type} model...")
        print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")

        start_time = time.time()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                result = perform_object_detection(model, device, frame, model_type)
                if result is not None:
                    frame_with_objects, predictions = result
                    out.write(frame_with_objects)
                    frame_count += 1
                    if frame_count % 30 == 0:  # Progress update every 30 frames
                        progress = (frame_count / total_frames) * 100
                        elapsed_time = time.time() - start_time
                        fps_current = frame_count / elapsed_time
                        print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) | FPS: {fps_current:.1f}")
            else:
                break

        end_time = time.time()
        processing_time = end_time - start_time
        average_fps = frame_count / processing_time

        cap.release()
        out.release()

        print(f"\nProcessing complete:")
        print(f"- Model used: {model_type}")
        print(f"- Total frames processed: {frame_count}")
        print(f"- Processing time: {processing_time:.2f} seconds")
        print(f"- Average FPS: {average_fps:.2f}")
        print(f"- Output saved to: {output_video_path}")

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()

class MainWindow(Gtk.ApplicationWindow):
    """
    Main application window with model selection and capture options.

    This window provides a user interface for:
    - Selecting between YOLO and Faster R-CNN models
    - Processing images for object detection
    - Running live camera detection
    - Processing video files
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_capture_window = None

        self.stack = Gtk.Stack()
        self.set_child(self.stack)

        # Main Page with model selection
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.stack.add_named(main_box, "main_page")

        # Model selector container
        selector_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        selector_box.set_margin_start(10)
        selector_box.set_margin_end(10)
        selector_box.set_margin_top(10)

        # Model selector label and combo box
        model_label = Gtk.Label(label="Detection Model:")
        self.model_selector = Gtk.ComboBoxText()
        self.model_selector.append_text("Faster R-CNN")
        self.model_selector.append_text("YOLO")
        self.model_selector.set_active(0)

        selector_box.append(model_label)
        selector_box.append(self.model_selector)
        main_box.append(selector_box)

        # Grid for buttons
        self.grid = Gtk.Grid()
        self.grid.set_margin_top(10)
        main_box.append(self.grid)

        # Connect model selector change signal
        self.model_selector.connect('changed', self.on_model_changed)

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

            # Update grid position to start after model selector
            self.grid.attach(btn, i % 3, (i // 3) + 1, 1, 1)

            btn.set_size_request(button_size, button_size)
            btn.set_hexpand(True)
            btn.set_vexpand(True)
            btn.set_halign(Gtk.Align.CENTER)
            btn.set_valign(Gtk.Align.CENTER)

    def on_LiveCaptureStop_clicked(self, widget):
        """Stop the live capture process and clean up resources"""
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
        """
        Handle image file selection and processing.
        Opens a file chooser dialog and processes the selected image using
        the current detection model.
        """
        filechooser = Gtk.FileChooserDialog(title="Open image", parent=self, action=Gtk.FileChooserAction.OPEN)
        filechooser.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        filechooser.add_button("_Open", Gtk.ResponseType.ACCEPT)
        filechooser.show()
        def on_response(dialog, response_id):
            if response_id == Gtk.ResponseType.ACCEPT:
                input_image_path = filechooser.get_file().get_path()
                output_image_path = os.path.splitext(input_image_path)[0] + "_output.png"
                print(f"Processing image: {input_image_path}")
                model_type = self.model_selector.get_active_text().lower().replace(' ', '')
                detect_objects_in_image(input_image_path, output_image_path, model_type)
                print(f"Output image saved to {output_image_path}")
            filechooser.hide()
        filechooser.connect("response", on_response)

    def on_livecapture_clicked(self, widget):
        """
        Initialize live camera capture with object detection.
        Opens the default camera (index 0) and starts real-time detection
        using the selected model.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open the camera.")
            return

        capture_thread = FrameCaptureThread(cap)
        capture_thread.start()

        app = Gtk.Application.new("org.DeepActionExperimental.GtkApplication", Gio.ApplicationFlags.FLAGS_NONE)
        model_type = self.model_selector.get_active_text().lower().replace(' ', '')
        win = livecapture(app, capture_thread, model_type)
        self.live_capture_window = win  # Save the reference to the live capture window
        win.present()
        app.run()

    def on_videocapture_clicked(self, widget):
        """
        Handle video file selection and processing.
        Opens a file chooser dialog and processes the selected video using
        the current detection model.
        """
        filechooser = Gtk.FileChooserDialog(title="Open Video", parent=self, action=Gtk.FileChooserAction.OPEN)
        filechooser.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        filechooser.add_button("_Open", Gtk.ResponseType.ACCEPT)

        def on_response(dialog, response_id):
            if response_id == Gtk.ResponseType.ACCEPT:
                input_video_path = filechooser.get_file().get_path()
                output_video_path = os.path.splitext(input_video_path)[0] + "_output.mp4"
                print(f"Processing video: {input_video_path}")
                model_type = self.model_selector.get_active_text().lower().replace(' ', '')
                detect_objects_in_video(input_video_path, output_video_path, model_type)
                print(f"Output video saved to {output_video_path}")

            filechooser.hide()

        filechooser.connect("response", on_response)
        filechooser.show()

    def on_exit_clicked(self, widget):
        """Clean up resources and exit the application"""
        self.get_application().quit()

    def on_contactus_clicked(self, widget):
        """Open default email client for support contact"""
        webbrowser.open('mailto:exploit0xffff@gmail.com')

class ObjectDetection(Gtk.Application):
    """
    Main application class that initializes the GTK application.

    This class handles:
    - Application initialization
    - Main window creation
    - CSS styling application
    """
    def __init__(self, **kwargs):
        """Initialize the GTK application"""
        super().__init__(**kwargs)
        self.connect('activate', self.on_activate)

    def on_activate(self, app):
        """
        Create and display the main application window.
        Also loads custom CSS styling for the application.
        """
        # Create and show the main window
        self.win = MainWindow(application=app, title="Deep Actions Experimental")
        self.win.present()

        # Load custom CSS styling
        style_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(style_css_path)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

if __name__ == "__main__":
    # Create and run the application
    app = ObjectDetection(application_id='org.PenetrationApp.GtkApplication')
    app.run(sys.argv)