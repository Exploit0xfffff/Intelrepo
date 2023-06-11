## Title: Penetration App -  cyber security  in one base

⬇️ **Download**

```
    git clone https://github.com/Exploit0xfffff/Deep-Actions-Experimental
    cd Deep-Actions-Experimental
    pip install -r requirements.txt
    cd main/
```

Code Description and Future Features :

This Python script, `main.py`, implements real-time object detection using a pre-trained Faster R-CNN model. It captures frames from a video source, performs object detection on each frame, and displays the processed frames with bounding boxes around detected objects in a graphical user interface (GUI) window. The script is built using OpenCV, TorchVision, and GTK.

The main features of the current code include:

1. Real-time object detection: The script utilizes a pre-trained Faster R-CNN model with a ResNet-50 backbone to perform object detection on video frames. Detected objects are visualized with bounding boxes and labels.

2. Multi-threading: Two custom thread classes, `FrameCaptureThread` and `UpdateFrameThread`, enable efficient frame capturing and continuous updating of the GUI window.

3. GUI window: The `ObjectDetectionWindow` class extends the GTK `Gtk.ApplicationWindow` and displays the video stream with detected objects. It utilizes the GTK `Gtk.Image` widget to render frames and the `numpy_to_pixbuf` method to convert frames from numpy arrays to `GdkPixbuf.Pixbuf` objects.

Future Development:

1. Menu System: Implement a menu system to provide additional functionality and options to the user. For example, menus could include options for selecting different object detection models, adjusting detection thresholds, and enabling/disabling specific object classes.

2. Home Object View: Enhance the object detection functionality to include a "Home Object View" mode. In this mode, the script could utilize camera feeds from multiple sources (e.g., security cameras) and display the detected objects in a centralized dashboard or grid view. This feature would provide a comprehensive overview of the detected objects across different camera feeds.

3. GPS View Integration: Integrate Google Maps or other GPS services to overlay the detected objects' locations on a map. This integration would provide geographical context to the detected objects and enable users to visualize their distribution across different areas.

4. Mobile Connectivity: Develop a companion mobile application that can connect to the script running on a remote machine. The mobile app could display the video stream and detected objects, allowing users to monitor the object detection in real-time from their smartphones or tablets.

By adding these future features, the script can be expanded into a more comprehensive object detection and monitoring system with enhanced user interaction and geographical visualization capabilities.

before ->

![OIP](https://github.com/Exploit0xfffff/Deep-Actions-Experimental/assets/81065703/ae960bdc-79f1-4bd7-8659-ab774d0c8684)


after ->

![image](https://github.com/Exploit0xfffff/Deep-Actions-Experimental/assets/81065703/05cdb353-4d81-4b92-93f5-126dd941052c)

before ->

https://github.com/Exploit0xfffff/Deep-Actions-Experimental/assets/81065703/d8a39a03-2458-4f28-9099-15a227669734


after ->

https://github.com/Exploit0xfffff/Deep-Actions-Experimental/assets/81065703/f0bf5d72-f4e9-4728-8ff6-5cc8d4c02ae8
