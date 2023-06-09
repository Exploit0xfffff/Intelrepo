# Deep Actions Experimental
The project aims to develop a real-time object detection application using live camera feed and the GTK 4.0 library. The application captures video frames from a connected camera, processes them using a custom-trained TensorFlow model, and displays the processed frames with object detection annotations in a GTK window.

The main Desktop components of the project are as follows:

1. **LiveCameraWindow**: This class represents the main GTK window where the live camera feed and object detection results will be displayed. It initializes the window, sets its title and size, and creates a Gtk.Image widget to show the processed frames. The `update_frame` method is responsible for continuously capturing frames from the camera, processing them, and updating the Gtk.Image widget with the processed frames.

2. **load_custom_model**: This function loads a custom-trained TensorFlow model from a specified model path. The model should be trained for object detection, and its path needs to be provided to the function.

3. **process_frame**: This method is a placeholder for the custom object detection logic using the custom-trained TensorFlow model. You need to implement your own object detection algorithm within this method to detect objects in the frames. The method takes a frame as input and should return the processed frame with the object detection annotations.

In the main section of the code, the video capture object (`cap`) is initialized to capture frames from the default camera (index 0). The path to the custom-trained TensorFlow model (`custom_model_path`) is provided, and the model is loaded using the `load_custom_model` function.

Then, a GTK application is created, and the `activate` signal is connected to a lambda function that creates an instance of the `LiveCameraWindow` class, passing the video capture object and the custom model as arguments. The GTK application runs, displaying the main window with the live camera feed and object detection results.

Once the GTK application is terminated, the video capture object is released to free the camera resources.

To use this code for your project, you need to implement the `process_frame` method with your custom object detection logic based on your trained TensorFlow model. Additionally, you should ensure that you have the necessary dependencies installed, such as OpenCV, TensorFlow, and GTK 4.0. 

