import cv2
import torch
import torchvision
from torchvision import models
import numpy as np

def satellite():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Object Detection Model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    model.eval()

    # Load class labels for object detection
    class_labels = [
    'background', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'rickshaw', 'motor-rickshaw',
    'traffic cone', 'street sign', 'road barrier', 'wheelbarrow',
    'shopping cart', 'baby stroller', 'tractor', 'helicopter',
    'jet ski', 'airship', 'aircraft carrier', 'hoverboard',
    'scooter', 'tricycle', 'crane', 'bulldozer', 'excavator',
    'forklift', 'golf cart', 'trolley', 'wagon', 'binoculars',
    'umbrella', 'walking stick', 'crutch', 'fishing rod',
    'hammock', 'tent', 'camping stove', 'campfire', 'marshmallow',
    'flashlight', 'compass', 'map', 'backpack', 'suitcase',
    'briefcase', 'wallet', 'purse', 'hand fan', 'umbrella hat',
    'sun hat', 'fedora', 'top hat', 'bicycle helmet',
    'hard hat', 'swim cap', 'bandana', 'headband', 'sunglasses',
    't-shirt', 'pants', 'dress', 'shirt', 'jacket', 'coat',
    'sweater', 'hoodie', 'scarf', 'gloves', 'socks', 'shoes',
    'boots', 'sandals', 'sneakers', 'hat', 'tie', 'belt',
    'glasses', 'watch', 'bracelet', 'earrings', 'ring'
    ]


    # Function to perform object detection on a frame
    def perform_object_detection(frame):
        # Convert the frame to a numpy array
        frame = frame.numpy()

        # Preprocess the frame
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        input_tensor = transform(frame)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # Perform object detection
        with torch.no_grad():
            predictions = model(input_tensor)

        # Move predictions to CPU for further processing
        predictions = [{k: v.to(torch.device('cpu')) for k, v in pred.items()} for pred in predictions]

        # Extract the bounding boxes, labels, and scores
        boxes = predictions[0]['boxes'].numpy().astype(int)
        labels = predictions[0]['labels'].numpy()
        scores = predictions[0]['scores'].numpy()

        # Draw bounding boxes and labels on the frame
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:  # Set a threshold for the confidence score
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_labels[label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

    # Main loop
    cap = cv2.VideoCapture(0)  # Open the camera

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Move frame to GPU for object detection
        frame = torch.from_numpy(frame).to(device)

        # Perform object detection on the frame
        frame_with_objects = perform_object_detection(frame)

        # Display the frame
        cv2.imshow('Satellite-Oriented Object Detection', frame_with_objects)

        # Check for key press to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

# Call the satellite function
satellite()
