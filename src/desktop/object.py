import cv2
import torch
import torchvision
from torchvision import models

def object():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Object Detection Model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    model.eval()

    # Open the file to append the detected objects
    file = open("dataset/train.txt", "a")

    # Function to perform object detection on a frame
    def perform_object_detection(frame):
        # Move the frame to the CPU
        frame = frame.cpu()

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

        # Write detected objects' IDs to the file
        for pred in predictions:
            labels = pred['labels'].numpy()
            scores = pred['scores'].numpy()
            boxes = pred['boxes'].numpy().astype(int)
            for label, score, box in zip(labels, scores, boxes):
                if score > 0.5:  # Set a threshold for the confidence score
                    file.write(f"item {{\n  id: {label}\n}}\n")

        # Draw bounding boxes on the frame
        for pred in predictions:
            labels = pred['labels'].numpy()
            scores = pred['scores'].numpy()
            boxes = pred['boxes'].numpy().astype(int)
            for label, score, box in zip(labels, scores, boxes):
                if score > 0.5:  # Set a threshold for the confidence score
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

    # Close the file
    file.close()

# Call the object function
object()
