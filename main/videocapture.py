import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import os
import numpy as np

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

input_video_path = "path/to/your/input/video.mp4"
output_video_path = "path/to/your/output/video.mp4"
detect_objects_in_video(input_video_path, output_video_path)
