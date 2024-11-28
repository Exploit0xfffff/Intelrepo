import sys
import os
import cv2
import torch
import numpy as np
import time
import pytest
from pathlib import Path

# Add parent directory to path to import main modules
sys.path.append(str(Path(__file__).parent.parent))
from main.model_factory import ModelFactory
from main.yolo_detector import YOLODetector

@pytest.fixture(params=['fasterrcnn', 'yolo'])
def model_type(request):
    return request.param

def test_model_performance(model_type):
    """Test performance of object detection models"""
    print(f"\n=== Testing {model_type.upper()} Model ===")

    # Initialize model
    try:
        model, device = ModelFactory.create_model(model_type)
        print(f"✓ Model initialization successful on {device}")
    except Exception as e:
        print(f"✗ Model initialization failed: {str(e)}")
        return False

    # Create a test image
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (500, 500), (255, 255, 255), -1)

    # Warm-up run
    print("Performing warm-up inference...")
    _ = model.detect(test_image) if model_type == 'yolo' else model([torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0])

    # Performance test
    print("\nRunning performance test...")
    num_iterations = 10
    total_time = 0

    for i in range(num_iterations):
        start_time = time.time()
        if model_type == 'yolo':
            predictions = model.detect(test_image)
        else:
            frame_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0
            predictions = model([frame_tensor.to(device)])

        inference_time = time.time() - start_time
        total_time += inference_time
        print(f"Iteration {i+1}/{num_iterations}: {inference_time:.3f}s")

    avg_time = total_time / num_iterations
    avg_fps = 1 / avg_time

    print(f"\nPerformance Results:")
    print(f"- Average inference time: {avg_time:.3f}s")
    print(f"- Average FPS: {avg_fps:.2f}")
    print(f"- Device: {device}")
    if device.type == 'cuda':
        print(f"- GPU Memory Used: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

    assert avg_fps > 0, "FPS should be greater than 0"
    return True

def main():
    print("=== Deep Actions Experimental Model Tests ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    models = ['fasterrcnn', 'yolo']
    results = {}

    for model_type in models:
        results[model_type] = test_model_performance(model_type)

    print("\n=== Test Summary ===")
    for model_type, success in results.items():
        print(f"{model_type}: {'✓ PASSED' if success else '✗ FAILED'}")

if __name__ == "__main__":
    main()    export PYTHONPATH=$PYTHONPATH:/home/kasinadhsarma/Documents/Intelrepo-main
    pytest