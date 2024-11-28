"""
Model Factory for Object Detection
================================

This module provides a factory class for creating and optimizing object detection models.
It demonstrates key concepts in deep learning model management:
1. Hardware detection and optimization
2. Model initialization and configuration
3. Memory management
4. Performance optimization techniques

The factory supports multiple model architectures (Faster R-CNN and YOLO) and
automatically configures them for optimal performance on available hardware.
"""

import torch
import gc
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from main.yolo_detector import YOLODetector

class ModelFactory:
    """
    Factory class for creating optimized object detection models.

    This class demonstrates the Factory design pattern, which provides a clean interface
    for creating complex objects (in this case, deep learning models) while hiding the
    complexity of their initialization and configuration.
    """

    @staticmethod
    def create_model(model_type='fasterrcnn', quantize=True, batch_size=4):
        """
        Create and configure an object detection model based on specified parameters.

        This method demonstrates several important deep learning concepts:
        1. Memory Management:
           - Garbage collection
           - CUDA cache clearing
        2. Hardware Detection:
           - GPU availability checking
           - CPU thread optimization
        3. Model Optimization:
           - Quantization for CPU
           - CUDA benchmarking for GPU

        Args:
            model_type (str): Type of model to create ('fasterrcnn' or 'yolo')
            quantize (bool): Whether to apply quantization on CPU
            batch_size (int): Batch size for YOLO model

        Returns:
            tuple: (model, device) where model is the initialized neural network
                  and device is the torch device (CPU/GPU) to run it on

        Example:
            >>> model, device = ModelFactory.create_model('yolo', quantize=True)
            >>> print(f"Model created on device: {device}")
        """
        # Step 1: Memory Management
        # Clear unused memory to ensure smooth model loading
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Step 2: Hardware Detection and Setup
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            device = torch.device('cpu')
            print("Using CPU for inference")
            # Optimize CPU thread count for better performance
            torch.set_num_threads(min(torch.get_num_threads(), 4))
            print(f"Using {torch.get_num_threads()} CPU threads")

        # Step 3: Model Initialization
        # Create the requested model with pre-trained weights
        if model_type == 'fasterrcnn':
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        elif model_type == 'yolo':
            model = YOLODetector(batch_size=batch_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Step 4: Device Configuration
        # Move model to appropriate device and set to evaluation mode
        model = model.to(device)
        model.eval()  # Set model to evaluation mode (disables training-specific operations)

        # Step 5: Model Optimization
        # Apply quantization for CPU models to reduce memory usage and improve speed
        if quantize and device.type == 'cpu':
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )

        # Enable CUDA optimization if using GPU
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        return model, device