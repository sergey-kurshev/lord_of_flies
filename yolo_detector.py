"""
To-do
- 
"""

from ultralytics import YOLO
import os
from typing import Tuple, Optional
import torch
import numpy as np
import cv2

class YOLODetector:
    def __init__(self, model_name: str, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        try:
            self.model = YOLO(model_name)  # This will download if not available
        except:
            self.model = YOLO('yolov8n.pt')  # This will download if not available

    def detect_objects(self, image: np.ndarray) -> list:
        """
        Detect objects in the given image using the YOLO model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries with 'bbox', 'confidence', and 'class' keys
        """
        if self.model is None:
            raise Exception("Model not loaded")
            
        # Validate input image
        if image is None or image.size == 0:
            raise ValueError("Image cannot be empty")
            
        # Run inference
        results = self.model(image)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.model.names[class_id]
                    
                    if confidence >= self.confidence_threshold:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class': class_name
                        })
        
        return detections

