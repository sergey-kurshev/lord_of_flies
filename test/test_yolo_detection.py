import pytest
import numpy as np
import cv2
import torch
from pathlib import Path
from yolo_detector import YOLODetector

class TestYoloDetection:
    @pytest.fixture(scope="class")
    def detector(self):
        detector = YOLODetector(model_path = "./models/yolov7-e6e.pt")
        return detector

    def test_detect_objects(self, detector):
        # Load image in color mode (3 channels) for YOLO
        img = cv2.imread('test/frame.jpg', cv2.IMREAD_COLOR)
        if img is None:
            # Create a dummy image if the test image doesn't exist
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Ensure the image has 3 channels (RGB) for YOLO
        if len(img.shape) == 2:
            # Convert grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            # Convert RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        assert img is not None, "Failed to load or create test image"
        assert len(img.shape) == 3 and img.shape[2] == 3, "Image must have 3 channels (RGB)"

        objects = detector.detect_objects(img)
        
        # Verify the results - we expect a list of detections
        assert isinstance(objects, list), "Expected list of detections"
        
        # Print results for debugging
        print(f"Found {len(objects)} detections:")
        for i, obj in enumerate(objects):
            print(f"Detection {i+1}: {obj}")
        
        # Basic validation of detection format
        for obj in objects:
            assert 'bbox' in obj, "Detection missing 'bbox' key"
            assert 'confidence' in obj, "Detection missing 'confidence' key"
            assert 'class' in obj, "Detection missing 'class' key"
            assert len(obj['bbox']) == 4, "Bbox should have 4 coordinates"
            assert 0 <= obj['confidence'] <= 1, "Confidence should be between 0 and 1"

