from yolo_detector import YOLODetector
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
import pytest

class TestYOLODetector:
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        with patch('yolo_detector.YOLO') as mock_yolo:
            # Create a mock YOLO model
            self.mock_model = Mock()
            self.mock_model.names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'}
            mock_yolo.return_value = self.mock_model
            
            self.detector = YOLODetector(model_path="./models/yolov7-e6e.pt")
            self.mock_yolo = mock_yolo
            yield

    def test_detect_object_with_valid_input_under_mock(self):
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        expected_detections = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.95, 'class': 'person'},
            {'bbox': [300, 300, 400, 400], 'confidence': 0.87, 'class': 'car'},
        ]

        # Mock the model inference results
        mock_result = Mock()
        mock_box1 = Mock()
        mock_box1.xyxy = [torch.tensor([100, 100, 200, 200])]
        mock_box1.conf = [torch.tensor(0.95)]
        mock_box1.cls = [torch.tensor(0)]
        
        mock_box2 = Mock()
        mock_box2.xyxy = [torch.tensor([300, 300, 400, 400])]
        mock_box2.conf = [torch.tensor(0.87)]
        mock_box2.cls = [torch.tensor(2)]
        
        mock_boxes = Mock()
        mock_boxes.__iter__ = Mock(return_value=iter([mock_box1, mock_box2]))
        
        mock_result.boxes = mock_boxes
        self.mock_model.return_value = [mock_result]

        result = self.detector.detect_objects(test_image)

        assert len(result) == len(expected_detections), 'Wrong number of bounded boxes'
        assert result[0]['bbox'] == expected_detections[0]['bbox'], 'First bbox mismatch'
        assert abs(result[0]['confidence'] - expected_detections[0]['confidence']) < 0.001, 'First confidence mismatch'
        assert result[0]['class'] == expected_detections[0]['class'], 'First class mismatch'
        assert result[1]['bbox'] == expected_detections[1]['bbox'], 'Second bbox mismatch'
        assert abs(result[1]['confidence'] - expected_detections[1]['confidence']) < 0.001, 'Second confidence mismatch'
        assert result[1]['class'] == expected_detections[1]['class'], 'Second class mismatch'
        print("Test passed! Detections:", result)

    def test_detectobject_with_empty_image(self):
        # Reset the mock model to return an empty list (no detections)
        self.mock_model.return_value = []
        
        empty_image = np.array([])
        with pytest.raises(ValueError, match="Image cannot be empty"):
            self.detector.detect_objects(empty_image)

    def test_detect_object_with_no_detection(self):
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        self.mock_model.return_value = []

        result = self.detector.detect_objects(test_image)

        assert result == []
        assert len(result) == 0

