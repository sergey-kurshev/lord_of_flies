import cv2
import numpy as np
from fly_detect import find_objects, extrema2d, detect_edges, find_objects_canny


def test_detect_edges():
    # Load the test frame
    img = cv2.imread('test/frame.jpg', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "Failed to load test/frame.jpg"

    # Call detect_edges
    edges = detect_edges(img)

    # Check output shape matches input
    assert edges.shape== img.shape[:2] , "Output shape mismatch"

    # Check if edges contain non-zero values
    assert np.any(edges > 0), "No edges detected"

    print("test_detect_edges passed!")

def test_extrema2d():
    # Create a synthetic test image with peaks
    img = np.zeros((100, 100))
    img[50, 50] = 1  # Add a peak at the center
    img[20, 20] = 0.8  # Add another peak

    # Call extrema2d
    max_coords = extrema2d(img, mode='max')

    # Check if the peaks are detected
    assert (50, 50) in zip(max_coords[0], max_coords[1]), "Center peak not detected"
    assert (20, 20) in zip(max_coords[0], max_coords[1]), "Corner peak not detected"

    print("test_extrema2d passed!")

def test_find_objects():
    # Load the test frame
    img = cv2.imread('test/frame.jpg', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "Failed to load test/frame.jpg"

    # Call find_objects
    X, Y = find_objects(img)

    # Check if outputs are arrays
    assert isinstance(X, np.ndarray), "X is not a numpy array"
    assert isinstance(Y, np.ndarray), "Y is not a numpy array"

    # Check if detections are within image bounds
    assert np.all(X >= 0) and np.all(X < img.shape[0]), "X coordinates out of bounds"
    assert np.all(Y >= 0) and np.all(Y < img.shape[1]), "Y coordinates out of bounds"

    print("test_find_objects passed!")


def test_find_objects_canny_synthetic():
    # Create simple synthetic image with two rectangles
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (20, 20), 255, -1)
    cv2.rectangle(img, (60, 60), (80, 80), 255, -1)

    X, Y = find_objects_canny(img, canny_low=50, canny_high=150, min_area=10)

    # Should detect two components roughly centered
    assert len(X) == len(Y) and len(X) >= 2
    # Check centers are near expected
    centers = list(zip(X.tolist(), Y.tolist()))
    # Expect around (15,15) and (70,70)
    assert any(abs(x-15) <= 3 and abs(y-15) <= 3 for x, y in centers)
    assert any(abs(x-70) <= 5 and abs(y-70) <= 5 for x, y in centers)
    print("test_find_objects_canny_synthetic passed!")
