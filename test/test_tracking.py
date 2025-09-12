import numpy as np
from math import isclose
from fly_track_rewrite import (
    detect_objects,
    predict_tracks,
    match_detections_to_tracks,
    update_tracks_with_assignments,
    calculate_strikes,
    init_new_tracks_for_unmatched,
    prune_lost_tracks,
    initialize_kalman_filter,
)


def test_detect_objects_monkeypatched(monkeypatch):
    # Force deterministic output by monkeypatching find_objects used in module
    def fake_find_objects(_img):
        return np.array([10, 20]), np.array([30, 40])  # Xs, Ys

    import fly_track_rewrite as mod

    monkeypatch.setattr(mod, "find_objects", fake_find_objects)
    detections = detect_objects(np.zeros((50, 50, 3), dtype=np.uint8))
    assert detections.shape == (2, 2)
    assert np.array_equal(detections, np.array([[10, 30], [20, 40]]))


def test_predict_tracks_empty():
    preds = predict_tracks([])
    assert preds.shape == (0, 2)


def test_predict_tracks_with_state():
    kf = initialize_kalman_filter()
    # state: x=10, y=20, vx=1, vy=2
    kf.statePre = np.array([[10], [20], [1], [2]], dtype=np.float32)
    kf.statePost = kf.statePre.copy()
    preds = predict_tracks([kf])
    assert preds.shape == (1, 2)
    # After one predict: x' = 11, y' = 22
    assert isclose(float(preds[0, 0]), 11.0, rel_tol=1e-4, abs_tol=1e-3)
    assert isclose(float(preds[0, 1]), 22.0, rel_tol=1e-4, abs_tol=1e-3)


def test_match_detections_to_tracks_basic():
    predictions = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    detections = np.array([[1.0, 1.0], [102.0, 98.0]], dtype=np.float32)
    assignments = match_detections_to_tracks(predictions, detections, distance_threshold=50)
    # Expect track0->det0 (1), track1->det1 (2)
    assert assignments.tolist() == [1, 2]


def test_match_detections_to_tracks_threshold_rejects():
    predictions = np.array([[0.0, 0.0]], dtype=np.float32)
    detections = np.array([[100.0, 100.0]], dtype=np.float32)
    assignments = match_detections_to_tracks(predictions, detections, distance_threshold=10)
    # Should be rejected as -1
    assert assignments.tolist() == [-1]


def test_update_and_strikes():
    # Two tracks
    kf1 = initialize_kalman_filter()
    kf2 = initialize_kalman_filter()
    # Initialize around (0,0) and (100,100)
    kf1.statePre = np.array([[0], [0], [0], [0]], dtype=np.float32)
    kf1.statePost = kf1.statePre.copy()
    kf2.statePre = np.array([[100], [100], [0], [0]], dtype=np.float32)
    kf2.statePost = kf2.statePre.copy()

    kalmans = [kf1, kf2]
    detections = np.array([[1.0, 1.0]], dtype=np.float32)  # only one detection
    # assignments: first gets det0 (1), second rejected (-1)
    assignments = np.array([1, -1], dtype=int)
    strikes = [0, 0]

    # Capture pre-correction distance to detection for track 0
    x0, y0 = float(kalmans[0].statePost[0]), float(kalmans[0].statePost[1])
    d0_x = abs(x0 - 1.0)
    d0_y = abs(y0 - 1.0)

    predict_tracks(kalmans)
    update_tracks_with_assignments(kalmans, detections, assignments)
    calculate_strikes(strikes, assignments)

    # After correction, state should move closer to the detection than before
    x1, y1 = float(kalmans[0].statePost[0]), float(kalmans[0].statePost[1])
    d1_x = abs(x1 - 1.0)
    d1_y = abs(y1 - 1.0)
    assert d1_x < d0_x  and d1_y < d0_y
    # Second accumulated strike
    assert strikes == [0, 1]


def test_init_new_tracks_for_unmatched():
    kalmans = []
    strikes = []
    detections = np.array([[5.0, 6.0], [50.0, 60.0]], dtype=np.float32)
    # No assignments yet
    assignments = np.array([], dtype=int)

    init_new_tracks_for_unmatched(kalmans, strikes, detections, assignments)
    assert len(kalmans) == 2 and len(strikes) == 2
    # Check first track initialized near detection
    x0, y0 = float(kalmans[0].statePost[0]), float(kalmans[0].statePost[1])
    assert isclose(x0, 5.0, abs_tol=1e-5) and isclose(y0, 6.0, abs_tol=1e-5)


def test_prune_lost_tracks():
    k1 = initialize_kalman_filter()
    k2 = initialize_kalman_filter()
    kalmans = [k1, k2]
    strikes = [0, 7]
    prune_lost_tracks(kalmans, strikes, max_strikes=6)
    assert len(kalmans) == 1 and len(strikes) == 1

