import time

import numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from fly_detect import find_objects, read_frame, find_objects_canny


def assignment_optimal(cost_matrix):
    """
    Hungarian algorithm for optimal assignment
    Returns assignment indices and total cost
    """
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_indices, col_indices].sum()

    # Create assignment array (1-indexed like MATLAB, 0 means no assignment)
    assignment = np.zeros(cost_matrix.shape[0], dtype=int)
    for i, j in zip(row_indices, col_indices):
        assignment[i] = j + 1  # +1 for MATLAB-style 1-indexing

    return assignment, total_cost


def initialize_kalman_filter():
    """
    Initialize OpenCV Kalman filter for 2D tracking.
    """
    kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, vx, vy), 2 measurements (x, y)
    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman


def detect_objects(img_bgr):
    """
    Run detection on input image and return Nx2 array of [x, y] detections.
    """
    try:
        # sigma = 0.33
        # v = np.median(img_bgr)
        # # apply automatic Canny edge detection using the computed median
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))

        current_x, current_y = find_objects_canny(img_bgr)
        if len(current_x) > 0:
            return np.column_stack([current_x, current_y])
        return np.array([]).reshape(0, 2)
    except Exception:
        return np.array([]).reshape(0, 2)


def predict_tracks(kalman_filters):
    """
    Run Kalman predict for all existing tracks and return Nx2 predictions.
    """
    if len(kalman_filters) == 0:
        return np.array([]).reshape(0, 2)
    predictions = []
    for kalman in kalman_filters:
        prediction = kalman.predict()
        predictions.append(prediction[:2].flatten())
    return np.array(predictions)


def match_detections_to_tracks(predictions, detections, distance_threshold=50):
    """
    Compute optimal assignment using Hungarian algorithm and reject large distances.
    Returns assignments array of length N_pred:
      - value > 0: 1-indexed detection id assigned to prediction index
      - value == -1: rejected due to distance threshold
      - value == 0: no assignment (e.g., no detections/predictions)
    """
    if len(detections) == 0 or len(predictions) == 0:
        return np.zeros(len(predictions), dtype=int)

    distances = squareform(pdist(np.vstack([predictions, detections])))
    cost_matrix = distances[:len(predictions), len(predictions):]
    assignments, _ = assignment_optimal(cost_matrix)

    # Reject assignments with large distances
    for i, assignment in enumerate(assignments):
        if assignment > 0:
            detection_idx = assignment - 1
            if cost_matrix[i, detection_idx] > distance_threshold:
                assignments[i] = -1
    return assignments


def update_tracks_with_assignments(kalman_filters, detections, assignments):
    """
    Correct Kalman filters with assigned detections.
    """
    for i, kalman in enumerate(kalman_filters):
        if i < len(assignments) and assignments[i] > 0:
            detection_idx = assignments[i] - 1
            measurement = detections[detection_idx].astype(np.float32)
            kalman.correct(measurement)


def calculate_strikes(track_strikes, assignments):
    """
    Update strike counters based on assignment results.
    - reset to 0 on successful assignment (>0)
    - increment when rejected (-1)
    - unchanged when 0 (no decision)
    """
    for i in range(len(track_strikes)):
        if i < len(assignments):
            if assignments[i] > 0:
                track_strikes[i] = 0
            elif assignments[i] == -1:
                track_strikes[i] += 1


def init_new_tracks_for_unmatched(kalman_filters, track_strikes, detections, assignments):
    """
    Initialize new tracks for detections that are not assigned to any track.
    """
    assigned_detection_ids = set(a - 1 for a in assignments if a > 0)
    for idx in range(len(detections)):
        if idx not in assigned_detection_ids:
            kalman = initialize_kalman_filter()
            kalman.statePre = np.array([[detections[idx][0]], [detections[idx][1]], [0], [0]], np.float32)
            kalman.statePost = np.array([[detections[idx][0]], [detections[idx][1]], [0], [0]], np.float32)
            kalman_filters.append(kalman)
            track_strikes.append(0)


def prune_lost_tracks(kalman_filters, track_strikes, max_strikes=6):
    """
    Remove tracks whose strike count exceeds max_strikes.
    """
    for i in range(len(track_strikes) - 1, -1, -1):
        if track_strikes[i] > max_strikes:
            del kalman_filters[i]
            del track_strikes[i]


def draw_tracks_on_frame(frame, Q_loc_estimateX, Q_loc_estimateY, frame_idx, track_colors):
    """
    Draw bounding boxes, ids, and short trails for all valid track positions on the frame.
    Modifies frame in place and also returns it for convenience.
    """
    center_x = Q_loc_estimateX[frame_idx,:]
    center_y = Q_loc_estimateY[frame_idx,:]
    for track_idx in range(Q_loc_estimateX.shape[1]):
        color = track_colors[track_idx%len(track_colors)]
        if np.isnan(center_x[track_idx]) or np.isnan(center_y[track_idx]):
            continue

        x = int(center_x[track_idx])
        y = int(center_y[track_idx])
        tl = (y - 10, x - 10)
        br = (y + 10, x + 10)
        cv2.rectangle(frame, tl, br, color, 1)
        cv2.putText(frame, str(track_idx), (y - 20, x - 10), 0, 0.5, color, 2)
        for k in range(10):
            prev_idx = frame_idx - k
            if prev_idx < 0:
                break
            x1 = Q_loc_estimateX[prev_idx, track_idx]
            y1 = Q_loc_estimateY[prev_idx, track_idx]
            if np.isnan(x1) or np.isnan(y1):
                break
            x1 = int(x1)
            y1 = int(y1)
            cv2.circle(frame, (y1, x1), 3, color, 1)
        cv2.circle(frame, (y, x), 6, color, 1)
    return frame


def main():
    # Clear all variables and close plots
    plt.close('all')

    # Open the video file
    video_path = "flies.webm"
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames
    f_list = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Found {f_list} frames in the video.")

    ## Initialize Kalman filters for multiple tracks
    kalman_filters = []
    track_strikes = []
    MAX_TRACKS = 2000

    ## Initialize result variables
    Q_loc_estimateX = np.full((f_list, MAX_TRACKS), np.nan)
    Q_loc_estimateY = np.full((f_list, MAX_TRACKS), np.nan)

    MAX_FRAME = 50
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (127, 127, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127), (127, 10, 255), (0, 255, 127)]
    for t in range(5, MAX_FRAME):
        print(f"Processing frame {t}")

        # Read the current frame
        try:
            frame, img_tmp = read_frame(cap, t)
        except Exception as e:
            print(f"Error reading frame {t}: {e}")
            continue

        # Detect objects in the current frame
        detections = detect_objects(img_tmp)

        # Kalman Filter Prediction
        predictions = predict_tracks(kalman_filters)

        ## Assignment of detections to tracks
        assignments = match_detections_to_tracks(predictions, detections, distance_threshold=50)

        # Update Kalman filters and strike counters based on assignments
        update_tracks_with_assignments(kalman_filters, detections, assignments)
        calculate_strikes(track_strikes, assignments)

        ## Handle new detections and unmatched tracks
        init_new_tracks_for_unmatched(kalman_filters, track_strikes, detections, assignments)

        # Remove tracks with too many strikes
        prune_lost_tracks(kalman_filters, track_strikes, max_strikes=6)

        # Store data
        for i, kalman in enumerate(kalman_filters):
            Q_loc_estimateX[t, i] = kalman.statePost[0].item()  # Extract scalar
            Q_loc_estimateY[t, i] = kalman.statePost[1].item()  # Extract scalar

        # Live visualization as we go
        try:
            vis_frame = frame.copy()
            vis_frame = draw_tracks_on_frame(vis_frame, Q_loc_estimateX, Q_loc_estimateY, t, track_colors)
            cv2.imshow('image', vis_frame)
            time.sleep(0.03)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        except Exception:
            pass

    print("Tracking complete!")

    while True:
        for frame_idx in range(5, MAX_FRAME):
            frame, img_tmp = read_frame(cap, frame_idx)
            frame = draw_tracks_on_frame(frame, Q_loc_estimateX, Q_loc_estimateY, frame_idx, track_colors)
            cv2.imshow('image', frame)
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    return Q_loc_estimateX, Q_loc_estimateY



if __name__ == "__main__":
    main()