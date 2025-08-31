import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from fly_detect import find_objects, read_frame


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
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman


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
    for t in range(5, MAX_FRAME):
        print(f"Processing frame {t}")

        # Read the current frame
        try:
            frame, img_tmp = read_frame(cap, t)
        except Exception as e:
            print(f"Error reading frame {t}: {e}")
            continue

        # Detect objects in the current frame
        try:
            current_x, current_y = find_objects(img_tmp)
            if len(current_x) > 0:
                detections = np.column_stack([current_x, current_y])
            else:
                detections = np.array([]).reshape(0, 2)
        except Exception as e:
            print(f"Error detecting objects in frame {t}: {e}")
            detections = np.array([]).reshape(0, 2)

        # Kalman Filter Prediction
        predictions = []
        for i, kalman in enumerate(kalman_filters):
            prediction = kalman.predict()
            predictions.append(prediction[:2].flatten())  # Ensure predictions are 2D

        predictions = np.array(predictions)

        ## Assignment of detections to tracks
        if len(detections) > 0 and len(predictions) > 0:
            # Create distance matrix
            distances = squareform(pdist(np.vstack([predictions, detections])))
            cost_matrix = distances[:len(predictions), len(predictions):]

            # Optimal assignment
            assignments, _ = assignment_optimal(cost_matrix)

            # # Reject assignments with large distances
            distance_threshold = 50
            for i, assignment in enumerate(assignments):
                if assignment > 0:  # Check if the track is assigned
                    detection_idx = assignment - 1  # Convert to 0-indexing
                    if cost_matrix[i, detection_idx] > distance_threshold:
                        assignments[i] = 0

            # Update Kalman filters with assigned detections
            for i, kalman in enumerate(kalman_filters):
                if assignments[i] > 0:
                    detection_idx = assignments[i] - 1  # Convert to 0-indexing
                    measurement = detections[detection_idx].astype(np.float32)
                    kalman.correct(measurement)
                    track_strikes[i] = 0  # Reset strike counter
                else:
                    track_strikes[i] += 1  # Increment strike counter for unmatched tracks
        else:
            assignments = np.zeros(len(predictions), dtype=int)

        ## Handle new detections and unmatched tracks
        unmatched_detections = [i for i in range(len(detections)) if i + 1 not in assignments]
        for idx in unmatched_detections:
            kalman = initialize_kalman_filter()
            kalman.statePre = np.array([[detections[idx][0]], [detections[idx][1]], [0], [0]], np.float32)
            kalman.statePost = np.array([[detections[idx][0]], [detections[idx][1]], [0], [0]], np.float32)
            kalman_filters.append(kalman)
            track_strikes.append(0)

        # Remove tracks with too many strikes
        for i in range(len(track_strikes) - 1, -1, -1):
            if track_strikes[i] > 6:
                del kalman_filters[i]
                del track_strikes[i]

        # Store data
        for i, kalman in enumerate(kalman_filters):
            Q_loc_estimateX[t, i] = kalman.statePost[0].item()  # Extract scalar
            Q_loc_estimateY[t, i] = kalman.statePost[1].item()  # Extract scalar

    print("Tracking complete!")

    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (127, 127, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127), (127, 10, 255), (0, 255, 127)]
    while True:
        for frame_idx in range(5, MAX_FRAME):
            center_x = Q_loc_estimateX[frame_idx,:]
            center_y = Q_loc_estimateY[frame_idx,:]
            frame, img_tmp = read_frame(cap, frame_idx)


            for track_idx in range(Q_loc_estimateX.shape[1]):
                color = track_colors[track_idx%len(track_colors)]
                if np.isnan(center_x[track_idx]) or np.isnan(center_y[track_idx]):
                    continue

                x = int(center_x[track_idx])
                y = int(center_y[track_idx])
                tl = (y - 10, x - 10)
                br = (y + 10, x + 10)
                cv2.rectangle(frame, tl, br, color, 1)
                cv2.putText(frame, str(track_idx), (y - 20, x - 10), 0, 0.5,color, 2)
                for k in range(10):
                    x1 = Q_loc_estimateX[frame_idx - k, track_idx]
                    y1 = Q_loc_estimateY[frame_idx - k, track_idx]
                    if np.isnan(x1) or np.isnan(y1):
                        break

                    x1 = int(x1)
                    y1 = int(y1)

                    cv2.circle(frame, (y1, x1), 3, color, 1)
                cv2.circle(frame, (y, x), 6, color, 1)

            cv2.imshow('image', frame)
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    return Q_loc_estimateX, Q_loc_estimateY



if __name__ == "__main__":
    main()