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


def main():
    # Clear all variables and close plots
    plt.close('all')

    # Open the video file
    video_path = "flies.webm"
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames
    f_list = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Found {f_list} frames in the video.")

    ## Define main variables for KALMAN FILTER
    dt = 1  # sampling rate
    S_frame = 5  # starting frame

    # Define process and measurement parameters
    u = 0  # acceleration magnitude
    HexAccel_noise_mag = 1  # process noise magnitude
    tkn_x = 0.1  # measurement noise in x direction
    tkn_y = 0.1  # measurement noise in y direction

    # Measurement noise covariance matrix
    Ez = np.array([[tkn_x, 0], [0, tkn_y]])

    # Process noise covariance matrix
    Ex = np.array([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2]
    ]) * HexAccel_noise_mag**2

    P = Ex.copy()  # initial position variance estimate

    ## Define update equations in 2D
    # State transition matrix
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Control input matrix
    B = np.array([[dt**2/2], [dt**2/2], [dt], [dt]])

    # Measurement function matrix
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    ## Initialize result variables
    Q_loc_meas = []  # fly detections

    ## Initialize estimation variables
    Q_estimate = np.full((4, 2000), np.nan)
    Q_loc_estimateX = np.full((f_list, 2000), np.nan)
    Q_loc_estimateY = np.full((f_list, 2000), np.nan)

    P_estimate = P.copy()
    strk_trks = np.zeros(2000)  # strike counter for tracks

    nF = 0  # number of track estimates

    MAX_FRAME = 100
    for t in range(S_frame, MAX_FRAME):
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
                Q_loc_meas = np.column_stack([current_x, current_y])
                nD = len(current_x)
            else:
                Q_loc_meas = np.array([]).reshape(0, 2)
                nD = 0
        except Exception as e:
            print(f"Error detecting objects in frame {t}: {e}")
            Q_loc_meas = np.array([]).reshape(0, 2)
            nD = 0

        ## Kalman Filter Prediction
        # Predict next state for all active tracks
        for F in range(nF):
            if not np.isnan(Q_estimate[0, F]):
                Q_estimate[:, F] = A @ Q_estimate[:, F] + (B * u).flatten()

        # Predict next covariance
        P = A @ P @ A.T + Ex

        # Kalman Gain
        K = P @ C.T @ np.linalg.inv(C @ P @ C.T + Ez)

        ## Assignment of detections to tracks
        if nD > 0 and nF > 0:
            # Create distance matrix
            active_tracks = []
            active_positions = []

            for F in range(nF):
                if not np.isnan(Q_estimate[0, F]):
                    active_tracks.append(F)
                    active_positions.append([Q_estimate[0, F], Q_estimate[1, F]])

            if len(active_positions) > 0:
                active_positions = np.array(active_positions)

                # Calculate distances between tracks and detections
                all_points = np.vstack([active_positions, Q_loc_meas])
                distances = squareform(pdist(all_points))
                est_dist = distances[:len(active_positions), len(active_positions):]

                # Optimal assignment
                asgn, cost = assignment_optimal(est_dist)

                # Reject assignments that are too far (distance > 50)
                rej = np.zeros(len(asgn), dtype=bool)
                for i, assignment in enumerate(asgn):
                    if assignment > 0:
                        if est_dist[i, assignment-1] < 50:  # -1 for 0-indexing
                            rej[i] = True

                asgn = asgn * rej.astype(int)

                # Apply updates
                for i, (F, assignment) in enumerate(zip(active_tracks, asgn)):
                    if assignment > 0:
                        measurement = Q_loc_meas[assignment-1, :]  # -1 for 0-indexing
                        innovation = measurement - C @ Q_estimate[:, F]
                        Q_estimate[:, F] = Q_estimate[:, F] + K @ innovation
        else:
            asgn = np.array([])

        # Update covariance
        P = (np.eye(4) - K @ C) @ P

        ## Store data
        Q_loc_estimateX[t, :nF] = Q_estimate[0, :nF]
        Q_loc_estimateY[t, :nF] = Q_estimate[1, :nF]

        ## Handle new detections and lost tracks
        if nD > 0:
            # Find unassigned detections (new tracks)
            if len(asgn) > 0:
                assigned_detections = asgn[asgn > 0] - 1  # Convert to 0-indexing
                unassigned = []
                for i in range(nD):
                    if i not in assigned_detections:
                        unassigned.append(i)
            else:
                unassigned = list(range(nD))

            # Create new tracks for unassigned detections
            if unassigned:
                new_positions = Q_loc_meas[unassigned, :]
                n_new = len(unassigned)

                # Add new tracks
                Q_estimate[0, nF:nF+n_new] = new_positions[:, 0]
                Q_estimate[1, nF:nF+n_new] = new_positions[:, 1]
                Q_estimate[2, nF:nF+n_new] = 0  # zero velocity
                Q_estimate[3, nF:nF+n_new] = 0  # zero velocity

                nF += n_new

        # Give strikes to unmatched tracks
        if len(asgn) > 0:
            no_match = np.where(asgn == 0)[0]
            if len(no_match) > 0:
                active_track_indices = [i for i in range(min(nF, len(strk_trks)))
                                      if not np.isnan(Q_estimate[0, i])]
                for idx in no_match:
                    if idx < len(active_track_indices):
                        track_idx = active_track_indices[idx]
                        strk_trks[track_idx] += 1

        # Remove tracks with too many strikes
        bad_tracks = np.where(strk_trks > 6)[0]
        if len(bad_tracks) > 0:
            Q_estimate[:, bad_tracks] = np.nan

    # Save results
    results = {
        'Q_loc_estimateX': Q_loc_estimateX,
        'Q_loc_estimateY': Q_loc_estimateY
    }

    print("Tracking complete!")


    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (127, 127, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127), (127, 10, 255), (0, 255, 127)]
    while True:
        for frame_idx in range(S_frame, MAX_FRAME):
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