import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from scipy.signal import convolve2d
from skimage.feature import peak_local_max

import scipy.io

def create_log_filter(hsize, sigma):
    """
    Create Laplacian of Gaussian (LoG) filter
    Equivalent to MATLAB's fspecial('log', hsize, sigma)
    """
    # Create coordinate arrays
    x = np.arange(-hsize//2. + 1, hsize//2. + 1)
    y = np.arange(-hsize//2. + 1, hsize//2. + 1)
    X, Y = np.meshgrid(x, y)
    
    # Calculate LoG filter
    sigma2 = sigma**2
    kernel = -(1./(np.pi * sigma2**2.)) * (1. - (X**2. + Y**2.)/(2.*sigma2)) * np.exp(-(X**2. + Y**2.)/(2.*sigma2))
    
    # Normalize so that sum is zero (typical for LoG)
    kernel = kernel - np.mean(kernel)
    
    return kernel

def extrema2d(image, mode='max'):
    """
    Find local maxima/minima in 2D image
    Equivalent to extrema2.m function
    """
    # Remove NaN values for processing
    valid_mask = ~np.isnan(image)
    
    if mode == 'max':
        # Find local maxima
        local_max = peak_local_max(image,threshold_abs=0.1)
                                   #  threshold_rel=0.1, exclude_border=True,
                                   #  indices=True, num_peaks=np.inf)
        return local_max.T

    elif mode == 'min':
        # Find local minima by inverting the image
        inverted_image = -image
        local_min = peak_local_max(inverted_image, threshold_abs=0.1)
        return local_min.T

def main():
    """
    Main function for fly detection using blob analysis
    """
    # Clear all and close plots
    plt.close('all')
    
    import cv2

    # Open the video file
    cap = cv2.VideoCapture("flies.webm")

    # get total number of frames
    f_list = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Found {f_list} image files")

    ## Initialize detection storage
    X = []  # Detection X coordinates (as list of arrays)
    Y = []  # Detection Y coordinates (as list of arrays)
    
    ## Process each frame
    for i in range(f_list-30):
        # print(f"Processing frame {i+1}/{f_list}")

        frame, img_tmp = read_frame(cap, i)

        try:
            X_frame, Y_frame = find_objects(img_tmp)

        except Exception as e:
            print(f"Error in extrema detection for frame {i}: {e}")
            X_frame = np.array([])
            Y_frame = np.array([])
        
        # Store detections
        X.append(X_frame)
        Y.append(Y_frame)
        
        # Visualization (optional - uncomment to enable)

        visualise_frame_detection(X_frame, Y_frame, i, img_tmp, frame, )
    
    # Save results in MATLAB format
    # Convert to the same format as MATLAB (cell arrays become object arrays)
    detection_data = {
        'X': np.array(X, dtype=object),
        'Y': np.array(Y, dtype=object)
    }
    
    scipy.io.savemat('raw_fly_detections.mat', detection_data)
    print("Detection complete! Results saved to 'raw_fly_detections.mat'")
    
    # Print summary statistics
    total_detections = sum(len(x) for x in X)
    avg_detections = total_detections / f_list if f_list else 0
    
    print(f"\nSummary:")
    print(f"Total frames processed: {f_list}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {avg_detections:.2f}")

    cap.release()
    return X, Y


def find_objects(img_tmp):
    blob_img_thresh = detect_edges(img_tmp)
    # Find local maxima (blob peaks)
    # Try using the simple extrema finder
    max_coords = extrema2d(blob_img_thresh)
    if len(max_coords[0]) > 0:
        X_frame = max_coords[0]  # Row coordinates
        Y_frame = max_coords[1]  # Column coordinates
    else:
        X_frame = np.array([])
        Y_frame = np.array([])
    return X_frame, Y_frame


def find_objects_canny(img_tmp, canny_low=100, canny_high=200, min_area=15, blur_kernel_size=5):
    """
    Alternative detector using Gaussian blur + Canny edges + contour centroids.
    Returns (X_frame, Y_frame) with row (x) and col (y) coordinates.
    """
    # Convert to grayscale (match detect_edges channel behavior)
    if len(img_tmp.shape) == 3:
        gray = img_tmp[:, :, 0]
    else:
        gray = img_tmp

    # Ensure uint8 for Canny
    if gray.dtype != np.uint8:
        gmin, gmax = float(np.nanmin(gray)), float(np.nanmax(gray))
        if gmax > gmin:
            gray_u8 = ((gray - gmin) / (gmax - gmin) * 255.0).astype(np.uint8)
        else:
            gray_u8 = np.zeros_like(gray, dtype=np.uint8)
    else:
        gray_u8 = gray

    # Apply Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray_u8, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    # Connect edges slightly to form contours
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xs = []
    ys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cy = m["m10"] / m["m00"]  # column
        cx = m["m01"] / m["m00"]  # row
        xs.append(int(round(cx)))
        ys.append(int(round(cy)))

    if len(xs) == 0:
        return np.array([]), np.array([])
    return np.array(xs), np.array(ys)

## Initialize Gaussian filter
# Parameters for LoG filter - you may need to adjust these values
hsizeh = 30  # Filter size - larger values detect larger blobs
sigmah = 6  # Standard deviation - affects blob size sensitivity

# Create LoG filter
h = create_log_filter(hsizeh, sigmah)

def detect_edges(img_tmp):
    # Convert to grayscale (take first channel)
    if len(img_tmp.shape) == 3:
        img = img_tmp[:, :, 0]  # Use first channel
    else:
        img = img_tmp
    # Apply blob filter (LoG convolution)
    blob_img = convolve2d(img, h, mode='same', boundary='symm')
    # Threshold the image - adjust threshold as needed
    # In the original code, values < 0.7 are set to NaN
    threshold = 0.7
    blob_img_thresh = blob_img.copy()
    blob_img_thresh[blob_img < threshold] = 0
    return blob_img_thresh


def read_frame(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    # Read the first frame to confirm reading
    ret, frame = cap.read()
    if not ret:
        raise
    # Load image
    img_tmp = frame.astype(np.float64)  # For processing
    return frame, img_tmp


def visualise_frame_detection(X_frame, Y_frame, i, img, img_real):
    plt.clf()
    # Show original image with detections
    if img_real is not None:
        plt.imshow(cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    # Plot detections
    if len(X_frame) > 0:
        plt.plot(Y_frame, X_frame, 'or', markersize=8, markerfacecolor='red')
    plt.axis('off')
    plt.title(f'Frame {i + 1}: {len(X_frame)} flies detected')
    plt.pause(0.1)


def visualize_filter(h):
    # Visualize the filter
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(h, cmap='jet')
    plt.title('LoG Filter')
    plt.colorbar()
    plt.subplot(122)
    x = np.arange(h.shape[1])
    y = np.arange(h.shape[0])
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, h, cmap='jet')
    ax.set_title('LoG Filter 3D')
    plt.show()


def visualize_detections(X, Y, base_dir=None, frame_indices=None):
    """
    Visualize detection results for specified frames
    """
    if base_dir:
        os.chdir(base_dir)
    
    f_list = sorted(glob.glob('*.jpeg'))
    
    if frame_indices is None:
        frame_indices = range(min(10, len(f_list)))  # Show first 10 frames by default
    
    plt.figure(figsize=(15, 10))
    
    for idx, i in enumerate(frame_indices):
        if i >= len(f_list) or i >= len(X):
            continue
            
        plt.subplot(2, 5, idx + 1)
        
        # Load and display image
        img = cv2.imread(f_list[i])
        if img is not None:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Plot detections
        if len(X[i]) > 0:
            plt.plot(Y[i], X[i], 'or', markersize=6, markerfacecolor='red', alpha=0.7)
        
        plt.axis('off')
        plt.title(f'Frame {i+1}: {len(X[i])} flies')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run main detection
    X, Y = main()
    
    # Optionally visualize some results
    visualize_detections(X, Y, frame_indices=range(0, 20, 2))
