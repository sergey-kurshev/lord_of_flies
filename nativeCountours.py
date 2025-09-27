import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def read_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        raise Exception(f"Failed to read frame {frame_number}")
    return frame

video_path = "flies.webm"
video_path = "fly1.mp4"
cap = cv2.VideoCapture(video_path)
f_list = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame = read_frame(cap, 0)

# DoG
Gsc = cv2.GaussianBlur(frame, (0,0), sigmaX=1)
Gss = cv2.GaussianBlur(frame, (0,0), sigmaX=2)
dog = Gsc - Gss

# Sobel / Prewitt filters
Sx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
Sy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
Smag = np.sqrt(Sx**2 + Sy**2)

# Prewitt with custom kernel
Kx = np.array([
    [-1, 0, 1],
    [-1, 0, 1], 
    [-1, 0, 1]
])
Ky = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])
Px = cv2.filter2D(frame, cv2.CV_32F, Kx)
Py = cv2.filter2D(frame, cv2.CV_32F, Ky)
Pmag = np.sqrt(Px**2 + Py**2)

edges = cv2.Canny(frame, 100, 200)

# Define simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 13 * 13, 10)  # for 28x28 input

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.softmax(self.fc1(x), dim=1)
        return x

# Instantiate model
model = SimpleCNN()

# Access filters from first conv layer
filters = model.conv1.weight.data.clone()  # shape [out_channels, in_channels, H, W]

# Normalize filters to 0–1 for visualization
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

#plt.figure(figsize=(8,4))
#for i in range(filters.shape[0]):  # out_channels
#    plt.subplot(2, 4, i+1)
#    plt.imshow(filters[i,0,:,:].cpu().numpy(), cmap="gray")
#    plt.axis("off")
#plt.suptitle("First Conv Layer Filters (untrained)")
#plt.show()

img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
overlay = frame.copy()
cv2.drawContours(overlay, contours, -1, (0,255,0),1)

plt.figure(figsize=(14, 8)) 
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.title("Contours on Original")
plt.show()
