#!/usr/bin/env python3
import sys
sys.path.append('.')
from PIL import Image
import cv2
import numpy as np

img = Image.open('/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg')
bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# Same thresholds
mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
red_mask = cv2.bitwise_or(mask1, mask2)

contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} total red contours")

# Check ALL sizes
valid = []
for c in contours:
    area = cv2.contourArea(c)
    if area > 0:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            valid.append((cx, cy, area))

valid.sort(key=lambda x: x[2], reverse=True)  # Sort by area

print(f"\nTop 10 red regions by area:")
for i, (x, y, area) in enumerate(valid[:10]):
    print(f"  {i+1}. Position: ({x}, {y}), Area: {area:.1f} px")
    
# Check around the marker area specifically
marker_x, marker_y = 362, 462  # From logs
print(f"\nRed regions within 200px of marker ({marker_x}, {marker_y}):")
near_marker = []
for x, y, area in valid:
    dist = ((x - marker_x)**2 + (y - marker_y)**2)**0.5
    if dist < 200:
        near_marker.append((x, y, area, dist))
        
near_marker.sort(key=lambda x: x[3])  # Sort by distance
for x, y, area, dist in near_marker[:5]:
    print(f"  Position: ({x}, {y}), Area: {area:.1f} px, Distance: {dist:.1f} px")
