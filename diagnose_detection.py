#!/usr/bin/env python3
"""Diagnose why only one puncture site is being detected"""

import sys
sys.path.append('.')
from main_v3 import detect_papules_red, segment_skin_region, detect_markers
from PIL import Image
import cv2
import numpy as np

# Load baseline image  
img = Image.open('/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg')
bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Run detection
print("Running puncture site detection...")
sites = detect_papules_red(img)
print(f"\nDetected {len(sites)} puncture site(s)")
for i, (x, y) in enumerate(sites, 1):
    print(f"  Site {i}: ({x}, {y})")

# Also check HSV detection directly
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# Find all red contours
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nFound {len(contours)} red contours in HSV mask")

# Filter by area
valid_contours = []
for c in contours:
    area = cv2.contourArea(c)
    if 3 < area < 100:  # Same thresholds as detect_papules_red
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            valid_contours.append((cx, cy, area))
            
print(f"\n{len(valid_contours)} contours meet size criteria (3-100 px):")
for i, (x, y, area) in enumerate(valid_contours[:10], 1):
    print(f"  Contour {i}: ({x}, {y}), area={area:.1f}px")

# Check marker detection
markers, centroid = detect_markers(img)
if centroid:
    print(f"\nMarker centroid at: {centroid}")
    print(f"Found {len(markers)} marker regions")
    
print("\nPossible issues:")
print("- Second puncture site might be <3px or >100px area")
print("- Pairing logic might be rejecting valid pairs") 
print("- Site might not meet HSV threshold (S>100, V>100)")
