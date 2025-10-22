#!/usr/bin/env python3
"""Visualize what the detection algorithm found."""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Load the cropped baseline image from pipeline
baseline_cropped = cv2.imread("output_debug/intermediate_baseline_precrop.jpg")
h, w = baseline_cropped.shape[:2]

print(f"Baseline precrop size: {w}×{h}")

# Apply shared region crop (from debug log: top=135)
shared_top = 135
baseline_final = baseline_cropped[shared_top:, :]
h_final, w_final = baseline_final.shape[:2]

print(f"After shared crop (top={shared_top}): {w_final}×{h_final}")

# Convert to PIL for drawing
img_pil = Image.fromarray(cv2.cvtColor(baseline_final, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)

# From debug log:
marker_pos = (363, 458)
contour1_pos = (398, 421)
contour2_pos = (383, 407)

# Draw marker (large green circle)
draw.ellipse([marker_pos[0]-30, marker_pos[1]-30, marker_pos[0]+30, marker_pos[1]+30],
             outline='lime', width=5)
draw.text((marker_pos[0]+35, marker_pos[1]-15), "MARKER", fill='lime')

# Draw detected contours (red circles)
draw.ellipse([contour1_pos[0]-15, contour1_pos[1]-15, contour1_pos[0]+15, contour1_pos[1]+15],
             outline='red', width=4)
draw.text((contour1_pos[0]+20, contour1_pos[1]-10), "Contour1", fill='red')

draw.ellipse([contour2_pos[0]-15, contour2_pos[1]-15, contour2_pos[0]+15, contour2_pos[1]+15],
             outline='red', width=4)
draw.text((contour2_pos[0]+20, contour2_pos[1]-10), "Contour2", fill='red')

# Draw ROI box (from debug log: 420×420 around (398, 421))
roi_center = contour1_pos
roi_size = 420
roi_left = roi_center[0] - roi_size//2
roi_top = roi_center[1] - roi_size//2
roi_right = roi_left + roi_size
roi_bottom = roi_top + roi_size

draw.rectangle([roi_left, roi_top, roi_right, roi_bottom], outline='yellow', width=3)
draw.text((roi_left+10, roi_top+10), "ROI (420×420)", fill='yellow')

# Draw line from marker to contour1
draw.line([marker_pos, contour1_pos], fill='cyan', width=2)
offset_x = contour1_pos[0] - marker_pos[0]
offset_y = contour1_pos[1] - marker_pos[1]
mid_x = (marker_pos[0] + contour1_pos[0]) // 2
mid_y = (marker_pos[1] + contour1_pos[1]) // 2
draw.text((mid_x, mid_y), f"Δ=({offset_x}, {offset_y})", fill='cyan')

# Save
result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
cv2.imwrite("detection_visualization.jpg", result)

print(f"\nDetection summary:")
print(f"  Marker position: {marker_pos}")
print(f"  Contour 1: {contour1_pos} (offset from marker: ({offset_x}, {offset_y}))")
print(f"  Contour 2: {contour2_pos}")
print(f"  ROI: {roi_left},{roi_top} to {roi_right},{roi_bottom}")
print(f"\nSaved: detection_visualization.jpg")
print(f"  → Green circle = Detected marker")
print(f"  → Red circles = Detected contours")
print(f"  → Yellow box = ROI shown in timeline composite")
