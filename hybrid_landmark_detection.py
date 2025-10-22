#!/usr/bin/env python3
"""Hybrid approach: VLM describes features, CV finds precise coordinates."""

import cv2
import numpy as np
import json
from pathlib import Path


def find_marker_center(image_path):
    """Find the X or + marker using computer vision."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to find dark regions (markers are black)
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours that might be the X/+ marker
    # Markers are typically small, compact, dark marks
    marker_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 < area < 5000:  # Marker size range
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Check compactness
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter ** 2)
                    if compactness > 0.3:  # Reasonably compact
                        marker_candidates.append((cx, cy, area, compactness))

    if not marker_candidates:
        return None

    # Return the most compact marker in central region
    h, w = img.shape[:2]
    central_candidates = [
        (cx, cy, area, comp) for cx, cy, area, comp in marker_candidates
        if w*0.2 < cx < w*0.8 and h*0.2 < cy < h*0.8
    ]

    if central_candidates:
        # Sort by compactness
        central_candidates.sort(key=lambda x: x[3], reverse=True)
        return central_candidates[0][:2]  # (cx, cy)

    # Fallback: most compact overall
    marker_candidates.sort(key=lambda x: x[3], reverse=True)
    return marker_candidates[0][:2]


def find_prominent_vein_junction(image_path):
    """Find prominent vein junction using edge detection."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance veins (they're darker/bluish)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Edge detection
    edges = cv2.Canny(enhanced, 30, 100)

    # Find junctions (points where multiple edges meet)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours on veins
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Look for Y-shaped or T-shaped junctions
    # For now, find central large contour
    if contours:
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for cnt in largest_contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)

    return None


def find_dark_freckle(image_path):
    """Find largest dark freckle/mole."""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect dark brown/black spots (freckles)
    # Freckles are low saturation, low value
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 100, 80])

    mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find small circular dark spots (freckles)
    freckle_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 500:  # Freckle size range
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.5:
                        freckle_candidates.append((cx, cy, area))

    if not freckle_candidates:
        return None

    # Return largest freckle
    freckle_candidates.sort(key=lambda x: x[2], reverse=True)
    return freckle_candidates[0][:2]


def find_arm_edge_point(image_path):
    """Find distinctive point on arm edge."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find edges
    edges = cv2.Canny(gray, 50, 150)

    # Find leftmost edge points (arm edge)
    h, w = edges.shape
    left_edge_points = []

    for y in range(h):
        for x in range(w):
            if edges[y, x] > 0:
                left_edge_points.append((x, y))
                break  # Found leftmost point for this row

    if not left_edge_points:
        return None

    # Find middle of arm edge
    left_edge_points.sort(key=lambda p: p[1])
    mid_idx = len(left_edge_points) // 2
    return left_edge_points[mid_idx]


if __name__ == "__main__":
    images = {
        "day0": "/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg",
        "day1": "/Users/davidbrewster/Documents/Documents_Brewster/15 August 13_17.jpg",
        "day2": "/Users/davidbrewster/Documents/Documents_Brewster/16 August 10_04.jpg"
    }

    print("="*100)
    print("HYBRID LANDMARK DETECTION: Computer Vision-based precise coordinates")
    print("="*100)

    landmarks = {}

    for day, img_path in images.items():
        print(f"\nProcessing {day}...")
        landmarks[day] = {}

        marker = find_marker_center(img_path)
        if marker:
            landmarks[day]['marker'] = list(marker)
            print(f"  marker: {marker}")
        else:
            print(f"  marker: NOT FOUND")

        vein = find_prominent_vein_junction(img_path)
        if vein:
            landmarks[day]['vein'] = list(vein)
            print(f"  vein: {vein}")
        else:
            print(f"  vein: NOT FOUND")

        freckle = find_dark_freckle(img_path)
        if freckle:
            landmarks[day]['freckle'] = list(freckle)
            print(f"  freckle: {freckle}")
        else:
            print(f"  freckle: NOT FOUND")

        arm_edge = find_arm_edge_point(img_path)
        if arm_edge:
            landmarks[day]['arm_edge'] = list(arm_edge)
            print(f"  arm_edge: {arm_edge}")
        else:
            print(f"  arm_edge: NOT FOUND")

    # Validation
    print("\n" + "="*100)
    print("VALIDATION:")
    print("="*100)

    valid_count = 0
    for feature in ['marker', 'vein', 'freckle', 'arm_edge']:
        found_in = [day for day in ['day0', 'day1', 'day2'] if feature in landmarks.get(day, {})]
        if len(found_in) == 3:
            print(f"✓ {feature:12s}: Found in all 3 images")
            valid_count += 1
        else:
            print(f"✗ {feature:12s}: Only found in {found_in}")

    if valid_count >= 3:
        output_file = "landmarks_cv.json"
        with open(output_file, 'w') as f:
            json.dump(landmarks, f, indent=2)
        print(f"\n✓ Saved {valid_count} CV-detected landmarks to: {output_file}")
    else:
        print(f"\n✗ FAILED: Only {valid_count} features found in all 3 images")
