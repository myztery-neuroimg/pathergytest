#!/usr/bin/env python3
"""Verify that landmarks are correctly aligned in the final pipeline output."""

import cv2
import json
import numpy as np
from PIL import Image, ImageDraw


def load_intermediate_image(path):
    """Load intermediate pipeline image."""
    img = cv2.imread(str(path))
    return img


def draw_landmarks_adjusted(img_array, landmarks, day, crop_bbox, color='green'):
    """Draw landmarks on image, adjusted for crop offset."""
    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    day_landmarks = landmarks[day]
    left, top = crop_bbox[0], crop_bbox[1]
    radius = 15

    for feature, coord in day_landmarks.items():
        # Adjust for crop offset
        x = coord[0] - left
        y = coord[1] - top

        # Draw circle
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, width=4)
        # Draw crosshair
        draw.line([x-radius*2, y, x+radius*2, y], fill=color, width=2)
        draw.line([x, y-radius*2, x, y+radius*2], fill=color, width=2)
        # Draw label
        draw.text((x+20, y-10), f"{feature}", fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    print("="*100)
    print("VERIFYING PIPELINE LANDMARK ALIGNMENT")
    print("="*100)

    # Crop bboxes from pipeline output (from logs)
    baseline_bbox = (0, 66, 702, 1152)
    early_bbox = (0, 67, 702, 1145)
    late_bbox = (0, 66, 702, 1150)

    print(f"\n1. Pre-crop bounding boxes:")
    print(f"   Baseline: {baseline_bbox}")
    print(f"   Early: {early_bbox}")
    print(f"   Late: {late_bbox}")

    # Load landmarks
    with open("/Users/davidbrewster/Documents/workspace/2025/pathergytest/landmarks.json", 'r') as f:
        landmarks = json.load(f)

    print(f"\n2. VLM-extracted landmarks (original image coordinates):")
    for day in ['day0', 'day1', 'day2']:
        print(f"\n   {day}:")
        for feature, coord in landmarks[day].items():
            print(f"     {feature:12s}: {coord}")

    # Load pre-cropped images
    print(f"\n3. Loading pre-cropped images from pipeline...")
    baseline_precrop = load_intermediate_image(
        "/Users/davidbrewster/Documents/workspace/2025/pathergytest/output_timeline_fixed/intermediate_baseline_precrop.jpg"
    )
    early_precrop = load_intermediate_image(
        "/Users/davidbrewster/Documents/workspace/2025/pathergytest/output_timeline_fixed/intermediate_early_precrop.jpg"
    )
    late_precrop = load_intermediate_image(
        "/Users/davidbrewster/Documents/workspace/2025/pathergytest/output_timeline_fixed/intermediate_late_precrop.jpg"
    )

    print(f"   Baseline precrop: {baseline_precrop.shape}")
    print(f"   Early precrop: {early_precrop.shape}")
    print(f"   Late precrop: {late_precrop.shape}")

    # Draw landmarks on pre-cropped images (adjusted for crop offset)
    print(f"\n4. Drawing adjusted landmarks on pre-cropped images...")
    baseline_marked = draw_landmarks_adjusted(baseline_precrop.copy(), landmarks, 'day0', baseline_bbox, 'green')
    early_marked = draw_landmarks_adjusted(early_precrop.copy(), landmarks, 'day1', early_bbox, 'blue')
    late_marked = draw_landmarks_adjusted(late_precrop.copy(), landmarks, 'day2', late_bbox, 'orange')

    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_pipeline_baseline_landmarks.jpg", baseline_marked)
    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_pipeline_early_landmarks.jpg", early_marked)
    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_pipeline_late_landmarks.jpg", late_marked)
    print("   ✓ Saved pre-cropped images with adjusted landmarks")

    # Compute adjusted landmark coordinates
    print(f"\n5. Adjusted landmark coordinates (for pre-cropped images):")
    for day, bbox in [('day0', baseline_bbox), ('day1', early_bbox), ('day2', late_bbox)]:
        print(f"\n   {day} (offset: left={bbox[0]}, top={bbox[1]}):")
        for feature, coord in landmarks[day].items():
            adjusted_x = coord[0] - bbox[0]
            adjusted_y = coord[1] - bbox[1]
            print(f"     {feature:12s}: {coord} → ({adjusted_x}, {adjusted_y})")

    # Create composite
    print(f"\n6. Creating verification composite...")
    scale = 0.4
    h, w = baseline_precrop.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    b_resized = cv2.resize(baseline_marked, (new_w, new_h))
    e_resized = cv2.resize(early_marked, (new_w, new_h))
    l_resized = cv2.resize(late_marked, (new_w, new_h))

    composite = np.hstack([b_resized, e_resized, l_resized])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(composite, "Day 0 - Green Landmarks", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(composite, "Day 1 - Blue Landmarks", (new_w + 10, 30), font, 0.8, (0, 0, 255), 2)
    cv2.putText(composite, "Day 2 - Orange Landmarks", (2*new_w + 10, 30), font, 0.8, (0, 165, 255), 2)

    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_pipeline_landmarks_composite.jpg", composite)
    print("   ✓ Saved verification composite")

    print("\n" + "="*100)
    print("PIPELINE VERIFICATION COMPLETE")
    print("="*100)
    print("\nCheck these files:")
    print("  - debug_pipeline_baseline_landmarks.jpg")
    print("  - debug_pipeline_early_landmarks.jpg")
    print("  - debug_pipeline_late_landmarks.jpg")
    print("  - debug_pipeline_landmarks_composite.jpg")
    print("\nLandmarks should appear at correct anatomical locations on pre-cropped images.")
