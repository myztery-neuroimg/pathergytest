#!/usr/bin/env python3
"""Verify landmark-based registration by visualizing landmarks on original and registered images."""

import cv2
import json
import numpy as np
from PIL import Image, ImageDraw

def draw_landmarks_on_image(img_array, landmarks, day, color='red'):
    """Draw landmarks on image array."""
    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    day_landmarks = landmarks[day]
    radius = 15

    for feature, coord in day_landmarks.items():
        x, y = coord
        # Draw circle
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, width=4)
        # Draw crosshair
        draw.line([x-radius*2, y, x+radius*2, y], fill=color, width=2)
        draw.line([x, y-radius*2, x, y+radius*2], fill=color, width=2)
        # Draw label
        draw.text((x+20, y-10), feature, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def apply_transform(src_img, transform_matrix):
    """Apply affine transform to image."""
    h, w = src_img.shape[:2]
    warped = cv2.warpAffine(src_img, transform_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return warped


if __name__ == "__main__":
    print("="*100)
    print("VERIFYING LANDMARK REGISTRATION")
    print("="*100)

    # Load images
    baseline_path = "/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg"
    early_path = "/Users/davidbrewster/Documents/Documents_Brewster/15 August 13_17.jpg"
    late_path = "/Users/davidbrewster/Documents/Documents_Brewster/16 August 10_04.jpg"

    baseline_img = cv2.imread(baseline_path)
    early_img = cv2.imread(early_path)
    late_img = cv2.imread(late_path)

    print(f"\n1. Loaded images:")
    print(f"   Baseline: {baseline_img.shape}")
    print(f"   Early: {early_img.shape}")
    print(f"   Late: {late_img.shape}")

    # Load landmarks
    with open("/Users/davidbrewster/Documents/workspace/2025/pathergytest/landmarks.json", 'r') as f:
        landmarks = json.load(f)

    print(f"\n2. Landmarks extracted by VLM:")
    for day in ['day0', 'day1', 'day2']:
        print(f"\n   {day}:")
        for feature, coord in landmarks[day].items():
            print(f"     {feature:12s}: {coord}")

    # Draw landmarks on ORIGINAL images
    print("\n3. Drawing landmarks on ORIGINAL images...")
    baseline_marked = draw_landmarks_on_image(baseline_img.copy(), landmarks, 'day0', color='green')
    early_marked = draw_landmarks_on_image(early_img.copy(), landmarks, 'day1', color='blue')
    late_marked = draw_landmarks_on_image(late_img.copy(), landmarks, 'day2', color='orange')

    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_baseline_original_landmarks.jpg", baseline_marked)
    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_early_original_landmarks.jpg", early_marked)
    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_late_original_landmarks.jpg", late_marked)
    print("   ✓ Saved original images with landmarks")

    # Compute transforms
    print("\n4. Computing affine transforms...")

    # Get landmark points
    baseline_points = np.array([landmarks['day0'][f] for f in ['marker', 'vein', 'freckle', 'arm_edge']], dtype=np.float32)
    early_points = np.array([landmarks['day1'][f] for f in ['marker', 'vein', 'freckle', 'arm_edge']], dtype=np.float32)
    late_points = np.array([landmarks['day2'][f] for f in ['marker', 'vein', 'freckle', 'arm_edge']], dtype=np.float32)

    # Compute transforms
    M_early = cv2.getAffineTransform(early_points[:3], baseline_points[:3])
    M_late = cv2.getAffineTransform(late_points[:3], baseline_points[:3])

    print(f"   Early→Baseline transform:")
    print(f"   {M_early}")
    print(f"   Late→Baseline transform:")
    print(f"   {M_late}")

    # Apply transforms
    print("\n5. Warping images...")
    h, w = baseline_img.shape[:2]
    early_warped = cv2.warpAffine(early_img, M_early, (w, h), flags=cv2.INTER_LINEAR)
    late_warped = cv2.warpAffine(late_img, M_late, (w, h), flags=cv2.INTER_LINEAR)

    # Now draw BASELINE landmarks (day0) on all warped images to verify alignment
    print("\n6. Drawing BASELINE landmarks on warped images to verify alignment...")
    baseline_check = draw_landmarks_on_image(baseline_img.copy(), landmarks, 'day0', color='green')
    early_warped_check = draw_landmarks_on_image(early_warped.copy(), landmarks, 'day0', color='green')
    late_warped_check = draw_landmarks_on_image(late_warped.copy(), landmarks, 'day0', color='green')

    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_baseline_check.jpg", baseline_check)
    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_early_warped_check.jpg", early_warped_check)
    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_late_warped_check.jpg", late_warped_check)
    print("   ✓ Saved warped images with BASELINE landmarks")
    print("   → If registration worked, landmarks should be at SAME positions in all images")

    # Create side-by-side comparison
    print("\n7. Creating verification composite...")

    # Resize for display
    scale = 0.5
    new_h, new_w = int(h * scale), int(w * scale)

    b_resized = cv2.resize(baseline_check, (new_w, new_h))
    e_resized = cv2.resize(early_warped_check, (new_w, new_h))
    l_resized = cv2.resize(late_warped_check, (new_w, new_h))

    # Stack horizontally
    composite = np.hstack([b_resized, e_resized, l_resized])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(composite, "Day 0 (Baseline)", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(composite, "Day 1 (Registered)", (new_w + 10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(composite, "Day 2 (Registered)", (2*new_w + 10, 30), font, 1, (0, 255, 0), 2)

    cv2.imwrite("/Users/davidbrewster/Documents/workspace/2025/pathergytest/debug_registration_verification.jpg", composite)
    print("   ✓ Saved verification composite")

    print("\n" + "="*100)
    print("VERIFICATION COMPLETE")
    print("="*100)
    print("\nFiles to check:")
    print("  1. debug_baseline_original_landmarks.jpg - Day 0 original with VLM landmarks")
    print("  2. debug_early_original_landmarks.jpg - Day 1 original with VLM landmarks")
    print("  3. debug_late_original_landmarks.jpg - Day 2 original with VLM landmarks")
    print("  4. debug_registration_verification.jpg - All images with SAME baseline landmarks")
    print("\nIf registration worked correctly:")
    print("  - Green crosshairs should be at SAME anatomical locations in verification composite")
    print("  - Marker, vein, freckle, arm_edge should align across all 3 panels")
