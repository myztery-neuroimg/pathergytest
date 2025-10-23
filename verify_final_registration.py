#!/usr/bin/env python3
"""Verify final registration by computing and visualizing warped images with landmarks."""

import cv2
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def draw_landmarks(img_array, landmarks, color='green'):
    """Draw landmarks on image array."""
    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    radius = 15

    for feature, coord in landmarks.items():
        x, y = coord
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
    print("FINAL REGISTRATION VERIFICATION")
    print("="*100)

    # Load landmarks
    landmarks_path = Path("landmarks.json")
    with open(landmarks_path, 'r') as f:
        landmarks = json.load(f)

    # Crop bboxes from pipeline
    baseline_bbox = (0, 66, 702, 1152)
    early_bbox = (0, 67, 702, 1145)
    late_bbox = (0, 66, 702, 1150)

    print(f"\n1. Computing adjusted landmark coordinates for pre-cropped images...")

    # Adjust landmarks for crop offsets
    def adjust_landmarks(lm, bbox):
        adjusted = {}
        for feature, coord in lm.items():
            adjusted[feature] = [coord[0] - bbox[0], coord[1] - bbox[1]]
        return adjusted

    baseline_lm_adjusted = adjust_landmarks(landmarks['day0'], baseline_bbox)
    early_lm_adjusted = adjust_landmarks(landmarks['day1'], early_bbox)
    late_lm_adjusted = adjust_landmarks(landmarks['day2'], late_bbox)

    print("   Day 0 adjusted landmarks:")
    for k, v in baseline_lm_adjusted.items():
        print(f"     {k:12s}: {v}")

    # Load pre-cropped images
    print(f"\n2. Loading pre-cropped images...")
    output_dir = Path(".")  # Use current directory

    baseline_img = cv2.imread(str(output_dir / "intermediate_baseline_precrop.jpg"))
    early_img = cv2.imread(str(output_dir / "intermediate_early_precrop.jpg"))
    late_img = cv2.imread(str(output_dir / "intermediate_late_precrop.jpg"))

    h, w = baseline_img.shape[:2]
    print(f"   Baseline: {baseline_img.shape}")

    # Compute transforms using adjusted landmarks
    print(f"\n3. Computing affine transforms...")

    baseline_pts = np.array([baseline_lm_adjusted[f] for f in ['marker', 'vein', 'freckle']], dtype=np.float32)
    early_pts = np.array([early_lm_adjusted[f] for f in ['marker', 'vein', 'freckle']], dtype=np.float32)
    late_pts = np.array([late_lm_adjusted[f] for f in ['marker', 'vein', 'freckle']], dtype=np.float32)

    M_early = cv2.getAffineTransform(early_pts, baseline_pts)
    M_late = cv2.getAffineTransform(late_pts, baseline_pts)

    print(f"   Early→Baseline: computed")
    print(f"   Late→Baseline: computed")

    # Warp images
    print(f"\n4. Warping images to baseline frame...")
    early_warped = cv2.warpAffine(early_img, M_early, (w, h), flags=cv2.INTER_LINEAR)
    late_warped = cv2.warpAffine(late_img, M_late, (w, h), flags=cv2.INTER_LINEAR)

    # Draw BASELINE landmarks on all three (to verify alignment)
    print(f"\n5. Drawing BASELINE landmarks on all images (should align if registration worked)...")
    baseline_marked = draw_landmarks(baseline_img.copy(), baseline_lm_adjusted, 'lime')
    early_warped_marked = draw_landmarks(early_warped.copy(), baseline_lm_adjusted, 'lime')
    late_warped_marked = draw_landmarks(late_warped.copy(), baseline_lm_adjusted, 'lime')

    # Save individual marked images
    cv2.imwrite("verify_final_baseline.jpg", baseline_marked)
    cv2.imwrite("verify_final_early_warped.jpg", early_warped_marked)
    cv2.imwrite("verify_final_late_warped.jpg", late_warped_marked)

    print("   ✓ Saved individual verification images")

    # Create composite
    print(f"\n6. Creating composite...")
    scale = 0.5
    new_h, new_w = int(h * scale), int(w * scale)

    b_resized = cv2.resize(baseline_marked, (new_w, new_h))
    e_resized = cv2.resize(early_warped_marked, (new_w, new_h))
    l_resized = cv2.resize(late_warped_marked, (new_w, new_h))

    composite = np.hstack([b_resized, e_resized, l_resized])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(composite, "Day 0 (Baseline)", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(composite, "Day 1 (Registered)", (new_w + 10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(composite, "Day 2 (Registered)", (2*new_w + 10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(composite, "LIME landmarks = Day 0 positions (should align across all 3 panels)",
                (10, new_h - 10), font, 0.6, (0, 255, 0), 1)

    cv2.imwrite("verify_final_composite.jpg", composite)

    print("   ✓ Saved composite")

    print("\n" + "="*100)
    print("VERIFICATION COMPLETE")
    print("="*100)
    print("\nFiles created:")
    print("  - verify_final_baseline.jpg")
    print("  - verify_final_early_warped.jpg")
    print("  - verify_final_late_warped.jpg")
    print("  - verify_final_composite.jpg")
    print("\nIf registration is correct:")
    print("  → LIME landmarks should be at SAME anatomical positions in all 3 panels")
    print("  → Marker, vein, freckle, arm_edge should align perfectly")
