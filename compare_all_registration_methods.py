#!/usr/bin/env python3
"""Comprehensive visual comparison of ALL registration methods."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path):
    """Load image as numpy array."""
    return cv2.imread(str(path))


def segment_skin_region(pil_img):
    """Segment skin region (from main.py)."""
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    lower_hsv = np.array([0, 20, 70])
    upper_hsv = np.array([20, 150, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    lower_ycrcb = np.array([0, 133, 77])
    upper_ycrcb = np.array([255, 173, 127])
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    return (skin_mask > 0).astype(np.float32)


def ecc_register(src_pil, dst_pil):
    """ECC-based registration (from main.py)."""
    src = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2GRAY)
    dst = cv2.cvtColor(np.array(dst_pil), cv2.COLOR_RGB2GRAY)

    src_h, src_w = src.shape
    dst_h, dst_w = dst.shape

    if (src_h, src_w) != (dst_h, dst_w):
        src = cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    skin_mask = segment_skin_region(dst_pil)
    mask = (skin_mask * 255).astype(np.uint8)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-8)

    try:
        (cc, warp_matrix) = cv2.findTransformECC(
            templateImage=dst,
            inputImage=src,
            warpMatrix=warp_matrix,
            motionType=cv2.MOTION_AFFINE,
            criteria=criteria,
            inputMask=mask,
            gaussFiltSize=5
        )
        return warp_matrix, cc
    except cv2.error as e:
        print(f"    ECC failed: {e}")
        return np.eye(2, 3, dtype=np.float32), 0.0


def vlm_landmark_register(src_landmarks, dst_landmarks):
    """VLM landmark-based registration."""
    src_pts = []
    dst_pts = []

    for feature in ['marker', 'vein', 'freckle', 'skin_feature']:
        if feature in src_landmarks and feature in dst_landmarks:
            src_pts.append(src_landmarks[feature])
            dst_pts.append(dst_landmarks[feature])

    if len(src_pts) < 3:
        print(f"    VLM: Insufficient landmarks ({len(src_pts)})")
        return None

    src_pts = np.array(src_pts[:3], dtype=np.float32)
    dst_pts = np.array(dst_pts[:3], dtype=np.float32)

    M = cv2.getAffineTransform(src_pts, dst_pts)
    return M


def sift_register_masked(src_path, dst_path):
    """SIFT with arm mask."""
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)

    # Quick skin mask
    src_pil = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    dst_pil = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    src_mask = (segment_skin_region(src_pil) * 255).astype(np.uint8)
    dst_mask = (segment_skin_region(dst_pil) * 255).astype(np.uint8)

    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(src_gray, mask=src_mask)
    kp2, des2 = sift.detectAndCompute(dst_gray, mask=dst_mask)

    if len(kp1) < 10 or len(kp2) < 10:
        print(f"    SIFT: Too few keypoints ({len(kp1)}, {len(kp2)})")
        return None

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50)
    )

    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) < 10:
        print(f"    SIFT: Too few matches ({len(good)})")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )

    if M is None:
        print(f"    SIFT: RANSAC failed")
        return None

    inliers = sum(mask.ravel().tolist())
    print(f"    SIFT: {inliers}/{len(good)} inliers")
    return M


def create_comparison_panel(baseline_img, early_warped_dict, late_warped_dict,
                           detection_point, method_names):
    """Create comprehensive comparison panel."""
    h, w = baseline_img.shape[:2]

    # Resize for display
    scale = 0.4
    new_h, new_w = int(h * scale), int(w * scale)

    # Number of methods + baseline + no-registration
    n_methods = len(method_names) + 2

    # Create grid: baseline + methods for early, then baseline + methods for late
    grid_h = new_h * 2 + 100  # 2 rows (early, late)
    grid_w = new_w * n_methods + 50

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(grid, "REGISTRATION METHOD COMPARISON", (10, 30),
                font, 1.2, (0, 0, 0), 2)

    # Draw detection point on all images
    det_x, det_y = detection_point

    def draw_box(img, x, y, color):
        img_copy = img.copy()
        box_size = 30
        cv2.rectangle(img_copy, (x-box_size, y-box_size), (x+box_size, y+box_size),
                     color, 3)
        return img_copy

    # Row 1: Day 1 registrations
    row1_y = 60
    cv2.putText(grid, "Day 1:", (10, row1_y + new_h//2), font, 0.8, (0, 0, 200), 2)

    x_offset = 80

    # Baseline
    baseline_resized = cv2.resize(baseline_img, (new_w, new_h))
    baseline_with_box = draw_box(baseline_resized,
                                  int(det_x * scale), int(det_y * scale), (0, 255, 0))
    grid[row1_y:row1_y+new_h, x_offset:x_offset+new_w] = baseline_with_box
    cv2.putText(grid, "Baseline", (x_offset, row1_y-5), font, 0.5, (0, 150, 0), 1)
    x_offset += new_w + 10

    # No registration
    if 'none' in early_warped_dict:
        early_none = cv2.resize(early_warped_dict['none'], (new_w, new_h))
        early_none_box = draw_box(early_none, int(det_x * scale), int(det_y * scale),
                                   (0, 0, 255))
        grid[row1_y:row1_y+new_h, x_offset:x_offset+new_w] = early_none_box
        cv2.putText(grid, "No Reg", (x_offset, row1_y-5), font, 0.5, (0, 0, 200), 1)
        x_offset += new_w + 10

    # Each method
    for method in method_names:
        if method in early_warped_dict:
            warped = cv2.resize(early_warped_dict[method], (new_w, new_h))
            warped_box = draw_box(warped, int(det_x * scale), int(det_y * scale),
                                 (255, 0, 0))
            grid[row1_y:row1_y+new_h, x_offset:x_offset+new_w] = warped_box
            cv2.putText(grid, method, (x_offset, row1_y-5), font, 0.5, (200, 0, 0), 1)
        else:
            # Mark as FAILED
            cv2.putText(grid, method, (x_offset, row1_y-5), font, 0.5, (100, 100, 100), 1)
            cv2.putText(grid, "FAILED", (x_offset + 20, row1_y + new_h//2),
                       font, 0.8, (0, 0, 200), 2)
        x_offset += new_w + 10

    # Row 2: Day 2 registrations
    row2_y = row1_y + new_h + 20
    cv2.putText(grid, "Day 2:", (10, row2_y + new_h//2), font, 0.8, (200, 0, 200), 2)

    x_offset = 80

    # Baseline
    grid[row2_y:row2_y+new_h, x_offset:x_offset+new_w] = baseline_with_box
    x_offset += new_w + 10

    # No registration
    if 'none' in late_warped_dict:
        late_none = cv2.resize(late_warped_dict['none'], (new_w, new_h))
        late_none_box = draw_box(late_none, int(det_x * scale), int(det_y * scale),
                                (0, 0, 255))
        grid[row2_y:row2_y+new_h, x_offset:x_offset+new_w] = late_none_box
        x_offset += new_w + 10

    # Each method
    for method in method_names:
        if method in late_warped_dict:
            warped = cv2.resize(late_warped_dict[method], (new_w, new_h))
            warped_box = draw_box(warped, int(det_x * scale), int(det_y * scale),
                                 (255, 0, 255))
            grid[row2_y:row2_y+new_h, x_offset:x_offset+new_w] = warped_box
        else:
            cv2.putText(grid, "FAILED", (x_offset + 20, row2_y + new_h//2),
                       font, 0.8, (0, 0, 200), 2)
        x_offset += new_w + 10

    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive registration method comparison")
    parser.add_argument("baseline", help="Path to baseline (day 0) image")
    parser.add_argument("early", help="Path to early (day 1) image")
    parser.add_argument("late", help="Path to late (day 2) image")
    parser.add_argument("--output-dir", default=".", help="Directory to save output images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*100)
    print("COMPREHENSIVE REGISTRATION COMPARISON")
    print("="*100)

    # Load images
    print("\n1. Loading images...")
    baseline_img = load_image(args.baseline)
    early_img = load_image(args.early)
    late_img = load_image(args.late)

    h, w = baseline_img.shape[:2]
    print(f"   Image size: {w}×{h}")

    # Detection point (from main.py logs: marker at ~363, 458 after shared crop at top=135)
    # So original coordinates before crop
    detection_point = (363, 458 + 135)  # Add back the crop offset

    print(f"   Using detection point: {detection_point}")

    # Dictionary to store warped images
    early_warped = {}
    late_warped = {}

    # 0. No registration (baseline)
    print("\n2. No Registration (identity transform)...")
    early_warped['none'] = early_img
    late_warped['none'] = late_img

    # 1. ECC Registration
    print("\n3. ECC Registration...")
    baseline_pil = Image.fromarray(cv2.cvtColor(baseline_img, cv2.COLOR_BGR2RGB))
    early_pil = Image.fromarray(cv2.cvtColor(early_img, cv2.COLOR_BGR2RGB))
    late_pil = Image.fromarray(cv2.cvtColor(late_img, cv2.COLOR_BGR2RGB))

    print("   Day 1 → Day 0...")
    M_ecc_early, cc_early = ecc_register(early_pil, baseline_pil)
    print(f"   Correlation: {cc_early:.4f}")
    early_warped['ECC'] = cv2.warpAffine(early_img, M_ecc_early, (w, h))

    print("   Day 2 → Day 0...")
    M_ecc_late, cc_late = ecc_register(late_pil, baseline_pil)
    print(f"   Correlation: {cc_late:.4f}")
    late_warped['ECC'] = cv2.warpAffine(late_img, M_ecc_late, (w, h))

    # 2. VLM Landmark Registration
    print("\n4. VLM Landmark Registration...")
    landmarks_file = Path("landmarks.json")
    if landmarks_file.exists():
        with open(landmarks_file, '0', encoding='utf-8') as f:
            landmarks = json.load(f)

        print("   Day 1 → Day 0...")
        M_vlm_early = vlm_landmark_register(landmarks['day1'], landmarks['day0'])
        if M_vlm_early is not None:
            early_warped['VLM'] = cv2.warpAffine(early_img, M_vlm_early, (w, h))

        print("   Day 2 → Day 0...")
        M_vlm_late = vlm_landmark_register(landmarks['day2'], landmarks['day0'])
        if M_vlm_late is not None:
            late_warped['VLM'] = cv2.warpAffine(late_img, M_vlm_late, (w, h))
    else:
        print("   landmarks.json not found - skipping")

    # 3. SIFT Masked Registration
    print("\n5. SIFT Masked Registration...")
    print("   Day 1 → Day 0...")
    M_sift_early = sift_register_masked(args.early, args.baseline)
    if M_sift_early is not None:
        early_warped['SIFT'] = cv2.warpAffine(early_img, M_sift_early, (w, h))

    print("   Day 2 → Day 0...")
    M_sift_late = sift_register_masked(args.late, args.baseline)
    if M_sift_late is not None:
        late_warped['SIFT'] = cv2.warpAffine(late_img, M_sift_late, (w, h))

    # Create comparison visualization
    print("\n6. Creating comparison visualization...")
    method_names = ['ECC', 'VLM', 'SIFT']

    comparison = create_comparison_panel(
        baseline_img, early_warped, late_warped,
        detection_point, method_names
    )

    output_path = "registration_comparison.jpg"
    cv2.imwrite(output_path, comparison)
    print(f"   ✓ Saved: {output_path}")

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print("\nMethods tested:")
    print("  - No Registration: Identity transform (shows misalignment)")
    print("  - ECC: Intensity-based correlation")
    print("  - VLM: Vision Language Model landmark extraction")
    print("  - SIFT: Feature matching with arm masking")

    print("\nResults:")
    print(f"  Day 1: {len(early_warped)} methods completed")
    print(f"  Day 2: {len(late_warped)} methods completed")

    print("\nVisual Guide:")
    print("  - Green box = Baseline detection location")
    print("  - Red box = Day 1 tracked location")
    print("  - Magenta box = Day 2 tracked location")
    print("  - Boxes should overlap if registration works correctly")

    print(f"\n✓ View comparison: {output_path}")
