#!/usr/bin/env python3
"""SIFT registration with arm-only mask to exclude background."""

import argparse
import cv2
import numpy as np
import json
from pathlib import Path


def segment_arm_mask(image_path):
    """Create binary mask of arm region using skin segmentation."""
    img = cv2.imread(image_path)

    # Convert to HSV and YCrCb for skin detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # HSV skin range
    lower_hsv = np.array([0, 20, 70])
    upper_hsv = np.array([20, 150, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb skin range
    lower_ycrcb = np.array([0, 133, 77])
    upper_ycrcb = np.array([255, 173, 127])
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Combine masks
    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # Find largest connected component (the arm)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skin_mask, connectivity=8)

    if num_labels > 1:
        # Find largest component (excluding background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        arm_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        arm_mask = skin_mask

    return arm_mask


def sift_register_masked(src_path, dst_path):
    """Register source to destination using SIFT with arm-only mask."""

    # Load images
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)

    # Create arm masks
    print("  Creating arm masks...")
    src_mask = segment_arm_mask(src_path)
    dst_mask = segment_arm_mask(dst_path)

    # Save masks for debugging
    cv2.imwrite(f"{Path(src_path).stem}_mask.jpg", src_mask)
    cv2.imwrite(f"{Path(dst_path).stem}_mask.jpg", dst_mask)

    # Convert to grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT
    sift = cv2.SIFT_create(nfeatures=5000)

    # Detect keypoints ONLY in arm regions
    print("  Detecting SIFT features on arm regions...")
    kp1, des1 = sift.detectAndCompute(src_gray, mask=src_mask)
    kp2, des2 = sift.detectAndCompute(dst_gray, mask=dst_mask)

    print(f"  Source arm: {len(kp1)} keypoints")
    print(f"  Destination arm: {len(kp2)} keypoints")

    if len(kp1) < 10 or len(kp2) < 10:
        print("  ERROR: Not enough keypoints detected on arms")
        return None

    # Match features
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    print("  Matching arm features...")
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"  Good matches: {len(good_matches)}")

    if len(good_matches) < 10:
        print("  ERROR: Not enough good matches")
        return None

    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC affine estimation
    print("  Computing affine transform with RANSAC...")
    M, mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=2000,
        confidence=0.995
    )

    if M is None:
        print("  ERROR: RANSAC failed")
        return None

    # Count inliers
    inliers = mask.ravel().tolist()
    inlier_count = sum(inliers)
    print(f"  Inliers: {inlier_count}/{len(good_matches)}")

    return M


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIFT registration with arm-only masking")
    parser.add_argument("baseline", help="Path to baseline (day 0) image")
    parser.add_argument("early", help="Path to early (day 1) image")
    parser.add_argument("late", help="Path to late (day 2) image")
    args = parser.parse_args()

    baseline_path = args.baseline
    early_path = args.early
    late_path = args.late

    print("="*100)
    print("SIFT + RANSAC REGISTRATION (ARM-MASKED)")
    print("="*100)

    # Register early → baseline
    print("\n1. Registering Day 1 → Day 0 (ARM ONLY)...")
    M_early = sift_register_masked(early_path, baseline_path)

    if M_early is not None:
        print(f"\n  Transform matrix:")
        print(f"    [{M_early[0,0]:.4f}  {M_early[0,1]:.4f} | {M_early[0,2]:.2f}]")
        print(f"    [{M_early[1,0]:.4f}  {M_early[1,1]:.4f} | {M_early[1,2]:.2f}]")

        angle = np.arctan2(M_early[1,0], M_early[0,0]) * 180 / np.pi
        scale_x = np.sqrt(M_early[0,0]**2 + M_early[1,0]**2)
        scale_y = np.sqrt(M_early[0,1]**2 + M_early[1,1]**2)
        tx, ty = M_early[0,2], M_early[1,2]

        print(f"\n  Rotation: {angle:.2f}°")
        print(f"  Scale: x={scale_x:.4f}, y={scale_y:.4f}")
        print(f"  Translation: x={tx:.2f}px, y={ty:.2f}px")

    # Register late → baseline
    print("\n2. Registering Day 2 → Day 0 (ARM ONLY)...")
    M_late = sift_register_masked(late_path, baseline_path)

    if M_late is not None:
        print(f"\n  Transform matrix:")
        print(f"    [{M_late[0,0]:.4f}  {M_late[0,1]:.4f} | {M_late[0,2]:.2f}]")
        print(f"    [{M_late[1,0]:.4f}  {M_late[1,1]:.4f} | {M_late[1,2]:.2f}]")

        angle = np.arctan2(M_late[1,0], M_late[0,0]) * 180 / np.pi
        scale_x = np.sqrt(M_late[0,0]**2 + M_late[1,0]**2)
        scale_y = np.sqrt(M_late[0,1]**2 + M_late[1,1]**2)
        tx, ty = M_late[0,2], M_late[1,2]

        print(f"\n  Rotation: {angle:.2f}°")
        print(f"  Scale: x={scale_x:.4f}, y={scale_y:.4f}")
        print(f"  Translation: x={tx:.2f}px, y={ty:.2f}px")

    # Test warping
    if M_early is not None and M_late is not None:
        print("\n3. Warping images...")
        baseline = cv2.imread(baseline_path)
        early = cv2.imread(early_path)
        late = cv2.imread(late_path)

        h, w = baseline.shape[:2]

        early_warped = cv2.warpAffine(early, M_early, (w, h), flags=cv2.INTER_LINEAR)
        late_warped = cv2.warpAffine(late, M_late, (w, h), flags=cv2.INTER_LINEAR)

        cv2.imwrite("sift_masked_early_warped.jpg", early_warped)
        cv2.imwrite("sift_masked_late_warped.jpg", late_warped)

        print("  ✓ Saved sift_masked_early_warped.jpg")
        print("  ✓ Saved sift_masked_late_warped.jpg")

        # Save transforms
        transforms = {
            "early_to_baseline": M_early.tolist(),
            "late_to_baseline": M_late.tolist()
        }

        with open("sift_masked_transforms.json", 'r', encoding='utf-8') as f:
            json.dump(transforms, f, indent=2)

        print("  ✓ Saved sift_masked_transforms.json")

    print("\n" + "="*100)
    print("MASKED SIFT REGISTRATION COMPLETE")
    print("="*100)
