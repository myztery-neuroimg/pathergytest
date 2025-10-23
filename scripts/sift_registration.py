#!/usr/bin/env python3
"""Use SIFT + RANSAC for automatic feature-based registration."""

import argparse
import cv2
import numpy as np
from pathlib import Path


def sift_register(src_path, dst_path):
    """Register source image to destination using SIFT + RANSAC."""

    # Load images
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)

    # Convert to grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=5000)

    # Detect keypoints and compute descriptors
    print("  Detecting SIFT features...")
    kp1, des1 = sift.detectAndCompute(src_gray, None)
    kp2, des2 = sift.detectAndCompute(dst_gray, None)

    print(f"  Source: {len(kp1)} keypoints")
    print(f"  Destination: {len(kp2)} keypoints")

    # Match features using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    print("  Matching features...")
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"  Good matches: {len(good_matches)}")

    if len(good_matches) < 10:
        print("  ERROR: Not enough good matches")
        return None, []

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find affine transform using RANSAC
    print("  Computing affine transform with RANSAC...")
    M, mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=2000,
        confidence=0.995
    )

    if M is None:
        print("  ERROR: RANSAC failed to find transform")
        return None, []

    # Count inliers
    inliers = mask.ravel().tolist()
    inlier_count = sum(inliers)
    print(f"  Inliers: {inlier_count}/{len(good_matches)}")

    # Extract inlier matches for visualization
    inlier_matches = [m for m, inlier in zip(good_matches, inliers) if inlier]

    return M, inlier_matches


def visualize_matches(src_path, dst_path, matches, kp1, kp2, output_path):
    """Visualize feature matches."""
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)

    # Draw matches
    img_matches = cv2.drawMatches(
        src, kp1, dst, kp2, matches[:50],  # Show top 50 matches
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(output_path, img_matches)
    print(f"  Saved match visualization: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIFT + RANSAC registration for pathergy test images")
    parser.add_argument("baseline", help="Path to baseline (day 0) image")
    parser.add_argument("early", help="Path to early (day 1) image")
    parser.add_argument("late", help="Path to late (day 2) image")
    args = parser.parse_args()

    baseline_path = args.baseline
    early_path = args.early
    late_path = args.late

    print("="*100)
    print("SIFT + RANSAC REGISTRATION")
    print("="*100)

    # Register early → baseline
    print("\n1. Registering Day 1 → Day 0...")
    M_early, matches_early = sift_register(early_path, baseline_path)

    if M_early is not None:
        print(f"\n  Transform matrix:")
        print(f"    [{M_early[0,0]:.4f}  {M_early[0,1]:.4f} | {M_early[0,2]:.2f}]")
        print(f"    [{M_early[1,0]:.4f}  {M_early[1,1]:.4f} | {M_early[1,2]:.2f}]")

        # Decompose
        angle = np.arctan2(M_early[1,0], M_early[0,0]) * 180 / np.pi
        scale_x = np.sqrt(M_early[0,0]**2 + M_early[1,0]**2)
        scale_y = np.sqrt(M_early[0,1]**2 + M_early[1,1]**2)
        tx, ty = M_early[0,2], M_early[1,2]

        print(f"\n  Rotation: {angle:.2f}°")
        print(f"  Scale: x={scale_x:.4f}, y={scale_y:.4f}")
        print(f"  Translation: x={tx:.2f}px, y={ty:.2f}px")

    # Register late → baseline
    print("\n2. Registering Day 2 → Day 0...")
    M_late, matches_late = sift_register(late_path, baseline_path)

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
        print("\n3. Testing image warping...")
        baseline = cv2.imread(baseline_path)
        early = cv2.imread(early_path)
        late = cv2.imread(late_path)

        h, w = baseline.shape[:2]

        early_warped = cv2.warpAffine(early, M_early, (w, h), flags=cv2.INTER_LINEAR)
        late_warped = cv2.warpAffine(late, M_late, (w, h), flags=cv2.INTER_LINEAR)

        cv2.imwrite("sift_early_warped.jpg", early_warped)
        cv2.imwrite("sift_late_warped.jpg", late_warped)

        print("  ✓ Saved sift_early_warped.jpg")
        print("  ✓ Saved sift_late_warped.jpg")

        # Save transforms
        transforms = {
            "early_to_baseline": M_early.tolist(),
            "late_to_baseline": M_late.tolist()
        }

        import json
        with open("sift_transforms.json", 'r', encoding='utf-8') as f:
            json.dump(transforms, f, indent=2)

        print("  ✓ Saved sift_transforms.json")

    print("\n" + "="*100)
    print("SIFT REGISTRATION COMPLETE")
    print("="*100)
