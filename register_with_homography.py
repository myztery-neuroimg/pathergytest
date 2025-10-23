#!/usr/bin/env python3
"""Improved registration using homography for better 3D arm rotation handling."""

import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import logging
import argparse


def load_landmarks(landmarks_file='landmarks.json'):
    """Load landmarks from JSON file."""
    with open(landmarks_file, 'r') as f:
        return json.load(f)


def register_with_homography(src_img, dst_img, src_landmarks, dst_landmarks,
                             src_crop_offset=None, dst_crop_offset=None):
    """
    Register images using homography (projective transform) instead of affine.
    Homography can handle perspective changes from 3D arm rotations better.

    Args:
        src_img: Source PIL image
        dst_img: Destination PIL image
        src_landmarks: Dict of source landmarks
        dst_landmarks: Dict of destination landmarks
        src_crop_offset: (left, top) offset if source was pre-cropped
        dst_crop_offset: (left, top) offset if destination was pre-cropped

    Returns:
        Warped source image aligned to destination
    """
    # Convert to numpy arrays
    src_np = np.array(src_img)
    dst_np = np.array(dst_img)

    # Collect corresponding points
    src_points = []
    dst_points = []

    for feature in ['marker', 'vein', 'freckle', 'arm_edge']:
        if feature in src_landmarks and feature in dst_landmarks:
            src_pt = list(src_landmarks[feature])
            dst_pt = list(dst_landmarks[feature])

            # Adjust for crop offsets
            if src_crop_offset:
                src_pt[0] -= src_crop_offset[0]
                src_pt[1] -= src_crop_offset[1]
            if dst_crop_offset:
                dst_pt[0] -= dst_crop_offset[0]
                dst_pt[1] -= dst_crop_offset[1]

            src_points.append(src_pt)
            dst_points.append(dst_pt)

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    if len(src_points) < 4:
        logging.error(f"Need at least 4 points for homography, got {len(src_points)}")
        return src_img

    # Compute homography matrix (8 DOF vs 6 for affine)
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Warp the source image
    height, width = dst_np.shape[:2]
    warped = cv2.warpPerspective(src_np, H, (width, height))

    # Calculate reprojection error
    src_points_homo = np.column_stack([src_points, np.ones(len(src_points))])
    projected = (H @ src_points_homo.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    errors = np.linalg.norm(projected - dst_points, axis=1)
    logging.info(f"Homography registration: {len(src_points)} points, "
                f"mean error={np.mean(errors):.2f}px, max error={np.max(errors):.2f}px")

    # Check if homography is reasonable (not too distorted)
    det = np.linalg.det(H[:2, :2])
    if det < 0.5 or det > 2.0:
        logging.warning(f"Homography may be unstable (determinant={det:.3f})")

    return Image.fromarray(warped)


def register_with_tps(src_img, dst_img, src_landmarks, dst_landmarks,
                     src_crop_offset=None, dst_crop_offset=None):
    """
    Register using Thin Plate Splines for non-rigid registration.
    Better for handling local deformations from arm muscle/skin movement.

    Uses cv2.createThinPlateSplineShapeTransformer if available.
    """
    try:
        # Check if TPS is available (requires opencv-contrib-python)
        tps = cv2.createThinPlateSplineShapeTransformer()

        # Prepare points
        src_points = []
        dst_points = []

        for feature in ['marker', 'vein', 'freckle', 'arm_edge']:
            if feature in src_landmarks and feature in dst_landmarks:
                src_pt = list(src_landmarks[feature])
                dst_pt = list(dst_landmarks[feature])

                if src_crop_offset:
                    src_pt[0] -= src_crop_offset[0]
                    src_pt[1] -= src_crop_offset[1]
                if dst_crop_offset:
                    dst_pt[0] -= dst_crop_offset[0]
                    dst_pt[1] -= dst_crop_offset[1]

                src_points.append(src_pt)
                dst_points.append(dst_pt)

        src_points = np.array(src_points, dtype=np.float32).reshape(1, -1, 2)
        dst_points = np.array(dst_points, dtype=np.float32).reshape(1, -1, 2)

        # Estimate TPS transform
        tps.estimateTransformation(dst_points, src_points, [])

        # Apply transform
        src_np = np.array(src_img)
        height, width = src_np.shape[:2]
        warped = tps.warpImage(src_np)

        logging.info("TPS registration complete")
        return Image.fromarray(warped)

    except AttributeError:
        logging.warning("TPS not available (need opencv-contrib-python), falling back to homography")
        return register_with_homography(src_img, dst_img, src_landmarks, dst_landmarks,
                                       src_crop_offset, dst_crop_offset)


def test_registration_methods():
    """Test different registration methods and compare results."""

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, help='Baseline image path')
    parser.add_argument('--early', required=True, help='Early follow-up image')
    parser.add_argument('--late', required=True, help='Late follow-up image')
    parser.add_argument('--method', choices=['affine', 'homography', 'tps'],
                       default='homography', help='Registration method')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load images
    baseline = Image.open(args.baseline).convert('RGB')
    early = Image.open(args.early).convert('RGB')
    late = Image.open(args.late).convert('RGB')

    # Load landmarks
    landmarks = load_landmarks()

    # Register images
    if args.method == 'homography':
        early_registered = register_with_homography(
            early, baseline,
            landmarks['day1'], landmarks['day0']
        )
        late_registered = register_with_homography(
            late, baseline,
            landmarks['day2'], landmarks['day0']
        )
    elif args.method == 'tps':
        early_registered = register_with_tps(
            early, baseline,
            landmarks['day1'], landmarks['day0']
        )
        late_registered = register_with_tps(
            late, baseline,
            landmarks['day2'], landmarks['day0']
        )
    else:  # affine
        logging.info("Using standard affine registration")
        # Would call existing affine_register function
        pass

    # Save results
    output_dir = Path(args.baseline).parent
    early_registered.save(output_dir / 'early_homography_registered.jpg')
    late_registered.save(output_dir / 'late_homography_registered.jpg')
    logging.info(f"Saved registered images to {output_dir}")


if __name__ == '__main__':
    test_registration_methods()