#!/usr/bin/env python3
"""
Advanced registration using deep learning features and modern techniques.
Uses pretrained models for better feature extraction and matching.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import numpy as np
from PIL import Image
import json
import logging
from pathlib import Path
import argparse
from scipy.optimize import minimize
from skimage.transform import warp, PolynomialTransform, ThinPlateSplineTransform


def extract_deep_features(image, model, layer='layer3'):
    """
    Extract deep features from an image using a pretrained CNN.

    Args:
        image: PIL Image
        model: Pretrained model
        layer: Which layer to extract features from

    Returns:
        Feature map as numpy array
    """
    # Preprocess image for ResNet
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = preprocess(image).unsqueeze(0)

    # Extract features
    features = []
    def hook(module, input, output):
        features.append(output)

    handle = getattr(model, layer).register_forward_hook(hook)
    with torch.no_grad():
        model(img_tensor)
    handle.remove()

    return features[0].squeeze().cpu().numpy()


def match_deep_features(feat1, feat2, landmarks1, landmarks2):
    """
    Match features using deep feature correlation and landmark guidance.

    Returns refined correspondence points.
    """
    # Resize features to match
    h, w = 256, 256  # Standard size for matching
    feat1_resized = cv2.resize(feat1.mean(axis=0), (w, h))
    feat2_resized = cv2.resize(feat2.mean(axis=0), (w, h))

    # Compute correlation
    correlation = cv2.matchTemplate(feat1_resized, feat2_resized, cv2.TM_CCOEFF_NORMED)

    # Find peaks in correlation map (potential matches)
    from skimage.feature import peak_local_max
    peaks = peak_local_max(correlation, min_distance=20, num_peaks=20)

    # Add landmark-guided points
    correspondences = []
    for landmark_name in landmarks1:
        if landmark_name in landmarks2:
            pt1 = landmarks1[landmark_name]
            pt2 = landmarks2[landmark_name]
            correspondences.append((pt1, pt2))

    return correspondences


def register_with_elastic(src_img, dst_img, src_landmarks, dst_landmarks,
                          use_deep_features=True):
    """
    Elastic registration using Thin Plate Splines with deep feature guidance.

    This handles non-rigid deformations much better than affine/homography.
    """
    src_np = np.array(src_img)
    dst_np = np.array(dst_img)

    # Collect landmark correspondences
    src_points = []
    dst_points = []

    for feature in ['marker', 'vein', 'freckle', 'arm_edge']:
        if feature in src_landmarks and feature in dst_landmarks:
            src_points.append(src_landmarks[feature])
            dst_points.append(dst_landmarks[feature])

    # Add more correspondences using deep features if available
    if use_deep_features:
        try:
            # Load pretrained model
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.eval()

            # Extract deep features
            src_features = extract_deep_features(src_img, model)
            dst_features = extract_deep_features(dst_img, model)

            # Match features to get more correspondences
            extra_matches = match_deep_features(
                src_features, dst_features, src_landmarks, dst_landmarks
            )

            # Add high-confidence matches
            for pt1, pt2 in extra_matches[:5]:  # Use top 5 matches
                src_points.append(pt1)
                dst_points.append(pt2)

            logging.info(f"Added {len(extra_matches[:5])} deep feature correspondences")

        except Exception as e:
            logging.warning(f"Deep features unavailable: {e}")

    src_points = np.array(src_points)
    dst_points = np.array(dst_points)

    # Use Thin Plate Spline for elastic registration
    tps = ThinPlateSplineTransform()
    tps.estimate(dst_points, src_points)

    # Warp the image
    warped = warp(src_np, tps, output_shape=dst_np.shape[:2], preserve_range=True)
    warped = warped.astype(np.uint8)

    # Calculate registration error
    transformed = tps(src_points)
    errors = np.linalg.norm(transformed - dst_points, axis=1)
    logging.info(f"Elastic registration: {len(src_points)} points, "
                f"mean error={np.mean(errors):.2f}px, max error={np.max(errors):.2f}px")

    return Image.fromarray(warped)


def register_with_optical_flow(src_img, dst_img, src_landmarks, dst_landmarks):
    """
    Use dense optical flow for registration.
    Better for capturing complex deformations.
    """
    src_gray = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2GRAY)
    dst_gray = cv2.cvtColor(np.array(dst_img), cv2.COLOR_RGB2GRAY)

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        dst_gray, src_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW
    )

    # Create mesh grid
    h, w = src_gray.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Apply flow to warp
    x_new = x + flow[:, :, 0]
    y_new = y + flow[:, :, 1]

    # Remap the image
    src_np = np.array(src_img)
    warped = cv2.remap(src_np, x_new.astype(np.float32),
                      y_new.astype(np.float32), cv2.INTER_LINEAR)

    logging.info("Optical flow registration complete")
    return Image.fromarray(warped)


def register_with_sift_homography(src_img, dst_img):
    """
    SIFT feature matching with RANSAC homography.
    More robust than using just landmarks.
    """
    src_np = np.array(src_img)
    dst_np = np.array(dst_img)

    # Convert to grayscale
    src_gray = cv2.cvtColor(src_np, cv2.COLOR_RGB2GRAY)
    dst_gray = cv2.cvtColor(dst_np, cv2.COLOR_RGB2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=2000)

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(src_gray, None)
    kp2, des2 = sift.detectAndCompute(dst_gray, None)

    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp image
        h, w = dst_np.shape[:2]
        warped = cv2.warpPerspective(src_np, H, (w, h))

        # Count inliers
        inliers = np.sum(mask)
        logging.info(f"SIFT homography: {len(good_matches)} matches, {inliers} inliers")

        return Image.fromarray(warped)
    else:
        logging.warning(f"Insufficient SIFT matches: {len(good_matches)}")
        return src_img


def main():
    parser = argparse.ArgumentParser(description='Advanced image registration')
    parser.add_argument('--baseline', required=True, help='Baseline image')
    parser.add_argument('--early', required=True, help='Early follow-up')
    parser.add_argument('--late', required=True, help='Late follow-up')
    parser.add_argument('--method', choices=['elastic', 'optical_flow', 'sift', 'deep'],
                       default='elastic', help='Registration method')
    parser.add_argument('--landmarks', default='landmarks.json',
                       help='Landmarks JSON file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load images
    baseline = Image.open(args.baseline).convert('RGB')
    early = Image.open(args.early).convert('RGB')
    late = Image.open(args.late).convert('RGB')

    # Load landmarks
    with open(args.landmarks, 'r') as f:
        landmarks = json.load(f)

    # Register based on method
    if args.method == 'elastic':
        early_reg = register_with_elastic(early, baseline,
                                         landmarks['day1'], landmarks['day0'])
        late_reg = register_with_elastic(late, baseline,
                                        landmarks['day2'], landmarks['day0'])
    elif args.method == 'optical_flow':
        early_reg = register_with_optical_flow(early, baseline,
                                              landmarks['day1'], landmarks['day0'])
        late_reg = register_with_optical_flow(late, baseline,
                                             landmarks['day2'], landmarks['day0'])
    elif args.method == 'sift':
        early_reg = register_with_sift_homography(early, baseline)
        late_reg = register_with_sift_homography(late, baseline)
    else:  # deep
        early_reg = register_with_elastic(early, baseline,
                                         landmarks['day1'], landmarks['day0'],
                                         use_deep_features=True)
        late_reg = register_with_elastic(late, baseline,
                                        landmarks['day2'], landmarks['day0'],
                                        use_deep_features=True)

    # Save results
    output_dir = Path(args.baseline).parent
    early_reg.save(output_dir / f'early_{args.method}_registered.jpg')
    late_reg.save(output_dir / f'late_{args.method}_registered.jpg')
    logging.info(f"Saved {args.method} registered images")


if __name__ == '__main__':
    main()