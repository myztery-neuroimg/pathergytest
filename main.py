#!/usr/bin/env python3
"""Utilities for aligning and visualising serial pathergy test photographs."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import morphology, measure

Coordinate = Tuple[int, int]


def configure_logging(level: str) -> None:
    """Configure the root logger with a sensible default format."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_image(path: Path) -> Image.Image:
    """Open an image path as RGB with basic error handling."""

    logging.debug("Loading image from %s", path)
    try:
        return Image.open(path).convert("RGB")
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise FileNotFoundError(f"Input image not found: {path}") from exc
    except OSError as exc:  # pragma: no cover - defensive
        raise OSError(f"Unable to open image '{path}': {exc}") from exc


def segment_skin_region(pil_img: Image.Image) -> np.ndarray:
    """Segment skin region using multi-color-space approach.

    Uses HSV and YCrCb color spaces to robustly detect skin regardless of lighting.
    Returns a binary mask where skin pixels are True.
    """

    logging.debug("Segmenting skin region")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # HSV color space - good for hue-based skin detection
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Skin hue typically in range 0-50 in H channel
    lower_hsv = np.array([0, 20, 50], dtype=np.uint8)
    upper_hsv = np.array([50, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb color space - robust to illumination changes
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    # Skin typically has Cr: 133-173, Cb: 77-127
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Combine both masks (AND operation for high confidence)
    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove small components
    skin_mask = morphology.remove_small_objects(skin_mask.astype(bool), min_size=500)

    return skin_mask


def find_forearm_bbox(pil_img: Image.Image, margin_percent: float = 0.15) -> Tuple[int, int, int, int] | None:
    """Find bounding box of the forearm region with safety margin for registration.

    Args:
        pil_img: Input image
        margin_percent: Safety margin as percentage of bbox dimensions (default 15%)

    Returns:
        (left, top, right, bottom) bounding box, or None if no forearm found
    """

    logging.debug("Finding forearm bounding box")

    # Segment skin
    skin_mask = segment_skin_region(pil_img)

    if not np.any(skin_mask):
        logging.warning("No skin region detected")
        return None

    # Label connected components
    labeled = measure.label(skin_mask)
    regions = measure.regionprops(labeled)

    if not regions:
        logging.warning("No connected skin regions found")
        return None

    # Find the largest region (likely the forearm)
    largest_region = max(regions, key=lambda r: r.area)

    # Get bounding box
    min_row, min_col, max_row, max_col = largest_region.bbox

    # Add safety margin for registration (expand by margin_percent)
    height = max_row - min_row
    width = max_col - min_col

    margin_h = int(height * margin_percent)
    margin_w = int(width * margin_percent)

    img_height, img_width = skin_mask.shape

    # Expand with margin, clipping to image bounds
    left = max(0, min_col - margin_w)
    top = max(0, min_row - margin_h)
    right = min(img_width, max_col + margin_w)
    bottom = min(img_height, max_row + margin_h)

    logging.info(
        "Forearm region: area=%d px², bbox=(%d,%d,%d,%d), margin=%d%%",
        largest_region.area, left, top, right, bottom, int(margin_percent * 100)
    )

    return (left, top, right, bottom)


def intelligent_precrop(pil_img: Image.Image, margin_percent: float = 0.15) -> Tuple[Image.Image, Tuple[int, int, int, int] | None]:
    """Intelligently pre-crop image to forearm region with registration margin.

    Returns:
        Tuple of (cropped_image, bbox_used)
        If no forearm found, returns (original_image, None)
    """

    bbox = find_forearm_bbox(pil_img, margin_percent=margin_percent)

    if bbox is None:
        logging.warning("Could not identify forearm; using full image")
        return pil_img, None

    cropped = pil_img.crop(bbox)
    logging.debug("Pre-cropped to forearm region: %d x %d", cropped.size[0], cropped.size[1])

    return cropped, bbox


def detect_arm_orientation(pil_img: Image.Image) -> float:
    """Detect the long axis of the arm using PCA on the skin mask.

    This identifies the principal direction of the forearm, which is critical
    for pathergy tests where injection sites are arranged lengthwise along the arm.

    Returns:
        Angle in degrees of the arm's principal axis (0-180 degrees from horizontal)
    """

    logging.debug("Detecting arm orientation using PCA")
    skin_mask = segment_skin_region(pil_img)

    # Get coordinates of all skin pixels
    ys, xs = np.nonzero(skin_mask)
    if len(xs) < 10:
        logging.warning("Insufficient skin pixels for orientation detection; defaulting to 0°")
        return 0.0

    # Stack coordinates for PCA [x, y]
    coords = np.column_stack([xs, ys])

    # Compute mean-centered coordinates
    mean = np.mean(coords, axis=0)
    centered = coords - mean

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Principal axis is the eigenvector with largest eigenvalue
    principal_idx = np.argmax(eigenvalues)
    principal_axis = eigenvectors[:, principal_idx]

    # Calculate angle from horizontal (in degrees)
    # Angle is measured counter-clockwise from positive x-axis
    angle = np.arctan2(principal_axis[1], principal_axis[0]) * 180 / np.pi

    # Normalize to 0-180 range (we don't care about direction, just orientation)
    if angle < 0:
        angle += 180

    logging.info("Detected arm orientation: %.1f° from horizontal", angle)
    return float(angle)


def compute_local_arm_orientation(pil_img: Image.Image, center_point: Coordinate, radius: int = 100) -> float:
    """Compute local arm orientation at a specific location.

    Uses PCA on skin pixels within a local region around the test site,
    rather than global arm orientation. This accounts for arm curvature.

    Args:
        pil_img: Input image
        center_point: (x, y) center of local region
        radius: Radius of local region to analyze (pixels)

    Returns:
        Local arm orientation angle in degrees (0-180 from horizontal)
    """

    logging.debug("Computing local arm orientation at (%d, %d)", center_point[0], center_point[1])
    skin_mask = segment_skin_region(pil_img)

    height, width = skin_mask.shape
    cx, cy = center_point

    # Define local region
    x_min = max(0, cx - radius)
    x_max = min(width, cx + radius)
    y_min = max(0, cy - radius)
    y_max = min(height, cy + radius)

    # Extract local skin mask
    local_mask = skin_mask[y_min:y_max, x_min:x_max]

    # Get local coordinates
    ys, xs = np.nonzero(local_mask)
    if len(xs) < 10:
        logging.warning("Insufficient local skin pixels; using global orientation")
        return detect_arm_orientation(pil_img)

    # Convert to global coordinates
    coords = np.column_stack([xs + x_min, ys + y_min])

    # Compute PCA
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    principal_idx = np.argmax(eigenvalues)
    principal_axis = eigenvectors[:, principal_idx].real

    angle = np.arctan2(principal_axis[1], principal_axis[0]) * 180 / np.pi
    if angle < 0:
        angle += 180

    logging.debug("Local arm orientation at test site: %.1f°", angle)
    return float(angle)


def estimate_arm_width_px(pil_img: Image.Image, center_y: int | None = None) -> float:
    """Estimate forearm width in pixels at a given vertical position.

    Used for scale calibration. Typical forearm width is 6-8 cm.

    Args:
        pil_img: Input image
        center_y: Y-coordinate to measure width at (defaults to center)

    Returns:
        Estimated arm width in pixels
    """

    logging.debug("Estimating arm width for scale calibration")
    skin_mask = segment_skin_region(pil_img)
    height, width = skin_mask.shape

    if center_y is None:
        center_y = height // 2

    # Sample a band around center_y
    band_height = 20
    y_min = max(0, center_y - band_height // 2)
    y_max = min(height, center_y + band_height // 2)

    band = skin_mask[y_min:y_max, :]

    # For each row, find the width of skin
    widths = []
    for row in band:
        xs = np.nonzero(row)[0]
        if len(xs) > 0:
            widths.append(xs.max() - xs.min())

    if not widths:
        logging.warning("Could not estimate arm width; using default")
        return 200.0  # Fallback

    arm_width_px = np.median(widths)
    logging.info("Estimated arm width: %.1f pixels", arm_width_px)
    return float(arm_width_px)


def detect_markers(pil_img: Image.Image) -> Tuple[List[Coordinate], Coordinate | None]:
    """Detect + and - markers drawn by physician.

    Markers are typically larger than injection sites and may be darker/pen ink.
    Returns marker positions and the centroid of the marker region.

    Args:
        pil_img: Input image

    Returns:
        Tuple of (marker_positions, marker_centroid)
        marker_centroid is the center point where markers are located
    """

    logging.debug("Detecting + and - markers")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    markers = []

    # Try detecting dark markers (pen ink)
    # Pen marks are typically very dark (low V in HSV, low grayscale)
    _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.medianBlur(dark_mask, 5)

    # Try detecting red markers
    red_mask = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255)) | cv2.inRange(
        hsv, (170, 60, 60), (180, 255, 255)
    )
    red_mask = cv2.medianBlur(red_mask, 3)

    # Combine both masks
    combined_mask = cv2.bitwise_or(dark_mask, red_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Look for larger regions that could be markers
    # Markers are typically larger than injection sites
    # Adjusted threshold for lower resolution images
    marker_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:  # Larger than injection sites (adjusted for low-res)
            moments = cv2.moments(contour)
            if moments["m00"]:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                marker_candidates.append((cx, cy, area))

    if marker_candidates:
        logging.info("Found %d potential marker regions", len(marker_candidates))
        # Sort by area (largest first)
        marker_candidates.sort(key=lambda t: t[2], reverse=True)
        markers = [(x, y) for x, y, _ in marker_candidates]

        # Find the closest pair of markers (these should be the + and - symbols)
        # This is more reliable than using all markers
        if len(markers) >= 2:
            min_dist = float('inf')
            best_pair = (markers[0], markers[1])

            for i, m1 in enumerate(markers):
                for m2 in markers[i+1:]:
                    dist = math.hypot(m2[0] - m1[0], m2[1] - m1[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (m1, m2)

            # Use centroid of closest pair as test site
            centroid_x = int((best_pair[0][0] + best_pair[1][0]) / 2)
            centroid_y = int((best_pair[0][1] + best_pair[1][1]) / 2)
            marker_centroid = (centroid_x, centroid_y)
            logging.info(
                "Using closest marker pair at distance %.1f px: (%d, %d) and (%d, %d)",
                min_dist, best_pair[0][0], best_pair[0][1], best_pair[1][0], best_pair[1][1]
            )
        else:
            marker_centroid = markers[0]

        logging.info("Marker region centroid: (%d, %d)", marker_centroid[0], marker_centroid[1])
        return markers, marker_centroid
    else:
        logging.warning("No markers detected")
        return [], None


def illumination_correction(pil_img: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """Apply illumination correction using LAB color space and CLAHE on L channel."""

    logging.debug("Applying illumination correction with clip_limit=%.1f", clip_limit)
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_channel_enhanced, a_channel, b_channel])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB))


def bilateral_filter_preprocess(
    pil_img: Image.Image, d: int = 9, sigma_color: int = 75, sigma_space: int = 75
) -> Image.Image:
    """Apply bilateral filtering for edge-preserving noise reduction."""

    logging.debug(
        "Applying bilateral filter: d=%d, sigma_color=%d, sigma_space=%d",
        d, sigma_color, sigma_space
    )
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    filtered = cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)
    return Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))


def normalize_colors(pil_img: Image.Image) -> Image.Image:
    """Normalize color channels to improve consistency across images."""

    logging.debug("Normalizing color channels")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Normalize each channel to 0-255 range
    l_normalized = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)

    lab_normalized = cv2.merge([l_normalized, a_channel, b_channel])
    bgr_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(bgr_normalized, cv2.COLOR_BGR2RGB))


def unsharp_mask(pil_img: Image.Image, kernel_size: int = 5, strength: float = 1.5) -> Image.Image:
    """Apply unsharp masking for edge enhancement."""

    logging.debug("Applying unsharp mask: kernel_size=%d, strength=%.1f", kernel_size, strength)
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Create Gaussian blurred version
    gaussian = cv2.GaussianBlur(bgr, (kernel_size, kernel_size), 0)

    # Calculate unsharp mask
    unsharp = cv2.addWeighted(bgr, 1.0 + strength, gaussian, -strength, 0)

    return Image.fromarray(cv2.cvtColor(unsharp, cv2.COLOR_BGR2RGB))


def preprocess_image(
    pil_img: Image.Image,
    *,
    apply_illumination: bool = True,
    apply_bilateral: bool = True,
    apply_normalize: bool = True,
    apply_unsharp: bool = False,
    illumination_clip: float = 2.0,
    bilateral_d: int = 9,
    bilateral_sigma_color: int = 75,
    bilateral_sigma_space: int = 75,
    unsharp_kernel: int = 5,
    unsharp_strength: float = 1.5,
) -> Image.Image:
    """Apply a comprehensive pre-processing pipeline to enhance image quality.

    Args:
        pil_img: Input PIL Image
        apply_illumination: Apply illumination correction
        apply_bilateral: Apply bilateral filtering
        apply_normalize: Apply color normalization
        apply_unsharp: Apply unsharp masking
        illumination_clip: CLAHE clip limit for illumination correction (range: 0.0-10.0, typical: 1.0-4.0)
        bilateral_d: Diameter of pixel neighborhood for bilateral filter (must be positive odd integer, typical: 5-15)
        bilateral_sigma_color: Filter sigma in color space (range: 10-200, typical: 50-100)
        bilateral_sigma_space: Filter sigma in coordinate space (range: 10-200, typical: 50-100)
        unsharp_kernel: Kernel size for unsharp mask (must be positive odd integer)
        unsharp_strength: Strength of unsharp mask effect (range: 0.5-3.0)

    Returns:
        Preprocessed PIL Image

    Raises:
        ValueError: If parameters are out of valid ranges
    """

    logging.debug("Starting image pre-processing pipeline")

    try:
        processed = pil_img

        if apply_illumination:
            processed = illumination_correction(processed, clip_limit=illumination_clip)

        if apply_bilateral:
            processed = bilateral_filter_preprocess(
                processed, d=bilateral_d, sigma_color=bilateral_sigma_color, sigma_space=bilateral_sigma_space
            )

        if apply_normalize:
            processed = normalize_colors(processed)

        if apply_unsharp:
            processed = unsharp_mask(processed, kernel_size=unsharp_kernel, strength=unsharp_strength)

        logging.debug("Pre-processing pipeline complete")
        return processed
    except Exception as exc:
        logging.error("Pre-processing failed: %s", exc)
        raise ValueError(f"Image pre-processing failed: {exc}") from exc


def affine_register(src_pil: Image.Image, dst_pil: Image.Image) -> np.ndarray:
    """Estimate an affine transform aligning ``src`` → ``dst`` using structural features.

    Uses edge-enhanced SIFT on arm contours and structural boundaries for robust
    registration based on arm geometry (outline, elbow, creases) rather than
    interior skin texture which may vary with lighting.
    """

    logging.debug("Computing affine registration using structural features")
    src = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2GRAY)
    dst = cv2.cvtColor(np.array(dst_pil), cv2.COLOR_RGB2GRAY)

    # Apply edge detection to emphasize structural features (arm outline, elbow, creases)
    # Using Canny edge detection to find high-gradient regions
    src_edges = cv2.Canny(src, 50, 150)
    dst_edges = cv2.Canny(dst, 50, 150)

    # Dilate edges slightly to create better feature extraction regions
    kernel = np.ones((3, 3), np.uint8)
    src_edges = cv2.dilate(src_edges, kernel, iterations=1)
    dst_edges = cv2.dilate(dst_edges, kernel, iterations=1)

    logging.debug("Extracting SIFT features from structural edges")
    sift = cv2.SIFT_create()

    # Use edge masks to focus SIFT detection on structural features
    # This prioritizes arm outline, elbow contours over interior skin texture
    keypoints_src, descriptors_src = sift.detectAndCompute(src, mask=src_edges)
    keypoints_dst, descriptors_dst = sift.detectAndCompute(dst, mask=dst_edges)

    if descriptors_src is None or descriptors_dst is None:
        logging.warning("Edge-based SIFT failed; falling back to full-image SIFT")
        # Fallback to regular SIFT if edge-based detection fails
        keypoints_src, descriptors_src = sift.detectAndCompute(src, None)
        keypoints_dst, descriptors_dst = sift.detectAndCompute(dst, None)

        if descriptors_src is None or descriptors_dst is None:
            raise ValueError("Unable to find distinctive features for alignment")

    logging.debug(
        "Found %d structural keypoints in source, %d in destination",
        len(keypoints_src), len(keypoints_dst)
    )

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors_src, descriptors_dst)
    if not matches:
        raise ValueError("Could not match keypoints between images")

    logging.debug("Matched %d keypoint pairs", len(matches))

    # Use top matches for RANSAC-based affine estimation
    matches = sorted(matches, key=lambda x: x.distance)[:60]
    src_pts = np.float32([keypoints_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, _ = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4
    )
    if matrix is None:
        raise ValueError("Affine transformation could not be estimated")

    logging.debug("Successfully estimated affine transformation from structural features")
    return matrix


def detect_papules_red(
    pil_img: Image.Image,
    *,
    min_area: int = 10,
    max_area: int = 500,
) -> List[Coordinate]:
    """Detect TWO injection sites arranged lengthwise along the arm axis.

    Uses geomorphological scale calibration and marker detection to:
    1. Detect +/- markers to establish test site location
    2. Calculate pixel-to-cm scale from arm width (typical forearm: 6-8 cm)
    3. Compute local arm orientation at the marker location
    4. Search for injection sites near markers at expected spacing (2-3 cm real-world)
    5. Verify sites are parallel to local arm axis at that location

    Args:
        pil_img: Input image
        min_area: Minimum area for a valid injection site (pixels)
        max_area: Maximum area for a valid injection site (pixels)

    Returns:
        List of two (x, y) coordinates for injection sites, or empty list if not found
    """

    logging.info("Detecting injection sites using geomorphological scale calibration")

    # Step 1: Detect markers to establish test site location
    markers, marker_centroid = detect_markers(pil_img)

    if marker_centroid is None:
        logging.warning("No markers detected; using global approach")
        # Fallback to image center
        height, width = np.array(pil_img).shape[:2]
        marker_centroid = (width // 2, height // 2)

    # Step 2: Establish scale using arm width
    # Typical forearm width is 6-8 cm, we'll use 7 cm as median
    arm_width_px = estimate_arm_width_px(pil_img, center_y=marker_centroid[1])
    assumed_arm_width_cm = 7.0  # Typical forearm width in cm
    pixels_per_cm = arm_width_px / assumed_arm_width_cm

    logging.info(
        "Scale calibration: %.1f px / cm (arm width: %.1f px ≈ %.1f cm)",
        pixels_per_cm, arm_width_px, assumed_arm_width_cm
    )

    # Step 3: Convert real-world pathergy test spacing to pixels
    # Standard protocol: injection sites are VERY close together (0.5-1.5 cm apart)
    # They are adjacent to the + and - markers
    min_distance_cm = 0.5
    max_distance_cm = 1.5
    min_distance_px = int(min_distance_cm * pixels_per_cm)
    max_distance_px = int(max_distance_cm * pixels_per_cm)

    logging.info(
        "Injection site spacing: %.1f-%.1f cm → %d-%d pixels",
        min_distance_cm, max_distance_cm, min_distance_px, max_distance_px
    )

    # Step 4: Compute LOCAL arm orientation at marker location
    # This accounts for arm curvature - orientation at test site, not global
    local_arm_angle = compute_local_arm_orientation(pil_img, marker_centroid, radius=150)

    # Step 5: Detect red regions (injection sites, excluding large markers)
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Relaxed thresholds for subtle redness in low-res images
    mask = cv2.inRange(hsv, (0, 40, 40), (15, 255, 255)) | cv2.inRange(
        hsv, (165, 40, 40), (180, 255, 255)
    )
    mask = cv2.medianBlur(mask, 3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract potential injection sites (filter by size and proximity to markers)
    candidates: List[Tuple[int, int, float, float]] = []
    search_radius_px = int(3.0 * pixels_per_cm)  # Search within 3 cm of markers (sites are very close)

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:  # Filter by size (excludes large markers)
            moments = cv2.moments(contour)
            if moments["m00"]:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                # Check proximity to marker region
                dist_to_markers = math.hypot(cx - marker_centroid[0], cy - marker_centroid[1])
                logging.debug(
                    "Candidate at (%d, %d): area=%.1f, dist_to_markers=%.1f (max=%.1f)",
                    cx, cy, area, dist_to_markers, search_radius_px
                )
                if dist_to_markers > search_radius_px:
                    logging.debug("  -> Rejected: too far from markers")
                    continue  # Too far from markers

                # Calculate circularity (injection sites should be roughly circular)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * math.pi * area) / (perimeter * perimeter + 1e-6)

                logging.debug("  -> Accepted: circularity=%.2f", circularity)
                candidates.append((cx, cy, area, circularity))

    if len(candidates) < 2:
        logging.warning("Found fewer than 2 injection site candidates near markers")
        return [(x, y) for x, y, _, _ in candidates]

    logging.debug(
        "Found %d injection site candidates within %.1f cm of markers",
        len(candidates), search_radius_px / pixels_per_cm
    )

    # Step 6: Find best pair aligned LENGTHWISE with local arm axis at test site
    best_pair = None
    best_score = float('inf')
    best_alignment = 0.0

    for i, (x1, y1, area1, circ1) in enumerate(candidates):
        for j, (x2, y2, area2, circ2) in enumerate(candidates[i + 1:], start=i + 1):
            # Calculate distance between potential injection sites
            distance = math.hypot(x2 - x1, y2 - y1)

            if min_distance_px <= distance <= max_distance_px:
                # Calculate angle of line connecting the two points
                pair_angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
                if pair_angle < 0:
                    pair_angle += 180

                # Angular difference from LOCAL arm axis at test site
                # We want sites arranged PARALLEL to arm at marker location
                angle_diff = min(
                    abs(pair_angle - local_arm_angle),
                    abs(pair_angle - local_arm_angle + 180),
                    abs(pair_angle - local_arm_angle - 180)
                )

                # Score components (all lower is better):
                # 1. Distance from ideal spacing (2.5 cm center of range)
                ideal_distance_cm = 2.5
                ideal_distance_px = ideal_distance_cm * pixels_per_cm
                distance_score = abs(distance - ideal_distance_px)

                # 2. Size similarity (both sites should be similar size)
                size_diff = abs(area1 - area2) / max(area1, area2)

                # 3. Circularity (both should be circular)
                circ_score = 2.0 - (circ1 + circ2)

                # 4. Alignment with LOCAL arm axis (CRITICAL - must be parallel)
                # Heavy penalty for perpendicular arrangements
                alignment_score = angle_diff

                # Combined score (lower is better)
                # Weight alignment most heavily - sites MUST be lengthwise
                score = (
                    distance_score +
                    (size_diff * 50) +
                    (circ_score * 100) +
                    (alignment_score * 5.0)  # 5x weight for parallelism
                )

                if score < best_score:
                    best_score = score
                    best_pair = ((x1, y1), (x2, y2))
                    best_alignment = angle_diff

    if best_pair:
        pair_distance_px = math.hypot(
            best_pair[1][0] - best_pair[0][0],
            best_pair[1][1] - best_pair[0][1]
        )
        pair_distance_cm = pair_distance_px / pixels_per_cm

        logging.info(
            "Found injection site pair: distance=%.2f cm (%.1f px), "
            "alignment=%.1f° from local arm axis, score=%.2f",
            pair_distance_cm, pair_distance_px, best_alignment, best_score
        )
        # Sort by position along arm axis for consistent ordering
        return sorted(best_pair, key=lambda p: p[0])
    else:
        logging.warning("No valid lengthwise injection site pair found; returning largest candidates")
        # Fallback: return two largest candidates
        candidates.sort(key=lambda t: t[2], reverse=True)
        return [(x, y) for x, y, _, _ in candidates[:2]]


def detect_papules_dark(
    pil_img: Image.Image, *, scan_region: Sequence[int] | None = None
) -> List[Coordinate]:
    """Detect darker papular regions using CLAHE + adaptive thresholding."""

    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    if not scan_region:
        x0, y0 = int(width * 0.25), int(height * 0.45)
        x1, y1 = int(width * 0.80), int(height * 0.85)
    else:
        x0, y0, x1, y1 = scan_region
    crop = gray[y0:y1, x0:x1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(crop)
    threshold_value = int(np.mean(enhanced) - 10)
    _, threshold = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY_INV)
    threshold = cv2.medianBlur(threshold, 3)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 18 <= area <= 420:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * math.pi * area) / (perimeter * perimeter + 1e-6)
            blobs.append((cx, cy, area, circularity))
    best_pair = None
    best_score = float("inf")
    for i, blob_a in enumerate(blobs):
        for blob_b in blobs[i + 1 :]:
            distance = math.hypot(blob_a[0] - blob_b[0], blob_a[1] - blob_b[1])
            if 15 <= distance <= 65:
                score = distance - 5 * (blob_a[3] + blob_b[3])
                if score < best_score:
                    best_score = score
                    best_pair = (blob_a, blob_b)
    if not best_pair:
        return []
    blob_a, blob_b = best_pair
    return [
        (x0 + int(blob_a[0]), y0 + int(blob_a[1])),
        (x0 + int(blob_b[0]), y0 + int(blob_b[1])),
    ]


def visualize_hsv_mask(pil_img: Image.Image, min_area: int = 30) -> Image.Image:
    """Create a visualization overlay showing HSV-based red detection mask."""

    logging.debug("Creating HSV mask visualization")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Create red mask (same as detect_papules_red)
    mask = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255)) | cv2.inRange(
        hsv, (170, 60, 60), (180, 255, 255)
    )
    mask = cv2.medianBlur(mask, 3)

    # Create colored overlay: red mask on semi-transparent original
    overlay = bgr.copy()
    overlay[mask > 0] = [0, 0, 255]  # Red in BGR

    # Blend with original image
    result = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def visualize_contours(pil_img: Image.Image, points: Sequence[Coordinate]) -> Image.Image:
    """Create a visualization showing detected contours and their properties."""

    logging.debug("Creating contour visualization for %d points", len(points))
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255)) | cv2.inRange(
        hsv, (170, 60, 60), (180, 255, 255)
    )
    mask = cv2.medianBlur(mask, 3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours in green
    result = bgr.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Draw circles at detected points
    for x, y in points:
        cv2.circle(result, (x, y), 10, (255, 0, 255), 2)  # Magenta circles
        cv2.circle(result, (x, y), 2, (255, 0, 255), -1)   # Center dot

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def visualize_dark_detection(
    pil_img: Image.Image, points: Sequence[Coordinate], scan_region: Sequence[int] | None = None
) -> Image.Image:
    """Create a visualization showing dark papule detection process."""

    logging.debug("Creating dark detection visualization")
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if not scan_region:
        x0, y0 = int(width * 0.25), int(height * 0.45)
        x1, y1 = int(width * 0.80), int(height * 0.85)
    else:
        x0, y0, x1, y1 = scan_region

    # Draw scan region rectangle
    result = bgr.copy()
    cv2.rectangle(result, (x0, y0), (x1, y1), (255, 255, 0), 2)  # Yellow rectangle

    # Apply CLAHE and thresholding
    crop = gray[y0:y1, x0:x1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(crop)
    threshold_value = int(np.mean(enhanced) - 10)
    _, threshold = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY_INV)
    threshold = cv2.medianBlur(threshold, 3)

    # Create colored threshold overlay
    threshold_colored = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    threshold_colored[threshold > 0] = [0, 255, 255]  # Cyan for detected regions

    # Blend threshold with result in the scan region
    result[y0:y1, x0:x1] = cv2.addWeighted(
        result[y0:y1, x0:x1], 0.7, threshold_colored, 0.3, 0
    )

    # Draw detected points
    for x, y in points:
        cv2.circle(result, (x, y), 10, (255, 0, 255), 2)  # Magenta circles
        cv2.circle(result, (x, y), 2, (255, 0, 255), -1)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def visualize_structural_edges(pil_img: Image.Image) -> Image.Image:
    """Create visualization showing structural edges used for registration.

    Shows Canny edges (arm outline, elbow, creases) that are used for
    structural feature-based registration.
    """

    logging.debug("Creating structural edge visualization")
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Apply edge detection (same as registration)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Create colored overlay: edges in cyan on semi-transparent original
    overlay = bgr.copy()
    overlay[edges > 0] = [255, 255, 0]  # Cyan (yellow in BGR) for edges

    # Blend with original image
    result = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def visualize_arm_orientation(pil_img: Image.Image) -> Image.Image:
    """Create visualization showing detected arm orientation axis.

    Overlays the principal axis (long axis) of the arm in green,
    showing the direction used for lengthwise injection site detection.
    """

    logging.debug("Creating arm orientation visualization")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    skin_mask = segment_skin_region(pil_img)

    # Get coordinates and compute PCA
    ys, xs = np.nonzero(skin_mask)
    if len(xs) < 10:
        return pil_img  # Return original if no skin found

    coords = np.column_stack([xs, ys])
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Get principal axis
    principal_idx = np.argmax(eigenvalues)
    principal_axis = eigenvectors[:, principal_idx].real

    # Draw the principal axis as a line through the centroid
    height, width = skin_mask.shape
    axis_length = min(width, height) // 2

    # Calculate endpoints of the axis line
    center_x, center_y = int(mean[0]), int(mean[1])
    dx = principal_axis[0] * axis_length
    dy = principal_axis[1] * axis_length

    pt1 = (int(center_x - dx), int(center_y - dy))
    pt2 = (int(center_x + dx), int(center_y + dy))

    # Draw on image
    result = bgr.copy()
    cv2.line(result, pt1, pt2, (0, 255, 0), 3)  # Green line for arm axis
    cv2.circle(result, (center_x, center_y), 8, (0, 255, 0), -1)  # Green dot at centroid

    # Add angle label
    angle = np.arctan2(principal_axis[1], principal_axis[0]) * 180 / np.pi
    if angle < 0:
        angle += 180
    cv2.putText(
        result, f"Arm axis: {angle:.1f}deg",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def visualize_scale_calibration(pil_img: Image.Image) -> Image.Image:
    """Create visualization showing scale calibration and marker detection.

    Shows:
    - Detected +/- markers
    - Arm width measurement
    - Calculated scale (pixels per cm)
    - Local arm axis at test site
    """

    logging.debug("Creating scale calibration visualization")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result = bgr.copy()

    # Detect markers
    markers, marker_centroid = detect_markers(pil_img)

    if marker_centroid is None:
        height, width = bgr.shape[:2]
        marker_centroid = (width // 2, height // 2)

    # Draw markers if detected
    for i, (mx, my) in enumerate(markers[:5], start=1):  # Show up to 5 markers
        cv2.circle(result, (mx, my), 10, (255, 0, 255), 2)  # Magenta circles
        cv2.putText(
            result, f"M{i}",
            (mx + 15, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
        )

    # Draw marker centroid
    if marker_centroid:
        cv2.circle(result, marker_centroid, 15, (0, 255, 255), 3)  # Cyan for centroid
        cv2.putText(
            result, "Test site",
            (marker_centroid[0] + 20, marker_centroid[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

    # Measure and draw arm width
    arm_width_px = estimate_arm_width_px(pil_img, center_y=marker_centroid[1])
    assumed_arm_width_cm = 7.0
    pixels_per_cm = arm_width_px / assumed_arm_width_cm

    # Draw arm width measurement line
    skin_mask = segment_skin_region(pil_img)
    cy = marker_centroid[1]
    row = skin_mask[cy, :]
    xs = np.nonzero(row)[0]
    if len(xs) > 0:
        x_left, x_right = xs.min(), xs.max()
        cv2.line(result, (x_left, cy), (x_right, cy), (0, 255, 0), 2)
        cv2.circle(result, (x_left, cy), 5, (0, 255, 0), -1)
        cv2.circle(result, (x_right, cy), 5, (0, 255, 0), -1)

        # Add width label
        mid_x = (x_left + x_right) // 2
        cv2.putText(
            result, f"{arm_width_px:.0f}px ≈ {assumed_arm_width_cm}cm",
            (mid_x - 80, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # Draw local arm axis at test site
    local_angle = compute_local_arm_orientation(pil_img, marker_centroid, radius=150)
    axis_length = 100

    angle_rad = (local_angle * np.pi / 180)
    dx = np.cos(angle_rad) * axis_length
    dy = np.sin(angle_rad) * axis_length

    pt1 = (int(marker_centroid[0] - dx), int(marker_centroid[1] - dy))
    pt2 = (int(marker_centroid[0] + dx), int(marker_centroid[1] + dy))

    cv2.line(result, pt1, pt2, (255, 255, 0), 2)  # Yellow line for local axis

    # Add scale info
    y_offset = 30
    cv2.putText(
        result, f"Scale: {pixels_per_cm:.1f} px/cm",
        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        result, f"Local axis: {local_angle:.1f}deg",
        (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def visualize_sift_features(
    src_pil: Image.Image, dst_pil: Image.Image, num_features: int = 30
) -> Tuple[Image.Image, Image.Image]:
    """Create visualizations showing SIFT keypoints on both images."""

    logging.debug("Creating SIFT feature visualization")
    src = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2GRAY)
    dst = cv2.cvtColor(np.array(dst_pil), cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    keypoints_src, _ = sift.detectAndCompute(src, None)
    keypoints_dst, _ = sift.detectAndCompute(dst, None)

    # Draw keypoints (limit to top N by response)
    keypoints_src_sorted = sorted(keypoints_src, key=lambda k: k.response, reverse=True)[:num_features]
    keypoints_dst_sorted = sorted(keypoints_dst, key=lambda k: k.response, reverse=True)[:num_features]

    src_bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    dst_bgr = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    src_with_kp = cv2.drawKeypoints(
        src_bgr, keypoints_src_sorted, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )
    dst_with_kp = cv2.drawKeypoints(
        dst_bgr, keypoints_dst_sorted, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )

    return (
        Image.fromarray(cv2.cvtColor(src_with_kp, cv2.COLOR_BGR2RGB)),
        Image.fromarray(cv2.cvtColor(dst_with_kp, cv2.COLOR_BGR2RGB))
    )


def visualize_morphological_operations(
    pil_img: Image.Image, threshold: int = 5, kernel_size: int = 7
) -> Image.Image:
    """Create visualization showing morphological closing and opening operations."""

    logging.debug("Creating morphological operations visualization")
    mask = _content_mask(pil_img, threshold=threshold)
    mask_uint8 = mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply closing
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply opening
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Create colored visualization
    # Original mask: Blue, Closed: Green, Final (opened): Red
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result = bgr.copy()

    # Create colored overlays
    overlay = np.zeros_like(bgr)
    overlay[mask_uint8 > 0] = [255, 0, 0]  # Original in blue
    overlay[closed > 0] = [0, 255, 0]      # Closed in green
    overlay[opened > 0] = [0, 0, 255]      # Final in red

    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def create_diagnostic_panel(
    original: Image.Image,
    preprocessed: Image.Image,
    hsv_overlay: Image.Image,
    contour_overlay: Image.Image,
    *,
    padding: int = 10,
) -> Image.Image:
    """Create a 2x2 diagnostic panel showing preprocessing and detection stages."""

    logging.debug("Creating diagnostic panel")

    # Resize all images to same size
    target_width = min(original.size[0], 800)
    target_height = min(original.size[1], 600)

    images = [original, preprocessed, hsv_overlay, contour_overlay]
    resized = []
    for img in images:
        resized.append(img.resize((target_width, target_height), Image.Resampling.LANCZOS))

    # Create 2x2 grid
    grid_width = 2 * target_width + padding
    grid_height = 2 * target_height + padding
    panel = Image.new("RGB", (grid_width, grid_height), (0, 0, 0))

    # Add labels
    labels = ["Original", "Preprocessed", "HSV Mask", "Contours"]
    for idx, (img, label) in enumerate(zip(resized, labels)):
        x = (idx % 2) * (target_width + padding)
        y = (idx // 2) * (target_height + padding)

        # Add label to image
        labeled = img.copy()
        draw = ImageDraw.Draw(labeled)
        draw.text((10, 10), label, fill=(255, 255, 255))

        panel.paste(labeled, (x, y))

    return panel


def transform_points(points: Iterable[Coordinate], matrix: np.ndarray) -> List[Coordinate]:
    """Apply an affine transformation matrix to a sequence of points."""

    transformed: List[Coordinate] = []
    for x, y in points:
        vector = np.dot(matrix, np.array([x, y, 1]))
        transformed.append((int(vector[0]), int(vector[1])))
    return transformed


def warp_to_base(src_pil: Image.Image, matrix: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    """Warp ``src`` image to ``size`` using the provided affine ``matrix``."""

    src = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2BGR)
    warped = cv2.warpAffine(src, matrix, size, flags=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def common_content_bbox(panels: Sequence[Image.Image]) -> Tuple[int, int, int, int] | None:
    """Compute a bounding box that contains shared non-empty content across ``panels``."""

    if not panels:
        return None

    masks = []
    for panel in panels:
        gray = np.array(panel.convert("L"))
        mask = gray > 0
        if not np.any(mask):
            return None
        masks.append(mask)

    combined = np.logical_and.reduce(masks)
    if not np.any(combined):
        return None

    ys, xs = np.nonzero(combined)
    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()
    # Pillow crop uses half-open coordinates, so include the final pixel by +1.
    return int(left), int(top), int(right) + 1, int(bottom) + 1
def _content_mask(pil_img: Image.Image, *, threshold: int = 5) -> np.ndarray:
    """Return a boolean mask of pixels that contain visual content."""

    array = np.array(pil_img)
    if array.ndim == 3:  # RGB
        mask = np.any(array > threshold, axis=2)
    else:
        mask = array > threshold
    return mask


def _refine_mask(
    mask: np.ndarray,
    *,
    kernel_size: int = 5,
    min_component_area: int = 500,
) -> np.ndarray:
    """Use morphological operations to clean noisy mask regions."""

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)

    if min_component_area <= 0:
        return refined

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined)
    filtered = np.zeros_like(refined)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_component_area:
            filtered[labels == idx] = 255
    return filtered


def _component_areas(mask: np.ndarray) -> List[int]:
    """Return the areas of connected components within ``mask``."""

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, num_labels)]


def common_content_bbox(
    images: Sequence[Image.Image],
    *,
    threshold: int = 5,
    kernel_size: int = 7,
    min_component_area: int = 1000,
) -> Tuple[int, int, int, int] | None:
    """Find the bounding box shared by non-empty pixels across ``images``.

    The shared mask is computed with morphological closing/opening and component
    filtering to emphasise the dominant overlapping footprint while suppressing
    small mismatches.
    """

    if not images:
        return None

    logger = logging.getLogger(__name__)

    # Verify all images have the same size (they should after warping and cropping)
    sizes = [img.size for img in images]
    if len(set(sizes)) > 1:
        logger.error(
            "CRITICAL: Images have different sizes: %s. This indicates a bug in warping/cropping!",
            sizes
        )
        raise ValueError(
            f"Images must have the same size for mask intersection. Got sizes: {sizes}. "
            "This suggests the warping or cropping step failed to produce aligned images."
        )

    refined_masks = []
    for idx, image in enumerate(images, start=1):
        raw_mask = _content_mask(image, threshold=threshold)
        refined = _refine_mask(
            raw_mask,
            kernel_size=kernel_size,
            min_component_area=min_component_area,
        )
        refined_masks.append(refined)

        component_areas = _component_areas(refined)
        if component_areas:
            logger.debug(
                "Content mask %d retains %d component(s); minimum area=%d px",
                idx,
                len(component_areas),
                min(component_areas),
            )
        else:
            logger.debug("Content mask %d retains no components", idx)

    intersection = refined_masks[0]
    for mask in refined_masks[1:]:
        intersection = cv2.bitwise_and(intersection, mask)

    intersection = _refine_mask(
        intersection,
        kernel_size=kernel_size,
        min_component_area=min_component_area,
    )

    shared_component_areas = _component_areas(intersection)
    if shared_component_areas:
        logger.info(
            "Shared visual-footprint retains %d component(s); minimum shared component area=%d px",
            len(shared_component_areas),
            min(shared_component_areas),
        )
    else:
        logger.info(
            "Shared visual-footprint retains no components after refinement"
        )

    coords = cv2.findNonZero(intersection)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if w == 0 or h == 0:
        return None

    left, top = int(x), int(y)
    right, bottom = int(x + w), int(y + h)
    return (left, top, right, bottom)


def crop_images(
    images: Sequence[Image.Image],
    points: Sequence[Sequence[Coordinate]],
    bbox: Tuple[int, int, int, int],
) -> Tuple[List[Image.Image], List[List[Coordinate]]]:
    """Crop ``images`` and translate ``points`` into the cropped frame."""

    left, top, right, bottom = bbox
    image_list = list(images)
    points_list = [list(point_seq) for point_seq in points]

    if len(image_list) != len(points_list):
        raise ValueError("Number of image panels and point collections must match")

    cropped_images = [image.crop(bbox) for image in image_list]
    translated_points = [
        [(x - left, y - top) for x, y in point_seq]
        for point_seq in points_list
    ]

    return cropped_images, translated_points


def draw_boxes(
    pil_img: Image.Image,
    points: Sequence[Coordinate],
    label: str,
    *,
    radius: int,
) -> Image.Image:
    """Annotate the supplied image with bounding boxes and a label."""

    annotated = pil_img.copy()
    drawer = ImageDraw.Draw(annotated)
    for i, (x, y) in enumerate(points[:2], start=1):
        drawer.rectangle([x - radius, y - radius, x + radius, y + radius], outline=(255, 0, 0), width=4)
        drawer.text((x + radius + 6, y - 10), f"P{i}", fill=(255, 0, 0))
    drawer.text((10, 10), label, fill=(255, 255, 255))
    return annotated


def build_montage(
    panels: Sequence[Image.Image],
    *,
    padding: int,
    caption: str | None = None,
) -> Image.Image:
    """Concatenate annotated panels into a single montage."""

    if not panels:
        raise ValueError("No panels were provided for montage generation")

    width, height = panels[0].size
    montage_width = width * len(panels) + padding * (len(panels) - 1)
    montage = Image.new("RGB", (montage_width, height), (0, 0, 0))
    for idx, panel in enumerate(panels):
        montage.paste(panel, (idx * (width + padding), 0))

    if caption:
        drawer = ImageDraw.Draw(montage)
        drawer.text((10, height - 30), caption, fill=(255, 255, 255))
    return montage


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Align and annotate serial pathergy test images into a composite timeline."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("day1_0h.png"),
        help="Baseline image path (default: day1_0h.png)",
    )
    parser.add_argument(
        "--early",
        type=Path,
        default=Path("day1_24h.png"),
        help="Early follow-up image path (default: day1_24h.png)",
    )
    parser.add_argument(
        "--late",
        type=Path,
        default=Path("day2_48h.png"),
        help="Late follow-up image path (default: day2_48h.png)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pathergy_timeline_composite.jpg"),
        help="Output montage filename (default: pathergy_timeline_composite.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write outputs (default: same directory as baseline image)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=22,
        help="Bounding box half-size in pixels (default: 22)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Horizontal padding between panels in pixels (default: 20)",
    )
    parser.add_argument(
        "--content-threshold",
        type=int,
        default=5,
        help=(
            "Pixel intensity threshold for shared visual-footprint detection "
            "(default: 5)"
        ),
    )
    parser.add_argument(
        "--content-kernel-size",
        type=int,
        default=7,
        help=(
            "Morphological kernel size for shared visual-footprint detection "
            "(default: 7)"
        ),
    )
    parser.add_argument(
        "--content-min-component-area",
        type=int,
        default=1000,
        help=(
            "Minimum connected component area kept during shared visual-"
            "footprint detection (default: 1000)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--enable-preprocessing",
        action="store_true",
        help="Enable comprehensive pre-processing pipeline (illumination correction, bilateral filtering, color normalization).",
    )
    parser.add_argument(
        "--enable-unsharp",
        action="store_true",
        help="Enable unsharp masking for edge enhancement (requires --enable-preprocessing).",
    )
    parser.add_argument(
        "--generate-diagnostic-panels",
        action="store_true",
        help="Generate diagnostic visualization panels showing morphological features and processing steps.",
    )
    parser.add_argument(
        "--save-morphological-overlays",
        action="store_true",
        help="Save individual morphological overlay images (HSV masks, contours, SIFT features).",
    )
    parser.add_argument(
        "--illumination-clip",
        type=float,
        default=2.0,
        help="CLAHE clip limit for illumination correction (default: 2.0).",
    )
    parser.add_argument(
        "--bilateral-d",
        type=int,
        default=9,
        help="Diameter of pixel neighborhood for bilateral filter (default: 9).",
    )
    parser.add_argument(
        "--bilateral-sigma-color",
        type=int,
        default=75,
        help="Filter sigma in color space for bilateral filter (default: 75).",
    )
    parser.add_argument(
        "--bilateral-sigma-space",
        type=int,
        default=75,
        help="Filter sigma in coordinate space for bilateral filter (default: 75).",
    )
    return parser.parse_args(argv)


def ensure_inputs_exist(paths: Iterable[Path]) -> None:
    """Ensure that each path exists before processing."""

    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required input file(s): {joined}")


def run_pipeline(args: argparse.Namespace) -> Path:
    """Execute the alignment and montage generation pipeline."""

    configure_logging(args.log_level)
    ensure_inputs_exist([args.baseline, args.early, args.late])

    # Set output directory to baseline image directory if not specified
    if args.output_dir is None:
        args.output_dir = args.baseline.parent
        logging.info("Output directory not specified; using baseline image directory: %s", args.output_dir)

    logging.info("Loading images")
    baseline_original = load_image(args.baseline)
    early_original = load_image(args.early)
    late_original = load_image(args.late)

    logging.info("Performing intelligent pre-crop to identify forearm regions independently")
    # Identify forearm in each image independently using skin segmentation
    # This removes background noise/hands/elbows while keeping enough for registration
    baseline_precrop, baseline_bbox = intelligent_precrop(baseline_original, margin_percent=0.15)
    early_precrop, early_bbox = intelligent_precrop(early_original, margin_percent=0.15)
    late_precrop, late_bbox = intelligent_precrop(late_original, margin_percent=0.15)

    logging.info("Pre-crop complete - forearm regions identified with 15%% safety margin")

    # Save intermediate pre-cropped images for debugging
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_precrop_path = output_dir / "intermediate_baseline_precrop.jpg"
    early_precrop_path = output_dir / "intermediate_early_precrop.jpg"
    late_precrop_path = output_dir / "intermediate_late_precrop.jpg"

    baseline_precrop.save(baseline_precrop_path, quality=95)
    early_precrop.save(early_precrop_path, quality=95)
    late_precrop.save(late_precrop_path, quality=95)

    logging.info("Saved intermediate pre-cropped images to %s", output_dir)

    # Apply pre-processing if enabled (on pre-cropped images for better registration)
    if args.enable_preprocessing:
        logging.info("Applying pre-processing pipeline to pre-cropped images")
        baseline = preprocess_image(
            baseline_precrop,
            apply_illumination=True,
            apply_bilateral=True,
            apply_normalize=True,
            apply_unsharp=args.enable_unsharp,
            illumination_clip=args.illumination_clip,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
        )
        early = preprocess_image(
            early_precrop,
            apply_illumination=True,
            apply_bilateral=True,
            apply_normalize=True,
            apply_unsharp=args.enable_unsharp,
            illumination_clip=args.illumination_clip,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
        )
        late = preprocess_image(
            late_precrop,
            apply_illumination=True,
            apply_bilateral=True,
            apply_normalize=True,
            apply_unsharp=args.enable_unsharp,
            illumination_clip=args.illumination_clip,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
        )
    else:
        # Use pre-cropped images even without additional preprocessing
        baseline = baseline_precrop
        early = early_precrop
        late = late_precrop

    base_width, base_height = baseline.size
    logging.debug("Pre-cropped baseline image size: %s x %s", base_width, base_height)

    logging.info("Registering pre-cropped follow-up images to baseline")
    # Registration now uses pre-cropped images (forearm + margin) for cleaner SIFT matching
    # Background noise removed while keeping enough features for robust alignment
    matrix_early_to_base = affine_register(early, baseline)
    matrix_late_to_base = affine_register(late, baseline)

    logging.info("Warping follow-up images to baseline coordinate frame")
    early_warped = warp_to_base(early, matrix_early_to_base, (base_width, base_height))
    late_warped = warp_to_base(late, matrix_late_to_base, (base_width, base_height))

    logging.info("Identifying shared anatomical region across all ALIGNED images")
    logging.info(
        "Shared content detection params: threshold=%d, kernel_size=%d, min_component_area=%d",
        args.content_threshold,
        args.content_kernel_size,
        args.content_min_component_area,
    )
    # Find shared visual footprint AFTER registration on aligned images
    # This identifies the forearm region and eliminates hands, elbows, background
    shared_bbox = common_content_bbox(
        [baseline, early_warped, late_warped],
        threshold=args.content_threshold,
        kernel_size=args.content_kernel_size,
        min_component_area=args.content_min_component_area,
    )

    if shared_bbox:
        left, top, right, bottom = shared_bbox
        logging.info(
            "Shared anatomical region found: left=%d, top=%d, right=%d, bottom=%d",
            left, top, right, bottom
        )
        # Crop all aligned images to shared forearm region
        baseline_cropped = baseline.crop(shared_bbox)
        early_warped_cropped = early_warped.crop(shared_bbox)
        late_warped_cropped = late_warped.crop(shared_bbox)
        logging.info("Images cropped to shared forearm region: %d x %d", baseline_cropped.size[0], baseline_cropped.size[1])
    else:
        logging.warning("No shared anatomical region found; using full aligned images")
        logging.warning("This may include hands, elbows, or background in detection")
        baseline_cropped = baseline
        early_warped_cropped = early_warped
        late_warped_cropped = late_warped

    logging.info("Detecting puncture sites on cropped baseline image (Day 0)")
    baseline_points = detect_papules_red(baseline_cropped)
    if not baseline_points:
        logging.warning("No puncture sites detected in baseline image")
        logging.warning("Consider adjusting detection parameters or checking image quality")

    logging.info("Tracking Day 0 puncture sites across all registered, cropped timepoints")
    # Since all images are in baseline coordinate space and cropped to same region,
    # we use the same anatomical locations across all timepoints
    early_points_base = baseline_points
    late_points_base = baseline_points
    logging.info(
        "Tracking %d puncture site(s) at same anatomical locations across all timepoints",
        len(baseline_points)
    )

    logging.info("Creating annotated panels")
    annotated_baseline = draw_boxes(baseline_cropped, baseline_points, "Day 0 (baseline)", radius=args.radius)
    annotated_early = draw_boxes(
        early_warped_cropped, early_points_base, "Day 1 (~24h)", radius=args.radius
    )
    annotated_late = draw_boxes(
        late_warped_cropped, late_points_base, "Day 2 (~48h)", radius=args.radius
    )

    panels = [annotated_baseline, annotated_early, annotated_late]

    # Final refinement: crop annotated panels to remove any black borders from warping
    logging.debug("Applying final crop to remove warping artifacts from annotated panels")
    bbox = common_content_bbox(panels, threshold=args.content_threshold)
    if bbox is None:
        logging.debug("No refinement crop needed; panels are clean")
    else:
        left, top, right, bottom = bbox
        bbox_width = right - left
        bbox_height = bottom - top
        logging.debug(
            "Final refinement crop: left=%d, top=%d, right=%d, bottom=%d",
            left, top, right, bottom
        )
        panels = [panel.crop(bbox) for panel in panels]

    logging.info("Building montage")
    montage = build_montage(
        panels,
        padding=args.padding,
        caption="Pathergy Test Timeline (Baseline-aligned)",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    logging.info("Saving montage to %s", output_path)
    montage.save(output_path, quality=95)

    # Generate morphological overlays if requested
    if args.save_morphological_overlays or args.generate_diagnostic_panels:
        logging.info("Generating morphological visualizations")

        # Generate HSV mask overlays (on original baseline, shows what was detected)
        baseline_hsv = visualize_hsv_mask(baseline_cropped)

        # Generate structural feature visualizations
        baseline_edges = visualize_structural_edges(baseline)
        baseline_arm_axis = visualize_arm_orientation(baseline)
        baseline_scale = visualize_scale_calibration(baseline_cropped)

        # Generate contour overlays showing tracked puncture sites
        # For baseline: show detection on original image
        baseline_contours = visualize_contours(baseline_cropped, baseline_points)
        # For warped images: show the same anatomical locations being tracked
        early_contours = visualize_contours(early_warped_cropped, early_points_base)
        late_contours = visualize_contours(late_warped_cropped, late_points_base)

        # Generate SIFT visualizations - use the images that were actually registered
        # If preprocessing enabled, show SIFT on preprocessed images; otherwise on originals
        sift_early_src = early if args.enable_preprocessing else early_original
        sift_late_src = late if args.enable_preprocessing else late_original
        sift_baseline_src = baseline if args.enable_preprocessing else baseline_original

        early_sift, baseline_sift_early = visualize_sift_features(sift_early_src, sift_baseline_src)
        late_sift, baseline_sift_late = visualize_sift_features(sift_late_src, sift_baseline_src)

        if args.save_morphological_overlays:
            logging.info("Saving individual morphological overlays")
            baseline_hsv.save(output_dir / "baseline_hsv_mask.jpg", quality=95)
            baseline_edges.save(output_dir / "baseline_structural_edges.jpg", quality=95)
            baseline_arm_axis.save(output_dir / "baseline_arm_orientation.jpg", quality=95)
            baseline_scale.save(output_dir / "baseline_scale_calibration.jpg", quality=95)
            baseline_contours.save(output_dir / "baseline_contours_detected.jpg", quality=95)
            early_contours.save(output_dir / "early_tracked_sites.jpg", quality=95)
            late_contours.save(output_dir / "late_tracked_sites.jpg", quality=95)
            early_sift.save(output_dir / "early_sift_features.jpg", quality=95)
            late_sift.save(output_dir / "late_sift_features.jpg", quality=95)
            logging.info("Morphological overlays saved to %s", output_dir)

    # Generate diagnostic panels if requested
    if args.generate_diagnostic_panels:
        logging.info("Generating diagnostic panels")

        # Baseline diagnostic panel: Shows scale calibration and detection process
        baseline_panel = create_diagnostic_panel(
            baseline_original,
            baseline_scale,
            baseline_hsv,
            baseline_contours,
            padding=10,
        )
        baseline_panel_path = output_dir / "baseline_diagnostic_panel.jpg"
        baseline_panel.save(baseline_panel_path, quality=95)
        logging.info("Baseline diagnostic panel saved to %s", baseline_panel_path)

        # Early diagnostic panel: Shows tracking of Day 0 sites on Day 1
        early_panel = create_diagnostic_panel(
            early_original,
            early_warped,
            early_sift,
            early_contours,
            padding=10,
        )
        early_panel_path = output_dir / "early_diagnostic_panel.jpg"
        early_panel.save(early_panel_path, quality=95)
        logging.info("Early diagnostic panel saved to %s", early_panel_path)

        # Late diagnostic panel: Shows tracking of Day 0 sites on Day 2+
        late_panel = create_diagnostic_panel(
            late_original,
            late_warped,
            late_sift,
            late_contours,
            padding=10,
        )
        late_panel_path = output_dir / "late_diagnostic_panel.jpg"
        late_panel.save(late_panel_path, quality=95)
        logging.info("Late diagnostic panel saved to %s", late_panel_path)

    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(argv)
    try:
        output_path = run_pipeline(args)
    except Exception as exc:  # pragma: no cover - CLI surface
        logging.getLogger(__name__).exception("Pipeline failed: %s", exc)
        return 1
    print(f"Composite saved to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())

