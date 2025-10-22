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
        illumination_clip: CLAHE clip limit for illumination correction
        bilateral_d: Diameter of pixel neighborhood for bilateral filter
        bilateral_sigma_color: Filter sigma in color space
        bilateral_sigma_space: Filter sigma in coordinate space
        unsharp_kernel: Kernel size for unsharp mask
        unsharp_strength: Strength of unsharp mask effect

    Returns:
        Preprocessed PIL Image
    """

    logging.debug("Starting image pre-processing pipeline")
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


def affine_register(src_pil: Image.Image, dst_pil: Image.Image) -> np.ndarray:
    """Estimate an affine transform aligning ``src`` → ``dst`` using SIFT + RANSAC."""

    logging.debug("Computing affine registration")
    src = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2GRAY)
    dst = cv2.cvtColor(np.array(dst_pil), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints_src, descriptors_src = sift.detectAndCompute(src, None)
    keypoints_dst, descriptors_dst = sift.detectAndCompute(dst, None)

    if descriptors_src is None or descriptors_dst is None:
        raise ValueError("Unable to find distinctive features for alignment")

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors_src, descriptors_dst)
    if not matches:
        raise ValueError("Could not match keypoints between images")

    matches = sorted(matches, key=lambda x: x.distance)[:60]
    src_pts = np.float32([keypoints_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, _ = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4
    )
    if matrix is None:
        raise ValueError("Affine transformation could not be estimated")

    return matrix


def detect_papules_red(
    pil_img: Image.Image, *, min_area: int = 30, max_cnt: int = 2
) -> List[Coordinate]:
    """Detect red papular regions using HSV thresholding."""

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255)) | cv2.inRange(
        hsv, (170, 60, 60), (180, 255, 255)
    )
    mask = cv2.medianBlur(mask, 3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points: List[Tuple[int, int, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            moments = cv2.moments(contour)
            if moments["m00"]:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                points.append((cx, cy, area))
    points.sort(key=lambda t: t[2], reverse=True)
    return [(x, y) for x, y, _ in points[:max_cnt]]


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
        default=Path.cwd(),
        help="Directory to write outputs (default: current directory)",
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
        "--skip-dark-detection",
        action="store_true",
        help="Disable the late-stage dark lesion detector and reuse early detections.",
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

    logging.info("Loading images")
    baseline_original = load_image(args.baseline)
    early_original = load_image(args.early)
    late_original = load_image(args.late)

    # Apply pre-processing if enabled
    if args.enable_preprocessing:
        logging.info("Applying pre-processing pipeline")
        baseline = preprocess_image(
            baseline_original,
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
            early_original,
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
            late_original,
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
        baseline = baseline_original
        early = early_original
        late = late_original

    base_width, base_height = baseline.size
    logging.debug("Baseline image size: %s x %s", base_width, base_height)

    logging.info("Detecting papules")
    baseline_points = detect_papules_red(baseline)
    if not baseline_points:
        logging.warning("No papules detected in baseline image")

    early_points = detect_papules_red(early)
    if not early_points:
        logging.warning("No papules detected in early follow-up image")
    if args.skip_dark_detection:
        logging.warning("Skipping dark lesion detector; using early detections for late image")
        late_points = early_points
    else:
        late_points = detect_papules_dark(late)
        if not late_points:
            logging.warning("No papules detected in late follow-up image")

    logging.info("Registering follow-up images to baseline")
    matrix_early_to_base = affine_register(early, baseline)
    matrix_late_to_base = affine_register(late, baseline)

    logging.info("Warping follow-up images")
    early_warped = warp_to_base(early, matrix_early_to_base, (base_width, base_height))
    late_warped = warp_to_base(late, matrix_late_to_base, (base_width, base_height))

    logging.info("Transforming lesion coordinates to baseline frame")
    early_points_base = transform_points(early_points, matrix_early_to_base)
    late_points_base = transform_points(late_points, matrix_late_to_base)

    logging.info("Cropping images to their shared visual footprint")
    logging.info(
        "Shared content detection params: threshold=%d, kernel_size=%d, min_component_area=%d",
        args.content_threshold,
        args.content_kernel_size,
        args.content_min_component_area,
    )
    bbox = common_content_bbox(
        [baseline, early_warped, late_warped],
        threshold=args.content_threshold,
        kernel_size=args.content_kernel_size,
        min_component_area=args.content_min_component_area,
    )
    if bbox:
        (baseline, early_warped, late_warped), (
            baseline_points,
            early_points_base,
            late_points_base,
        ) = crop_images(
            [baseline, early_warped, late_warped],
            [baseline_points, early_points_base, late_points_base],
            bbox,
        )

    logging.info("Creating annotated panels")
    annotated_baseline = draw_boxes(baseline, baseline_points, "Day 0 (baseline)", radius=args.radius)
    annotated_early = draw_boxes(
        early_warped, early_points_base, "Day 1 (~24h)", radius=args.radius
    )
    annotated_late = draw_boxes(
        late_warped, late_points_base, "Day 2 (~48h)", radius=args.radius
    )

    panels = [annotated_baseline, annotated_early, annotated_late]
    original_sizes = [panel.size for panel in panels]
    for idx, (width, height) in enumerate(original_sizes, start=1):
        logging.debug("Panel %d pre-crop size: %d x %d", idx, width, height)

    bbox = common_content_bbox(panels)
    if bbox is None:
        logging.warning("No shared content bounding box found; skipping crop")
    else:
        left, top, right, bottom = bbox
        bbox_width = right - left
        bbox_height = bottom - top
        bbox_area = bbox_width * bbox_height
        logging.info(
            "Shared content bbox: left=%d, top=%d, right=%d, bottom=%d (area=%d)",
            left,
            top,
            right,
            bottom,
            bbox_area,
        )
        for idx, (orig_w, orig_h) in enumerate(original_sizes, start=1):
            original_area = orig_w * orig_h
            retained_pct = (bbox_area / original_area * 100) if original_area else 0.0
            logging.info(
                "Panel %d retains %.2f%% of original area after crop", idx, retained_pct
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

        # Generate HSV mask overlays
        baseline_hsv = visualize_hsv_mask(baseline)
        early_hsv = visualize_hsv_mask(early)

        # Generate contour overlays
        baseline_contours = visualize_contours(baseline, baseline_points)
        early_contours = visualize_contours(early, early_points)

        # Generate dark detection overlay
        late_dark = visualize_dark_detection(late, late_points)

        # Generate SIFT visualizations
        early_sift, baseline_sift_early = visualize_sift_features(early_original, baseline_original)
        late_sift, baseline_sift_late = visualize_sift_features(late_original, baseline_original)

        if args.save_morphological_overlays:
            logging.info("Saving individual morphological overlays")
            baseline_hsv.save(output_dir / "baseline_hsv_mask.jpg", quality=95)
            early_hsv.save(output_dir / "early_hsv_mask.jpg", quality=95)
            baseline_contours.save(output_dir / "baseline_contours.jpg", quality=95)
            early_contours.save(output_dir / "early_contours.jpg", quality=95)
            late_dark.save(output_dir / "late_dark_detection.jpg", quality=95)
            early_sift.save(output_dir / "early_sift_features.jpg", quality=95)
            baseline_sift_early.save(output_dir / "baseline_sift_features_early.jpg", quality=95)
            late_sift.save(output_dir / "late_sift_features.jpg", quality=95)
            baseline_sift_late.save(output_dir / "baseline_sift_features_late.jpg", quality=95)
            logging.info("Morphological overlays saved to %s", output_dir)

    # Generate diagnostic panels if requested
    if args.generate_diagnostic_panels:
        logging.info("Generating diagnostic panels")

        # Baseline diagnostic panel
        baseline_panel = create_diagnostic_panel(
            baseline_original,
            baseline,
            baseline_hsv,
            baseline_contours,
            padding=10,
        )
        baseline_panel_path = output_dir / "baseline_diagnostic_panel.jpg"
        baseline_panel.save(baseline_panel_path, quality=95)
        logging.info("Baseline diagnostic panel saved to %s", baseline_panel_path)

        # Early diagnostic panel
        early_panel = create_diagnostic_panel(
            early_original,
            early,
            early_hsv,
            early_contours,
            padding=10,
        )
        early_panel_path = output_dir / "early_diagnostic_panel.jpg"
        early_panel.save(early_panel_path, quality=95)
        logging.info("Early diagnostic panel saved to %s", early_panel_path)

        # Late diagnostic panel (with dark detection instead of contours)
        late_panel = create_diagnostic_panel(
            late_original,
            late,
            visualize_hsv_mask(late),
            late_dark,
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

