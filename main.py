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
    baseline = load_image(args.baseline)
    early = load_image(args.early)
    late = load_image(args.late)

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

    logging.debug("Transforming lesion coordinates to baseline frame")
    early_points_base = transform_points(early_points, matrix_early_to_base)
    late_points_base = transform_points(late_points, matrix_late_to_base)

    logging.info("Creating annotated panels")
    annotated_baseline = draw_boxes(baseline, baseline_points, "Day 0 (baseline)", radius=args.radius)
    annotated_early = draw_boxes(
        early_warped, early_points_base, "Day 1 (~24h)", radius=args.radius
    )
    annotated_late = draw_boxes(
        late_warped, late_points_base, "Day 2 (~48h)", radius=args.radius
    )

    logging.info("Building montage")
    montage = build_montage(
        [annotated_baseline, annotated_early, annotated_late],
        padding=args.padding,
        caption="Pathergy Test Timeline (Baseline-aligned)",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    logging.info("Saving montage to %s", output_path)
    montage.save(output_path, quality=95)
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

