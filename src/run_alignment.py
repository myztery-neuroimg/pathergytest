#!/usr/bin/env python3
"""
Clean script to run pathergy test alignment using imported functions.
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Import the main functions directly
from src.main import (
    load_image,
    intelligent_precrop,
    preprocess_image,
    affine_register,
    warp_to_base,
    common_content_bbox,
    detect_papules_red,
    draw_boxes,
    build_montage,
    configure_logging
)

# Import landmark extraction if needed
from src.get_anatomical_landmarks_final import get_claude_anatomical_landmarks, parse_landmarks


def run_pathergy_alignment(baseline_path, early_path, late_path,
                          output_dir=None, enable_preprocessing=True,
                          log_level='INFO'):
    """
    Run the complete pathergy test alignment pipeline.

    Args:
        baseline_path: Path to baseline (day 0) image
        early_path: Path to early follow-up (day 1) image
        late_path: Path to late follow-up (day 2) image
        output_dir: Output directory (default: baseline image directory)
        enable_preprocessing: Whether to apply preprocessing
        log_level: Logging level
    """
    # Configure logging
    configure_logging(log_level)

    # Convert to Path objects
    baseline_path = Path(baseline_path)
    early_path = Path(early_path)
    late_path = Path(late_path)

    # Set output directory
    if output_dir is None:
        output_dir = baseline_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Output directory: {output_dir}")

    # Load images
    logging.info("Loading images...")
    baseline_original = load_image(baseline_path)
    early_original = load_image(early_path)
    late_original = load_image(late_path)

    # Pre-crop to forearm regions
    logging.info("Pre-cropping to forearm regions...")
    baseline_precrop, baseline_bbox = intelligent_precrop(baseline_original, margin_percent=0.15)
    early_precrop, early_bbox = intelligent_precrop(early_original, margin_percent=0.15)
    late_precrop, late_bbox = intelligent_precrop(late_original, margin_percent=0.15)

    # Apply preprocessing if enabled
    if enable_preprocessing:
        logging.info("Applying preprocessing...")
        baseline = preprocess_image(
            baseline_precrop,
            apply_illumination=True,
            apply_bilateral=True,
            apply_normalize=True
        )
        early = preprocess_image(
            early_precrop,
            apply_illumination=True,
            apply_bilateral=True,
            apply_normalize=True
        )
        late = preprocess_image(
            late_precrop,
            apply_illumination=True,
            apply_bilateral=True,
            apply_normalize=True
        )
    else:
        baseline = baseline_precrop
        early = early_precrop
        late = late_precrop

    base_width, base_height = baseline.size

    # Register images using landmarks
    logging.info("Registering images using VLM landmarks...")
    baseline_offset = (baseline_bbox[0], baseline_bbox[1])
    early_offset = (early_bbox[0], early_bbox[1])
    late_offset = (late_bbox[0], late_bbox[1])

    matrix_early_to_base = affine_register(
        early, baseline,
        src_timepoint='day1',
        src_crop_offset=early_offset,
        dst_crop_offset=baseline_offset
    )

    matrix_late_to_base = affine_register(
        late, baseline,
        src_timepoint='day2',
        src_crop_offset=late_offset,
        dst_crop_offset=baseline_offset
    )

    # Warp images
    logging.info("Warping images to baseline coordinate frame...")
    early_warped = warp_to_base(early, matrix_early_to_base, (base_width, base_height))
    late_warped = warp_to_base(late, matrix_late_to_base, (base_width, base_height))

    # Find shared content region
    logging.info("Finding shared anatomical region...")
    shared_bbox = common_content_bbox(
        [baseline, early_warped, late_warped],
        threshold=5,
        kernel_size=7,
        min_component_area=1000
    )

    if shared_bbox:
        baseline_cropped = baseline.crop(shared_bbox)
        early_warped_cropped = early_warped.crop(shared_bbox)
        late_warped_cropped = late_warped.crop(shared_bbox)
    else:
        baseline_cropped = baseline
        early_warped_cropped = early_warped
        late_warped_cropped = late_warped

    # Detect pathergy sites
    logging.info("Detecting puncture sites on baseline...")
    baseline_points = detect_papules_red(baseline_cropped)

    if not baseline_points:
        logging.warning("No puncture sites detected in baseline image")
    else:
        logging.info(f"Detected {len(baseline_points)} puncture site(s)")

    # Use same points for tracking
    early_points_base = baseline_points
    late_points_base = baseline_points

    # Create panels with tracked sites
    logging.info("Creating tracked site visualizations...")
    if baseline_points:
        baseline_with_boxes = draw_boxes(baseline_cropped, baseline_points, "Baseline Sites", radius=50)
        early_with_boxes = draw_boxes(early_warped_cropped, early_points_base, "Tracked Sites (Day 1)", radius=50)
        late_with_boxes = draw_boxes(late_warped_cropped, late_points_base, "Tracked Sites (Day 2)", radius=50)

        # Generate montage from panels
        logging.info("Generating pathergy timeline montage...")
        panels = [baseline_with_boxes, early_with_boxes, late_with_boxes]
        montage = build_montage(
            panels,
            padding=20,
            caption="Pathergy Test Timeline (Baseline-aligned)"
        )
    else:
        # No sites detected, create simple montage
        logging.warning("No puncture sites detected, creating simple montage")
        panels = [baseline_cropped, early_warped_cropped, late_warped_cropped]
        montage = build_montage(
            panels,
            padding=20,
            caption="Pathergy Test Timeline (No sites detected)"
        )
        baseline_with_boxes = baseline_cropped
        early_with_boxes = early_warped_cropped
        late_with_boxes = late_warped_cropped

    # Save montage
    montage.save(output_dir / "pathergy_timeline_composite.jpg", quality=95)
    logging.info("Composite saved to %s", output_dir / "pathergy_timeline_composite.jpg")

    # Save individual tracked sites images
    if baseline_points:

        baseline_with_boxes.save(output_dir / "baseline_tracked_sites.jpg", quality=95)
        early_with_boxes.save(output_dir / "early_tracked_sites.jpg", quality=95)
        late_with_boxes.save(output_dir / "late_tracked_sites.jpg", quality=95)

        logging.info("Saved tracked sites images")

    return montage


def extract_landmarks_for_images(baseline_path, early_path, late_path, use_claude=False):
    """
    Extract landmarks for the images.

    Args:
        baseline_path: Path to baseline image
        early_path: Path to early follow-up image
        late_path: Path to late follow-up image
        use_claude: Whether to use Claude API for landmark extraction

    Returns:
        Dictionary of landmarks
    """
    if use_claude:
        # Use Claude API for better landmark extraction
        api_key_file = Path.home() / '.ANTHROPIC_API_KEY'
        if not api_key_file.exists():
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("No API key found. Set ANTHROPIC_API_KEY environment variable.")
        else:
            api_key = api_key_file.read_text().strip()

        logging.info("Extracting landmarks using Claude API...")
        response = get_claude_anatomical_landmarks(
            baseline_path, early_path, late_path, api_key
        )
        landmarks = parse_landmarks(response)
    else:
        # Use existing landmarks.json file
        landmarks_file = Path('landmarks.json')
        if landmarks_file.exists():
            import json
            with open(landmarks_file, 'r', encoding='utf-8') as f:
                landmarks = json.load(f)
            logging.info("Using existing landmarks.json")
        else:
            raise FileNotFoundError("No landmarks.json file found. Run landmark extraction first.")

    return landmarks


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Align and annotate serial pathergy test images"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline image path (day 0)"
    )
    parser.add_argument(
        "--early",
        type=str,
        required=True,
        help="Early follow-up image (day 1)"
    )
    parser.add_argument(
        "--late",
        type=str,
        required=True,
        help="Late follow-up image (day 2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: baseline image directory)"
    )
    parser.add_argument(
        "--enable-preprocessing",
        action="store_true",
        help="Apply preprocessing to images"
    )
    parser.add_argument(
        "--extract-landmarks",
        action="store_true",
        help="Extract new landmarks using Claude API"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Extract landmarks if requested
    if args.extract_landmarks:
        landmarks = extract_landmarks_for_images(
            args.baseline, args.early, args.late, use_claude=True
        )
        import json
        with open('landmarks.json', 'r', encoding='utf-8') as f:
            json.dump(landmarks, f, indent=2)
        logging.info("Saved landmarks to landmarks.json")

    # Run alignment
    run_pathergy_alignment(
        args.baseline,
        args.early,
        args.late,
        args.output_dir,
        args.enable_preprocessing,
        args.log_level
    )


if __name__ == "__main__":
    main()
