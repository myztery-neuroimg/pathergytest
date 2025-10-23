#!/usr/bin/env python3
"""Landmark-based affine registration using VLM-extracted corresponding points."""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_landmarks(json_path: str) -> dict:
    """Load landmarks from JSON file."""
    with open(json_path, 'r') as f:
        landmarks = json.load(f)
    return landmarks


def landmarks_to_points(landmarks: dict, day: str) -> np.ndarray:
    """Convert landmarks dict to numpy array of points."""
    day_landmarks = landmarks[day]
    points = []
    for feature in ['marker', 'vein', 'freckle', 'arm_edge']:
        if feature in day_landmarks:
            points.append(day_landmarks[feature])
    return np.array(points, dtype=np.float32)


def compute_affine_transform(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute affine transform matrix from source to destination points.

    Uses the first 3 points for cv2.getAffineTransform().
    """
    # Need exactly 3 points for affine transform
    if len(src_points) < 3 or len(dst_points) < 3:
        raise ValueError(f"Need at least 3 points, got {len(src_points)} and {len(dst_points)}")

    # Use first 3 points
    src_3pts = src_points[:3]
    dst_3pts = dst_points[:3]

    # Compute affine transform
    M = cv2.getAffineTransform(src_3pts, dst_3pts)

    return M


def apply_affine_transform(image: np.ndarray, M: np.ndarray, output_shape: tuple) -> np.ndarray:
    """Apply affine transform to image."""
    h, w = output_shape
    warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped


def draw_landmarks(image_pil: Image.Image, landmarks: dict, day: str, color: str = 'red') -> Image.Image:
    """Draw landmarks on image."""
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)

    day_landmarks = landmarks[day]
    radius = 10

    for feature, coord in day_landmarks.items():
        x, y = coord
        # Draw circle
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, width=3)
        # Draw label
        draw.text((x+15, y-10), feature, fill=color)

    return img


def visualize_registration(baseline_img: np.ndarray, warped_img: np.ndarray,
                          baseline_landmarks: dict, warped_landmarks: dict,
                          title: str) -> Image.Image:
    """Create visualization showing landmark alignment."""
    h, w = baseline_img.shape[:2]

    # Convert to PIL for drawing
    baseline_pil = Image.fromarray(cv2.cvtColor(baseline_img, cv2.COLOR_BGR2RGB))
    warped_pil = Image.fromarray(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))

    # Draw landmarks
    baseline_with_marks = draw_landmarks(baseline_pil, baseline_landmarks, 'day0', color='green')
    warped_with_marks = draw_landmarks(warped_pil, warped_landmarks, 'day0', color='red')

    # Create side-by-side comparison
    canvas = Image.new('RGB', (w * 2 + 20, h + 60), 'white')

    # Add title
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill='black')

    # Paste images
    canvas.paste(baseline_with_marks, (10, 50))
    canvas.paste(warped_with_marks, (w + 10, 50))

    # Add labels
    draw.text((10, h + 40), "Baseline (Day 0) - Green", fill='green')
    draw.text((w + 10, h + 40), f"Warped ({title}) - Red", fill='red')

    return canvas


def create_overlay(baseline_img: np.ndarray, warped_img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create overlay visualization."""
    # Convert to same type
    baseline_float = baseline_img.astype(np.float32)
    warped_float = warped_img.astype(np.float32)

    # Blend
    overlay = cv2.addWeighted(baseline_float, alpha, warped_float, 1-alpha, 0)

    return overlay.astype(np.uint8)


if __name__ == "__main__":
    # Paths
    landmarks_path = "/Users/davidbrewster/Documents/workspace/2025/pathergytest/landmarks.json"
    baseline_path = "/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg"
    early_path = "/Users/davidbrewster/Documents/Documents_Brewster/15 August 13_17.jpg"
    late_path = "/Users/davidbrewster/Documents/Documents_Brewster/16 August 10_04.jpg"

    output_dir = Path(".")  # Output to current directory

    print("="*100)
    print("LANDMARK-BASED AFFINE REGISTRATION")
    print("="*100)

    # Load landmarks
    print("\n1. Loading landmarks...")
    landmarks = load_landmarks(landmarks_path)
    print(f"   Loaded landmarks for {len(landmarks)} timepoints")

    # Load images
    print("\n2. Loading images...")
    baseline_img = cv2.imread(baseline_path)
    early_img = cv2.imread(early_path)
    late_img = cv2.imread(late_path)

    print(f"   Baseline: {baseline_img.shape}")
    print(f"   Early: {early_img.shape}")
    print(f"   Late: {late_img.shape}")

    # Get landmark points
    baseline_points = landmarks_to_points(landmarks, 'day0')
    early_points = landmarks_to_points(landmarks, 'day1')
    late_points = landmarks_to_points(landmarks, 'day2')

    print(f"\n3. Extracted {len(baseline_points)} landmark points per image")
    print(f"   Features: marker, vein, freckle, arm_edge")

    # Compute affine transforms
    print("\n4. Computing affine transforms...")

    # Early (day1) → Baseline (day0)
    M_early = compute_affine_transform(early_points, baseline_points)
    print(f"   Early→Baseline transform matrix:")
    print(f"   {M_early}")

    # Late (day2) → Baseline (day0)
    M_late = compute_affine_transform(late_points, baseline_points)
    print(f"   Late→Baseline transform matrix:")
    print(f"   {M_late}")

    # Apply transforms
    print("\n5. Warping images...")
    h, w = baseline_img.shape[:2]

    early_warped = apply_affine_transform(early_img, M_early, (h, w))
    late_warped = apply_affine_transform(late_img, M_late, (h, w))

    print(f"   Warped early image: {early_warped.shape}")
    print(f"   Warped late image: {late_warped.shape}")

    # Save warped images
    print("\n6. Saving results...")
    cv2.imwrite(str(output_dir / "early_warped_landmark.jpg"), early_warped)
    cv2.imwrite(str(output_dir / "late_warped_landmark.jpg"), late_warped)
    print(f"   ✓ Saved warped images")

    # Create visualizations with landmarks
    print("\n7. Creating visualizations...")

    # Early comparison
    early_viz = visualize_registration(baseline_img, early_warped, landmarks, landmarks, "Day 1 → Day 0")
    early_viz.save(output_dir / "early_landmark_comparison.jpg")
    print(f"   ✓ Saved early comparison")

    # Late comparison
    late_viz = visualize_registration(baseline_img, late_warped, landmarks, landmarks, "Day 2 → Day 0")
    late_viz.save(output_dir / "late_landmark_comparison.jpg")
    print(f"   ✓ Saved late comparison")

    # Create overlays
    early_overlay = create_overlay(baseline_img, early_warped, alpha=0.5)
    late_overlay = create_overlay(baseline_img, late_warped, alpha=0.5)

    cv2.imwrite(str(output_dir / "early_overlay.jpg"), early_overlay)
    cv2.imwrite(str(output_dir / "late_overlay.jpg"), late_overlay)
    print(f"   ✓ Saved overlay images")

    # Create timeline composite
    print("\n8. Creating timeline composite...")

    # Resize to consistent height for timeline
    target_height = 400
    scale = target_height / h
    target_width = int(w * scale)

    baseline_resized = cv2.resize(baseline_img, (target_width, target_height))
    early_warped_resized = cv2.resize(early_warped, (target_width, target_height))
    late_warped_resized = cv2.resize(late_warped, (target_width, target_height))

    # Create horizontal timeline
    timeline_width = target_width * 3 + 40
    timeline_height = target_height + 80
    timeline = np.ones((timeline_height, timeline_width, 3), dtype=np.uint8) * 255

    # Place images
    timeline[60:60+target_height, 10:10+target_width] = baseline_resized
    timeline[60:60+target_height, 20+target_width:20+target_width*2] = early_warped_resized
    timeline[60:60+target_height, 30+target_width*2:30+target_width*3] = late_warped_resized

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(timeline, "Day 0 (Baseline)", (10, 30), font, 1, (0, 0, 0), 2)
    cv2.putText(timeline, "Day 1 (Registered)", (20+target_width, 30), font, 1, (0, 0, 0), 2)
    cv2.putText(timeline, "Day 2 (Registered)", (30+target_width*2, 30), font, 1, (0, 0, 0), 2)

    cv2.imwrite(str(output_dir / "timeline_landmark_registered.jpg"), timeline)
    print(f"   ✓ Saved timeline composite")

    print("\n" + "="*100)
    print("REGISTRATION COMPLETE")
    print("="*100)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles created:")
    print("  - early_warped_landmark.jpg")
    print("  - late_warped_landmark.jpg")
    print("  - early_landmark_comparison.jpg")
    print("  - late_landmark_comparison.jpg")
    print("  - early_overlay.jpg")
    print("  - late_overlay.jpg")
    print("  - timeline_landmark_registered.jpg")

    # Print transform details
    print("\n" + "="*100)
    print("TRANSFORM ANALYSIS")
    print("="*100)

    # Decompose affine transforms
    def analyze_affine(M, name):
        print(f"\n{name}:")
        print(f"  Matrix:")
        print(f"    [{M[0,0]:.4f}  {M[0,1]:.4f} | {M[0,2]:.2f}]")
        print(f"    [{M[1,0]:.4f}  {M[1,1]:.4f} | {M[1,2]:.2f}]")

        # Extract rotation angle
        angle = np.arctan2(M[1,0], M[0,0]) * 180 / np.pi

        # Extract scale
        scale_x = np.sqrt(M[0,0]**2 + M[1,0]**2)
        scale_y = np.sqrt(M[0,1]**2 + M[1,1]**2)

        # Translation
        tx, ty = M[0,2], M[1,2]

        print(f"  Rotation: {angle:.2f}°")
        print(f"  Scale: x={scale_x:.4f}, y={scale_y:.4f}")
        print(f"  Translation: x={tx:.2f}px, y={ty:.2f}px")

    analyze_affine(M_early, "Day 1 → Day 0")
    analyze_affine(M_late, "Day 2 → Day 0")
