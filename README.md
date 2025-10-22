# Pathergy Reaction Analysis Toolkit

A lightweight, reproducible pipeline for analyzing pathergy test reactions from clinical photographs.

This tool aligns serial images, detects papular responses, and generates overlays and longitudinal composites for visual interpretation.

This would be based on the following test https://behcet-zentrum.de/fuer-patienten/ 

---

## Overview

Purpose:
Provide an automated way to verify, document, and visualize a positive pathergy reaction — a papule or pustule appearing 24–48 hours after sterile puncture, typically used in Behçet spectrum disease evaluation.

Features:
- Automatic detection of injection sites arranged lengthwise along the forearm axis (per clinical protocol).
- Structural feature-based registration using arm outline, elbow contours, and anatomical edges.
- Comprehensive pre-processing pipeline for improved image quality:
  - Illumination correction using LAB color space and CLAHE.
  - Edge-preserving noise reduction via bilateral filtering.
  - Color normalization for consistent analysis.
  - Optional unsharp masking for edge enhancement.
- Alignment of all frames to a Day-0 baseline forearm using edge-enhanced SIFT + RANSAC.
- Generation of:
  - Individual annotated overlays for each timepoint.
  - A composite timeline panel showing morphological progression.
  - Morphological mapping visualizations (HSV masks, contours, SIFT features).
  - Structural feature visualizations (arm outline edges, principal axis orientation).
  - Diagnostic panels showing all processing stages.

---

## Installation

```
git clone https://github.com/myztery-neuroimg/pathergytest.git
cd pathergytest
# Copy your files as named below or change the filenames here
pip install -r requirements.txt
```

Requirements:
- python >= 3.12
- opencv-python
- numpy
- Pillow

---

## Usage

1. Prepare your images:
   - day0_0h.png   (baseline at point of injection at 2 sites)
   - day1_24h.png         (24 h)
   - day2_48h.png         (48 h)

2. Run the pipeline:

   **Basic usage:**
   ```bash
   python main.py \
       --baseline day1_0h.png \
       --early day1_24h.png \
       --late day2_48h.png \
       --output-dir outputs \
       --log-level INFO
   ```

   **With pre-processing and morphological visualizations:**
   ```bash
   python main.py \
       --baseline day1_0h.png \
       --early day1_24h.png \
       --late day2_48h.png \
       --output-dir outputs \
       --enable-preprocessing \
       --generate-diagnostic-panels \
       --save-morphological-overlays \
       --log-level DEBUG
   ```

   **Pre-processing options:**
   - `--enable-preprocessing`: Enable comprehensive pre-processing pipeline (illumination correction, bilateral filtering, color normalization).
   - `--enable-unsharp`: Enable unsharp masking for edge enhancement (requires `--enable-preprocessing`).
   - `--illumination-clip FLOAT`: CLAHE clip limit for illumination correction (default: 2.0).
   - `--bilateral-d INT`: Diameter of pixel neighborhood for bilateral filter (default: 9).
   - `--bilateral-sigma-color INT`: Filter sigma in color space (default: 75).
   - `--bilateral-sigma-space INT`: Filter sigma in coordinate space (default: 75).

   **Visualization options:**
   - `--generate-diagnostic-panels`: Generate diagnostic visualization panels showing preprocessing and detection stages.
   - `--save-morphological-overlays`: Save individual morphological overlay images (HSV masks, contours, SIFT features).

   **Other optional flags:**
   - `--radius`: Adjust the lesion bounding box size (pixels, default: 22).
   - `--padding`: Control spacing between montage panels (pixels, default: 20).
   - `--content-threshold INT`: Pixel intensity threshold for shared visual-footprint detection (default: 5).
   - `--content-kernel-size INT`: Morphological kernel size for shared visual-footprint detection (default: 7).
   - `--content-min-component-area INT`: Minimum connected component area kept during detection (default: 1000).

3. Outputs:
   - `outputs/pathergy_timeline_composite.jpg` - Main composite timeline showing Day 0 puncture sites tracked across all timepoints

   **Diagnostic panels** (if `--generate-diagnostic-panels` is used):
   - `outputs/baseline_diagnostic_panel.jpg` - Baseline detection process (original, preprocessed, HSV mask, detected sites)
   - `outputs/early_diagnostic_panel.jpg` - Day 1 tracking (original, warped, SIFT features, tracked sites)
   - `outputs/late_diagnostic_panel.jpg` - Day 2+ tracking (original, warped, SIFT features, tracked sites)

   **Morphological overlays** (if `--save-morphological-overlays` is used):
   - `outputs/baseline_hsv_mask.jpg` - HSV-based red detection mask on baseline
   - `outputs/baseline_structural_edges.jpg` - Structural edges (arm outline, elbow, creases) used for registration
   - `outputs/baseline_arm_orientation.jpg` - Detected arm principal axis (lengthwise direction)
   - `outputs/baseline_contours_detected.jpg` - Detected puncture sites on Day 0
   - `outputs/early_tracked_sites.jpg` - Day 0 sites tracked to Day 1 image
   - `outputs/late_tracked_sites.jpg` - Day 0 sites tracked to Day 2+ image
   - `outputs/early_sift_features.jpg` - SIFT keypoints used for Day 1 alignment
   - `outputs/late_sift_features.jpg` - SIFT keypoints used for Day 2+ alignment

Each overlay marks the same papule pair aligned to the Day-1 contour, showing their evolution over time. Logging provides traceability for each processing stage.

---

## Methodology

### Pipeline Order (Critical for Accuracy)

The pipeline executes in this specific order to ensure accurate tracking:

1. **Load Images**: Load Day 0, Day 1, and Day 2+ images
2. **Intelligent Pre-Crop**: Independently identify forearm region in each image using multi-color-space skin segmentation (HSV + YCrCb)
   - Removes background noise, hands, elbows from each image independently
   - Adds 15% safety margin to preserve features for registration
   - Handles unaligned images (different rotations/positions/scales)
3. **Pre-process** (optional): Apply illumination correction, bilateral filtering, color normalization on pre-cropped images
4. **Register**: Align pre-cropped follow-up images to pre-cropped baseline using structural feature-based SIFT + RANSAC
   - Canny edge detection identifies arm outline, elbow contours, skin creases
   - SIFT applied to structural edges for registration based on arm geometry
   - Cleaner feature matching focused on stable anatomical structure
   - Fallback to full-image SIFT if edge-based detection fails
5. **Warp**: Transform follow-up images to baseline coordinate frame
6. **Refine Shared Region** (optional): Final refinement to crop aligned images to exact shared forearm region
7. **Detect Arm Orientation**: Compute forearm's long axis using PCA on skin mask
8. **Detect Puncture Sites**: Detect Day 0 puncture sites on refined baseline image only
   - Finds two injection sites arranged LENGTHWISE along arm axis
   - Scores pairs heavily on alignment with arm orientation (3× weight)
   - Matches clinical pathergy test protocol
9. **Track Sites**: Use the same Day 0 coordinates across all aligned, refined timepoints

This order is critical because:
- **Intelligent pre-crop BEFORE registration**: Removes confounding features (hands, background) that create false SIFT matches, while keeping enough forearm detail for alignment
- **Independent skin segmentation**: Each image analyzed separately to find its own forearm region, regardless of rotation/position
- **Safety margin**: 15% expansion ensures SIFT has enough overlapping features between images for robust registration
- **Preprocessing after pre-crop**: Illumination correction helps edge detection quality, applied to relevant anatomy only
- **Structural feature registration**: Edge-based SIFT focuses on stable anatomical geometry (arm outline, elbow) not variable skin appearance
  - More robust to lighting changes (edges remain constant)
  - Prioritizes arm STRUCTURE over interior texture
  - Better alignment for tracking anatomical locations over time
- **Optional final refinement**: Additional crop to exact shared region after alignment
- **Arm orientation detection**: PCA on skin mask identifies forearm's long axis for lengthwise injection site detection
- **Lengthwise site detection**: Matches clinical protocol where sites are arranged parallel to arm axis (not perpendicular)
- **Stable coordinates**: All images in same coordinate space; Day 0 puncture coordinates apply directly

### Technical Details

**Intelligent Pre-Crop (Skin Segmentation):**
- **Multi-color-space approach**: Combines HSV and YCrCb color spaces for robust skin detection
  - HSV: Detects skin hue (0-50°) regardless of brightness
  - YCrCb: Detects skin chrominance (Cr: 133-173, Cb: 77-127) robust to illumination
  - Logical AND of both masks for high-confidence skin pixels
- **Morphological refinement**: Elliptical kernels (11×11) for closing/opening to remove noise
- **Connected component analysis**: Identifies largest skin region (forearm) using scikit-image
- **Bounding box with margin**: Adds 15% safety margin to preserve features for SIFT registration
- **Independent processing**: Each image analyzed separately, handles rotation/translation/scale differences

**Post-Alignment Refinement (Optional):**
- Uses morphological operations on ALIGNED images to find exact shared forearm region
- Computes intersection of content masks across all three aligned timepoints
- Final crop to clean anatomical region
- This is a refinement step after registration, not the primary cropping mechanism

**Pre-processing** (optional):
- **Illumination correction**: LAB color space conversion with CLAHE applied to L channel to normalize lighting conditions
- **Bilateral filtering**: Edge-preserving noise reduction that smooths while maintaining sharp boundaries
- **Color normalization**: Normalizes color channels to improve consistency across images taken under different lighting
- **Unsharp masking**: Edge enhancement technique to improve feature detection and alignment

**Registration (Structural Feature-Based):**
- **Edge-enhanced SIFT**: Uses Canny edge detection to identify structural features
  - Detects arm outline, elbow contours, skin creases
  - Applies SIFT to edge regions for registration based on arm STRUCTURE
  - Prioritizes geometric anatomy over interior skin texture
  - More robust to lighting variations (edges remain constant)
- **Scale-Invariant Feature Transform**: Works on PRE-CROPPED images
  - Focuses on forearm structural features only
  - 15% safety margin ensures sufficient overlapping features
  - Fallback to full-image SIFT if edge-based detection fails
- **Affine transformation via RANSAC**: Robust estimation to align follow-up images to baseline frame
- Handles rotation, translation, and scale differences between pre-cropped images
- Uses preprocessed images if preprocessing is enabled (improves edge detection quality)
- Benefits: Alignment based on stable anatomical structure, not variable skin appearance

**Arm Orientation Detection:**
- **Principal Component Analysis (PCA)**: Computes the long axis of the forearm
  - Analyzes skin mask pixel coordinates
  - Identifies principal direction (eigenvector with largest eigenvalue)
  - Returns angle in degrees from horizontal
- **Purpose**: Used to detect injection sites arranged LENGTHWISE along the arm
- **Clinical relevance**: Pathergy test protocol requires sites parallel to arm axis (not perpendicular)

**Puncture Site Detection and Tracking:**
- **Day 0 (Baseline)**: HSV-based red hue segmentation with lengthwise alignment constraint
  - Detects red regions (injection sites and +/- markers)
  - Filters by size (excludes large markers, keeps puncture sites)
  - Scores pairs based on:
    1. Distance similarity (20-150 pixels, typically 2-3 cm)
    2. Size similarity (both sites should be similar)
    3. Circularity (both should be circular)
    4. **Alignment with arm axis** (3× weight - sites MUST be lengthwise)
  - Selects best pair aligned parallel to forearm's long axis
- **Day 1-5+ (Follow-up)**: The same anatomical locations are tracked across all cropped, aligned images; no independent detection is performed
- Tracks the evolution of the SAME puncture sites over time
- Coordinates remain stable because all images are in the same aligned+cropped coordinate space

**Visualization:**
- **Standard output**: Uniform coordinate geometry with red bounding boxes (22 px radius) and side-by-side montage
- **Morphological overlays**:
  - HSV masks showing red detection regions
  - Structural edge overlays (arm outline, elbow, creases used for registration)
  - Arm orientation overlay (principal axis direction)
  - Contour visualizations on cropped images
  - SIFT keypoint displays
- **Diagnostic panels**: 2×2 grids showing original, arm orientation/warped, SIFT features, and tracked sites

---

## Example Interpretation

Timepoint | Morphology | Clinical Meaning
---------- | ----------- | ----------------
Day 0 (~) | Baseline  | Only puncture site
Day 1 (+24 h) | Papular / pustular | Autoimmune response
Day 2 (+48 h) | Papular / pustular | Meets positivity threshold

---

## Disclaimer

For research and educational use only.
This does not substitute for professional medical evaluation.
