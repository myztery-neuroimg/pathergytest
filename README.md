# Pathergy Reaction Analysis Toolkit

A lightweight, reproducible pipeline for analyzing pathergy test reactions from clinical photographs.

This tool aligns serial images, detects papular responses, and generates overlays and longitudinal composites for visual interpretation.

This would be based on the following test https://behcet-zentrum.de/fuer-patienten/ 

---

## Overview

Purpose:
Provide an automated way to verify, document, and visualize a positive pathergy reaction — a papule or pustule appearing 24–48 hours after sterile puncture, typically used in Behçet spectrum disease evaluation.

Features:
- Automatic detection of papular lesions (red → brown macules).
- Comprehensive pre-processing pipeline for improved image quality:
  - Illumination correction using LAB color space and CLAHE.
  - Edge-preserving noise reduction via bilateral filtering.
  - Color normalization for consistent analysis.
  - Optional unsharp masking for edge enhancement.
- Alignment of all frames to a Day-0 baseline forearm contour using SIFT + RANSAC.
- Generation of:
  - Individual annotated overlays for each timepoint.
  - A composite timeline panel showing morphological progression.
  - Morphological mapping visualizations (HSV masks, contours, SIFT features).
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

1. **Load and Pre-process** (optional illumination correction, bilateral filtering, color normalization)
2. **Geomorphological Feature Identification**: Identify shared anatomical region (forearm) across ALL images BEFORE any detection or registration
3. **Pre-crop to Shared Region**: Eliminate hands, elbows, and background from all images
4. **Detect Puncture Sites**: Detect Day 0 puncture sites on the cropped baseline image only
5. **Register and Warp**: Align cropped follow-up images to cropped baseline
6. **Track Sites**: Use the same Day 0 coordinates across all registered images

This order is critical because it:
- Prevents confounding features (hands/elbows) from interfering with puncture site detection
- Ensures puncture sites are never cropped out after detection
- Maintains stable coordinate system throughout the pipeline
- Focuses detection on the relevant forearm anatomy only

### Technical Details

**Geomorphological Feature Identification:**
- Uses morphological operations (closing/opening) with elliptical kernels to identify skin regions
- Filters connected components by minimum area to remove noise
- Computes intersection of content masks across all three timepoints
- Finds bounding box containing the shared forearm region
- This happens BEFORE detection or registration to establish the anatomical region of interest

**Pre-processing** (optional):
- **Illumination correction**: LAB color space conversion with CLAHE applied to L channel to normalize lighting conditions
- **Bilateral filtering**: Edge-preserving noise reduction that smooths while maintaining sharp boundaries
- **Color normalization**: Normalizes color channels to improve consistency across images taken under different lighting
- **Unsharp masking**: Edge enhancement technique to improve feature detection and alignment

**Puncture Site Detection and Tracking:**
- **Day 0 (Baseline)**: HSV-based red hue segmentation to detect puncture sites on the PRE-CROPPED baseline image
- **Day 1-5+ (Follow-up)**: The same anatomical locations are tracked across all registered images; no independent detection is performed
- Tracks the evolution of the SAME puncture sites over time through geometric mapping via affine registration

**Registration:**
- **SIFT feature matching**: Scale-Invariant Feature Transform to detect distinctive keypoints on cropped images
- **Affine transformation via RANSAC**: Robust estimation to align cropped follow-up images to cropped baseline frame

**Visualization:**
- **Standard output**: Uniform coordinate geometry with red bounding boxes (22 px radius) and side-by-side montage
- **Morphological overlays**: HSV masks, contour visualizations, SIFT keypoint displays on cropped images
- **Diagnostic panels**: 2×2 grids showing original, preprocessed/warped, SIFT features, and tracked sites

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
