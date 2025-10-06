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
- Alignment of all frames to a Day-0 baseline forearm contour using SIFT + RANSAC.
- Generation of:
  - Individual annotated overlays for each timepoint.
  - A composite timeline panel showing morphological progression.

---

## Installation

git clone https://github.com/myzteryneuro/pathergytest.git
cd pathergytest
pip install -r requirements.txt

Requirements:
- opencv-python
- numpy
- Pillow

---

## Usage

1. Prepare your images:
   - day0_0h.png   (baseline at point of injection at 2 sites)
   - day1_24h.png         (24 h)
   - day2_48h.png        (48g)

2. Run the pipeline:
   python main.py

3. Outputs:
   - out_day0_baseline.jpg
   - out_day0_aligned_24h.jpg
   - out_day0_aligned_48h.jpg
   - pathergy_timeline_composite.jpg

Each overlay marks the same papule pair aligned to the Day-1 contour, showing their evolution over time.

---

## Methodology

Detection:
- Early: HSV-based red hue segmentation.
- Late: CLAHE-enhanced gray thresholding for brown macules.

Registration:
- SIFT feature matching.
- Affine transformation via RANSAC.

Visualization:
- Uniform coordinate geometry.
- Red bounding boxes (22 px radius).
- Optional side-by-side montage output.

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
