# Technical Methodology - Pathergy Test Analysis

## Pipeline Order (Critical for Accuracy)

The pipeline executes in this specific order to ensure accurate tracking:

1. **Load Images**: Load Day 0, Day 1, and Day 2+ images
2. **Intelligent Pre-Crop**: Independently identify forearm region in each image using multi-color-space skin segmentation (HSV + YCrCb)
   - Removes background noise, hands, elbows from each image independently
   - Adds 15% safety margin to preserve features for registration
   - Handles unaligned images (different rotations/positions/scales)
3. **Pre-process** (optional): Apply illumination correction, bilateral filtering, color normalization on pre-cropped images
4. **Register**: Align pre-cropped follow-up images to pre-cropped baseline using anatomical landmark-based registration
   - VLM (Vision-Language Model) extracts anatomical landmarks
   - Affine transformation computed from corresponding points
   - Falls back to ECC if landmarks unavailable
5. **Warp**: Transform follow-up images to baseline coordinate frame
6. **Refine Shared Region** (optional): Final refinement to crop aligned images to exact shared forearm region
7. **Detect +/- Markers**: Identify physician-drawn markers to establish test site location
8. **Calibrate Scale**: Measure arm width and calculate pixels per cm (typical forearm: 7 cm)
9. **Compute Local Arm Orientation**: PCA on skin near markers to get arm axis at test site
10. **Detect Puncture Sites**: Detect Day 0 puncture sites on refined baseline image only
    - Searches within 5 cm of markers using calibrated scale
    - Uses real-world spacing (2-3 cm) converted to pixels
    - Finds two injection sites arranged PARALLEL to local arm axis
    - Scores pairs heavily on alignment with local arm orientation (5× weight)
    - Matches clinical pathergy test protocol
11. **Track Sites**: Use the same Day 0 coordinates across all aligned, refined timepoints

## Why This Order Matters

### Pre-crop BEFORE Registration
- **Removes confounding features** (hands, background) that create false matches
- **Independent skin segmentation**: Each image analyzed separately to find its own forearm region
- **Safety margin**: 15% expansion ensures sufficient overlapping features for robust registration

### Preprocessing After Pre-crop
- **Illumination correction** helps edge detection quality
- **Applied to relevant anatomy only** for efficiency

### Anatomical Landmark Registration
- **More robust than intensity-based**: Works across lighting changes
- **Uses stable features**: Veins, hair patterns, skin marks
- **Better than edge-based SIFT**: Edges can vary with lighting

### Marker Detection Establishes Test Site
- **Identifies where physician marked** the test location
- **Critical for scale calibration** and search area

### Geomorphological Scale Calibration
**CRITICAL - cannot use fixed pixel distances:**
- Camera distance and zoom vary between images
- Must measure arm width to establish scale (pixels per cm)
- Converts real-world protocol (2-3 cm) to image-specific pixel distances
- Example: 210 px arm width → 30 px/cm → 2.5 cm spacing = 75 px

### Local Orientation at Test Site
- **Accounts for arm curvature**: More accurate than global axis
- **Sites must align with arm direction** at that specific location

### Stable Coordinates
- All images in same coordinate space
- Day 0 puncture coordinates apply directly to aligned images

## Technical Implementation Details

### Intelligent Pre-Crop (Skin Segmentation)

**Multi-color-space approach**: Combines HSV and YCrCb color spaces for robust skin detection
- **HSV**: Detects skin hue (0-50°) regardless of brightness
- **YCrCb**: Detects skin chrominance (Cr: 133-173, Cb: 77-127) robust to illumination
- **Logical AND** of both masks for high-confidence skin pixels

**Morphological refinement**:
- Elliptical kernels (11×11) for closing/opening to remove noise
- Connected component analysis identifies largest skin region (forearm)
- Bounding box with 15% margin preserves features for registration
- Independent processing handles rotation/translation/scale differences

### Pre-processing Pipeline

**Illumination correction**:
- LAB color space conversion
- CLAHE applied to L channel
- Normalizes lighting conditions

**Bilateral filtering**:
- Edge-preserving noise reduction
- Smooths while maintaining sharp boundaries
- Parameters: d=9, sigmaColor=75, sigmaSpace=75

**Color normalization**:
- Normalizes color channels
- Improves consistency across different lighting

**Optional unsharp masking**:
- Edge enhancement for better feature detection

### Registration Methods

#### VLM Landmark-Based (Primary)
- **Claude Sonnet 4.5 with thinking mode** extracts anatomical landmarks
- **Anatomical features used**:
  - Hair transition boundaries
  - Vein junctions
  - Skin marks/freckles
  - Hair follicle patterns
- **NOT using test marks** as landmarks (circular dependency)
- **Affine transformation** computed from correspondences
- **Handles pre-crop offsets** automatically

#### ECC Registration (Fallback)
- Intensity-based correlation using skin mask
- Less robust but available if landmarks fail

### Arm Orientation Detection

**Principal Component Analysis (PCA)**:
- Computes the long axis of the forearm
- Analyzes skin mask pixel coordinates
- Identifies principal direction (eigenvector with largest eigenvalue)
- Returns angle in degrees from horizontal

**Purpose**: Detect injection sites arranged LENGTHWISE along arm

**Clinical relevance**: Pathergy test protocol requires sites parallel to arm axis

### Scale Calibration Process

**Problem**: Camera distance and zoom vary between images

**Solution**: Calibrate using anatomical features with known dimensions

**Marker Detection**:
- Detects physician-drawn +/- markers (pen ink or red marker)
- Combines grayscale thresholding and HSV color detection
- Identifies test site location as marker centroid
- Filters by area (>500 px) to distinguish from injection sites

**Arm Width Measurement**:
- Measures forearm width in pixels at test site y-coordinate
- Assumes typical forearm width: 7 cm
- Calculates pixels_per_cm = arm_width_px / 7.0
- Dynamically adapts to camera distance and image scale

**Real-World Distance Conversion**:
- Protocol spacing: 2-3 cm → converted to pixels using scale
- Search radius: 5 cm around markers → 5 * pixels_per_cm
- Example: If arm width is 210 px, scale is 30 px/cm, so 2.5 cm = 75 px

**Local Arm Orientation at Test Site**:
- PCA on skin pixels within 150 px radius of markers
- Computes LOCAL arm axis (accounts for curvature)
- More accurate than global orientation

### Puncture Site Detection

**Day 0 (Baseline)**: Scale-calibrated HSV segmentation with marker-relative search
- Detects red regions within 5 cm of markers
- Filters by size (30-500 px area, excludes large markers)
- Scores pairs based on:
  1. Distance from ideal spacing (2.5 cm in real-world units)
  2. Size similarity (both sites should be similar)
  3. Circularity (both should be circular)
  4. **Alignment with LOCAL arm axis** (5× weight - sites MUST be parallel)
- Selects best pair aligned parallel to arm at test site
- Logs detected spacing: "distance=2.5 cm (75.3 px), alignment=3.2° from local arm axis"

**Day 1-5+ (Follow-up)**:
- The same anatomical locations are tracked
- No independent detection performed
- Coordinates remain stable in aligned coordinate space

## Visualization Components

### Standard Output
- Uniform coordinate geometry with red bounding boxes (22-50 px radius)
- Side-by-side montage with timepoint labels

### Morphological Overlays
- HSV masks showing red detection regions
- Structural edge overlays (arm outline, elbow, creases)
- Arm orientation overlay (principal axis direction)
- Contour visualizations on cropped images
- SIFT keypoint displays

### Diagnostic Panels
- 2×2 grids showing:
  - Original image
  - Arm orientation/warped
  - Feature detection
  - Tracked sites

## Algorithm Parameters

### Detection Parameters
- **HSV Red Range**: H:0-10 or 160-180, S:70-255, V:50-255
- **Size Filter**: 30-500 px area
- **Circularity Threshold**: > 0.4
- **Spacing Ideal**: 2.5 cm (real-world)
- **Spacing Tolerance**: 0.5-1.5 cm
- **Search Radius**: 5 cm from markers
- **Alignment Weight**: 5× for parallel to arm axis

### Registration Parameters
- **Landmark Match Threshold**: Minimum 3 corresponding points
- **RANSAC Iterations**: 1000
- **RANSAC Threshold**: 3.0 pixels
- **ECC Iterations**: 5000 (fallback)
- **ECC Epsilon**: 1e-10 (fallback)

### Preprocessing Parameters
- **CLAHE Clip Limit**: 2.0
- **CLAHE Grid Size**: (8, 8)
- **Bilateral d**: 9
- **Bilateral Sigma Color**: 75
- **Bilateral Sigma Space**: 75
- **Unsharp Radius**: 1.0
- **Unsharp Amount**: 0.5

## Performance Considerations

### Image Size Limits
- **Maximum dimensions**: 8848×8848 pixels
- **Auto-resize target**: 884×884 pixels
- **Maximum file size**: 50MB

### API Constraints
- **Model**: Claude Sonnet 4.5
- **Max tokens**: 20,000 (output)
- **Thinking tokens**: 16,384
- **Timeout**: 30 seconds per request
- **Retries**: 3 with exponential backoff

### Processing Time
- **Landmark extraction**: 10-20 seconds (API call)
- **Pre-cropping**: < 1 second per image
- **Registration**: < 2 seconds per image pair
- **Site detection**: < 1 second
- **Total pipeline**: ~30-45 seconds for 3 images

## Validation Metrics

### Registration Quality
- **Mean landmark error**: Target < 50 pixels
- **Maximum landmark error**: Target < 100 pixels
- **Successful if**: At least 3 landmarks match

### Detection Quality
- **Sites detected**: Should find exactly 2
- **Spacing validation**: 0.5-1.5 cm (real-world)
- **Alignment validation**: < 30° from arm axis
- **Size similarity**: Ratio < 2.0

### Output Validation
- **Composite generated**: All 3 panels present
- **Sites tracked**: Same coordinates across timepoints
- **Visual inspection**: Marks align with pathergy reactions