# Registration Methods - Integration Status

## Overview
This document summarizes all registration approaches attempted and their integration status in the pathergy test analysis pipeline.

## Current Pipeline Status (main.py)

### Integrated Methods:
1. **VLM Landmark Registration** ✓ Integrated (Default)
   - Location: `main.py:565` - `affine_register()` function
   - Uses VLM-extracted anatomical landmarks from `landmarks.json`
   - Handles pre-crop offset adjustments automatically
   - Falls back to ECC if landmarks unavailable
   - **Issue**: VLM coordinate precision insufficient (~50-90px errors)

2. **ECC Registration** ✓ Integrated (Fallback)
   - Location: `main.py:665` - ECC fallback in `affine_register()`
   - Intensity-based correlation using skin mask
   - **Issue**: Creates massive warping artifacts, poor alignment

### Diagnostic Outputs:
- `--save-morphological-overlays`: Saves intermediate images
  - `intermediate_baseline_precrop.jpg`
  - `intermediate_early_precrop.jpg`
  - `intermediate_late_precrop.jpg`
  - `intermediate_early_warped.jpg`
  - `intermediate_late_warped.jpg`
  - `baseline_tracked_sites.jpg`
  - `early_tracked_sites.jpg`
  - `late_tracked_sites.jpg`

## Standalone Scripts

### 1. Landmark Extraction

#### `extract_landmarks.py` (Original)
- Basic VLM landmark extraction
- **Status**: Superseded by improved version
- **Issue**: Insufficient prompting, poor correspondence

#### `extract_landmarks_improved.py` ✓ Recommended
- Enhanced VLM prompting emphasizing CORRESPONDING features
- Requests descriptions to validate same anatomical location
- Validation checks ensure features found in all 3 images
- **Output**: `landmarks.json`
- **Issue**: VLM gives good descriptions but imprecise coordinates

### 2. Registration Testing

#### `landmark_registration.py`
- Standalone landmark-based registration tester
- Visualizes alignment with landmarks overlaid
- Useful for validating landmark quality
- **Usage**: Tests landmarks.json directly on full images

#### `hybrid_landmark_detection.py`
- Pure computer vision landmark detection
- No VLM, uses CV algorithms (threshold, edge detection, etc.)
- **Result**: Only found 2/4 features consistently
- **Issue**: Too brittle for varying image conditions

#### `sift_registration.py`
- SIFT feature matching + RANSAC
- **Result**: Matched background instead of arm
- **Issue**: Kitchen cabinets/windows are static, arm moved

#### `sift_registration_masked.py`
- SIFT with arm-only masking via skin segmentation
- **Result**: Failed - only 0-3 matches
- **Issue**: Skin texture too uniform for SIFT

### 3. Comparison & Visualization

#### `compare_registration_methods.py`
- Compares ECC vs VLM landmark registration
- Creates side-by-side visualizations
- Shows transform matrices and alignment metrics

#### `compare_all_registration_methods.py` ✓ **COMPREHENSIVE**
- Visual comparison of ALL methods in single panel
- Tests: No Registration, ECC, VLM, SIFT
- Shows Day 1 and Day 2 results in parallel
- Color-coded detection boxes show alignment quality
- **Output**: `registration_comparison.jpg`
- **Best for**: Understanding which methods work/fail at a glance

### 4. Verification Scripts

#### `verify_registration.py`
- Verifies landmark alignment on ORIGINAL full images
- Draws landmarks before any processing

#### `verify_pipeline_landmarks.py`
- Verifies landmarks on PRE-CROPPED pipeline images
- Adjusts coordinates for crop offsets

#### `verify_final_registration.py`
- Verifies final warped image alignment
- Draws BASELINE landmarks on all warped images
- Shows if registration achieved correct correspondence

#### `visualize_detection.py`
- Shows where detection algorithm found features
- Overlays marker, contours, and ROI box
- Useful for debugging detection vs registration issues

### 5. Testing Scripts

#### `test_vision_models.py`
- Tests multiple VLMs on single baseline image
- Compares: granite, gemma (multiple sizes), llama4, aya

#### `test_progression.py`
- Tests VLMs with all 3 images simultaneously
- Evaluates temporal progression understanding

#### `test_vlm_precision.py`
- Tests coordinate extraction precision
- Tests bounding box generation
- Tests code generation capabilities

#### `test_sapiens.py`
- Attempted Sapiens model integration
- **Result**: Failed - model expects full body, not arm crops

## Visual Comparison Results

From `registration_comparison.jpg`:

| Method | Day 1 Result | Day 2 Result | Assessment |
|--------|--------------|--------------|------------|
| No Registration | ✗ Wrong location | ✗ Wrong location | Baseline - proves registration needed |
| ECC | ✗ Black artifacts | ✗ Black artifacts | Massive warping, unusable |
| VLM | ⚠️ Some warping | ⚠️ Some warping | Better than ECC but still wrong |
| SIFT | ✗ Failed | ✗ Failed | Too few matches on skin |

## Integration Checklist

### ✓ Completed:
- [x] VLM landmark extraction integrated into pipeline
- [x] ECC fallback integrated
- [x] Diagnostic outputs for all intermediate steps
- [x] Visual comparison of all methods
- [x] Tracked sites visualization showing registration quality
- [x] Transform matrix logging with decomposition
- [x] Pre-crop offset handling for coordinate translation

### ⚠️ Known Issues:
- [ ] VLM coordinate precision insufficient for accurate registration
- [ ] ECC creates severe warping artifacts
- [ ] SIFT fails on uniform skin texture
- [ ] No working registration method currently available

### 🔴 Not Integrated:
- [ ] Registration method selector (command-line flag)
- [ ] Automatic method switching based on quality metrics
- [ ] Deep learning features (SuperPoint, LoFTR, etc.)
- [ ] Manual landmark annotation tool
- [ ] Iterative VLM coordinate refinement

## Recommended Next Steps

1. **Try Deep Learning Features**:
   - SuperPoint + SuperGlue
   - LoFTR (detector-free matching)
   - R2D2 features
   - These handle large viewpoint changes better than SIFT

2. **Hybrid VLM + CV Refinement**:
   - Use VLM to identify ROI around landmarks
   - Use template matching or local feature detection to refine coordinates
   - Iteratively improve landmark precision

3. **Use Pathergy Markers Themselves**:
   - The X/+ markers are drawn at specific locations
   - Detect markers in all images
   - Use marker centers as primary landmarks
   - Supplement with other features

4. **Interactive Annotation**:
   - Build simple GUI for manual landmark selection
   - Click same feature in all 3 images
   - Save coordinates, use for registration
   - One-time manual effort for this dataset

## Files Summary

### Core Pipeline:
- `main.py` - Main pipeline with VLM registration integrated

### Landmark Extraction:
- `extract_landmarks_improved.py` - Best VLM extraction approach
- `landmarks.json` - VLM-extracted coordinates (current, may be inaccurate)

### Registration Implementations:
- `landmark_registration.py` - Standalone landmark registration
- `sift_registration.py` - SIFT + RANSAC (failed)
- `sift_registration_masked.py` - SIFT with masking (failed)
- `hybrid_landmark_detection.py` - Pure CV detection (partial)

### Comparison & Debugging:
- `compare_all_registration_methods.py` - **Comprehensive visual comparison**
- `compare_registration_methods.py` - ECC vs VLM comparison
- `verify_*.py` (3 files) - Various verification scripts
- `visualize_detection.py` - Detection location visualization

### Testing:
- `test_vision_models.py` - VLM comparison
- `test_progression.py` - Temporal progression analysis
- `test_vlm_precision.py` - Coordinate precision testing
- `test_sapiens.py` - Sapiens integration attempt

## Usage Examples

### Run pipeline with current method:
```bash
python3 main.py --baseline day0.jpg --early day1.jpg --late day2.jpg \
    --output-dir output --save-morphological-overlays
```

### Generate comprehensive comparison:
```bash
python3 compare_all_registration_methods.py
# Output: registration_comparison.jpg
```

### Re-extract landmarks with improved prompting:
```bash
python3 extract_landmarks_improved.py
# Output: landmarks.json (overwrites existing)
```

### Verify landmark quality:
```bash
python3 verify_registration.py
# Output: verify_final_composite.jpg
```

## Conclusion

All registration approaches have been implemented, tested, and integrated where possible. Visual comparisons clearly show that **none of the current methods work adequately**. The VLM approach is closest but needs better coordinate precision. Deep learning-based feature matching or interactive annotation are the most promising paths forward.
