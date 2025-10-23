# Pathergy Test Alignment & Analysis Pipeline

A secure, automated pipeline for aligning and analyzing serial pathergy test images using anatomical landmark registration.

## 🔒 Security-First Implementation

This pipeline has been redesigned with security best practices:
- ✅ Environment variable API key storage (no plaintext files)
- ✅ Path traversal protection with directory whitelisting
- ✅ Input validation and automatic image resizing (max 884×884)
- ✅ Secure API calls using `requests` library with retries
- ✅ No hardcoded paths or credentials

## 🚀 Quick Start

### Prerequisites
- Python >= 3.10
- Anthropic API key (Claude Sonnet 4.5)

### Installation

#### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv via Homebrew (macOS/Linux)
brew install uv

# Clone the repository
git clone https://github.com/myztery-neuroimg/pathergytest.git
cd pathergytest

# Initialize uv environment (automatically uses pyproject.toml)
uv sync

# Set your API key
export ANTHROPIC_API_KEY='your-api-key-here'
```

#### Option 2: Using pip

```bash
git clone https://github.com/myztery-neuroimg/pathergytest.git
cd pathergytest
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Basic Usage

```bash
# Run with uv
uv run python3 src/run_alignment.py \
    --baseline path/to/baseline.jpg \
    --early path/to/day1.jpg \
    --late path/to/day2.jpg \
    --enable-preprocessing

# Or with regular python
python3 src/run_alignment.py \
    --baseline path/to/baseline.jpg \
    --early path/to/day1.jpg \
    --late path/to/day2.jpg \
    --enable-preprocessing
```

## 📁 Project Structure

```
pathergytest/
├── src/                          # Core source code
│   ├── main.py                  # Core alignment & detection logic
│   ├── run_alignment.py         # Main entry point
│   ├── secure_landmark_extraction.py  # VLM landmark extraction
│   ├── get_anatomical_landmarks_final.py  # Landmark API interface
│   └── detect_two_sites_simple.py  # Puncture site detection
├── scripts/                      # Utility scripts
│   ├── compare_*.py             # Registration comparison tools
│   ├── verify_*.py              # Verification scripts
│   ├── landmark_registration.py # Landmark-based registration
│   ├── sift_registration*.py   # SIFT-based registration
│   └── visualize_detection.py  # Visualization tools
├── docs/                         # Documentation
│   ├── SECURITY_REVIEW.md       # Security audit
│   ├── TECHNICAL_METHODOLOGY.md # Technical details
│   └── CLINICAL_INTERPRETATION.md  # Clinical guide
├── config/                       # Configuration files
│   └── landmarks*.json          # Landmark data
├── .pylintrc                    # Pylint configuration
├── pyproject.toml               # Project metadata & dependencies
└── README.md                    # This file
```

## 📋 Features

### Core Capabilities
- **Anatomical Landmark Detection**: Uses Claude Sonnet 4.5 with thinking mode for precise landmark identification
- **Automatic Image Alignment**: Registers day 1 and day 2 images to baseline using affine transformation
- **Pathergy Site Tracking**: Detects test marks on baseline and tracks same locations across timepoints
- **Security Hardened**: Input validation, path sanitization, secure API handling

### Image Processing
- **Intelligent Pre-cropping**: Automatically identifies and crops to forearm region
- **Multi-stage Preprocessing**:
  - Illumination correction (LAB color space + CLAHE)
  - Bilateral filtering (edge-preserving noise reduction)
  - Color normalization
  - Optional unsharp masking
- **Automatic Resizing**: Large images automatically resized to 884×884 while maintaining aspect ratio

### Visualization Outputs
- **Timeline Composite**: Side-by-side comparison showing pathergy test progression
- **Tracked Sites**: Individual images with marked injection sites
- **Diagnostic Panels**: Optional debug visualizations of processing stages

## 🏗️ Architecture

### Main Components

#### 1. `src/secure_landmark_extraction.py`
Primary landmark extraction with full security measures:
- Environment variable API key loading
- Path validation against whitelist
- Image size/dimension validation
- Automatic resizing for large images
- Secure API calls with retry logic

```bash
uv run python3 src/secure_landmark_extraction.py \
    --baseline img1.jpg \
    --early img2.jpg \
    --late img3.jpg \
    --output config/landmarks.json
```

#### 2. `src/run_alignment.py`
Clean runner using imported functions (no subprocess calls):
- Direct function imports from `src/main.py`
- Integrated landmark extraction
- Full pipeline execution

#### 3. `src/main.py`
Core alignment and detection logic:
- Pre-crop to forearm region
- VLM landmark-based registration
- Pathergy site detection
- Montage generation

## 🔐 Security Features

### API Key Management
```bash
# Preferred: Environment variable
export ANTHROPIC_API_KEY='your-key-here'

# Fallback: File (with warning)
echo 'your-key' > ~/.ANTHROPIC_API_KEY
```

### Path Validation
- Whitelist of allowed directories
- Symlink resolution to prevent traversal
- File extension validation (.jpg, .jpeg, .png, .bmp, .tiff)
- File size limits (50MB max)

### Input Validation
- Maximum image dimensions: 8848×8848
- Automatic resizing to 884×884 for large images
- Image verification using PIL before processing
- Sanitized error messages (no sensitive data exposure)

## 📊 Pipeline Workflow

```
1. Load Images
   ↓
2. Validate Paths & Resize if Needed
   ↓
3. Extract Anatomical Landmarks (Claude API)
   ↓
4. Pre-crop to Forearm Region
   ↓
5. Apply Preprocessing (Optional)
   ↓
6. Register Images Using Landmarks
   ↓
7. Detect Pathergy Sites on Baseline
   ↓
8. Track Sites Across Timepoints
   ↓
9. Generate Composite & Outputs
```

## 🛠️ Advanced Options

### Preprocessing Parameters
```bash
--enable-preprocessing       # Enable all preprocessing
--illumination-clip 2.0     # CLAHE clip limit
--bilateral-d 9             # Bilateral filter diameter
--bilateral-sigma-color 75  # Color space sigma
--bilateral-sigma-space 75  # Coordinate space sigma
```

### Visualization Options
```bash
--generate-diagnostic-panels    # Debug visualizations
--save-morphological-overlays  # Individual overlays
--radius 50                    # Detection radius (pixels)
```

### Logging Levels
```bash
--log-level DEBUG  # Verbose debugging
--log-level INFO   # Standard output
--log-level ERROR  # Errors only
```

## 📁 Output Files

```
outputs/
├── pathergy_timeline_composite.jpg  # Main result
├── baseline_tracked_sites.jpg       # Day 0 with marks
├── early_tracked_sites.jpg          # Day 1 aligned
├── late_tracked_sites.jpg           # Day 2 aligned
└── config/landmarks.json            # Detected landmarks
```

## 🧪 Testing

```bash
# Run with uv
uv run python3 -m pytest

# Run security checks
uv run python3 -m bandit -r src/
uv run python3 -m pylint src/

# Test with sample images
uv run python3 src/run_alignment.py \
    --baseline test_images/baseline.jpg \
    --early test_images/day1.jpg \
    --late test_images/day2.jpg
```

## 🔧 Utility Scripts

The `scripts/` directory contains various utility scripts for analysis and verification:

```bash
# Compare registration methods
uv run python3 scripts/compare_all_registration_methods.py \
    baseline.jpg early.jpg late.jpg

# Verify registration results
uv run python3 scripts/verify_registration.py \
    baseline.jpg early.jpg late.jpg \
    config/landmarks.json

# Run SIFT-based registration
uv run python3 scripts/sift_registration.py \
    baseline.jpg early.jpg late.jpg
```

## 📚 Documentation

### Core Documentation
- [`docs/SECURITY_REVIEW.md`](docs/SECURITY_REVIEW.md) - Security audit and current vulnerability status
- [`docs/TECHNICAL_METHODOLOGY.md`](docs/TECHNICAL_METHODOLOGY.md) - Detailed technical implementation
- [`docs/CLINICAL_INTERPRETATION.md`](docs/CLINICAL_INTERPRETATION.md) - Clinical guide for pathergy test interpretation

### Technical Details
- **Pipeline Order & Methodology** - See [Technical Methodology](docs/TECHNICAL_METHODOLOGY.md)
- **Algorithm Parameters** - See [Technical Methodology](docs/TECHNICAL_METHODOLOGY.md#algorithm-parameters)
- **Clinical Protocol** - See [Clinical Interpretation](docs/CLINICAL_INTERPRETATION.md#test-protocol)
- **API Documentation** - See function docstrings in source files

## ⚠️ Important Notes

1. **Clinical Use**: This tool is for research/educational purposes only. Not for diagnostic use.
2. **API Costs**: Uses Claude Sonnet 4.5 API (~$3/$15 per million tokens)
3. **Privacy**: Process images locally before API calls. Consider PHI implications.
4. **Landmarks**: System uses anatomical features (veins, hair patterns) NOT test marks for alignment

## 🔍 Code Quality

- **Pylint Score**: 9.50/10
- **Type Hints**: Extensive type annotations throughout
- **Documentation**: Comprehensive docstrings
- **Security**: Reviewed and hardened against common vulnerabilities

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Run quality checks:
   ```bash
   uv run python3 -m pylint src/
   uv run python3 -m pytest
   ```
4. Submit pull request with clear description

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Anthropic for Claude API
- OpenCV and scikit-image communities
- Behçet's disease research community

## 📧 Contact

For questions or issues, please open a GitHub issue.

---
*Last Updated: October 2025*
*Version: 2.1 (Restructured & Hardened)*
