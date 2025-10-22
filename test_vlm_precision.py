#!/usr/bin/env python3
"""Test Gemma3-27B's precision for coordinate and bounding box extraction."""

import base64
import json
import subprocess
from pathlib import Path


def query_gemma(model_name: str, image_path: str, prompt: str):
    """Query Gemma vision model with an image."""

    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Prepare request
    request = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_data],
        "stream": False
    }

    # Call ollama API
    result = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/generate',
         '-d', json.dumps(request)],
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode == 0:
        response = json.loads(result.stdout)
        return response.get('response', 'No response')
    else:
        return f"Error: {result.stderr}"


if __name__ == "__main__":
    baseline_image = "/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg"
    model = "gemma3:27b-it-q8_0"

    print("="*100)
    print("TEST 1: Precise Pixel Coordinate Extraction")
    print("="*100)

    prompt1 = """You are analyzing a pathergy test image. Please provide PRECISE pixel coordinates for the following landmarks:

1. **+ Marker center**: Estimate the (x, y) pixel coordinates of the center of the + or X marker drawn on the skin.

2. **Most prominent vein**: Identify the most visible vein and provide:
   - Starting point (x, y) coordinates
   - Ending point (x, y) coordinates

3. **Largest freckle/mole**: Provide (x, y) coordinates of its center.

4. **Each pathergy site**: For each small red dot/papule you can see, provide its (x, y) coordinate.

5. **Arm boundaries**: Estimate the bounding box for the visible arm:
   - Top-left corner: (x, y)
   - Bottom-right corner: (x, y)

**IMPORTANT**:
- Provide specific numerical coordinates like (450, 320)
- Assume the image is approximately 1024×768 pixels (or describe if different)
- Use format: "Feature Name: (x, y)" or "Feature: (x1, y1) to (x2, y2)"
- Be as precise as possible based on what you see"""

    print("\nPrompt:", prompt1[:200] + "...")
    print("\nQuerying Gemma3-27B...")
    result1 = query_gemma(model, baseline_image, prompt1)
    print("\nRESULT:")
    print(result1)

    print("\n" + "="*100)
    print("TEST 2: Bounding Box Generation")
    print("="*100)

    prompt2 = """Please provide bounding boxes (in pixel coordinates) for the following regions of interest in this pathergy test image:

1. **Region around + marker** (for pathergy site detection):
   - A rectangle approximately 6cm × 6cm around the marker
   - Format: x_min, y_min, x_max, y_max

2. **Arm segmentation box**:
   - Tight bounding box around just the visible arm/forearm
   - Format: x_min, y_min, x_max, y_max

3. **Each individual pathergy papule**:
   - Small boxes (e.g., 20×20 pixels) around each red dot
   - Format: papule_N: x_min, y_min, x_max, y_max

Provide numerical coordinates based on the image dimensions."""

    print("\nPrompt:", prompt2[:200] + "...")
    print("\nQuerying Gemma3-27B...")
    result2 = query_gemma(model, baseline_image, prompt2)
    print("\nRESULT:")
    print(result2)

    print("\n" + "="*100)
    print("TEST 3: Code Generation for Automation")
    print("="*100)

    prompt3 = """Based on this pathergy test image, please generate Python code using OpenCV that would:

1. Detect the + or X marker (black marking) and return its center coordinates
2. Detect the prominent vein and extract its path
3. Detect red papules (pathergy sites) near the marker
4. Return a list of landmark coordinates that could be used for image registration

The code should use:
- OpenCV (cv2) for image processing
- HSV color space for marker detection (black/dark colors)
- HSV for red papule detection
- Hough lines or edge detection for veins

Please provide complete, runnable Python code with comments.
Include a main function that takes an image path and returns:
```python
{
    'marker_center': (x, y),
    'vein_points': [(x1, y1), (x2, y2), ...],
    'papules': [(x1, y1), (x2, y2), ...],
    'freckles': [(x1, y1), (x2, y2), ...]
}
```"""

    print("\nPrompt:", prompt3[:200] + "...")
    print("\nQuerying Gemma3-27B...")
    result3 = query_gemma(model, baseline_image, prompt3)
    print("\nRESULT:")
    print(result3)

    print("\n" + "="*100)
    print("Summary: VLM Precision Test Complete")
    print("="*100)
