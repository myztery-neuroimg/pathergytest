#!/usr/bin/env python3
"""Test VLMs on pathergy test image progression across three timepoints."""

import base64
import json
import subprocess
from pathlib import Path
import time


def test_vision_model_multi_image(model_name: str, image_paths: list, prompt: str):
    """Test a vision model with multiple images using Ollama API."""

    # Read and encode all images
    images = []
    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            images.append(image_data)

    # Prepare request
    request = {
        "model": model_name,
        "prompt": prompt,
        "images": images,
        "stream": False
    }

    # Call ollama API
    result = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/generate',
         '-d', json.dumps(request)],
        capture_output=True,
        text=True,
        timeout=600  # 10 minutes max
    )

    if result.returncode == 0:
        response = json.loads(result.stdout)
        return response.get('response', 'No response')
    else:
        return f"Error: {result.stderr}"


if __name__ == "__main__":
    # Three pathergy test images across timepoints
    images = [
        "/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg",  # Day 0 baseline
        "/Users/davidbrewster/Documents/Documents_Brewster/15 August 13_17.jpg",  # Day 1 early
        "/Users/davidbrewster/Documents/Documents_Brewster/16 August 10_04.jpg"   # Day 2 late
    ]

    prompt = """These are three images of the SAME arm taken at different timepoints for a pathergy test:
- Image 1: Day 0 (baseline, immediately after test)
- Image 2: Day 1 (~24 hours later)
- Image 3: Day 2 (~48 hours later)

Please analyze and compare ALL THREE images:

1. **Arm position/angle**: How does the arm position differ across images? Is it rotated, moved, different lighting?

2. **Markers**: Identify the + or X markers drawn on the skin. Are they visible in all images? Where are they located?

3. **Pathergy test sites**: Look for small red dots/papules near the markers. How many can you identify in each image?

4. **Temporal progression**: How does the reaction change from Day 0 → Day 1 → Day 2?
   - Is the redness increasing or decreasing?
   - Are the papules getting larger/smaller?
   - Any new spots appearing?

5. **Anatomical landmarks**: Identify consistent features across all three images that could be used for alignment:
   - Freckles/moles at specific locations
   - Hair patterns
   - Vein positions
   - Skin texture patterns

6. **Registration challenge**: Based on the differences in arm position/angle/lighting, how difficult would it be to automatically align these images to track the same anatomical location?

Be very specific about spatial locations (e.g., "2cm above marker", "upper left quadrant", etc.)."""

    # Test top models
    models_to_test = [
        ("gemma3:27b-it-q8_0", "27B - Largest, most capable"),
        ("gemma3:12b-it-q8_0", "12B - Good balance"),
        ("gemma3n:e4b", "4B - Smallest, efficient")
    ]

    for model, description in models_to_test:
        print(f"\n{'='*100}")
        print(f"MODEL: {model} ({description})")
        print('='*100)

        start_time = time.time()

        try:
            result = test_vision_model_multi_image(model, images, prompt)
            elapsed = time.time() - start_time

            print(result)
            print(f"\n[Time taken: {elapsed:.1f}s]")

        except Exception as e:
            print(f"Error testing {model}: {e}")

        print()
