#!/usr/bin/env python3
"""Extract corresponding anatomical landmarks from all 3 pathergy images using Gemma3-27B."""

import base64
import json
import subprocess
from pathlib import Path
import re


def query_gemma_landmarks(model_name: str, image_paths: list):
    """Query Gemma for corresponding landmarks across all 3 images."""

    # Read and encode all images
    images = []
    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            images.append(image_data)

    prompt = """These are 3 images of the SAME arm at different timepoints (Day 0, Day 1, Day 2).

I need you to identify the SAME anatomical landmarks in ALL 3 images for image registration.

Please provide pixel coordinates (x, y) for these CORRESPONDING features:

1. **+ Marker center** - The black + or X marking drawn by the physician:
   - Day 0: (x, y)
   - Day 1: (x, y)
   - Day 2: (x, y)

2. **Most prominent vein midpoint** - The same vein visible in all images:
   - Day 0: (x, y)
   - Day 1: (x, y)
   - Day 2: (x, y)

3. **Largest freckle/mole** - The same freckle visible in all images:
   - Day 0: (x, y)
   - Day 1: (x, y)
   - Day 2: (x, y)

4. **Arm outline reference point** - Pick a distinctive point on the arm edge (like elbow crease):
   - Day 0: (x, y)
   - Day 1: (x, y)
   - Day 2: (x, y)

**CRITICAL**: These must be the SAME anatomical locations across all 3 images!
Provide only numerical coordinates in format: "Feature Day N: (x, y)"

Note: The arm position/angle changes between images, but the anatomical locations stay the same."""

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
        timeout=600
    )

    if result.returncode == 0:
        response = json.loads(result.stdout)
        return response.get('response', 'No response')
    else:
        return f"Error: {result.stderr}"


def parse_landmarks(response_text):
    """Parse coordinate responses into structured data."""

    landmarks = {
        'day0': {},
        'day1': {},
        'day2': {}
    }

    # Find all coordinate patterns like (x, y)
    coord_pattern = r'\((\d+),\s*(\d+)\)'

    lines = response_text.split('\n')

    current_feature = None
    for line in lines:
        # Identify feature type
        if 'marker' in line.lower() or '+' in line:
            current_feature = 'marker'
        elif 'vein' in line.lower():
            current_feature = 'vein'
        elif 'freckle' in line.lower() or 'mole' in line.lower():
            current_feature = 'freckle'
        elif 'outline' in line.lower() or 'edge' in line.lower() or 'elbow' in line.lower():
            current_feature = 'arm_edge'

        # Extract coordinates
        if current_feature:
            coords = re.findall(coord_pattern, line)
            if coords:
                x, y = int(coords[0][0]), int(coords[0][1])

                # Determine which day
                if 'day 0' in line.lower() or 'baseline' in line.lower():
                    landmarks['day0'][current_feature] = (x, y)
                elif 'day 1' in line.lower() or 'early' in line.lower():
                    landmarks['day1'][current_feature] = (x, y)
                elif 'day 2' in line.lower() or 'late' in line.lower():
                    landmarks['day2'][current_feature] = (x, y)

    return landmarks


if __name__ == "__main__":
    images = [
        "/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg",  # Day 0
        "/Users/davidbrewster/Documents/Documents_Brewster/15 August 13_17.jpg",  # Day 1
        "/Users/davidbrewster/Documents/Documents_Brewster/16 August 10_04.jpg"   # Day 2
    ]

    model = "gemma3:27b-it-q8_0"

    print("="*100)
    print("Extracting Corresponding Landmarks from All 3 Images")
    print("="*100)

    print("\nQuerying Gemma3-27B with all 3 images...")
    response = query_gemma_landmarks(model, images)

    print("\n" + "="*100)
    print("RAW RESPONSE:")
    print("="*100)
    print(response)

    print("\n" + "="*100)
    print("PARSED LANDMARKS:")
    print("="*100)

    landmarks = parse_landmarks(response)

    for day in ['day0', 'day1', 'day2']:
        print(f"\n{day.upper()}:")
        for feature, coord in landmarks[day].items():
            print(f"  {feature}: {coord}")

    # Save to JSON for use in registration
    output_file = "/Users/davidbrewster/Documents/workspace/2025/pathergytest/landmarks.json"
    with open(output_file, 'w') as f:
        json.dump(landmarks, f, indent=2)

    print(f"\n✓ Saved landmarks to: {output_file}")
