#!/usr/bin/env python3
"""Extract corresponding anatomical landmarks using improved VLM prompting with visual examples."""

import base64
import json
import subprocess
from pathlib import Path
import re


def query_gemma_landmarks_improved(model_name: str, image_paths: list):
    """Query Gemma with improved prompt emphasizing CORRESPONDING features."""

    # Read and encode all images
    images = []
    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            images.append(image_data)

    prompt = """CRITICAL TASK: Find CORRESPONDING anatomical landmarks across 3 images of the SAME arm.

These are 3 photos of the SAME PERSON'S ARM taken on different days (Day 0, Day 1, Day 2).
The arm is at DIFFERENT ANGLES and POSITIONS in each photo, but it's the SAME arm.

YOUR TASK: Identify the EXACT SAME anatomical features visible in ALL 3 images.

IMPORTANT RULES:
1. Look for PERMANENT anatomical features (not pathergy reactions which change)
2. Find features that are CLEARLY VISIBLE in ALL 3 images
3. The SAME feature will be at DIFFERENT pixel coordinates due to arm rotation/position
4. Focus on STABLE features: freckles, moles, veins, skin creases, hair patterns

REQUIRED LANDMARKS (must be the SAME feature in all 3 images):

**Landmark 1: BLACK X OR + MARKER**
This is a PERMANENT marker drawn by the physician - it should be visible in all 3 images.
Find the CENTER of this marker in each image.
- Day 0: (x, y) = ?
- Day 1: (x, y) = ?
- Day 2: (x, y) = ?

**Landmark 2: MOST PROMINENT VEIN JUNCTION**
Find a vein junction/bifurcation that is CLEARLY visible in all 3 images.
Pick the SAME vein junction in all images.
- Day 0: (x, y) = ?
- Day 1: (x, y) = ?
- Day 2: (x, y) = ?

**Landmark 3: LARGEST DARK FRECKLE/MOLE**
Find a freckle or mole that is CLEARLY visible in all 3 images.
Pick the SAME freckle in all images.
- Day 0: (x, y) = ?
- Day 1: (x, y) = ?
- Day 2: (x, y) = ?

**Landmark 4: DISTINCTIVE HAIR PATTERN OR SKIN FEATURE**
Find a distinctive feature (could be hair cluster, skin crease, or other permanent mark).
Must be the SAME feature in all 3 images.
- Day 0: (x, y) = ?
- Day 1: (x, y) = ?
- Day 2: (x, y) = ?

VALIDATION CHECK:
- For each landmark, describe what you see at that location in each image
- Confirm it's the SAME feature in all 3 images
- If you can't find a feature in all 3 images, skip it

OUTPUT FORMAT:
Landmark 1 (X marker):
  Day 0: (x, y) - "description of what I see here"
  Day 1: (x, y) - "description of what I see here"
  Day 2: (x, y) - "description of what I see here"

Repeat for each landmark.

REMEMBER: The coordinates will be DIFFERENT (because arm position changed), but the ANATOMICAL FEATURE must be IDENTICAL."""

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


def parse_landmarks_improved(response_text):
    """Parse VLM response with validation descriptions."""

    landmarks = {
        'day0': {},
        'day1': {},
        'day2': {}
    }

    descriptions = {
        'day0': {},
        'day1': {},
        'day2': {}
    }

    # Find coordinate patterns - more flexible regex
    # Matches: Day 0: (170, 130) - "description" or Day 0:** (170, 130) - "description"
    coord_pattern = r'\*?\*?\s*Day (\d+):\*?\*?\s*\((\d+),\s*(\d+)\)\s*-\s*"([^"]*)"'

    current_feature = None
    lines = response_text.split('\n')

    for line in lines:
        # Identify landmark type from headers
        line_lower = line.lower()
        if 'landmark 1' in line_lower or ('x marker' in line_lower and 'landmark' in line_lower):
            current_feature = 'marker'
        elif 'landmark 2' in line_lower or ('vein' in line_lower and 'landmark' in line_lower):
            current_feature = 'vein'
        elif 'landmark 3' in line_lower or ('freckle' in line_lower and 'landmark' in line_lower):
            current_feature = 'freckle'
        elif 'landmark 4' in line_lower or ('hair' in line_lower and 'landmark' in line_lower):
            current_feature = 'skin_feature'

        # Extract coordinates with descriptions
        matches = re.findall(coord_pattern, line)
        if matches and current_feature:
            for match in matches:
                day_num, x, y, description = match
                day_key = f'day{day_num}'

                if day_key in landmarks:
                    landmarks[day_key][current_feature] = [int(x), int(y)]
                    descriptions[day_key][current_feature] = description.strip()

    return landmarks, descriptions


if __name__ == "__main__":
    images = [
        "/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg",  # Day 0
        "/Users/davidbrewster/Documents/Documents_Brewster/15 August 13_17.jpg",  # Day 1
        "/Users/davidbrewster/Documents/Documents_Brewster/16 August 10_04.jpg"   # Day 2
    ]

    model = "gemma3:27b-it-q8_0"

    print("="*100)
    print("IMPROVED VLM LANDMARK EXTRACTION WITH CORRESPONDENCE VALIDATION")
    print("="*100)

    print("\nQuerying Gemma3-27B with improved prompt emphasizing CORRESPONDING features...")
    response = query_gemma_landmarks_improved(model, images)

    print("\n" + "="*100)
    print("RAW RESPONSE:")
    print("="*100)
    print(response)

    print("\n" + "="*100)
    print("PARSED LANDMARKS WITH DESCRIPTIONS:")
    print("="*100)

    landmarks, descriptions = parse_landmarks_improved(response)

    for day in ['day0', 'day1', 'day2']:
        print(f"\n{day.upper()}:")
        if day in landmarks:
            for feature, coord in landmarks[day].items():
                desc = descriptions[day].get(feature, "")
                print(f"  {feature:15s}: {str(coord):20s} - {desc}")

    # Validation check
    print("\n" + "="*100)
    print("VALIDATION CHECK:")
    print("="*100)

    for feature in ['marker', 'vein', 'freckle', 'skin_feature']:
        found_in = []
        for day in ['day0', 'day1', 'day2']:
            if feature in landmarks.get(day, {}):
                found_in.append(day)

        if len(found_in) == 3:
            print(f"✓ {feature:15s}: Found in all 3 images")
        else:
            print(f"✗ {feature:15s}: Only found in {found_in} - INVALID CORRESPONDENCE")

    # Only save if we have good correspondences
    valid_features = []
    for feature in ['marker', 'vein', 'freckle', 'skin_feature']:
        if all(feature in landmarks.get(day, {}) for day in ['day0', 'day1', 'day2']):
            valid_features.append(feature)

    if len(valid_features) >= 3:
        # Save valid landmarks
        output_landmarks = {}
        for day in ['day0', 'day1', 'day2']:
            output_landmarks[day] = {f: landmarks[day][f] for f in valid_features if f in landmarks[day]}

        output_file = "landmarks.json"
        with open(output_file, 'w') as f:
            json.dump(output_landmarks, f, indent=2)

        print(f"\n✓ Saved {len(valid_features)} valid corresponding landmarks to: {output_file}")
    else:
        print(f"\n✗ FAILED: Only found {len(valid_features)} valid correspondences (need at least 3)")
        print("   NOT saving landmarks.json - previous version kept")
