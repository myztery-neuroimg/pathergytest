#!/usr/bin/env python3
"""
Final script to get proper anatomical landmarks using Claude API.
Uses pre-cropped images and focuses on anatomical features only.
"""

import base64
import json
import requests
from pathlib import Path
import argparse
import logging
import numpy as np


def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_claude_anatomical_landmarks(baseline_path, early_path, late_path, api_key):
    """
    Get anatomical landmarks from Claude for pre-cropped forearm images.
    """
    # Encode images
    images = []
    for path in [baseline_path, early_path, late_path]:
        img_data = encode_image(path)
        images.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_data
            }
        })

    # Create request
    messages = [
        {
            "role": "user",
        "content": [
            {
                "type": "text",
                "text": """These are 3 pre-cropped images showing ONLY the forearm region from different days.
The arm is in different positions/angles in each image.

CRITICAL: Find ANATOMICAL landmarks only - NOT the pathergy test marks!

Look for these features:
1. Hair pattern boundary where dense hair meets less dense area
2. Distinctive vein or skin crease (NOT near the test mark)
3. A specific cluster of 3-4 hairs in a recognizable pattern
4. Edge of the arm or wrist area if visible

For EACH feature, identify the SAME anatomical point in all 3 images.
The coordinates will be DIFFERENT because the arm moves!

Provide coordinates as a JSON object with this exact structure:
{
  "day0": {
    "hair_boundary": [x, y],
    "vein": [x, y],
    "hair_cluster": [x, y],
    "arm_edge": [x, y]
  },
  "day1": {
    "hair_boundary": [x, y],
    "vein": [x, y],
    "hair_cluster": [x, y],
    "arm_edge": [x, y]
  },
  "day2": {
    "hair_boundary": [x, y],
    "vein": [x, y],
    "hair_cluster": [x, y],
    "arm_edge": [x, y]
  }
}"""
            },
            images[0],
            {"type": "text", "text": "Image 1: Day 0 (baseline) - pre-cropped forearm"},
            images[1],
            {"type": "text", "text": "Image 2: Day 1 - pre-cropped forearm"},
            images[2],
            {"type": "text", "text": "Image 3: Day 2 - pre-cropped forearm"}
        ]
    }]

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",  # Standard API version
        "content-type": "application/json"
    }

    data = {
        "model": "claude-sonnet-4-5-20250929",  # Claude Sonnet 4.5 exact version
        "max_tokens": 20000,  # Must be > thinking budget
        "messages": messages,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 16384  # Token budget for thinking
        },
        "temperature": 1.0  # Must be 1 when thinking is enabled
    }

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data,
        timeout=60
    )

    if response.status_code == 200:
        # Get the response text
        response_data = response.json()
        # Handle both thinking and non-thinking response formats
        if 'content' in response_data and response_data['content']:
            for item in response_data['content']:
                if item.get('type') == 'text':
                    return item['text']
        return ""
    else:
        error_msg = response.json() if response.content else response.text
        print(f"API Error Response: {error_msg}")
        raise Exception(f"API error: {response.status_code}")


def parse_landmarks(response):
    """Parse the JSON landmark response."""
    try:
        # Remove markdown code blocks if present
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0].strip()
        elif '```' in response:
            response = response.split('```')[1].split('```')[0].strip()

        # Parse JSON response
        data = json.loads(response)

        # Map to expected format
        landmarks = {'day0': {}, 'day1': {}, 'day2': {}}

        # Map feature names to expected keys
        feature_map = {
            'hair_boundary': 'marker',
            'vein': 'vein',
            'hair_cluster': 'freckle',
            'arm_edge': 'arm_edge'
        }

        for day in ['day0', 'day1', 'day2']:
            if day in data:
                for feature, mapped in feature_map.items():
                    if feature in data[day]:
                        landmarks[day][mapped] = data[day][feature]

        return landmarks
    except json.JSONDecodeError:
        # Fallback to regex parsing if not valid JSON
        print("Warning: Response was not valid JSON, falling back to regex parsing")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, help='Baseline pre-cropped image')
    parser.add_argument('--early', required=True, help='Early pre-cropped image')
    parser.add_argument('--late', required=True, help='Late pre-cropped image')
    parser.add_argument('--output', default='landmarks.json')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load API key
    api_key_file = Path.home() / '.ANTHROPIC_API_KEY'
    api_key = api_key_file.read_text().strip()

    # Get landmarks
    logging.info("Requesting anatomical landmarks from Claude...")
    response = get_claude_anatomical_landmarks(
        args.baseline, args.early, args.late, api_key
    )

    print("\n" + "="*80)
    print("CLAUDE RESPONSE:")
    print("="*80)
    print(response)

    # Parse
    landmarks = parse_landmarks(response)

    print("\n" + "="*80)
    print("PARSED LANDMARKS:")
    print("="*80)
    print(json.dumps(landmarks, indent=2))

    # Validate movement
    print("\n" + "="*80)
    print("VALIDATION:")
    print("="*80)
    for feature in ['marker', 'vein', 'freckle', 'arm_edge']:
        if all(feature in landmarks[day] for day in ['day0', 'day1', 'day2']):
            coords = [landmarks[day][feature] for day in ['day0', 'day1', 'day2']]
            dist_01 = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
            dist_02 = np.linalg.norm(np.array(coords[0]) - np.array(coords[2]))
            print(f"{feature}: Day0→Day1: {dist_01:.1f}px, Day0→Day2: {dist_02:.1f}px")

    # Save
    with open(args.output, 'w') as f:
        json.dump(landmarks, f, indent=2)

    logging.info(f"Saved landmarks to {args.output}")


if __name__ == '__main__':
    main()