#!/usr/bin/env python3
"""Test various Ollama vision models for pathergy test image analysis."""

import base64
import json
import subprocess
from pathlib import Path


def test_vision_model(model_name: str, image_path: str, prompt: str):
    """Test a vision model with an image using Ollama API."""

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
        text=True
    )

    if result.returncode == 0:
        response = json.loads(result.stdout)
        return response.get('response', 'No response')
    else:
        return f"Error: {result.stderr}"


if __name__ == "__main__":
    baseline_image = "/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg"

    prompt = """This is a dermatological test image showing an arm with a pathergy test.
Please describe:
1. What body part is visible (arm/forearm/wrist)?
2. Any visible markings or symbols drawn on the skin
3. Any red spots, lesions, or skin reactions (look carefully - there are small red dots)
4. Anatomical features like freckles, moles, hair patterns
5. The general skin texture and any distinctive features

Be specific about locations (e.g., upper left, center, near wrist, etc.)."""

    # Test remaining models
    models_to_test = [
        "aya:35b-23-q8_0",
        "gemma3n:e4b",
        "gemma3:27b-it-q8_0"  # Bonus: larger gemma model
    ]

    for model in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing {model}...")
        print('='*80)

        try:
            result = test_vision_model(model, baseline_image, prompt)
            print(result)
        except Exception as e:
            print(f"Error testing {model}: {e}")

        print()
