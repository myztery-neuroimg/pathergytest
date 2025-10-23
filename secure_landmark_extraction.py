#!/usr/bin/env python3
"""
Secure landmark extraction with improved security practices.
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import requests
from PIL import Image
import numpy as np


# Security constants
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB max file size
MAX_IMAGE_DIMENSION = 8848  # Max width or height (8848x8848)
RESIZE_DIMENSION = 884  # Target dimension for large images
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
ALLOWED_DIRECTORIES = [
    Path.home() / 'Documents',
    Path.home() / 'Downloads',
    Path.home() / 'Desktop',
    Path('/tmp'),
    Path('/var/tmp')
]


def validate_file_path(file_path: Path, allowed_dirs: Optional[List[Path]] = None) -> bool:
    """
    Validate that a file path is safe to access.

    Args:
        file_path: Path to validate
        allowed_dirs: List of allowed directories (uses ALLOWED_DIRECTORIES if None)

    Returns:
        True if path is valid, raises exception otherwise
    """
    if allowed_dirs is None:
        allowed_dirs = ALLOWED_DIRECTORIES

    # Resolve the path (follows symlinks, makes absolute)
    try:
        resolved_path = file_path.resolve(strict=False)
    except Exception as e:
        raise ValueError(f"Invalid path: {file_path} - {e}")

    # Check if file extension is allowed
    if resolved_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type not allowed: {resolved_path.suffix}")

    # Check if path is within allowed directories
    if allowed_dirs:
        is_allowed = any(
            resolved_path.is_relative_to(allowed_dir.resolve())
            for allowed_dir in allowed_dirs
            if allowed_dir.exists()
        )
        if not is_allowed:
            raise ValueError(f"Path outside allowed directories: {resolved_path}")

    # Check file size
    if resolved_path.exists():
        file_size = resolved_path.stat().st_size
        if file_size > MAX_IMAGE_SIZE:
            raise ValueError(f"File too large: {file_size} bytes (max {MAX_IMAGE_SIZE})")

    return True


def resize_image_if_needed(image: Image.Image, max_dimension: int = RESIZE_DIMENSION) -> Image.Image:
    """
    Resize image if it exceeds the maximum dimension while maintaining aspect ratio.

    Args:
        image: PIL Image to resize
        max_dimension: Maximum width or height

    Returns:
        Resized image or original if within limits
    """
    width, height = image.size

    # Check if resizing is needed
    if width <= max_dimension and height <= max_dimension:
        return image

    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))

    logging.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")

    # Use high-quality Lanczos resampling
    resized = image.resize((new_width, new_height), Image.LANCZOS)
    return resized


def validate_and_load_image(image_path: str, resize: bool = True) -> Tuple[Image.Image, Path]:
    """
    Safely load and validate an image file.

    Args:
        image_path: Path to the image file
        resize: Whether to resize large images

    Returns:
        Tuple of (PIL Image, resolved Path)
    """
    # Convert to Path and validate
    path = Path(image_path)
    validate_file_path(path)

    # Load and verify the image
    try:
        with Image.open(path) as img:
            # Verify the image is valid
            img.verify()

        # Re-open after verify (verify closes the file)
        img = Image.open(path).convert('RGB')

        # Check dimensions
        width, height = img.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ValueError(f"Image dimensions too large: {width}x{height} (max {MAX_IMAGE_DIMENSION})")

        # Resize if needed and requested
        if resize:
            img = resize_image_if_needed(img)

        return img, path.resolve()

    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")


def get_api_key() -> str:
    """
    Securely get the API key from environment variable.

    Returns:
        API key string
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        # Fallback to file if env var not set (with warning)
        api_key_file = Path.home() / '.ANTHROPIC_API_KEY'
        if api_key_file.exists():
            logging.warning("Using API key from file. Please set ANTHROPIC_API_KEY environment variable instead.")
            api_key = api_key_file.read_text().strip()
        else:
            raise ValueError("No API key found. Set ANTHROPIC_API_KEY environment variable.")

    return api_key


def encode_image_safely(image: Image.Image, format: str = 'JPEG', quality: int = 85) -> str:
    """
    Safely encode image to base64.

    Args:
        image: PIL Image to encode
        format: Output format (JPEG or PNG)
        quality: JPEG quality (1-100)

    Returns:
        Base64 encoded string
    """
    import io

    # Save to bytes buffer
    buffer = io.BytesIO()
    if format.upper() == 'JPEG':
        # Convert RGBA to RGB for JPEG
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
            image = rgb_image
        image.save(buffer, format=format, quality=quality, optimize=True)
    else:
        image.save(buffer, format=format, optimize=True)

    # Encode to base64
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def call_claude_api(images: List[Image.Image], prompt: str, api_key: str) -> Dict:
    """
    Call Claude API using requests library instead of subprocess.

    Args:
        images: List of PIL Images
        prompt: Text prompt
        api_key: API key

    Returns:
        API response dictionary
    """
    # Prepare image data
    image_contents = []
    for i, img in enumerate(images):
        img_data = encode_image_safely(img)
        image_contents.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_data
            }
        })
        image_contents.append({
            "type": "text",
            "text": f"Image {i+1} (Day {i})"
        })

    # Prepare request
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}] + image_contents
    }]

    data = {
        "model": "claude-3-5-sonnet-latest",  # Use latest stable model
        "max_tokens": 2048,
        "messages": messages,
        "temperature": 0.3  # Lower temperature for more consistent results
    }

    # Make request with timeout and retries
    max_retries = 3
    timeout = 30  # 30 second timeout

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=timeout
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit
                import time
                wait_time = min(2 ** attempt, 10)  # Exponential backoff
                logging.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                error_msg = response.json() if response.content else response.text
                logging.error(f"API error: {error_msg}")
                if attempt == max_retries - 1:
                    raise Exception(f"API error after {max_retries} attempts: {response.status_code}")

        except requests.exceptions.Timeout:
            logging.warning(f"Request timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise Exception("API request timed out after all retries")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {e}")
            if attempt == max_retries - 1:
                raise

    raise Exception("Failed to get response from API")


def extract_landmarks(baseline_path: str, early_path: str, late_path: str,
                     output_path: str = "landmarks.json",
                     allowed_dirs: Optional[List[str]] = None) -> Dict:
    """
    Extract anatomical landmarks from three images with full security measures.

    Args:
        baseline_path: Path to baseline image
        early_path: Path to early follow-up image
        late_path: Path to late follow-up image
        output_path: Where to save landmarks JSON
        allowed_dirs: Optional list of allowed directories

    Returns:
        Dictionary of landmarks
    """
    # Convert allowed directories if provided
    if allowed_dirs:
        allowed_dirs = [Path(d) for d in allowed_dirs]
    else:
        allowed_dirs = None

    # Load and validate images
    logging.info("Loading and validating images...")
    baseline_img, baseline_resolved = validate_and_load_image(baseline_path, resize=True)
    early_img, early_resolved = validate_and_load_image(early_path, resize=True)
    late_img, late_resolved = validate_and_load_image(late_path, resize=True)

    logging.info(f"Images loaded successfully:")
    logging.info(f"  Baseline: {baseline_resolved} ({baseline_img.size})")
    logging.info(f"  Early: {early_resolved} ({early_img.size})")
    logging.info(f"  Late: {late_resolved} ({late_img.size})")

    # Get API key securely
    api_key = get_api_key()

    # Prepare prompt
    prompt = """Identify anatomical landmarks in these 3 forearm images.
Find features like hair patterns, veins, or skin marks.
Do NOT use the pathergy test marks as landmarks.

Return a JSON object with this structure:
{
  "day0": {"hair_boundary": [x, y], "vein": [x, y], "skin_mark": [x, y]},
  "day1": {"hair_boundary": [x, y], "vein": [x, y], "skin_mark": [x, y]},
  "day2": {"hair_boundary": [x, y], "vein": [x, y], "skin_mark": [x, y]}
}"""

    # Call API
    logging.info("Calling Claude API for landmark extraction...")
    response = call_claude_api([baseline_img, early_img, late_img], prompt, api_key)

    # Parse response
    content = response['content'][0]['text'] if response.get('content') else ""

    # Extract JSON from response
    try:
        # Remove markdown code blocks if present
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()

        landmarks = json.loads(content)
    except json.JSONDecodeError:
        logging.error("Failed to parse JSON response")
        landmarks = {"day0": {}, "day1": {}, "day2": {}}

    # Validate output path for saving
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.json':
        raise ValueError("Output must be a JSON file")

    # Save landmarks
    with open(output_path, 'w') as f:
        json.dump(landmarks, f, indent=2)

    logging.info(f"Landmarks saved to {output_path}")
    return landmarks


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Securely extract anatomical landmarks from pathergy test images"
    )
    parser.add_argument("--baseline", required=True, help="Baseline image path")
    parser.add_argument("--early", required=True, help="Early follow-up image path")
    parser.add_argument("--late", required=True, help="Late follow-up image path")
    parser.add_argument("--output", default="landmarks.json", help="Output JSON file")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--allowed-dirs", nargs="+", help="Additional allowed directories")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    try:
        # Extract landmarks
        landmarks = extract_landmarks(
            args.baseline,
            args.early,
            args.late,
            args.output,
            args.allowed_dirs
        )

        # Print summary
        print("\n" + "="*60)
        print("LANDMARK EXTRACTION COMPLETE")
        print("="*60)
        for day in ['day0', 'day1', 'day2']:
            if day in landmarks:
                print(f"{day}: {len(landmarks[day])} landmarks found")
        print(f"\nSaved to: {args.output}")

    except Exception as e:
        logging.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())