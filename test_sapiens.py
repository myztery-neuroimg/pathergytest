#!/usr/bin/env python3
"""Test Sapiens model for arm segmentation on pathergy test images."""

import torch
from transformers import AutoModelForImageSegmentation, AutoImageProcessor
from PIL import Image
import numpy as np
import cv2


def test_sapiens_segmentation(image_path: str, output_path: str = None):
    """Test Sapiens segmentation on a single image."""

    print(f"Loading Sapiens model from HuggingFace...")

    # Try to load Sapiens model
    # Note: The exact model name might need adjustment based on what's available
    try:
        # Sapiens models are under facebook/sapiens
        model_name = "facebook/sapiens-segmentation-1b"  # Start with 1B model

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageSegmentation.from_pretrained(model_name)

        print(f"Model loaded: {model_name}")

    except Exception as e:
        print(f"Error loading Sapiens model: {e}")
        print("\nTrying alternative approach with standard segmentation model...")

        # Fallback to a general person segmentation model
        model_name = "mattmdjaga/segformer_b2_clothes"  # Human segmentation model
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageSegmentation.from_pretrained(model_name)
        print(f"Using fallback model: {model_name}")

    # Load and process image
    print(f"\nProcessing image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # Prepare image for model
    inputs = processor(images=image, return_tensors="pt")

    # Run inference
    print("Running segmentation...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get segmentation mask
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs[0]

    # Convert to segmentation mask
    segmentation = logits.argmax(dim=1)[0].cpu().numpy()

    # Resize to original image size
    segmentation_resized = cv2.resize(
        segmentation.astype(np.uint8),
        (image.width, image.height),
        interpolation=cv2.INTER_NEAREST
    )

    print(f"Segmentation shape: {segmentation_resized.shape}")
    print(f"Unique classes: {np.unique(segmentation_resized)}")

    # Visualize segmentation
    img_array = np.array(image)

    # Create colored mask overlay
    mask_colored = np.zeros_like(img_array)

    # Color different body parts differently
    unique_classes = np.unique(segmentation_resized)
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
    ]

    for idx, class_id in enumerate(unique_classes):
        if class_id == 0:  # Background
            continue
        color = colors[idx % len(colors)]
        mask_colored[segmentation_resized == class_id] = color

    # Blend with original image
    alpha = 0.5
    overlay = cv2.addWeighted(img_array, 1-alpha, mask_colored, alpha, 0)

    if output_path:
        Image.fromarray(overlay).save(output_path)
        print(f"Saved visualization to: {output_path}")

    # Also save just the mask
    mask_path = output_path.replace('.jpg', '_mask.jpg')
    Image.fromarray((segmentation_resized * 50).astype(np.uint8)).save(mask_path)
    print(f"Saved mask to: {mask_path}")

    return segmentation_resized


if __name__ == "__main__":
    images = [
        ("/Users/davidbrewster/Documents/Documents_Brewster/14 August 10_10.jpg",
         "/Users/davidbrewster/Documents/Documents_Brewster/sapiens_baseline.jpg"),
        ("/Users/davidbrewster/Documents/Documents_Brewster/15 August 13_17.jpg",
         "/Users/davidbrewster/Documents/Documents_Brewster/sapiens_early.jpg"),
        ("/Users/davidbrewster/Documents/Documents_Brewster/16 August 10_04.jpg",
         "/Users/davidbrewster/Documents/Documents_Brewster/sapiens_late.jpg"),
    ]

    for input_path, output_path in images:
        print(f"\n{'='*80}")
        try:
            mask = test_sapiens_segmentation(input_path, output_path)
            print("✓ Success")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
