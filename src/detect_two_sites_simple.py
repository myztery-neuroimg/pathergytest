#!/usr/bin/env python3
"""Simple detection that finds TWO puncture sites without complex pairing logic"""

import logging

import cv2
import numpy as np


def detect_two_puncture_sites_simple(pil_img):
    """Detect TWO injection sites using simple HSV thresholding.

    Returns the two most prominent red regions without complex pairing constraints.
    """

    logging.info("Simple detection: Finding TWO most prominent red regions")

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red color detection in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Score each contour by area and redness
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Be more permissive with area
        if area < 2 or area > 500:  # Very small or very large
            continue

        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Calculate average redness in the region
            mask = np.zeros(red_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(hsv, mask=mask)
            redness_score = mean_val[1] + mean_val[2]  # S + V channels

            candidates.append({
                'x': cx,
                'y': cy,
                'area': area,
                'score': area * redness_score  # Simple score
            })

    # Sort by score and take top 2
    candidates.sort(key=lambda c: c['score'], reverse=True)

    if len(candidates) >= 2:
        logging.info("Found %d candidates, returning top 2", len(candidates))
        return [
            (candidates[0]['x'], candidates[0]['y']),
            (candidates[1]['x'], candidates[1]['y'])
        ]
    if len(candidates) == 1:
        logging.warning("Only found 1 red region, returning it")
        return [(candidates[0]['x'], candidates[0]['y'])]

    logging.warning("No red regions found")
    return []
