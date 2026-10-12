"""Colored anchor tape: picks each tuft's root end and rejects unanchored strays.

Ported from the aditya-wip detector (tape_root / _one_per_anchor) into the
backend-agnostic design: the tape distance map is computed once per image and
handed to extract_tufts(), so it works the same for the classical and U-Net masks.
"""
import cv2
import numpy as np

TAPE_HSV_LOWER = (15, 80, 80)  # default: yellow tape
TAPE_HSV_UPPER = (40, 255, 255)


def tape_mask(image, hsv_lower=TAPE_HSV_LOWER, hsv_upper=TAPE_HSV_UPPER):
    """uint8 {0,255} mask of anchor-tape pixels, or None for grayscale images."""
    if image.ndim != 3:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tape = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    return cv2.morphologyEx(tape, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def tape_distance(image, hsv_lower=TAPE_HSV_LOWER, hsv_upper=TAPE_HSV_UPPER):
    """Distance (px) from every pixel to the nearest tape pixel, or None when
    the image has no color or no tape."""
    tape = tape_mask(image, hsv_lower, hsv_upper)
    if tape is None or not np.any(tape):
        return None
    return cv2.distanceTransform(cv2.bitwise_not(tape), cv2.DIST_L2, 5)


def one_per_anchor(tufts, max_dist):
    """Each tape patch anchors a single tuft, so when several candidates root at
    the same patch (e.g. a glare streak touching the tape), keep only the longest
    and renumber top-to-bottom, left-to-right."""
    kept = []
    for t in sorted(tufts, key=lambda t: t.length_px, reverse=True):
        if all(np.hypot(t.root[0] - k.root[0], t.root[1] - k.root[1]) > max_dist for k in kept):
            kept.append(t)
    kept.sort(key=lambda t: (t.root[1], t.root[0]))
    for i, t in enumerate(kept):
        t.id = i
    return kept
