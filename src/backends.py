from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MaskResult:
    tuft: np.ndarray                  # HxW, uint8 {0,255} or float probability
    root: Optional[np.ndarray] = None  # same shape; marks the taped root end, if known


class ClassicalBackend:
    """CLAHE + adaptive threshold (the original detector's mask stage)."""

    name = "classical"

    def __init__(self, block_size=25, c=5, clahe_clip=3.0, clahe_tile=8, polarity="dark"):
        if polarity not in ("dark", "light"):
            raise ValueError("polarity must be 'dark' (dark tufts on a light surface) or 'light'")
        self.polarity = polarity
        if not isinstance(block_size, int) or block_size <= 1 or block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer greater than 1")
        self.block_size = block_size
        self.c = c
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile

    def threshold(self, gray):
        # CLAHE equalizes contrast per tile so shadowed regions of a curved,
        # unevenly lit surface threshold as reliably as bright ones.
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile)
        )
        blurred = cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)
        # `c` is the contrast a tuft pixel needs against its neighborhood mean.
        # OpenCV keeps pixels where (pixel > mean - C) for THRESH_BINARY, so a
        # light-tuft threshold needs C negated to require "brighter by c".
        # (CLAHE is not symmetric under inversion, so this is not the same as
        # thresholding 255 - gray.)
        if self.polarity == "dark":
            mode, offset = cv2.THRESH_BINARY_INV, self.c
        else:
            mode, offset = cv2.THRESH_BINARY, -self.c
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, self.block_size, offset,
        )
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def predict(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return MaskResult(tuft=self.threshold(gray))


class ColorBackend:
    """Saturated-colour mask: tufts whose colour, not brightness, sets them apart
    (e.g. orange yarn on olive paint over busy terrain, where any brightness
    threshold also picks up fields and water). Hue in OpenCV units (0-180);
    a range like [170, 180] covers red wrapping past 0."""

    name = "color"

    def __init__(self, hue_ranges=((0, 15), (170, 180)), s_min=80, v_min=110, close_px=3):
        self.hue_ranges = [tuple(r) for r in hue_ranges]
        if not self.hue_ranges or any(not (0 <= lo <= hi <= 180) for lo, hi in self.hue_ranges):
            raise ValueError("hue_ranges must be [lo, hi] pairs within 0..180")
        self.s_min, self.v_min, self.close_px = s_min, v_min, close_px

    def predict(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        hue = np.zeros(h.shape, bool)
        for lo, hi in self.hue_ranges:
            hue |= (h >= lo) & (h <= hi)
        m = (hue & (s >= self.s_min) & (v >= self.v_min)).astype(np.uint8) * 255
        if self.close_px > 1:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((self.close_px, self.close_px), np.uint8))
        return MaskResult(tuft=m)


def build_backend(name="classical", weights=None, **kwargs):
    """Factory used by the CLI. The U-Net import is lazy so torch stays optional."""
    if name == "classical":
        return ClassicalBackend(**kwargs)
    if name == "unet":
        if not weights:
            raise ValueError("the unet backend needs --weights path/to/model.pt")
        from .unet import UNetBackend  # imports torch
        return UNetBackend(weights, **kwargs)
    raise ValueError(f"unknown backend {name!r} (expected 'classical' or 'unet')")
