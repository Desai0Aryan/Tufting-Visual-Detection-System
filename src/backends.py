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
        if self.polarity == "light":
            gray = 255 - gray  # light tufts on a dark surface: flip so tufts are dark again
        blurred = cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            self.block_size, self.c,
        )
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def predict(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return MaskResult(tuft=self.threshold(gray))


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
