import numpy as np

from src.anchor import TAPE_HSV_LOWER, TAPE_HSV_UPPER, one_per_anchor, tape_distance
from src.backends import ClassicalBackend
from src.mask_to_tufts import VECTOR_MODES, extract_tufts
from src.overlay import draw_tufts  # re-exported: `from detector import draw_tufts`
from src.tuft import Tuft  # re-exported: `from detector import Tuft`

__all__ = ["TuftDetector", "Tuft", "draw_tufts"]


class TuftDetector:
    def __init__(
        self,
        min_area=15,
        max_area=600,
        min_elongation=2.0,
        min_length_px=12,
        max_length_px=110,
        block_size=25,
        c=5,
        clahe_clip=3.0,
        clahe_tile=8,
        roi=None,  # (x, y, w, h) restricting the search area, or None for full image
        backend=None,  # any object with .predict(image) -> MaskResult; default classical
        vector_mode="chord",
        flow_prior=None,  # e.g. (1, 0): expected bulk flow direction in image coords
        polarity="dark",  # default classical backend: "dark" or "light" tufts
        tape_root=False,  # use colored anchor tape to pick the root end and reject strays
        tape_max_dist=60,  # max px from a tuft end to tape for it to count as anchored
        tape_hsv_lower=TAPE_HSV_LOWER,  # HSV range of the anchor tape (default: yellow)
        tape_hsv_upper=TAPE_HSV_UPPER,
    ):
        if min_area < 0 or max_area <= min_area:
            raise ValueError("max_area must be greater than min_area >= 0")
        if min_length_px < 0 or max_length_px <= min_length_px:
            raise ValueError("max_length_px must be greater than min_length_px >= 0")
        if min_elongation <= 0:
            raise ValueError("min_elongation must be greater than 0")
        if vector_mode not in VECTOR_MODES:
            raise ValueError(f"vector_mode must be one of {VECTOR_MODES}")

        self.min_area = min_area
        self.max_area = max_area
        self.min_elongation = min_elongation
        self.min_length_px = min_length_px
        self.max_length_px = max_length_px
        self.roi = roi
        self.vector_mode = vector_mode
        self.flow_prior = flow_prior
        self.tape_root = tape_root
        self.tape_max_dist = tape_max_dist
        self.tape_hsv_lower = tape_hsv_lower
        self.tape_hsv_upper = tape_hsv_upper
        # Validates block_size and polarity even when a custom backend is supplied.
        default = ClassicalBackend(block_size, c, clahe_clip, clahe_tile, polarity=polarity)
        self.backend = backend or default

    def detect(self, image):
        """Detect tufts in a BGR (or grayscale) image and return a list of Tuft."""
        if image is None or not hasattr(image, "shape") or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")

        if self.roi is not None:
            x, y, w, h = self.roi
            height, width = image.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise ValueError(
                    f"ROI {self.roi} is outside the image bounds ({width}x{height})"
                )
            work = image[y:y + h, x:x + w]
            offset = (x, y)
        else:
            work = image
            offset = (0, 0)

        result = self.backend.predict(work)
        anchor = (tape_distance(work, self.tape_hsv_lower, self.tape_hsv_upper)
                  if self.tape_root else None)
        tufts = extract_tufts(
            result.tuft,
            result.root,
            offset=offset,
            min_area=self.min_area,
            max_area=self.max_area,
            min_elongation=self.min_elongation,
            min_length_px=self.min_length_px,
            max_length_px=self.max_length_px,
            vector_mode=self.vector_mode,
            flow_prior=self.flow_prior,
            source=getattr(self.backend, "name", "custom"),
            anchor_dist=anchor,
            anchor_max_dist=self.tape_max_dist,
        )
        if anchor is not None:
            tufts = one_per_anchor(tufts, self.tape_max_dist)
        return tufts
