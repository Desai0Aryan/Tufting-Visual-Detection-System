"""Plain data returned by the pipeline. No Qt, no drawing."""
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class TuftResult:
    """One tracked tuft in one frame (image pixel coordinates, y down)."""
    track_id: int
    state: str                 # confirmed / tentative / missing
    root: tuple                # (x, y)
    tip: tuple
    vector: tuple              # unit flow vector (tip tangent), image coords
    angle_deg: float           # math convention: 0 = +x, counter-clockwise, y flipped
    length_px: float
    length_ratio: float        # current length / rolling baseline
    circ_std_deg: float        # angular unsteadiness over the rolling window
    reversed: bool
    stall: str                 # attached / unsteady / separated / unknown
    confidence: float = 1.0
    path: Optional[np.ndarray] = None   # (N, 2) centre-line, root -> tip
    detected: bool = True      # False when the track is coasting (missing this frame)


@dataclass
class FrameResult:
    index: int
    time_s: float
    tufts: List[TuftResult] = field(default_factory=list)
    mask_png: Optional[bytes] = None    # tuft mask, PNG-encoded (compact for caching)
    shape: tuple = (0, 0)
    detections: int = 0
    process_ms: float = 0.0

    def mask(self):
        """uint8 0/255 HxW mask, or None."""
        if self.mask_png is None:
            return None
        return cv2.imdecode(np.frombuffer(self.mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)

    def visible(self, include_tentative=False):
        """Tufts worth drawing: confirmed (and optionally tentative), detected this frame."""
        keep = ("confirmed", "tentative") if include_tentative else ("confirmed",)
        return [t for t in self.tufts if t.state in keep and t.detected]

    def by_id(self, track_id):
        return next((t for t in self.tufts if t.track_id == track_id), None)


def encode_mask(mask):
    if mask is None:
        return None
    ok, buf = cv2.imencode(".png", (np.asarray(mask) > 0).astype(np.uint8) * 255)
    return buf.tobytes() if ok else None
