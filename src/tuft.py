"""The Tuft record shared by every detection backend, the exporter and the overlay."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Tuft:
    id: int
    root: tuple       # (x, y) anchor/tape end, in image pixel coords
    tip: tuple        # (x, y) free/tail end, in image pixel coords
    vector: tuple     # 2D unit flow vector in image coords (y points down)
    angle_deg: float  # standard math angle (0 = +x, CCW positive, y flipped)
    length_px: float  # geodesic (along-string) length, not straight-line chord
    contour: np.ndarray
    vector_z: float = 0.0     # out-of-image component; 0 until a 3D model exists
    confidence: float = 1.0   # 0..1, how much to trust root/tip orientation + skeleton
    curl: float = 1.0         # chord / geodesic length: 1 = straight, ->0 = tightly curled
    path: Optional[np.ndarray] = None  # (N, 2) smoothed centerline, root -> tip
    source: str = "classical"  # which backend produced it ("classical", "unet", ...)

    @property
    def vector3(self) -> tuple:
        """(x, y, z) unit vector. z is 0 for the planar (image-plane) estimate."""
        return (self.vector[0], self.vector[1], self.vector_z)
