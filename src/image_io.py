"""
image_io.py
"""
from pathlib import Path
import cv2
import numpy as np


class ImageLoadError(Exception):
    """Raised when an image cannot be read or is invalid."""


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk as a BGR numpy array (OpenCV convention).
    """
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"File not found: {path}")

    image = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(
            f"Could not decode image (unsupported format or corrupt file): {path}"
        )
    return image


def get_image_info(image: np.ndarray) -> dict:
    """Return basic metadata about an image array."""
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    return {"width": width, "height": height, "channels": channels}


def resize_max_dim(image: np.ndarray, max_dim: int = 1400) -> np.ndarray:
    """
    Resize an image so its largest dimension is at most max_dim,
    preserving aspect ratio. Keeps processing/display consistent
    regardless of the input resolution (SRS requirement: accept
    various standard resolutions).
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image.copy()

    scale = max_dim / float(longest)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_image(image: np.ndarray, path: str) -> None:
    """Save an image array to disk, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), image)
    if not ok:
        raise ImageLoadError(f"Failed to write image to: {path}")


def annotate_info_banner(image: np.ndarray, text: str) -> np.ndarray:
    """
    Draw a small text banner in the top-left corner of a copy of the image.
    """
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    return out
