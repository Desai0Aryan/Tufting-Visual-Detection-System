"""Frame sources. The pipeline and the app only use the FrameSource interface,
so a live camera (CameraSource) can be added later without changing them."""
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


class FrameSource(ABC):
    """Sequential source of BGR frames with optional random access."""

    name: str = ""
    fps: float = 30.0
    frame_count: int = 0      # 0 when unknown / unbounded (live camera)
    size: tuple = (0, 0)      # (width, height)
    seekable: bool = False

    @abstractmethod
    def read(self):
        """Next frame as (index, BGR ndarray), or None at the end."""

    def read_at(self, index):
        """Frame at `index` (seekable sources only)."""
        raise NotImplementedError(f"{type(self).__name__} is not seekable")

    def close(self):
        pass

    def __iter__(self):
        while True:
            item = self.read()
            if item is None:
                return
            yield item

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class VideoFileSource(FrameSource):
    seekable = True

    def __init__(self, path):
        self.path = Path(path)
        # np.fromfile-free path handling: OpenCV's VideoCapture accepts the str path
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"cannot open video {self.path}")
        self.name = self.path.name
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._next = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        i = self._next
        self._next += 1
        return i, frame

    def read_at(self, index):
        index = int(index)
        if index != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next = index
        item = self.read()
        return None if item is None else item[1]

    def close(self):
        self.cap.release()


class CameraSource(FrameSource):
    """Placeholder for live camera input (planned). Same interface as a file:
    read() returns (index, frame) until stopped; not seekable."""

    def __init__(self, device=0):
        raise NotImplementedError("live camera input is planned; use VideoFileSource for now")

    def read(self):  # pragma: no cover
        raise NotImplementedError


def open_source(path_or_device):
    if isinstance(path_or_device, int):
        return CameraSource(path_or_device)
    p = Path(path_or_device)
    if p.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"unsupported video type {p.suffix} (expected one of {', '.join(VIDEO_EXTS)})")
    return VideoFileSource(p)
