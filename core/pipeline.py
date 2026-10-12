"""The processing pipeline the app calls: process_frame(frame) -> FrameResult.

    detector (backend: unet or classical) -> geometry (root / tip / tip tangent)
    -> tracker (persistent IDs, stall labels)

Frames must be fed in order (the tracker carries state); call reset() to start
over. Nothing here imports Qt.
"""
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .results import FrameResult, TuftResult, encode_mask
from .tracker import StallConfig, Tracker, TrackerConfig


@dataclass
class PipelineConfig:
    backend: str = "unet"              # "unet" or "classical"
    checkpoint: str = ""               # U-Net weights (.pt); required for unet
    threshold: float = 0.0             # U-Net probability threshold; 0 = checkpoint's own value
    device: str = ""                   # "" = cuda if available else cpu
    polarity: str = "light"            # classical backend: dark / light tufts
    block_size: int = 51               # classical adaptive threshold
    c: float = 10.0
    min_area: float = 20
    max_area: float = 1e6
    min_elongation: float = 2.5
    min_length_px: float = 15
    max_length_px: float = 5000
    keep_mask: bool = True             # store the tuft mask in each FrameResult
    max_dist_px: float = 0.0           # tracker gate; 0 = automatic
    n_confirm: int = 3
    max_missing: int = 15
    stall: dict = field(default_factory=lambda: asdict(StallConfig()))

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


class Pipeline:
    def __init__(self, config=None, fps=30.0):
        self.config = config or PipelineConfig()
        self.fps = fps or 30.0
        self.detector = self._build_detector()
        self.tracker = Tracker(TrackerConfig(max_dist_px=self.config.max_dist_px or None,
                                             n_confirm=self.config.n_confirm,
                                             max_missing=self.config.max_missing,
                                             stall=StallConfig(**self.config.stall)), fps=self.fps)
        self.index = 0

    # ------------------------------------------------------------ setup
    def _build_detector(self):
        from detector import TuftDetector
        from src.backends import ClassicalBackend
        c = self.config
        if c.backend == "unet":
            if not c.checkpoint or not Path(c.checkpoint).exists():
                raise FileNotFoundError(f"U-Net checkpoint not found: {c.checkpoint!r}")
            from src.unet import UNetBackend
            backend = UNetBackend(c.checkpoint, device=c.device or None, threshold=c.threshold or None)
        elif c.backend == "classical":
            backend = ClassicalBackend(int(c.block_size) | 1, c.c, polarity=c.polarity)
        else:
            raise ValueError(f"unknown backend {c.backend!r}")
        self.backend = backend
        return TuftDetector(min_area=c.min_area, max_area=c.max_area, min_elongation=c.min_elongation,
                            min_length_px=c.min_length_px, max_length_px=c.max_length_px,
                            backend=backend, vector_mode="tip")

    @property
    def device(self):
        dev = str(getattr(self.backend, "device", "cpu"))
        if dev.startswith("cuda"):
            import torch
            return f"GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"

    def reset(self):
        self.tracker.reset()
        self.index = 0

    # ------------------------------------------------------------ the API
    def process_frame(self, frame, index=None, time_s=None):
        """frame: BGR uint8 HxWx3 -> FrameResult (tufts with track IDs and stall labels)."""
        t0 = time.perf_counter()
        index = self.index if index is None else int(index)
        tufts, mask_result, offset = self.detector.detect_full(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        tracked = self.tracker.update(gray, tufts, index)
        out = []
        for tt in tracked:
            t = tt.tuft
            out.append(TuftResult(
                track_id=tt.track_id, state=tt.state, root=tt.root, tip=tt.tip,
                vector=tuple(t.vector) if t is not None else (0.0, 0.0),
                angle_deg=tt.angle_deg, length_px=tt.length_px, length_ratio=tt.length_ratio,
                circ_std_deg=tt.circ_std_deg, reversed=tt.reversed, stall=tt.stall,
                confidence=float(t.confidence) if t is not None else 0.0,
                path=None if t is None or t.path is None else np.asarray(t.path, np.float32),
                detected=t is not None))
        mask = None
        if self.config.keep_mask:
            m = mask_result.tuft
            m = (m > 127) if m.dtype == np.uint8 else (m >= 0.5)
            mask = np.zeros(frame.shape[:2], bool)
            ox, oy = offset
            mask[oy:oy + m.shape[0], ox:ox + m.shape[1]] = m
        self.index = index + 1
        return FrameResult(index=index, time_s=time_s if time_s is not None else index / self.fps,
                           tufts=out, mask_png=encode_mask(mask), shape=frame.shape[:2],
                           detections=len(tufts), process_ms=(time.perf_counter() - t0) * 1000)

    def history(self, track_id):
        return self.tracker.history(track_id)
