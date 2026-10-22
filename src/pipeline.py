"""One frame in -> tufts out. Used once per frame by the video path (src/video_pipeline.py).

Mirrors what main.py does for a single still image: detection runs at the image's
native resolution (downscaling thins tufts away), the tuft-field filter is opt-in
(--clean), and --tape-root is honoured. Results are in original image pixels.
"""
import math
from dataclasses import dataclass
from typing import Optional

from detector import TuftDetector
from src.backends import build_backend
from src.filtering import clean_tufts


@dataclass
class FrameResult:
    tufts: list            # Tuft list in image pixels (ids are 0..n-1 for this frame)
    raw_count: int         # detections before the optional tuft-field filter
    backend: object


class FramePipeline:
    def __init__(self, args):
        self.args = args
        self.flow_prior = None
        if args.flow_prior_deg is not None:
            rad = math.radians(args.flow_prior_deg)
            self.flow_prior = (math.cos(rad), -math.sin(rad))  # image coords: y points down
        self._polarity = args.polarity            # "auto" is resolved once, then locked (video)
        self._backend = None                      # classical backend for the locked polarity
        self._unet = None

    # ------------------------------------------------------------ helpers
    def _classical(self, polarity):
        a = self.args
        return build_backend("classical", block_size=a.block_size, c=a.c,
                             clahe_clip=a.clahe_clip, clahe_tile=a.clahe_tile,
                             polarity=polarity)

    def _run(self, image, roi, backend):
        a = self.args
        detector = TuftDetector(
            min_area=a.min_area, max_area=a.max_area, min_elongation=a.min_elongation,
            min_length_px=a.min_length_px, max_length_px=a.max_length_px,
            roi=roi, backend=backend, vector_mode=a.vector_mode,
            flow_prior=self.flow_prior, tape_root=a.tape_root,
            tape_max_dist=a.tape_max_dist,
        )
        found = detector.detect(image)
        raw_count = len(found)
        if a.clean:
            found = clean_tufts(image, found)
        return backend, found, raw_count

    # ------------------------------------------------------------ public
    def process(self, image, roi=None, lock_polarity=False):
        """Detect tufts in one BGR image. `roi` is (x, y, w, h) in image pixels.

        lock_polarity=True (video): when --polarity auto, pick dark/light on the first call
        and keep it, so the mask doesn't flip between frames.
        """
        a = self.args
        if a.backend == "classical":
            if self._polarity == "auto":
                runs = [self._run(image, roi, self._classical(p)) for p in ("dark", "light")]
                backend, tufts, raw_count = max(runs, key=lambda r: len(r[1]))
                if lock_polarity:
                    self._polarity = backend.polarity
                    self._backend = backend
            else:
                if self._backend is None or self._backend.polarity != self._polarity:
                    self._backend = self._classical(self._polarity)
                backend, tufts, raw_count = self._run(image, roi, self._backend)
        else:
            if self._unet is None:
                self._unet = build_backend("unet", weights=a.weights, threshold=a.threshold)
            backend, tufts, raw_count = self._run(image, roi, self._unet)

        for i, t in enumerate(tufts):
            t.id = i
        return FrameResult(tufts, raw_count, backend)
