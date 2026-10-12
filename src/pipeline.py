import math
from dataclasses import dataclass
from typing import Optional

from detector import TuftDetector
from src.backends import build_backend
from src.filtering import clean_tufts, normalize_scale, rescale_tufts, scale_roi


def clamp_roi(roi, shape):
    """Rounding when an ROI is rescaled can overshoot the image by a pixel."""
    if roi is None:
        return None
    x, y, w, h = roi
    H, W = shape[:2]
    x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
    return x, y, max(1, min(w, W - x)), max(1, min(h, H - y))


@dataclass
class FrameResult:
    tufts: list            # Tuft list in ORIGINAL image pixels (ids are 0..n-1 for this frame)
    raw_count: int         # detections before the tuft-field filter
    work: object           # the (possibly downscaled) image detection ran on
    work_roi: Optional[tuple]
    scale: float           # work size / original size
    backend: object
    sam_stats: Optional[dict] = None


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
        self._sam = None

    # ------------------------------------------------------------ helpers
    def _classical(self, polarity):
        a = self.args
        return build_backend("classical", block_size=a.block_size, c=a.c,
                             clahe_clip=a.clahe_clip, clahe_tile=a.clahe_tile,
                             polarity=polarity)

    def _run(self, work, work_roi, backend):
        a = self.args
        detector = TuftDetector(
            min_area=a.min_area, max_area=a.max_area, min_elongation=a.min_elongation,
            min_length_px=a.min_length_px, max_length_px=a.max_length_px,
            roi=work_roi, backend=backend, vector_mode=a.vector_mode,
            flow_prior=self.flow_prior,
        )
        found = detector.detect(work)
        raw_count = len(found)
        if not a.no_clean:
            found = clean_tufts(work, found)
        return backend, found, raw_count

    # ------------------------------------------------------------ public
    def process(self, image, roi=None, lock_polarity=False):
        """Detect tufts in one BGR image. `roi` is (x, y, w, h) in original pixels.

        lock_polarity=True (video): when --polarity auto, pick dark/light on the first call
        and keep it, so the mask doesn't flip between frames.
        """
        a = self.args
        work, scale = normalize_scale(image, a.norm_size)
        work_roi = clamp_roi(scale_roi(roi, scale), work.shape)

        if a.backend == "classical":
            if self._polarity == "auto":
                runs = [self._run(work, work_roi, self._classical(p)) for p in ("dark", "light")]
                backend, tufts, raw_count = max(runs, key=lambda r: len(r[1]))
                if lock_polarity:
                    self._polarity = backend.polarity
                    self._backend = backend
            else:
                if self._backend is None or self._backend.polarity != self._polarity:
                    self._backend = self._classical(self._polarity)
                backend, tufts, raw_count = self._run(work, work_roi, self._backend)
        else:
            if self._unet is None:
                self._unet = build_backend("unet", weights=a.weights, threshold=a.threshold)
            backend, tufts, raw_count = self._run(work, work_roi, self._unet)

        sam_stats = None
        if a.verify_sam2 and tufts:
            from src.sam2_labeler import load_sam2_predictor, verify_tufts_with_sam
            if self._sam is None:
                self._sam = load_sam2_predictor(a.sam2_model, config=a.sam2_config,
                                                checkpoint=a.sam2_checkpoint)
            tufts, sam_stats = verify_tufts_with_sam(work, tufts, self._sam)

        tufts = rescale_tufts(tufts, 1.0 / scale)
        for i, t in enumerate(tufts):
            t.id = i
        return FrameResult(tufts, raw_count, work, work_roi, scale, backend, sam_stats)
