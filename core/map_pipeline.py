"""Per-frame tuft measurement against a fixed tuft map (no Qt).

Same output as core.pipeline.Pipeline (FrameResult of TuftResult), so the
exporter, overlays and TuftView work unchanged. Differences:
  - one track per map anchor, ID = anchor ID, for the whole video (no churn);
  - a detection only counts when one of its ends is at a known root, so
    clutter away from the tufts (sky, sea, rivets, the pitot boom) is ignored;
  - the end at the root is the root, so arrows never point the wrong way;
  - near each root the probability threshold is relaxed (faint far tufts),
    since a false positive there has nowhere to attach;
  - an optional second detector (e.g. the classical skeleton detector, which
    finds more tufts but also lots of clutter) fills roots the model missed;
    the map throws its clutter away;
  - anchors still unmatched get a local search in a window around the root
    (touching tufts merged into one blob), else they are reported as not
    detected this frame (kept, never dropped).
Per frame: align the reference frame to this one (core.tuft_map.Aligner),
one U-Net pass, geometry via src.mask_to_tufts.extract_tufts.
"""
import time
from collections import deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.mask_to_tufts import extract_tufts

from .results import FrameResult, TuftResult, encode_mask
from .tracker import (ATTACHED, SEPARATED, UNKNOWN, UNSTEADY, StallConfig, ang_diff_deg, circ_mean_deg,
                      circ_std_deg, flip_detection)
from .tuft_map import Aligner, apply

GEOM = dict(min_area=4, max_area=10 ** 7, min_elongation=1.5, vector_mode="tip", source="unet")


class _AnchorTrack:
    def __init__(self, anchor, maxlen):
        self.anchor = anchor
        self.hist = deque(maxlen=maxlen)      # (frame, angle_deg, length_px)
        self.last = None


class MapPipeline:
    def __init__(self, tmap, backend, ref_frame, fps=30.0, threshold=None, local_scale=0.6,
                 root_tol=0.35, stall=None, keep_mask=True, align=True, extra_backend=None):
        """backend: object with predict_proba(bgr) -> (C, H, W) and .threshold (e.g. src.unet.UNetBackend).
        extra_backend: optional mask backend (.predict(bgr) -> MaskResult, e.g. src.backends.ClassicalBackend)
        for roots the model missed; only blobs within reach of those roots are measured."""
        self.map, self.backend, self.fps = tmap, backend, fps or 30.0
        self.threshold = threshold or getattr(backend, "threshold", 0.5)
        self.local_threshold = self.threshold * local_scale
        self.root_tol = root_tol
        self.stall = stall or StallConfig()
        self.keep_mask = keep_mask
        self.aligner = Aligner(ref_frame) if align else None
        self.extra = extra_backend
        maxlen = max(2, int(round(self.stall.window_s * self.fps)))
        self.tracks = [_AnchorTrack(a, maxlen) for a in tmap.anchors]
        self.L = tmap.median_length()
        self.index = 0
        self.device = str(getattr(backend, "device", "cpu"))

    def reset(self):
        for t in self.tracks:
            t.hist.clear()
            t.last = None
        self.index = 0

    # ------------------------------------------------------------ matching
    def _tol(self, a, s):
        return max(8.0, self.root_tol * (a.length_px or self.L) * s)

    def _match(self, dets, roots, tols, lengths=None):
        """Hungarian on min(root, tip) distance to each anchor. -> {anchor index: oriented detection}.
        lengths: expected tuft length per anchor; detections outside 0.4-2x it cannot match."""
        out = {}
        if not dets or not len(roots):
            return out
        r = np.array([d.root for d in dets], float)
        t = np.array([d.tip for d in dets], float)
        d_root = np.linalg.norm(roots[:, None] - r[None], axis=2)
        d_tip = np.linalg.norm(roots[:, None] - t[None], axis=2)
        cost = np.minimum(d_root, d_tip)
        if lengths is not None:
            dl = np.array([d.length_px for d in dets], float)[None, :] / np.asarray(lengths, float)[:, None]
            cost = np.where((dl >= 0.4) & (dl <= 2.0), cost, 1e9)
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] <= tols[i]:
                out[i] = flip_detection(dets[j]) if d_tip[i, j] < d_root[i, j] else dets[j]
        return out

    def _local(self, mask, root, a, s):
        """Window search around one root; the path end nearest the root is the root."""
        L = (a.length_px or self.L) * s
        R = int(1.4 * L + 12)
        H, W = mask.shape
        x0, y0 = max(int(root[0]) - R, 0), max(int(root[1]) - R, 0)
        x1, y1 = min(int(root[0]) + R + 1, W), min(int(root[1]) + R + 1, H)
        win = mask[y0:y1, x0:x1]
        if win.size == 0 or not win.any():
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - root[0], yy - root[1]).astype(np.float32)
        tol = self._tol(a, s)
        cands = extract_tufts(win.astype(np.uint8) * 255, offset=(x0, y0), anchor_dist=dist, anchor_max_dist=tol,
                              min_length_px=max(4.0, 0.2 * L), max_length_px=3.0 * L + 20, **GEOM)
        if not cands:
            return None
        return min(cands, key=lambda c: np.hypot(c.root[0] - root[0], c.root[1] - root[1]) - 0.1 * c.length_px)

    # ------------------------------------------------------------ stall
    def _analyse(self, tr, bulk):
        s = self.stall
        if not tr.hist:
            return float("nan"), float("nan"), False, UNKNOWN
        angles = [h[1] for h in tr.hist]
        lengths = [h[2] for h in tr.hist]
        base = float(np.percentile(lengths, s.baseline_pct))
        last_len = tr.last.length_px if tr.last is not None else lengths[-1]
        ratio = last_len / base if base > 0 else float("nan")
        cstd = circ_std_deg(angles)
        last_ang = tr.last.angle_deg if tr.last is not None else angles[-1]
        rev = bulk is not None and ang_diff_deg(last_ang, bulk) > s.reverse_deg
        if len(tr.hist) < s.min_samples:
            label = UNKNOWN
        elif rev or cstd > s.separated_std_deg or ratio < s.separated_ratio:
            label = SEPARATED
        elif cstd > s.unsteady_std_deg or ratio < s.unsteady_ratio:
            label = UNSTEADY
        else:
            label = ATTACHED
        return ratio, cstd, bool(rev), label

    # ------------------------------------------------------------ the API
    def process_frame(self, frame, index=None, time_s=None):
        t0 = time.perf_counter()
        index = self.index if index is None else int(index)
        M = self.aligner.ref_to_frame(frame) if self.aligner is not None else np.array([[1, 0, 0], [0, 1, 0]], float)
        s = float(np.sqrt(abs(np.linalg.det(M[:, :2])))) or 1.0
        prob = self.backend.predict_proba(frame)[0]
        local = prob >= self.local_threshold
        roots = apply(M, [tr.anchor.root for tr in self.tracks]) if self.tracks else np.zeros((0, 2))
        tols = [self._tol(tr.anchor, s) for tr in self.tracks]
        dets = extract_tufts(local.astype(np.uint8) * 255, min_length_px=max(4.0, 0.2 * self.L * s),
                             max_length_px=4.0 * self.L * s + 40, **GEOM)
        matched = self._match(dets, roots, tols)
        n_extra = 0
        if self.extra is not None and len(matched) < len(self.tracks):
            rest = [i for i in range(len(self.tracks)) if i not in matched]
            more = self._extra_match(frame, self.extra, rest, roots, tols, s)
            n_extra = len(more)
            matched.update(more)
        self.last_sources = {"model": len(matched) - n_extra, "extra": n_extra}
        for i, tr in enumerate(self.tracks):
            d = matched.get(i) or self._local(local, roots[i], tr.anchor, s)
            tr.last = d
            if d is not None:
                tr.hist.append((index, float(d.angle_deg), float(d.length_px)))
        live = [tr.last.angle_deg for tr in self.tracks if tr.last is not None]
        bulk = circ_mean_deg(live) if len(live) >= 3 else None
        out = []
        for i, tr in enumerate(self.tracks):
            ratio, cstd, rev, label = self._analyse(tr, bulk)
            d = tr.last
            root = tuple(float(v) for v in (d.root if d is not None else roots[i]))
            out.append(TuftResult(
                track_id=tr.anchor.id, state="confirmed" if d is not None else "missing", root=root,
                tip=tuple(d.tip) if d is not None else root,
                vector=tuple(d.vector) if d is not None else (0.0, 0.0),
                angle_deg=float(d.angle_deg) if d is not None else float("nan"),
                length_px=float(d.length_px) if d is not None else 0.0, length_ratio=ratio,
                circ_std_deg=cstd, reversed=rev, stall=label,
                confidence=float(d.confidence) if d is not None else 0.0,
                path=None if d is None or d.path is None else np.asarray(d.path, np.float32),
                detected=d is not None))
        mask = (prob >= self.threshold) if self.keep_mask else None
        self.index = index + 1
        return FrameResult(index=index, time_s=time_s if time_s is not None else index / self.fps, tufts=out,
                           mask_png=encode_mask(mask), shape=frame.shape[:2], detections=len(dets),
                           process_ms=(time.perf_counter() - t0) * 1000)

    def _extra_match(self, frame, extra, idx, roots, tols, s):
        """Second detector, only within reach of the given roots. -> {anchor index: detection}."""
        m = extra.predict(frame).tuft
        m = (m > 127) if m.dtype == np.uint8 else (m >= 0.5)
        reach = np.zeros(m.shape, np.uint8)
        for i in idx:
            r = int(1.3 * (self.tracks[i].anchor.length_px or self.L) * s) + 6
            cv2.circle(reach, (int(roots[i][0]), int(roots[i][1])), r, 1, -1)
        m &= reach.astype(bool)
        if not m.any():
            return {}
        dets = extract_tufts(m.astype(np.uint8) * 255, min_length_px=max(4.0, 0.2 * self.L * s),
                             max_length_px=4.0 * self.L * s + 40, **{**GEOM, "source": "classical"})
        lens = [(self.tracks[i].anchor.length_px or self.L) * s for i in idx]
        got = self._match(dets, roots[idx], [tols[i] for i in idx], lens)
        return {idx[k]: d for k, d in got.items()}

    def roots_matched_by(self, frame, extra):
        """How many map roots a second detector attaches to in this frame (used to pick its settings)."""
        M = self.aligner.ref_to_frame(frame) if self.aligner is not None else np.array([[1, 0, 0], [0, 1, 0]], float)
        s = float(np.sqrt(abs(np.linalg.det(M[:, :2])))) or 1.0
        roots = apply(M, [tr.anchor.root for tr in self.tracks])
        idx = list(range(len(self.tracks)))
        return len(self._extra_match(frame, extra, idx, roots, [self._tol(tr.anchor, s) for tr in self.tracks], s))

    def history(self, track_id):
        for tr in self.tracks:
            if tr.anchor.id == track_id:
                return list(tr.hist)
        return []
