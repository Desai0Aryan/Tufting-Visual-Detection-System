"""Classical tuft tracker: persistent IDs, angle/length history, stall labels.

Per frame:
  1. predict every live track's root with a PointPredictor (default: pyramidal
     Lucas-Kanade optical flow), so wing flex and camera shake are followed;
  2. match predictions to detections with the Hungarian algorithm
     (scipy.optimize.linear_sum_assignment), gated by a maximum distance. The
     cost is the distance from the predicted root to the NEARER end of each
     detection; when that is the detection's tip, the detection is flipped
     (root and tip swapped, tip tangent recomputed). A taped root does not
     move, so this keeps each tuft's orientation consistent from frame to
     frame even when thickness alone cannot tell root from tip;
  3. unmatched detections start TENTATIVE tracks, CONFIRMED after n_confirm hits;
  4. unmatched tracks become MISSING and are dropped after max_missing frames.

Stall analysis per confirmed track, over a rolling window:
  baseline length  = high percentile of the tuft's own length (handles perspective)
  length ratio     = current length / baseline (curling, shortening)
  circular std     = sqrt(-2 ln R) of the angle (sin/cos averaging, never raw degrees)
  reversed         = pointing more than reverse_deg away from the bulk flow direction
  label            = attached / unsteady / separated (thresholds in StallConfig)

To swap in a learned point tracker later (e.g. CoTracker), implement
PointPredictor.predict() and pass it to Tracker(predictor=...).
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Protocol

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

TENTATIVE, CONFIRMED, MISSING = "tentative", "confirmed", "missing"
ATTACHED, UNSTEADY, SEPARATED, UNKNOWN = "attached", "unsteady", "separated", "unknown"


# ---------------------------------------------------------------- point prediction
class PointPredictor(Protocol):
    def predict(self, prev_gray: np.ndarray, gray: np.ndarray, points: np.ndarray):
        """points (N, 2) in prev_gray -> (new_points (N, 2), ok (N,) bool)."""


class LKPredictor:
    def __init__(self, win=21, levels=3):
        self.params = dict(winSize=(win, win), maxLevel=levels,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def predict(self, prev_gray, gray, points):
        if len(points) == 0 or prev_gray is None:
            return points.copy(), np.zeros(len(points), bool)
        p0 = points.reshape(-1, 1, 2).astype(np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.params)
        p1 = p1.reshape(-1, 2)
        ok = st.reshape(-1).astype(bool)
        return np.where(ok[:, None], p1, points), ok


class StaticPredictor:
    """No motion model (for tests and as a fallback)."""

    def predict(self, prev_gray, gray, points):
        return points.copy(), np.ones(len(points), bool)


# ---------------------------------------------------------------- circular statistics
def circ_mean_deg(angles_deg, weights=None):
    a = np.radians(np.asarray(angles_deg, float))
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    return float(np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())))


def circ_std_deg(angles_deg):
    a = np.radians(np.asarray(angles_deg, float))
    if len(a) == 0:
        return float("nan")
    r = math.hypot(np.sin(a).mean(), np.cos(a).mean())
    return float(np.degrees(math.sqrt(-2.0 * math.log(max(r, 1e-12)))))


def ang_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def flip_detection(t):
    """The same tuft with root and tip swapped and the tip tangent recomputed
    at the new free end."""
    from dataclasses import replace
    path = None if getattr(t, "path", None) is None else np.asarray(t.path, float)[::-1].copy()
    if path is not None and len(path) >= 3:
        k = max(2, len(path) // 8)
        v = path[-1] - path[-1 - k]
    else:
        v = -np.asarray(t.vector, float)
    n = float(np.linalg.norm(v)) or 1.0
    v = v / n
    return replace(t, root=tuple(t.tip), tip=tuple(t.root), vector=(float(v[0]), float(v[1])),
                   angle_deg=float(np.degrees(np.arctan2(-v[1], v[0]))), path=path)


# ---------------------------------------------------------------- config and records
@dataclass
class StallConfig:
    window_s: float = 2.0           # rolling window for baseline / angle statistics
    baseline_pct: float = 90.0      # percentile of the tuft's own length
    min_samples: int = 8            # below this the state is "unknown"
    unsteady_std_deg: float = 15.0
    separated_std_deg: float = 40.0
    unsteady_ratio: float = 0.8
    separated_ratio: float = 0.5
    reverse_deg: float = 120.0


@dataclass
class TrackerConfig:
    max_dist_px: Optional[float] = None  # gate; None = 0.6 x median tuft length (>= 12 px)
    n_confirm: int = 3
    max_missing: int = 15
    stall: StallConfig = field(default_factory=StallConfig)


@dataclass
class TrackedTuft:
    track_id: int
    state: str              # tentative / confirmed / missing
    root: tuple
    tip: tuple
    angle_deg: float
    length_px: float
    length_ratio: float
    circ_std_deg: float
    reversed: bool
    stall: str              # attached / unsteady / separated / unknown
    tuft: object = None     # the detection this frame (None when missing)


class _Track:
    def __init__(self, tid, tuft, frame, maxlen):
        self.id, self.state = tid, TENTATIVE
        self.root = np.asarray(tuft.root, float)
        self.hits, self.misses = 1, 0
        self.hist = deque(maxlen=maxlen)  # (frame, angle_deg, length_px)
        self.last = tuft
        self.add(tuft, frame)

    def add(self, tuft, frame):
        self.hist.append((frame, float(tuft.angle_deg), float(tuft.length_px)))
        self.last = tuft


# ---------------------------------------------------------------- tracker
class Tracker:
    def __init__(self, config=None, predictor=None, fps=30.0, flow_prior_deg=None):
        self.cfg = config or TrackerConfig()
        self.predictor = predictor or LKPredictor()
        self.fps = fps or 30.0
        self.flow_prior_deg = flow_prior_deg
        self.tracks, self.next_id, self.prev_gray = [], 1, None
        self.window = max(2, int(round(self.cfg.stall.window_s * self.fps)))

    def reset(self):
        self.tracks, self.next_id, self.prev_gray = [], 1, None

    def _gate(self, tufts):
        if self.cfg.max_dist_px:
            return self.cfg.max_dist_px
        lengths = [t.length_px for t in tufts] + [tr.last.length_px for tr in self.tracks]
        return max(12.0, 0.6 * float(np.median(lengths))) if lengths else 30.0

    def update(self, gray, tufts, frame_index):
        tufts = list(tufts)
        """gray: uint8 HxW frame; tufts: list of detections (with .root, .tip,
        .angle_deg, .length_px). Returns list[TrackedTuft] for every live track."""
        # 1. predict
        if self.tracks:
            pts = np.array([t.root for t in self.tracks], np.float32)
            pred, _ = self.predictor.predict(self.prev_gray, gray, pts) if self.prev_gray is not None \
                else (pts, None)
        else:
            pred = np.zeros((0, 2), np.float32)
        # 2. match
        gate = self._gate(tufts)
        det = np.array([t.root for t in tufts], float).reshape(-1, 2)
        tips = np.array([t.tip for t in tufts], float).reshape(-1, 2)
        matched_t, matched_d = set(), set()
        if len(self.tracks) and len(det):
            d_root = np.linalg.norm(pred[:, None, :] - det[None, :, :], axis=2)
            d_tip = np.linalg.norm(pred[:, None, :] - tips[None, :, :], axis=2)
            cost = np.minimum(d_root, d_tip)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= gate:
                    tr = self.tracks[r]
                    t = tufts[c]
                    if d_tip[r, c] < d_root[r, c]:
                        t = flip_detection(t)  # keep the track's root end
                        tufts[c] = t
                    tr.root = np.asarray(t.root, float)
                    tr.hits += 1
                    tr.misses = 0
                    tr.add(t, frame_index)
                    if tr.state != CONFIRMED and tr.hits >= self.cfg.n_confirm:
                        tr.state = CONFIRMED
                    elif tr.state == MISSING:
                        tr.state = CONFIRMED
                    matched_t.add(r)
                    matched_d.add(c)
        # 4. unmatched tracks
        keep = []
        for i, tr in enumerate(self.tracks):
            if i in matched_t:
                keep.append(tr)
                continue
            if tr.state == TENTATIVE:
                continue  # a tentative track that misses is dropped at once
            tr.misses += 1
            tr.root = pred[i].astype(float)
            tr.state = MISSING
            if tr.misses <= self.cfg.max_missing:
                keep.append(tr)
        # 3. unmatched detections
        for j, t in enumerate(tufts):
            if j not in matched_d:
                keep.append(_Track(self.next_id, t, frame_index, self.window))
                self.next_id += 1
        self.tracks = keep
        self.prev_gray = gray
        return self._report(frame_index)

    # ------------------------------------------------------------ stall analysis
    def _bulk_flow(self):
        if self.flow_prior_deg is not None:
            return self.flow_prior_deg
        conf = [tr for tr in self.tracks if tr.state == CONFIRMED and tr.misses == 0]
        if len(conf) < 3:
            return None
        return circ_mean_deg([tr.last.angle_deg for tr in conf])

    def _analyse(self, tr, bulk):
        s = self.cfg.stall
        angles = [h[1] for h in tr.hist]
        lengths = [h[2] for h in tr.hist]
        base = float(np.percentile(lengths, s.baseline_pct)) if lengths else float("nan")
        ratio = tr.last.length_px / base if base and base > 0 else float("nan")
        cstd = circ_std_deg(angles)
        rev = bulk is not None and ang_diff_deg(tr.last.angle_deg, bulk) > s.reverse_deg
        if len(tr.hist) < s.min_samples or tr.state != CONFIRMED:
            label = UNKNOWN
        elif rev or cstd > s.separated_std_deg or ratio < s.separated_ratio:
            label = SEPARATED
        elif cstd > s.unsteady_std_deg or ratio < s.unsteady_ratio:
            label = UNSTEADY
        else:
            label = ATTACHED
        return ratio, cstd, bool(rev), label

    def _report(self, frame_index):
        bulk = self._bulk_flow()
        out = []
        for tr in self.tracks:
            ratio, cstd, rev, label = self._analyse(tr, bulk)
            t = tr.last
            out.append(TrackedTuft(tr.id, tr.state, tuple(float(v) for v in tr.root), tuple(t.tip),
                                   float(t.angle_deg), float(t.length_px), ratio, cstd, rev, label,
                                   t if tr.misses == 0 else None))
        return out

    def history(self, track_id):
        """[(frame, angle_deg, length_px)] for a live track (rolling window)."""
        for tr in self.tracks:
            if tr.id == track_id:
                return list(tr.hist)
        return []
