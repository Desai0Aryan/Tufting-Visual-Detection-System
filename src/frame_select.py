"""Pick diverse frames from videos for labelling.

Scan: every `stride`-th frame becomes a candidate unless it is a near-duplicate
of the last kept candidate. Each candidate gets a feature vector:
  - a 32x18 grey thumbnail (scene content / tuft layout),
  - brightness, contrast and a coarse hue histogram (lighting),
  - motion energy against a frame `motion_gap` frames later (fluttering and
    stalled tufts move a lot; the camera is fixed and stabilised).
Select: farthest-point sampling in standardised feature space, per clip, with
a minimum time gap, so a batch spreads across lighting, flow states and time.
"""
from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Candidate:
    clip_id: str
    frame_idx: int
    time_s: float
    brightness: float
    contrast: float
    motion: float
    thumb: np.ndarray
    hue: np.ndarray


def _thumb(gray):
    t = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (t - t.mean()) / (t.std() + 1e-6)


def _hue_hist(bgr):
    hsv = cv2.cvtColor(cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h = np.histogram(hsv[..., 0], bins=8, range=(0, 180), weights=hsv[..., 1].astype(np.float32))[0]
    return h / (h.sum() + 1e-6)


def frame_features(bgr, later_bgr=None, roi=None):
    if roi is not None:
        x, y, w, h = roi
        bgr = bgr[y:y + h, x:x + w]
        later_bgr = later_bgr[y:y + h, x:x + w] if later_bgr is not None else None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    motion = 0.0
    if later_bgr is not None:
        g2 = cv2.cvtColor(later_bgr, cv2.COLOR_BGR2GRAY)
        small = lambda g: cv2.resize(g, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
        motion = float(np.abs(small(gray) - small(g2)).mean())
    return float(gray.mean()), float(gray.std()), motion, _thumb(gray).ravel(), _hue_hist(bgr)


GROUP_WEIGHTS = {"thumb": 1.0, "brightness": 1.0, "contrast": 0.5, "motion": 1.0, "hue": 0.5}


def feature_matrix(cands, weights=GROUP_WEIGHTS):
    """Scale each feature GROUP as a block so a 576-pixel thumbnail does not
    outvote brightness or motion: every group has unit RMS spread x weight."""
    def block(X):
        X = np.asarray(X, np.float32).reshape(len(cands), -1)
        X = X - X.mean(0)
        rms = np.sqrt((X ** 2).sum(1).mean()) + 1e-6
        return X / rms
    parts = [block([c.thumb for c in cands]) * weights["thumb"],
             block([c.brightness for c in cands]) * weights["brightness"],
             block([c.contrast for c in cands]) * weights["contrast"],
             block([np.log1p(c.motion) for c in cands]) * weights["motion"],
             block([c.hue for c in cands]) * weights["hue"]]
    return np.hstack(parts)


def scan_video(path, clip_id, stride=15, motion_gap=2, dup_thresh=0.02, roi=None, max_frames=None):
    """Returns candidates (near-duplicates already skipped) and the video fps."""
    p = str(path)
    if any(p.lower().endswith(e) for e in IMAGE_EXTS):  # a still counts as a 1-frame clip
        img = cv2.imread(p)
        b, c, m, th, hue = frame_features(img, None, roi)
        return [Candidate(clip_id, 0, 0.0, b, c, m, th, hue)], 0.0
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n = min(n, max_frames)
    out, last_thumb = [], None
    for idx in range(0, max(n - motion_gap, 1), stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            break
        later = None
        for _ in range(motion_gap):
            ok2, later = cap.read()
            if not ok2:
                later = None
                break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        th = _thumb(gray if roi is None else gray[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]])
        if last_thumb is not None and float(np.mean((th - last_thumb) ** 2)) < dup_thresh:
            continue  # near-duplicate of the last kept candidate
        last_thumb = th
        b, c, m, thumb, hue = frame_features(f, later, roi)
        out.append(Candidate(clip_id, idx, idx / fps, b, c, m, thumb, hue))
    cap.release()
    return out, fps


def farthest_point_sample(features, k, min_gap_s=None, times=None, start=None):
    """Greedy k-centre selection on already-scaled features (see feature_matrix).
    Returns indices. Optional minimum time gap between picks."""
    Z = np.asarray(features, np.float32)
    n = len(Z)
    if n == 0 or k <= 0:
        return []
    first = start if start is not None else int(np.argmin(np.linalg.norm(Z - np.median(Z, 0), axis=1)))
    chosen = [first]
    dist = np.linalg.norm(Z - Z[first], axis=1)
    blocked = np.zeros(n, bool)

    def block(i):
        if min_gap_s and times is not None:
            blocked[np.abs(np.asarray(times) - times[i]) < min_gap_s] = True
        blocked[i] = True

    block(first)
    while len(chosen) < min(k, n):
        cand = np.where(blocked, -1.0, dist)
        i = int(np.argmax(cand))
        if cand[i] < 0:
            break  # time-gap constraint exhausted the clip
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(Z - Z[i], axis=1))
        block(i)
    return chosen


def per_clip_quota(available, total):
    """Split `total` across clips as evenly as possible, capped by availability."""
    quota = {c: 0 for c in available}
    left = total
    open_ = [c for c in available if available[c] > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for c in list(open_):
            take = min(share, available[c] - quota[c], left)
            quota[c] += take
            left -= take
            if quota[c] >= available[c]:
                open_.remove(c)
            if left == 0:
                break
    return quota
