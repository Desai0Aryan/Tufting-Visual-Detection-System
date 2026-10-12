"""SAM 2 refinement of tuft candidates at native resolution (training labels only).

Each candidate gets a window centred on it, sized to the candidate (>= 512 px),
so SAM always sees the whole tuft at high relative resolution, whatever its
length in pixels. (The older fixed-grid tiling in sam2_labeler.py only fits
tufts shorter than the tile overlap; longer ones were silently dropped.)
Nearby candidates whose boxes fit in an already-embedded window share it.

Prompts: points along the candidate's centre-line (positive), two points on
either side of the middle at a distance scaled to the measured tuft width
(negative), and the bounding box. SAM's multimask guess that best matches the
candidate is kept if it passes accept_mask(); otherwise the candidate is
rejected (strict) and left for the reviewer.

SAM 2 is never used at detection time.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .sam2_labeler import _best_sam_mask, accept_mask

DEFAULT_CHECKPOINT = "models/sam2/sam2.1_hiera_large.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def load_image_predictor(checkpoint=DEFAULT_CHECKPOINT, config=DEFAULT_CONFIG, device="cuda"):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))


def candidate_mask(tuft, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(m, [tuft.contour.astype(np.int32)], -1, 255, cv2.FILLED)
    return m


def half_width(tuft):
    """Max half-width (px) of the candidate, from its own distance transform."""
    x, y, w, h = cv2.boundingRect(tuft.contour.astype(np.int32))
    local = np.zeros((h + 4, w + 4), np.uint8)
    cv2.drawContours(local, [tuft.contour.astype(np.int32) - [x - 2, y - 2]], -1, 255, cv2.FILLED)
    return float(cv2.distanceTransform(local, cv2.DIST_L2, 5).max())


def prompt_for(tuft, offset, n_pos=5, box_pad=None):
    """(points, labels, box) in window coordinates."""
    hw = max(1.0, half_width(tuft))
    path = np.asarray(tuft.path if tuft.path is not None else [tuft.root, tuft.tip], float)
    idx = np.linspace(0, len(path) - 1, n_pos + 2)[1:-1].round().astype(int)
    pos = path[idx]
    mid = len(path) // 2
    tang = path[min(mid + 1, len(path) - 1)] - path[max(mid - 1, 0)]
    nrm = np.array([-tang[1], tang[0]]) / (np.linalg.norm(tang) + 1e-9)
    nd = max(6.0, 3.0 * hw)  # clear of the yarn at any resolution
    neg = np.stack([path[mid] + nrm * nd, path[mid] - nrm * nd])
    off = np.asarray(offset, float)
    pts = np.vstack([pos, neg]) - off
    labels = np.array([1] * len(pos) + [0] * 2, np.int32)
    pad = box_pad if box_pad is not None else max(4.0, 2.0 * hw)
    x, y, w, h = cv2.boundingRect(tuft.contour.astype(np.int32))
    box = np.array([x - pad, y - pad, x + w + pad, y + h + pad], float) - np.concatenate([off, off])
    return pts.astype(np.float32), labels, box.astype(np.float32)


def window_for(tuft, shape, min_side=512, margin=1.5):
    """Square window (x0, y0, x1, y1) centred on the candidate, inside the image."""
    H, W = shape[:2]
    x, y, w, h = cv2.boundingRect(tuft.contour.astype(np.int32))
    side = int(min(max(min_side, margin * max(w, h)), max(H, W)))
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(round(cx - side / 2), 0, max(W - side, 0)))
    y0 = int(np.clip(round(cy - side / 2), 0, max(H - side, 0)))
    return x0, y0, min(x0 + side, W), min(y0 + side, H)


def _drop_stray_components(mask, prop):
    """Keep only the connected pieces of SAM's mask that touch the candidate."""
    n, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.astype(bool)
    keep = np.unique(cc[prop & (cc > 0)])
    return np.isin(cc, keep[keep > 0])


@dataclass
class RefineResult:
    mask: np.ndarray                 # uint8 0/255, accepted SAM masks only
    status: list                     # per candidate: accepted / rejected / error
    per_tuft: list = field(default_factory=list)  # per candidate: (mask_full or None, iou)
    windows: int = 0


def refine(image_bgr, candidates, predictor, min_side=512, strict=True, **accept_kw):
    H, W = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out = np.zeros((H, W), np.uint8)
    status = ["rejected"] * len(candidates)
    per = [(None, 0.0)] * len(candidates)
    current, n_windows = None, 0
    # visit candidates top-to-bottom so neighbours can reuse an embedding
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i].root[1], candidates[i].root[0]))
    for i in order:
        t = candidates[i]
        bx, by, bw, bh = cv2.boundingRect(t.contour.astype(np.int32))
        if current is None or not (bx >= current[0] and by >= current[1]
                                   and bx + bw <= current[2] and by + bh <= current[3]):
            current = window_for(t, image_bgr.shape, min_side)
            predictor.set_image(rgb[current[1]:current[3], current[0]:current[2]])
            n_windows += 1
        x0, y0, x1, y1 = current
        prop = candidate_mask(t, image_bgr.shape)[y0:y1, x0:x1] > 0
        pts, labels, box = prompt_for(t, (x0, y0))
        try:
            masks, _, _ = predictor.predict(point_coords=pts, point_labels=labels, box=box,
                                            multimask_output=True)
            m = _drop_stray_components(_best_sam_mask(masks, prop), prop)
        except Exception:
            status[i] = "error"
            continue
        inter = (m & prop).sum()
        iou = float(inter / max((m | prop).sum(), 1))
        if accept_mask(m, prop, **accept_kw):
            out[y0:y1, x0:x1][m] = 255
            full = np.zeros((H, W), bool)
            full[y0:y1, x0:x1] = m
            status[i], per[i] = "accepted", (full, iou)
        elif not strict:
            out[y0:y1, x0:x1][prop] = 255
            status[i], per[i] = "fallback", (None, iou)
        else:
            per[i] = (None, iou)
    return RefineResult(out, status, per, n_windows)
