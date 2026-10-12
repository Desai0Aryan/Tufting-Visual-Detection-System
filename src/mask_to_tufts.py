from collections import deque

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize

from .tuft import Tuft

VECTOR_MODES = ("chord", "base", "tip")
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_ORIENT_DECISIVE = 0.2  # below this thickness-asymmetry confidence, consult the flow prior


def binarize(mask, threshold=0.5):
    """Accept bool / uint8 (0-255) / float probability masks -> uint8 {0, 255}."""
    m = np.asarray(mask)
    if m.dtype == bool:
        return m.astype(np.uint8) * 255
    if m.dtype == np.uint8:
        return ((m > 127) * 255).astype(np.uint8)
    return ((m >= threshold) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# skeleton graph helpers
# --------------------------------------------------------------------------
def _skeleton_graph(skel):
    ys, xs = np.nonzero(skel)
    pts = set(zip(ys.tolist(), xs.tolist()))
    return {
        p: [(p[0] + dy, p[1] + dx) for dy, dx in _NEIGHBORS if (p[0] + dy, p[1] + dx) in pts]
        for p in pts
    }


def _bfs(adj, start):
    parent = {start: None}
    queue = deque([start])
    last = start
    while queue:
        last = queue.popleft()
        for nxt in adj[last]:
            if nxt not in parent:
                parent[nxt] = last
                queue.append(nxt)
    return last, parent


def _largest_component_node(adj):
    seen, best_node, best_size = set(), None, 0
    for node in adj:
        if node in seen:
            continue
        _, parent = _bfs(adj, node)
        seen.update(parent)
        if len(parent) > best_size:
            best_node, best_size = node, len(parent)
    return best_node


def longest_skeleton_path(adj):
    """Longest simple path (graph diameter) via double-sweep BFS - O(n), versus
    the old all-endpoint-pairs search. Exact on trees (spurs never win), a very
    good approximation if the skeleton has a loop. Returns [(row, col), ...]."""
    if not adj:
        return None
    start = _largest_component_node(adj)
    a, _ = _bfs(adj, start)
    b, parent = _bfs(adj, a)
    path = [b]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path


def _path_length(path_xy):
    if len(path_xy) < 2:
        return 0.0
    return float(np.hypot(*np.diff(path_xy, axis=0).T).sum())


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def smooth_resample(path_xy, n_out=48):
    """Gaussian-smooth a pixel-staircase centerline and resample it uniformly
    by arc length so tangents / halfway points are well defined."""
    sigma = max(1.5, len(path_xy) / 20.0)
    sm = np.column_stack([
        gaussian_filter1d(path_xy[:, 0], sigma, mode="nearest"),
        gaussian_filter1d(path_xy[:, 1], sigma, mode="nearest"),
    ])
    seg = np.hypot(*np.diff(sm, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return sm
    t = np.linspace(0.0, s[-1], n_out)
    return np.column_stack([np.interp(t, s, sm[:, 0]), np.interp(t, s, sm[:, 1])])


def orient_path(path_xy, dist, root_prob=None, flow_prior=None):
    """Return (path_root_first, orientation_confidence).

    `path_xy` is (N, 2) in crop coords, arbitrary direction.
    `dist` is the distance transform of the crop mask (half-width in px).
    """
    n = len(path_xy)
    k = max(2, int(round(n * 0.2)))
    ij = np.clip(np.round(path_xy).astype(int), 0, np.array(dist.shape[::-1]) - 1)
    flip, conf = False, 0.0

    decided = False
    if root_prob is not None:
        r = root_prob[ij[:, 1], ij[:, 0]].astype(float)
        a, b = r[:k].max(), r[-k:].max()
        if max(a, b) >= 0.5 and abs(a - b) >= 0.25:
            flip, conf, decided = bool(b > a), float(abs(a - b)), True

    if not decided:
        w = dist[ij[:, 1], ij[:, 0]].astype(float)
        wa, wb = w[:k].mean(), w[-k:].mean()
        conf = min(1.0, 2.0 * abs(wa - wb) / (max(wa, wb) + 1e-6))
        flip = bool(wb > wa)
        if conf < _ORIENT_DECISIVE and flow_prior is not None:
            chord = path_xy[-1] - path_xy[0]
            flip = bool(np.dot(chord, flow_prior) < 0)  # make chord agree with prior
            conf = max(conf, 0.3)

    return (path_xy[::-1] if flip else path_xy), conf


def flow_vector(path, root, tip, mode="chord"):
    """2D unit flow vector from a root->tip centerline. Falls back to the chord."""
    root, tip = np.asarray(root, float), np.asarray(tip, float)
    if mode == "chord":
        v = tip - root
    elif mode == "base":
        v = path[len(path) // 2] - root
    elif mode == "tip":
        k = max(2, len(path) // 8)
        v = path[-1] - path[-1 - k]
    else:
        raise ValueError(f"vector_mode must be one of {VECTOR_MODES}")
    n = np.linalg.norm(v)
    if n < 1e-6:
        v = tip - root
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
    return v / n


def elongation(contour, area):
    """perimeter^2 / (4*pi*area): ~1 for round blobs, large for thin/curled strings."""
    if area < 1e-6:
        return 0.0
    perimeter = cv2.arcLength(contour, True)
    return (perimeter * perimeter) / (4.0 * np.pi * area)


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def extract_tufts(
    mask,
    root_mask=None,
    *,
    offset=(0, 0),
    min_area=15,
    max_area=600,
    min_elongation=2.0,
    min_length_px=12,
    max_length_px=110,
    vector_mode="chord",
    flow_prior=None,
    source="classical",
    anchor_dist=None,
    anchor_max_dist=60.0,
):
    """Mask -> list[Tuft]. `offset` (x, y) is added to every output coordinate so
    results from an ROI crop land in full-image coordinates.

    With `anchor_dist` (distance-to-tape map, same frame as `mask`), the root is
    the path end closest to the tape (more reliable than thickness for uniform
    yarn), and a candidate with neither end within `anchor_max_dist` of tape is
    rejected as an unanchored stray (reflections, glare streaks)."""
    if vector_mode not in VECTOR_MODES:
        raise ValueError(f"vector_mode must be one of {VECTOR_MODES}")
    prior = None
    if flow_prior is not None:
        prior = np.asarray(flow_prior, float)
        prior = prior / (np.linalg.norm(prior) + 1e-12)

    binary = binarize(mask)
    root_f = None
    if root_mask is not None:
        root_f = np.asarray(root_mask)
        root_f = (root_f.astype(np.float32) / 255.0) if root_f.dtype == np.uint8 else root_f.astype(np.float32)

    off = np.asarray(offset, dtype=np.float32)
    H, W = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    tufts = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if elongation(contour, area) < min_elongation:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        pad = 2
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
        local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.drawContours(local, [contour - np.array([x0, y0])], -1, 255, thickness=cv2.FILLED)

        skel = skeletonize(local > 0)
        adj = _skeleton_graph(skel)
        n_endpoints = sum(1 for nb in adj.values() if len(nb) == 1)
        path_rc = longest_skeleton_path(adj)
        if path_rc is None or len(path_rc) < 3:
            continue
        path_xy = np.array([(c, r) for r, c in path_rc], dtype=float)
        length = _path_length(path_xy)
        if length < min_length_px or length > max_length_px:
            continue

        if anchor_dist is not None:
            d_start = anchor_dist[int(path_xy[0][1]) + y0, int(path_xy[0][0]) + x0]
            d_end = anchor_dist[int(path_xy[-1][1]) + y0, int(path_xy[-1][0]) + x0]
            if min(d_start, d_end) > anchor_max_dist:
                continue
            if d_end < d_start:
                path_xy = path_xy[::-1]
            orient_conf = 1.0
        else:
            dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)
            local_root = root_f[y0:y1, x0:x1] if root_f is not None else None
            path_xy, orient_conf = orient_path(path_xy, dist, local_root, prior)

        root_xy, tip_xy = path_xy[0], path_xy[-1]
        smooth = smooth_resample(path_xy)
        vec = flow_vector(smooth, root_xy, tip_xy, vector_mode)
        if vec is None:
            continue

        chord = float(np.linalg.norm(tip_xy - root_xy))
        spurs = max(0, n_endpoints - 2)
        confidence = float(np.clip(orient_conf, 0.0, 1.0) / (1.0 + 0.5 * spurs))

        shift = np.array([x0, y0], dtype=float) + off
        tufts.append(Tuft(
            id=len(tufts),
            root=tuple((root_xy + shift).tolist()),
            tip=tuple((tip_xy + shift).tolist()),
            vector=tuple(vec.tolist()),
            angle_deg=float(np.degrees(np.arctan2(-vec[1], vec[0]))),
            length_px=length,
            contour=contour + off.astype(np.int32),
            confidence=confidence,
            curl=float(min(1.0, chord / length)),
            path=smooth + shift,
            source=source,
        ))
    return tufts
