from dataclasses import replace

import cv2
import numpy as np
from scipy.spatial import cKDTree

DEFAULT_NORM_SIZE = 1400  # long side (px) that the size limits / training crops are tuned for


# ------------------------------------------------------------------ scale
def normalize_scale(image, target_long_side=DEFAULT_NORM_SIZE):
    """Downscale (never upscale) so the long side is <= target. Returns (image, scale)
    where scale = new_size / old_size (1.0 when untouched)."""
    if not target_long_side or target_long_side <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long_side:
        return image, 1.0
    scale = target_long_side / float(long_side)
    out = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                     interpolation=cv2.INTER_AREA)
    return out, scale


def polygon_mask(shape, points):
    """uint8 {0,255} mask of the polygon `points` ((x, y) pairs) on an HxW canvas."""
    m = np.zeros(shape[:2], np.uint8)
    pts = np.round(np.asarray(points, float)).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 255)
    return m


def scale_points(points, scale):
    return [(x * scale, y * scale) for x, y in points]


def scale_roi(roi, scale):
    if roi is None or scale == 1.0:
        return roi
    x, y, w, h = roi
    return (int(round(x * scale)), int(round(y * scale)),
            max(1, int(round(w * scale))), max(1, int(round(h * scale))))


def rescale_tufts(tufts, inv_scale):
    """Map tufts found on a resized image back to the original image's coordinates."""
    if inv_scale == 1.0:
        return list(tufts)
    out = []
    for t in tufts:
        out.append(replace(
            t,
            root=(t.root[0] * inv_scale, t.root[1] * inv_scale),
            tip=(t.tip[0] * inv_scale, t.tip[1] * inv_scale),
            length_px=t.length_px * inv_scale,
            contour=np.round(t.contour.astype(np.float64) * inv_scale).astype(np.int32),
            path=None if t.path is None else t.path * inv_scale,
        ))
    return out


# ------------------------------------------------------------------ contrast
def local_contrast(gray, tuft, inner_px=1, ring_in=3, ring_out=8):
    """|mean(surrounding ring) - mean(string pixels)| in gray levels.

    Polarity-agnostic (works for dark or light strings on any surface).  Texture
    and compression noise have very little contrast against their own
    surroundings; a taped string has a lot.
    """
    pts = tuft.contour.astype(np.int32).reshape(-1, 2)
    x0, y0 = pts.min(axis=0) - ring_out - 1
    x1, y1 = pts.max(axis=0) + ring_out + 2
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), w), min(int(y1), h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    body = np.zeros(crop.shape, np.uint8)
    cv2.drawContours(body, [tuft.contour.astype(np.int32) - np.array([x0, y0])], -1, 255, cv2.FILLED)
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_in + 1, 2 * ring_in + 1))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_out + 1, 2 * ring_out + 1))
    ring = (cv2.dilate(body, k_out) > 0) & (cv2.dilate(body, k_in) == 0)
    core = body > 0
    if inner_px > 0:  # skip the anti-aliased edge so thin strings are not averaged with surface
        eroded = cv2.erode(body, np.ones((3, 3), np.uint8)) > 0
        if eroded.sum() >= 3:
            core = eroded
    if ring.sum() < 5 or core.sum() < 3:
        return 0.0
    return float(abs(np.median(crop[ring]) - np.median(crop[core])))


def contrast_filter(image, tufts, min_contrast=12.0):
    if min_contrast <= 0 or not tufts:
        return list(tufts)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return [t for t in tufts if local_contrast(gray, t) >= min_contrast]


# ------------------------------------------------------------------ tuft field
def tuft_centers(tufts):
    out = []
    for t in tufts:
        if t.path is not None and len(t.path):
            out.append(np.asarray(t.path, float)[len(t.path) // 2])
        else:
            out.append((np.asarray(t.root, float) + np.asarray(t.tip, float)) / 2.0)
    return np.asarray(out, float).reshape(-1, 2)


def _components(centers, radius):
    """Connected components of the 'closer than radius' graph -> label per point."""
    n = len(centers)
    labels = -np.ones(n, int)
    tree = cKDTree(centers)
    cur = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]
        labels[i] = cur
        while stack:
            j = stack.pop()
            for k in tree.query_ball_point(centers[j], radius):
                if labels[k] < 0:
                    labels[k] = cur
                    stack.append(k)
        cur += 1
    return labels


def field_filter(image, tufts, radius_factor=3.5, seed_frac=0.5,
                 min_cluster_frac=0.25, min_total=10, contrast_floor=3.0):
    """Keep only tufts that belong to the *tuft field*.

    1. Seeds = candidates with at least seed_frac x the median local contrast
       (scale-free, so it adapts from a crisp 4K photo to a soft 500 px one).
    2. Seeds closer than radius_factor x median-length are chained into clusters.
    3. Keep clusters holding at least min_cluster_frac of the biggest cluster:
       one instrumented surface beats a picture-in-picture inset or stray blobs,
       while two similar-sized surfaces both survive.
    4. Region-grow one hop: every candidate within that same radius of a kept seed
       (and above a tiny contrast floor) is kept, so fainter tufts at the edge of
       the field are not lost the way a hull-based test would lose them.

    Nothing here assumes a surface colour or material - only that tufts are many,
    similar in size and spread over one area.
    """
    if len(tufts) < min_total:
        return list(tufts)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    contrast = np.array([local_contrast(gray, t) for t in tufts])
    centers = tuft_centers(tufts)
    med_len = float(np.median([t.length_px for t in tufts]))
    radius = max(40.0, radius_factor * med_len)

    seed_thr = max(contrast_floor, seed_frac * float(np.median(contrast)))
    seed_idx = np.nonzero(contrast >= seed_thr)[0]
    if len(seed_idx) < 5:
        return [t for t, c in zip(tufts, contrast) if c >= contrast_floor]

    labels = _components(centers[seed_idx], radius)
    sizes = np.bincount(labels)
    good = {c for c, n in enumerate(sizes) if n >= max(3, min_cluster_frac * sizes.max())}
    seeds = np.array([centers[i] for i, l in zip(seed_idx, labels) if l in good], float)
    if len(seeds) < 3:
        return [t for t, c in zip(tufts, contrast) if c >= contrast_floor]

    dist, _ = cKDTree(seeds).query(centers)
    return [t for t, c, d in zip(tufts, contrast, dist) if c >= contrast_floor and d <= radius]


def clean_tufts(image, tufts, min_contrast=3.0, radius_factor=3.5, seed_frac=0.5,
                min_cluster_frac=0.25):
    """Field filter, then renumber ids so the CSV stays contiguous."""
    kept = field_filter(image, tufts, radius_factor, seed_frac, min_cluster_frac,
                        contrast_floor=min_contrast)
    return [replace(t, id=i) for i, t in enumerate(kept)]
