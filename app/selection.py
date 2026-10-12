"""Click -> tuft selection (pure functions, no Qt). Selection is for viewing only;
the app never edits tufts."""
import numpy as np


def distance_to_tuft(t, x, y):
    """Shortest distance (px) from (x, y) to the tuft's centre-line (or root-tip segment)."""
    pts = np.asarray(t.path, float) if t.path is not None and len(t.path) > 1 else \
        np.array([t.root, t.tip], float)
    p = np.array([x, y], float)
    a, b = pts[:-1], pts[1:]
    ab = b - a
    tt = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    proj = a + ab * tt[:, None]
    return float(np.min(np.linalg.norm(proj - p, axis=1)))


def nearest_tuft(tufts, x, y, max_dist=25.0):
    """track_id of the tuft nearest (x, y) within max_dist image pixels, else None."""
    best, bd = None, max_dist
    for t in tufts:
        d = distance_to_tuft(t, x, y)
        if d <= bd:
            best, bd = t.track_id, d
    return best


def update_selection(selected, hit, additive):
    """Plain click selects only `hit` (or clears on empty space); Ctrl+click toggles."""
    selected = set(selected)
    if additive:
        if hit is not None:
            selected ^= {hit}
        return selected
    return {hit} if hit is not None else set()
