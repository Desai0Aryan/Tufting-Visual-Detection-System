"""Locate the EC135 Fenestron duct opening as an ellipse (handles oblique views).

1. hub: the brightest, most circular blob near the centre of the tuft boxes
   (a white disc), found with HoughCircles at a radius scaled to the box cloud;
2. rays from the hub centre: on each ray, the first strong dark -> bright step
   beyond 1.6 hub radii is the edge between the duct opening and the white
   inner-ring surface;
3. RANSAC ellipse through those edge points (stator blades and tufts crossing a
   ray give outliers).
Returns ((cx, cy), (major, minor), angle) as cv2.fitEllipse does, plus a score,
or None when the fit is not trustworthy.
fit_and_check() adds a sanity check: the tuft boxes must surround the ellipse
(box centres just outside the duct in >= 6 of 8 directions).

Failure rate on all 982 EC135 frames (2026-10-12): 392 accepted (40 %);
578 with no ellipse (hub not found, or too few / inconsistent edge points:
oblique, far-away or blurred views) and 12 rejected by the sanity check.
Every accepted fit outside the split gaps (336) was checked by eye on contact
sheets; all sat on the duct ring. Frames without an accepted fit are dropped,
not labelled. A plain Hough circle found 85/121 on a sample but was often on
the wrong edge; an ellipse through the box centres was wrong too.
"""
import cv2
import numpy as np


def _hub(gray, boxes):
    B = np.array([b[1:] for b in boxes], float)
    bx0, by0 = B[:, :2].min(0)
    bx1, by1 = B[:, 2:].max(0)
    span = max(bx1 - bx0, by1 - by0)
    g = cv2.medianBlur(gray, 5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=span / 4, param1=100, param2=40,
                         minRadius=int(span * 0.04), maxRadius=int(span * 0.16))
    if c is None:
        return None
    best, best_s = None, -1
    for cx, cy, r in c[0]:
        if not (bx0 <= cx <= bx1 and by0 <= cy <= by1):
            continue
        m = np.zeros(gray.shape, np.uint8)
        cv2.circle(m, (int(cx), int(cy)), int(r * 0.8), 255, -1)
        ring = np.zeros(gray.shape, np.uint8)
        cv2.circle(ring, (int(cx), int(cy)), int(r * 1.6), 255, -1)
        cv2.circle(ring, (int(cx), int(cy)), int(r * 1.1), 0, -1)
        inside, around = gray[m > 0].mean(), gray[ring > 0].mean()
        s = inside - around  # a white disc on a darker duct background
        if s > best_s:
            best, best_s = (float(cx), float(cy), float(r)), s
    return best if best_s > 10 else None


def _edge_points(gray, hub, n_rays=180, max_k=4.5):
    cx, cy, r = hub
    g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    H, W = g.shape
    pts = []
    for a in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        d = np.arange(r * 1.6, r * max_k, 1.0)
        xs, ys = cx + d * np.cos(a), cy + d * np.sin(a)
        ok = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
        if ok.sum() < 20:
            continue
        prof = g[ys[ok].astype(int), xs[ok].astype(int)]
        step = prof[6:] - prof[:-6]  # dark (opening) -> bright (ring surface)
        k = int(np.argmax(step))
        if step[k] > 18:
            pts.append((xs[ok][k + 3], ys[ok][k + 3]))
    return np.array(pts, np.float32)


def fit_duct(img, boxes, iters=300, tol=0.04, seed=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hub = _hub(gray, boxes)
    if hub is None:
        return None
    pts = _edge_points(gray, hub)
    if len(pts) < 40:
        return None
    rng = np.random.default_rng(seed)
    best, best_in = None, None
    for _ in range(iters):
        sample = pts[rng.choice(len(pts), 6, replace=False)]
        try:
            e = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        (ex, ey), (ma, mi), ang = e
        if min(ma, mi) < 2 * hub[2] * 1.4 or max(ma, mi) > 2 * hub[2] * 9:
            continue
        inl = _inliers(pts, e, tol)
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = e, inl
    if best is None or best_in.sum() < 0.45 * len(pts):
        return None
    e = cv2.fitEllipse(pts[best_in])
    (ex, ey), _, _ = e
    if np.hypot(ex - hub[0], ey - hub[1]) > 0.6 * hub[2] * 2:  # must be centred on the hub
        return None
    return e, float(best_in.mean()), hub


def _inliers(pts, e, tol):
    (ex, ey), (ma, mi), ang = e
    t = np.radians(ang)
    dx, dy = pts[:, 0] - ex, pts[:, 1] - ey
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    rr = np.sqrt((u / (ma / 2)) ** 2 + (v / (mi / 2)) ** 2)
    return np.abs(rr - 1) < tol


def band_mask(shape, ellipse, inner=0.8, outer=1.0):
    """uint8 mask of the ring band between `inner` and `outer` times the duct ellipse."""
    (ex, ey), (ma, mi), ang = ellipse
    m = np.zeros(shape[:2], np.uint8)
    cv2.ellipse(m, ((ex, ey), (ma * outer, mi * outer), ang), 255, -1)
    cv2.ellipse(m, ((ex, ey), (ma * inner, mi * inner), ang), 0, -1)
    return m


def surround_check(ellipse, centers, min_octants=6, max_dist=2.2):
    """Sanity check: the tuft boxes must surround the fitted duct. Counts the
    octants (around the ellipse centre) holding a box centre that lies outside
    the duct but within max_dist x its radius in that direction. A fit on the
    stabiliser or the fin, or one shifted off the duct, fails."""
    (ex, ey), (ma, mi), ang = ellipse
    t = np.radians(ang)
    P = np.asarray(centers, float)
    dx, dy = P[:, 0] - ex, P[:, 1] - ey
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    rr = np.sqrt((u / (ma / 2)) ** 2 + (v / (mi / 2)) ** 2)
    near = (rr > 0.9) & (rr < max_dist)
    octs = set((np.floor((np.arctan2(dy[near], dx[near]) + np.pi) / (np.pi / 4)).astype(int) % 8).tolist())
    return len(octs) >= min_octants, len(octs)


def fit_and_check(img, boxes, min_octants=6):
    """-> dict(ok, reason, ellipse, hub, octants). `boxes` are (label, x0, y0, x1, y1)."""
    r = fit_duct(img, boxes)
    if r is None:
        return {"ok": False, "reason": "no ellipse fit", "ellipse": None, "hub": None, "octants": 0}
    e, score, hub = r
    centers = [((b[1] + b[3]) / 2, (b[2] + b[4]) / 2) for b in boxes]
    ok, n = surround_check(e, centers, min_octants)
    return {"ok": bool(ok), "reason": "ok" if ok else f"boxes cover only {n}/8 directions",
            "ellipse": e, "hub": hub, "octants": n, "inlier_frac": score}
