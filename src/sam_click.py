"""One click on a tuft -> its centre line, root -> tip (SAM 2, labelling only).

SAM 2 sees a window around the click at native resolution (tiny far tufts too),
returns a few nested guesses (string / string + tape / string + panel seam ...);
the smallest one that is still string-shaped is kept (the connected piece under
the click, elongated, not longer than max_len: panel seams are long and thin too). Its skeleton gives the centre line, and the end nearest the click is
the root, so click next to the tape end of the string.
"""
import cv2
import numpy as np

from .mask_to_tufts import extract_tufts

GEOM = dict(min_area=6, max_area=10 ** 7, min_elongation=1.5, min_length_px=8, max_length_px=10 ** 5,
            vector_mode="tip", source="sam2_click")


def click_window(xy, shape, size=512):
    H, W = shape[:2]
    s = min(size, H, W)
    x0 = int(np.clip(round(xy[0] - s / 2), 0, W - s))
    y0 = int(np.clip(round(xy[1] - s / 2), 0, H - s))
    return x0, y0, x0 + s, y0 + s


def _piece_at(mask, p, reach=6):
    """Connected piece of mask under (or within reach of) local point p, else None."""
    n, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    x, y = int(round(p[0])), int(round(p[1]))
    y0, y1 = max(y - reach, 0), min(y + reach + 1, mask.shape[0])
    x0, x1 = max(x - reach, 0), min(x + reach + 1, mask.shape[1])
    labs = cc[y0:y1, x0:x1]
    labs = labs[labs > 0]
    if not labs.size:
        return None
    return cc == np.bincount(labs).argmax()


def best_tuft(masks, p, max_len=None, min_elongation=3.0, max_area_frac=0.15):
    """Of SAM's guesses, the smallest string-shaped piece at p. -> Tuft (window coords) or None."""
    best, best_area = None, None
    for m in np.asarray(masks) > 0:
        piece = _piece_at(m, p)
        if piece is None or piece.sum() > max_area_frac * piece.size:
            continue
        yy, xx = np.mgrid[0:piece.shape[0], 0:piece.shape[1]]
        dist = np.hypot(xx - p[0], yy - p[1]).astype(np.float32)
        ts = extract_tufts(piece.astype(np.uint8) * 255, anchor_dist=dist, anchor_max_dist=1e9, **GEOM)
        if not ts:
            continue
        t = max(ts, key=lambda t: t.length_px)
        area = int(piece.sum())
        if t.length_px / max(area / max(t.length_px, 1.0), 1.0) < min_elongation:
            continue                                    # a blob, not a string
        if max_len is not None and t.length_px > max_len:
            continue                                    # string + seam / neighbour
        if best_area is None or area < best_area:
            best, best_area = t, area
    return best


def tuft_from_click(image_bgr, xy, predictor, window=512, max_len=None):
    """-> (N, 2) centre line in image coords, root (nearest the click) first; or None."""
    x0, y0, x1, y1 = click_window(xy, image_bgr.shape, window)
    rgb = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    p = (xy[0] - x0, xy[1] - y0)
    masks, _, _ = predictor.predict(point_coords=np.array([p], np.float32), point_labels=np.array([1], np.int32),
                                    multimask_output=True)
    t = best_tuft(masks, p, max_len)
    if t is None or t.path is None:
        return None
    return np.asarray(t.path, float) + [x0, y0]
