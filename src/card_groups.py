"""Look-alike groups of tuft cards, so one key press votes a whole group.

Features per card (src.tuft_cards.Card):
  - ImageNet resnet34 (the training encoder) on a crop around the candidate,
    pixels outside the candidate dimmed so the features describe the candidate,
    not the wing or sky around it; deepest map average-pooled, L2-normalised
  - a few shape numbers (length, width, elongation, contrast to its
    surroundings), z-scored and down-weighted, so merged double tufts and fat
    blobs separate from thin single strings
Groups: k-means (torch; sklearn is not a dependency), clusters larger than a
grid page split again, members ordered centre-first so the odd ones end up last.
"""
import math

import cv2
import numpy as np

CROP = 128
DIM = 0.4
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop_square(bbox, shape, factor=2.0, min_side=64):
    """Square window (x0, y0, x1, y1) around a bbox, inside the image."""
    x, y, w, h = bbox
    H, W = shape[:2]
    side = int(min(max(min_side, factor * max(w, h)), max(H, W)))
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(round(cx - side / 2), 0, max(W - side, 0)))
    y0 = int(np.clip(round(cy - side / 2), 0, max(H - side, 0)))
    return x0, y0, min(x0 + side, W), min(y0 + side, H)


def local_mask(card, box):
    x0, y0, x1, y1 = box
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.drawContours(m, [c - [x0, y0] for c in card.contours], -1, 255, cv2.FILLED)
    return m > 0


def crop_for_embedding(image, card, size=CROP):
    """size x size RGB float32 crop, outside the candidate dimmed."""
    box = crop_square(card.bbox, image.shape)
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1].astype(np.float32)
    m = local_mask(card, box)
    crop[~m] *= DIM
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) / 255.0


def shape_features(image, card, ring=6):
    """[log length, log width, log elongation, contrast to a ring around it]."""
    c = np.vstack(card.contours).reshape(-1, 2).astype(np.float32)
    (_, _), (a, b), _ = cv2.minAreaRect(c)
    length = max(a, b, 1.0)
    area = max(sum(cv2.contourArea(k) for k in card.contours), 1.0)
    width = area / length
    box = crop_square(card.bbox, image.shape, factor=1.0, min_side=1)
    x0, y0, x1, y1 = box
    pad = ring + 1
    x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
    x1, y1 = min(x1 + pad, image.shape[1]), min(y1 + pad, image.shape[0])
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = local_mask(card, (x0, y0, x1, y1))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    r = (cv2.dilate(m.astype(np.uint8), k) > 0) & ~m
    inside = gray[m].mean() if m.any() else 0.0
    around = gray[r].mean() if r.any() else inside
    return np.array([math.log(length), math.log(width + 1), math.log(length / max(width, 1.0)),
                     (inside - around) / 255.0], np.float32)


def load_encoder(device="cuda"):
    import segmentation_models_pytorch as smp
    enc = smp.encoders.get_encoder("resnet34", weights="imagenet")
    return enc.eval().to(device)


def embed(crops, encoder, device="cuda", batch=256):
    """(N, size, size, 3) RGB [0,1] -> (N, C) L2-normalised features."""
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(crops), batch):
            x = (np.asarray(crops[i:i + batch], np.float32) - MEAN) / STD
            x = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(device)
            f = encoder(x)[-1].mean((2, 3))
            out.append(torch.nn.functional.normalize(f, dim=1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def combine(cnn, shape, shape_weight=0.15):
    """CNN features + z-scored shape numbers at a small weight."""
    s = np.asarray(shape, np.float32)
    s = (s - s.mean(0)) / (s.std(0) + 1e-6)
    return np.hstack([cnn, shape_weight * s]).astype(np.float32)


# ---------------------------------------------------------------- clustering
def kmeans(x, k, iters=30, seed=0, device=None):
    """-> labels (N,) int. k-means++ init; torch on GPU when available."""
    import torch
    n = len(x)
    k = int(max(1, min(k, n)))
    if k == 1:
        return np.zeros(n, int)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(np.asarray(x, np.float32), device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = [X[torch.randint(n, (1,), generator=g).item()]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = (d2 / d2.sum().clamp_min(1e-12)).cpu()
        i = torch.multinomial(p, 1, generator=g).item() if p.sum() > 0 else torch.randint(n, (1,), generator=g).item()
        centers.append(X[i])
        d2 = torch.minimum(d2, ((X - X[i]) ** 2).sum(1))
    C = torch.stack(centers)
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        newC = torch.zeros_like(C).index_add_(0, lab, X)
        cnt = torch.bincount(lab, minlength=k).unsqueeze(1)
        keep = cnt.squeeze(1) > 0
        newC[keep] /= cnt[keep]
        newC[~keep] = C[~keep]
        if torch.allclose(newC, C):
            break
        C = newC
    return torch.cdist(X, C).argmin(1).cpu().numpy()


def _split(x, idx, max_size, seed):
    if len(idx) <= max_size:
        return [idx]
    lab = kmeans(x[idx], math.ceil(len(idx) / (0.8 * max_size)), seed=seed)
    parts = [idx[lab == j] for j in np.unique(lab)]
    if len(parts) == 1:  # identical points: k-means cannot split, chunk instead
        return [idx[i:i + max_size] for i in range(0, len(idx), max_size)]
    return [p for part in parts for p in _split(x, part, max_size, seed + 1)]


def make_groups(x, per_group=40, max_size=48, seed=0):
    """-> list of index arrays: each <= max_size, centre-first, largest group first."""
    x = np.asarray(x, np.float32)
    if len(x) == 0:
        return []
    lab = kmeans(x, round(len(x) / per_group), seed=seed)
    groups = []
    for j in np.unique(lab):
        for g in _split(x, np.flatnonzero(lab == j), max_size, seed):
            d = np.linalg.norm(x[g] - x[g].mean(0), axis=1)
            groups.append(g[np.argsort(d, kind="stable")])
    groups.sort(key=len, reverse=True)
    return groups


# ---------------------------------------------------------------- grid page
CYCLE = [None, "n", "y", "u"]


class GridPage:
    """One page of cards: clicked cells carry their own vote; a key fills the rest."""

    def __init__(self, card_ids):
        self.ids = list(card_ids)
        self.own = [None] * len(self.ids)

    def click(self, i):
        if 0 <= i < len(self.ids):
            self.own[i] = CYCLE[(CYCLE.index(self.own[i]) + 1) % len(CYCLE)]

    def commit(self, vote):
        """-> {card id: vote} for every cell."""
        return {cid: own or vote for cid, own in zip(self.ids, self.own)}
