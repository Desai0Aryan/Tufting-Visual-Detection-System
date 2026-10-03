from dataclasses import dataclass

import cv2
import numpy as np


# ------------------------------------------------------------------ loading
def load_sam2_predictor(model_id="facebook/sam2.1-hiera-small", device=None,
                        config=None, checkpoint=None):
    """Build a SAM2ImagePredictor.

    Either pass `model_id` (downloads from Hugging Face on first use) or a local
    `config` + `checkpoint` pair from the sam2 repo, e.g.
        config="configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint="checkpoints/sam2.1_hiera_small.pt".
    """
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device is None:
        mps = getattr(torch.backends, "mps", None)
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if mps is not None and mps.is_available()
                  else "cpu")
    if config and checkpoint:
        from sam2.build_sam import build_sam2
        return SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))
    return SAM2ImagePredictor.from_pretrained(model_id, device=device)


# ------------------------------------------------------------------ prompts
@dataclass
class Prompt:
    points: np.ndarray   # (N, 2) x,y in tile coords
    labels: np.ndarray   # (N,) 1 = on the tuft, 0 = not
    box: np.ndarray      # (4,) x0,y0,x1,y1


def build_prompt(tuft, offset=(0, 0), n_pos=4, neg_dist=6.0, box_pad=4):
    """Prompt for one proposal from its centerline. `offset` converts the
    tuft's full-image coords to tile coords (tile_xy = full_xy - offset)."""
    path = np.asarray(tuft.path, float) if tuft.path is not None else np.array([tuft.root, tuft.tip], float)
    idx = np.linspace(0, len(path) - 1, n_pos + 2)[1:-1].round().astype(int)  # skip the very ends
    pos = path[idx]
    # negatives: step off the string along its local normal, on both sides of the middle
    mid = len(path) // 2
    tang = path[min(mid + 1, len(path) - 1)] - path[max(mid - 1, 0)]
    nrm = np.array([-tang[1], tang[0]], float)
    nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
    neg = np.stack([path[mid] + nrm * neg_dist, path[mid] - nrm * neg_dist])
    pts = np.vstack([pos, neg]) - np.asarray(offset, float)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    lo = path.min(axis=0) - box_pad - np.asarray(offset, float)
    hi = path.max(axis=0) + box_pad - np.asarray(offset, float)
    return Prompt(pts.astype(np.float32), labels, np.concatenate([lo, hi]).astype(np.float32))


# ------------------------------------------------------------------ acceptance
def accept_mask(sam_mask, proposal_mask, min_iou=0.25, max_area_ratio=4.0, min_cover=0.6):
    """Decide whether SAM's mask is a plausible refinement of the proposal.

    - it must cover most of the proposal (SAM found *this* string),
    - overlap enough (IoU) that it isn't a different object that happens to
      include the string (wing, tape + a shadow, ...),
    - and not be wildly larger than the proposal.
    """
    s, p = sam_mask.astype(bool), proposal_mask.astype(bool)
    pa, sa = int(p.sum()), int(s.sum())
    if pa == 0 or sa == 0:
        return False
    inter = int((s & p).sum())
    iou = inter / float(pa + sa - inter)
    return iou >= min_iou and inter / pa >= min_cover and sa <= max_area_ratio * pa


def iter_tiles(h, w, tile=512, overlap=128):
    """Yield (x0, y0, x1, y1) tiles covering the image; the last row/col is
    pulled back so every tile is full-size where the image allows."""
    if max(h, w) <= tile:
        yield (0, 0, w, h)
        return
    stride = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y0 in ys:
        for x0 in xs:
            yield (x0, y0, min(x0 + tile, w), min(y0 + tile, h))


# ------------------------------------------------------------------ main
def refine_with_sam(image_bgr, proposals, proposal_mask, predictor,
                    tile=512, overlap=128, max_prompts=2000, **accept_kw):
    """Refine `proposals` (list[Tuft]) with SAM 2.

    Returns (refined_mask uint8 0/255, stats dict). Proposals SAM disagrees
    with fall back to the proposal's own pixels, so the result never loses a
    tuft the classical code found; it can only get cleaner or fuller.
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    refined = np.zeros((h, w), np.uint8)
    stats = {"proposals": len(proposals), "sam_accepted": 0, "fallback": 0, "skipped": 0}

    # each proposal is handled by the tile that contains its whole bounding box
    pending = list(enumerate(proposals[:max_prompts]))
    stats["skipped"] = max(0, len(proposals) - max_prompts)
    done = set()
    for (x0, y0, x1, y1) in iter_tiles(h, w, tile, overlap):
        here = []
        for i, t in pending:
            if i in done:
                continue
            bx, by, bw, bh = cv2.boundingRect(t.contour.astype(np.int32))
            if bx >= x0 and by >= y0 and bx + bw <= x1 and by + bh <= y1:
                here.append((i, t))
        if not here:
            continue
        predictor.set_image(rgb[y0:y1, x0:x1])
        for i, t in here:
            done.add(i)
            prop_local = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.drawContours(prop_local, [t.contour.astype(np.int32) - np.array([x0, y0])],
                             -1, 255, thickness=cv2.FILLED)
            pr = build_prompt(t, offset=(x0, y0))
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=pr.points, point_labels=pr.labels, box=pr.box,
                    multimask_output=False)
                m = np.asarray(masks)[0] > 0
            except Exception:  # a failing prompt must not kill a labelling run
                m = None
            if m is not None and accept_mask(m, prop_local > 0, **accept_kw):
                refined[y0:y1, x0:x1][m] = 255
                stats["sam_accepted"] += 1
            else:
                refined[y0:y1, x0:x1][prop_local > 0] = 255
                stats["fallback"] += 1

    # proposals not wholly inside any tile (rare, huge ones): keep their own pixels
    for i, t in pending:
        if i not in done:
            cv2.drawContours(refined, [t.contour.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
            stats["fallback"] += 1
    return refined, stats
