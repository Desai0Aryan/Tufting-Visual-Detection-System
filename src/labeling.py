import json
import zlib
from pathlib import Path

import cv2
import numpy as np

from .mask_to_tufts import extract_tufts

# relaxed filters for label generation: SAM masks can be bigger than the
# classical area limits, and the human reviewer is the real quality gate
LABEL_EXTRACT_KW = dict(min_area=10, max_area=2500, min_elongation=1.8,
                        min_length_px=8, max_length_px=260)


# ---------------------------------------------------------------- encode/decode
def encode_label(tuft, root, valid):
    """Stack three HxW masks (bool / 0-255 / 0-1) into an HxWx3 uint8 label."""
    def u8(m):
        m = np.asarray(m)
        return (m.astype(np.uint8) * 255) if m.dtype == bool else (m > 0).astype(np.uint8) * 255
    return np.dstack([u8(tuft), u8(root), u8(valid)])


def decode_label(label):
    """HxWx3 uint8 -> (tuft, root, valid) bool arrays."""
    return label[..., 0] > 127, label[..., 1] > 127, label[..., 2] > 127


# ---------------------------------------------------------------- rendering
def flip_root(tuft):
    """Swap root and tip (human override). Returns a new Tuft."""
    from dataclasses import replace
    path = tuft.path[::-1].copy() if tuft.path is not None else None
    return replace(tuft, root=tuft.tip, tip=tuft.root, path=path,
                   vector=(-tuft.vector[0], -tuft.vector[1]),
                   angle_deg=((tuft.angle_deg + 180.0 + 180.0) % 360.0) - 180.0,
                   confidence=1.0)


def root_radius(tuft):
    """Radius of the disk painted at a tuft's root (taped end), in px."""
    return float(np.clip(0.2 * tuft.length_px, 4.0, 9.0))


def render_label(shape, tufts, keep=None, trusted=None, min_trust=0.4):
    """Rasterise Tuft records into a 3-channel label.

    keep[i]    - False drops a tuft (rejected by the reviewer); default keep all.
    trusted[i] - True forces the root label to count (reviewer confirmed);
                 otherwise it counts only if tuft.confidence >= min_trust.
    A rejected tuft is simply left out, i.e. labelled as background, so the
    reviewer should reject only things that really are not tufts.
    Root-valid (ch2) is set around trusted roots and trusted bodies, and
    everywhere farther than ~7 px from any tuft; it stays 0 around untrusted
    tufts so the root head is not trained on a guess.
    """
    h, w = shape[:2]
    body = np.zeros((h, w), np.uint8)
    root = np.zeros((h, w), np.uint8)
    valid = np.zeros((h, w), np.uint8)
    for i, t in enumerate(tufts):
        if keep is not None and not keep[i]:
            continue
        cv2.drawContours(body, [t.contour.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
        is_trusted = (trusted[i] if trusted is not None else False) or t.confidence >= min_trust
        c = (int(round(t.root[0])), int(round(t.root[1])))
        r = int(round(root_radius(t)))
        if is_trusted:
            cv2.circle(root, c, r, 255, -1)
            cv2.circle(valid, c, r + 6, 255, -1)  # supervise the root halo (so "not root" is learned near it)
            # the rest of this tuft's body is trusted background for the root head
            cv2.drawContours(valid, [t.contour.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
    # non-tuft pixels far from any tuft are certain "not root": always valid
    far = cv2.dilate(body, np.ones((15, 15), np.uint8)) == 0
    valid[far] = 255
    return encode_label((body > 0), (root > 0), (valid > 0))


def label_from_mask(mask, **extract_kw):
    """Convenience: binary mask -> (label, tufts) using the Part-1 engine."""
    kw = dict(LABEL_EXTRACT_KW)
    kw.update(extract_kw)
    tufts = extract_tufts(mask, **kw)
    return render_label(mask.shape, tufts), tufts


# ---------------------------------------------------------------- dataset IO
def save_sample(root_dir, stem, image, label, meta=None):
    root_dir = Path(root_dir)
    (root_dir / "images").mkdir(parents=True, exist_ok=True)
    (root_dir / "labels").mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(root_dir / "images" / f"{stem}.png"), image):
        raise IOError(f"could not write image for {stem}")
    if not cv2.imwrite(str(root_dir / "labels" / f"{stem}.png"), label):
        raise IOError(f"could not write label for {stem}")
    if meta is not None:
        mp = root_dir / "meta.json"
        data = json.loads(mp.read_text()) if mp.exists() else {}
        data[stem] = meta
        mp.write_text(json.dumps(data, indent=1))


def list_samples(root_dir):
    root_dir = Path(root_dir)
    return sorted(p.stem for p in (root_dir / "images").glob("*.png")
                  if (root_dir / "labels" / f"{p.stem}.png").exists())


def load_sample(root_dir, stem):
    root_dir = Path(root_dir)
    img = cv2.imread(str(root_dir / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
    lab = cv2.imread(str(root_dir / "labels" / f"{stem}.png"), cv2.IMREAD_COLOR)
    if img is None or lab is None:
        raise IOError(f"could not read sample {stem} from {root_dir}")
    return img, lab


def split_of(stem, val_percent=20):
    """Deterministic train/val assignment from the *source image name*.

    The stem's `__` suffix (e.g. `wing__t03`, a tile/crop of `wing`) is ignored,
    so every tile of one photo lands in the same split - otherwise tiles that
    overlap would leak between train and validation and inflate the scores.
    """
    base = stem.split("__")[0]
    return "val" if zlib.crc32(base.encode()) % 100 < val_percent else "train"


def split_samples(stems, val_percent=20):
    out = {"train": [], "val": []}
    for s in stems:
        out[split_of(s, val_percent)].append(s)
    groups = {}
    for s in stems:
        groups.setdefault(s.split("__")[0], []).append(s)

    # Keep every tile from a source photo together when ensuring both splits exist.
    if not out["train"] and out["val"]:
        base = next(iter(groups))
        moved = groups[base]
        out["train"].extend(moved)
        out["val"] = [s for s in out["val"] if s not in moved]
    if not out["val"] and len(groups) > 1:
        base = next(reversed(groups))
        moved = groups[base]
        out["val"].extend(moved)
        out["train"] = [s for s in out["train"] if s not in moved]
    return out
