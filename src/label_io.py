"""Label files for the real-data pipeline.

A labelled frame lives in a dataset folder (same layout the trainer reads):
    images/<stem>.png       the frame, native resolution
    labels/<stem>.png       channels: tuft mask, root disks, root-valid
    json/<stem>.json        per-tuft geometry + provenance (this module)
    meta.json               {stem: {clip_id, frame_idx, split, use, source, status, ...}}

The tuft channel is stored pixel-exactly (never re-rasterised from filtered
contours, which silently dropped tufts in the old review tool).
"""
import json
from pathlib import Path

import numpy as np

from .labeling import encode_label, load_sample, render_label, save_sample
from .mask_to_tufts import extract_tufts

# Generous limits: geometry here is informational; the mask is the label.
GEOMETRY_KW = dict(min_area=10, max_area=10 ** 6, min_elongation=1.5, min_length_px=8,
                   max_length_px=10 ** 5, vector_mode="tip")


def tufts_from_mask(mask, anchor_dist=None, **kw):
    args = dict(GEOMETRY_KW, **kw)
    if anchor_dist is not None:
        args.update(anchor_dist=anchor_dist)
    return extract_tufts(mask, **args)


def build_label(mask, tufts):
    """3-channel label whose tuft channel is exactly `mask`."""
    rendered = render_label(mask.shape, tufts)
    return encode_label(mask > 0, rendered[..., 1] > 127, rendered[..., 2] > 127)


def tuft_records(tufts, extra=None):
    out = []
    for i, t in enumerate(tufts):
        r = {"id": i, "root": [round(v, 2) for v in t.root], "tip": [round(v, 2) for v in t.tip],
             "angle_deg": round(t.angle_deg, 2), "length_px": round(t.length_px, 2),
             "confidence": round(t.confidence, 3), "curl": round(t.curl, 3)}
        if extra:
            r.update(extra[i])
        out.append(r)
    return out


def save_frame(root, stem, image, mask, tufts, meta, rejected=None, extra=None):
    """Write image, label, per-frame JSON and meta entry."""
    root = Path(root)
    save_sample(root, stem, image, build_label(mask, tufts), meta=meta)
    (root / "json").mkdir(exist_ok=True)
    payload = {"stem": stem, **{k: meta.get(k) for k in ("clip_id", "frame_idx", "split", "use", "source")},
               "tufts": tuft_records(tufts, extra), "rejected_candidates": rejected or []}
    (root / "json" / f"{stem}.json").write_text(json.dumps(payload, indent=1))


def load_frame(root, stem):
    """(image, mask uint8 0/255, frame json dict, meta dict)."""
    img, lab = load_sample(root, stem)
    p = Path(root) / "json" / f"{stem}.json"
    info = json.loads(p.read_text()) if p.exists() else {"tufts": [], "rejected_candidates": []}
    meta_p = Path(root) / "meta.json"
    meta = json.loads(meta_p.read_text()).get(stem, {}) if meta_p.exists() else {}
    return img, ((lab[..., 0] > 127).astype(np.uint8) * 255), info, meta
