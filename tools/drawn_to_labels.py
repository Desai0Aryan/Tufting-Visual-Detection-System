"""Hand-drawn tuft lines (tools.draw_tufts) -> training labels.

    python -m tools.drawn_to_labels data/drawn/tufts_flight1 --clip flight1 --val-from 2900

Per drawn line (root -> tip):
  tuft      the line, drawn thicker than the yarn (--width, default ~10% of its
            length, 4-12 px): hand-drawn lines wander off the string a little,
            and a slightly fat label still covers it
  ignore    a border around it (--border, where exactly the yarn edge is) and, unless
            --no-model-ignore, anything the model sees as tuft-like that no line
            covers (a tuft you skipped is never taught as background)
  root      a disk at the line's start (the drawn root, so orientation is exact)
Frames with frame_idx >= --val-from become clip <clip>_val, the rest <clip>_train;
register both in sources.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.label_io import save_frame  # noqa: E402
from src.tuft import Tuft  # noqa: E402


def polyline_length(st):
    return float(np.linalg.norm(np.diff(st, axis=0), axis=1).sum()) if len(st) > 1 else 0.0


def stroke_tuft(st, shape, width=None, k=0):
    """One drawn line -> (mask bool, Tuft)."""
    st = np.asarray(st, float)
    L = polyline_length(st)
    w = int(round(width or np.clip(0.10 * L, 4, 12)))
    m = np.zeros(shape[:2], np.uint8)
    cv2.polylines(m, [np.round(st * 4).astype(np.int32)], False, 1, max(1, w), cv2.LINE_AA, shift=2)
    m = m > 0
    n = max(2, len(st) // 6)
    v = st[-1] - st[-1 - min(n, len(st) - 1)]
    v = v / (np.linalg.norm(v) + 1e-9)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(cs, key=cv2.contourArea) if cs else np.zeros((0, 1, 2), np.int32)
    t = Tuft(id=k, root=tuple(st[0]), tip=tuple(st[-1]), vector=(float(v[0]), float(v[1])),
             angle_deg=float(np.degrees(np.arctan2(-v[1], v[0]))), length_px=L, contour=contour,
             path=st, source="drawn")
    return m, t


def frame_labels(image, strokes, width=None, border=4, model_mask=None):
    """-> (tuft bool, ignore bool, [Tuft])."""
    shape = image.shape
    tuft = np.zeros(shape[:2], bool)
    tufts = []
    for k, st in enumerate(strokes):
        m, t = stroke_tuft(st, shape, width, k)
        tuft |= m
        tufts.append(t)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
    near = cv2.dilate(tuft.astype(np.uint8), ker) > 0
    ignore = near & ~tuft
    if model_mask is not None:
        covered = cv2.dilate(tuft.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        ignore |= model_mask & ~covered
    return tuft, ignore & ~tuft, tufts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("drawn", help="folder written by tools.draw_tufts")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--val-from", type=int, default=None, help="frames from here on are validation")
    ap.add_argument("--out", default=None, help="default data/drawn_labels/<clip>")
    ap.add_argument("--width", type=float, default=None, help="label line width in px (default ~10%% of length)")
    ap.add_argument("--border", type=int, default=4, help="ignore border around each line, px")
    ap.add_argument("--weights", default="models/tuft_unet.pt")
    ap.add_argument("--no-model-ignore", action="store_true", help="you drew every tuft: no model-based ignore")
    a = ap.parse_args(argv)

    drawn = Path(a.drawn)
    out = Path(a.out or f"data/drawn_labels/{a.clip}")
    backend = None
    if not a.no_model_ignore and Path(a.weights).exists():
        from src.unet import UNetBackend
        backend = UNetBackend(a.weights)
    n_frames = n_tufts = 0
    for jp in sorted((drawn / "json").glob("*.json")):
        d = json.loads(jp.read_text())
        if not d["strokes"] or not d.get("checked", True):   # pre-filled but never looked at: not a label
            continue
        img = cv2.imread(str(drawn / "images" / f"{d['stem']}.png"))
        model_mask = None
        if backend is not None:
            model_mask = backend.predict_proba(img)[0] >= backend.threshold * 0.5
        tuft, ignore, tufts = frame_labels(img, [np.array(s) for s in d["strokes"]], a.width, a.border,
                                           model_mask=model_mask)
        split = "val" if a.val_from is not None and d["frame_idx"] >= a.val_from else "train"
        clip = f"{a.clip}_{split}"
        meta = {"clip_id": clip, "frame_idx": d["frame_idx"], "split": split, "use": "tufts",
                "source": "drawn", "domain": "own"}
        save_frame(out, f"{clip}__f{d['frame_idx']:06d}", img, tuft.astype(np.uint8) * 255, tufts, meta,
                   ignore=ignore)
        n_frames += 1
        n_tufts += len(tufts)
    print(f"{n_frames} frames, {n_tufts} tufts -> {out}")
    print(f"register in sources.csv: {a.clip}_train -> train" + (f", {a.clip}_val -> val" if a.val_from else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
