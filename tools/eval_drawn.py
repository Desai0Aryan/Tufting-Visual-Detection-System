"""Score models against hand-corrected tufts (tools.draw_tufts), tuft by tuft.

    python -m tools.eval_drawn data/drawn_labels/flight1 --split val --weights old=models/tuft_unet.pt E=runs/<run>/checkpoints/best.pt
        --strokes data/drawn/tufts_flight1      # your exact (curved) lines; else straight root-tip lines

Per model, over the frames of the given split:
  found        share of your tufts whose line is mostly (>=60%) within 3 px of the prediction
               (thin and thick predicted masks are treated alike)
  extra/frame  predicted blobs touching none of your tufts and no ignore area
  axis err     median angle between the predicted tuft and your line, ignoring direction
               (0-90 deg): is the tuft's line right
  root found   (models with a root output) share of your roots with a root detection within
               8 px, and dir ok = share of found tufts whose root-to-tip direction is right
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.label_io import load_frame, load_ignore  # noqa: E402
from src.labeling import list_samples  # noqa: E402
from src.mask_to_tufts import extract_tufts  # noqa: E402

GEOM = dict(min_area=10, max_area=10 ** 7, min_elongation=1.5, min_length_px=8, max_length_px=10 ** 5,
            vector_mode="tip")


def _dir(a, b):
    v = np.subtract(b, a, dtype=float)
    return v / (np.linalg.norm(v) + 1e-9)


def score_frame(prob, root_prob, thr, gt_tufts, ignore, shape):
    pred = prob >= thr
    near_pred = cv2.dilate(pred.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    found, axis_err, root_hits, dir_ok, n_found = 0, [], 0, 0, 0
    covered_any = np.zeros(shape, bool)
    dets = extract_tufts(pred.astype(np.uint8) * 255, **GEOM)
    for t in gt_tufts:
        line = np.zeros(shape, np.uint8)
        path = np.asarray(t["path"] if t.get("path") else [t["root"], t["tip"]], float)
        cv2.polylines(line, [np.round(path).astype(np.int32)], False, 1, 1)
        on = line > 0
        cover = (near_pred & on).sum() / max(on.sum(), 1)
        covered_any |= cv2.dilate(line, np.ones((9, 9), np.uint8)) > 0
        if cover >= 0.6:
            found += 1
            gd = _dir(t["root"], t["tip"])
            mid = path[len(path) // 2]
            near = [d for d in dets if min(np.linalg.norm(np.asarray(d.path) - mid, axis=1)) < 6] if dets else []
            if near:
                d = max(near, key=lambda d: d.length_px)
                pdir = _dir(d.root, d.tip)
                axis_err.append(np.degrees(np.arccos(min(1.0, abs(float(pdir @ gd))))))
                if root_prob is not None:
                    n_found += 1
                    r_root = root_prob[int(np.clip(d.root[1], 0, shape[0] - 1)), int(np.clip(d.root[0], 0, shape[1] - 1))]
                    r_tip = root_prob[int(np.clip(d.tip[1], 0, shape[0] - 1)), int(np.clip(d.tip[0], 0, shape[1] - 1))]
                    if r_tip > r_root:
                        pdir = -pdir                     # the root output says the other end is the root
                    dir_ok += float(pdir @ gd) > 0
        if root_prob is not None:
            x, y = int(round(t["root"][0])), int(round(t["root"][1]))
            win = root_prob[max(y - 8, 0):y + 9, max(x - 8, 0):x + 9]
            root_hits += bool(win.size and win.max() >= 0.5)
    n, cc = cv2.connectedComponents(pred.astype(np.uint8))
    extra = 0
    ign = ignore if ignore is not None else np.zeros(shape, bool)
    for k in range(1, n):
        comp = cc == k
        if comp.sum() >= 20 and not (comp & covered_any).any() and not (comp & ign).any():
            extra += 1
    return dict(n=len(gt_tufts), found=found, axis=axis_err, extra=extra, root_hits=root_hits,
                dir_ok=dir_ok, n_dir=n_found)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", help="folder from tools.drawn_to_labels")
    ap.add_argument("--split", default="val")
    ap.add_argument("--weights", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", default=None, help="write the numbers as JSON")
    ap.add_argument("--strokes", default=None, help="tools.draw_tufts folder: score against the exact drawn lines")
    a = ap.parse_args(argv)

    from src.unet import UNetBackend
    meta = json.loads((Path(a.data) / "meta.json").read_text())
    stems = [s for s in list_samples(a.data) if meta.get(s, {}).get("split") == a.split]
    results = {}
    for spec in a.weights:
        name, path = spec.split("=", 1)
        b = UNetBackend(path)
        tot = dict(n=0, found=0, axis=[], extra=0, root_hits=0, dir_ok=0, n_dir=0)
        has_root = False
        for s in stems:
            img, _, info, _ = load_frame(a.data, s)
            if a.strokes:
                sj = next(Path(a.strokes, "json").glob(f"*__f{info.get('frame_idx', meta[s]['frame_idx']):06d}.json"), None)
                if sj is not None:
                    strokes = json.loads(sj.read_text())["strokes"]
                    info["tufts"] = [{"root": st[0], "tip": st[-1], "path": st} for st in strokes if len(st) > 1]
            ign = load_ignore(a.data, s)
            probs = b.predict_proba(img)
            has_root = probs.shape[0] > 1
            r = score_frame(probs[0], probs[1] if has_root else None, b.threshold, info["tufts"],
                            None if ign is None else ign > 127, img.shape[:2])
            for k in tot:
                tot[k] = tot[k] + r[k]
        res = {"frames": len(stems), "tufts": tot["n"], "found": round(tot["found"] / max(tot["n"], 1), 3),
               "extra_per_frame": round(tot["extra"] / max(len(stems), 1), 1),
               "axis_err_median_deg": round(float(np.median(tot["axis"])), 1) if tot["axis"] else None,
               "threshold": b.threshold}
        if has_root:
            res["root_found"] = round(tot["root_hits"] / max(tot["n"], 1), 3)
            res["dir_ok"] = round(tot["dir_ok"] / max(tot["n_dir"], 1), 3)
        results[name] = res
        line = (f"{name:10s} {res['frames']} frames {res['tufts']} tufts   found {res['found']:.3f}   "
                f"extra/frame {res['extra_per_frame']}   axis err {res['axis_err_median_deg']} deg")
        if has_root:
            line += f"   root found {res['root_found']:.3f}   dir ok {res['dir_ok']:.3f}"
        print(line)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
