"""Score checkpoints on the DLR validation frames. DLR VALIDATION ONLY: these
numbers say nothing about our own footage.

    python -m tools.eval_dlr --weights A360=runs/.../best.pt EC=runs/.../best.pt --out runs/eval_dlr.json

Whole frames (no resizing), each checkpoint's own threshold (from best.pt.json),
pixels in ignore/<stem>.png excluded. Sets:
  hableg_gp03          HABLEG GP03, human polygons
  ec135_val_polygons   EC135 val blocks, human outlines (band ignored)
  ec135_val_sam        EC135 val blocks, SAM 2 masks from DLR boxes (band, unsegmented and
                       shape-flagged boxes ignored): approximate labels, more frames
Metrics are pooled over all pixels of a set; per-frame Dice is reported as a median.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.unet import UNetBackend  # noqa: E402

SETS = {
    "hableg_gp03": "data/dlr/hableg_polygons",
    "ec135_val_polygons": "data/dlr/ec135_polygons",
    "ec135_val_sam": "data/dlr/ec135_sam2/proposed",
}


def val_stems(root):
    meta = json.loads((Path(root) / "meta.json").read_text())
    return sorted(s for s, m in meta.items() if m.get("split") == "val")


def score(be, root, stems):
    tp = fp = fn = 0
    per = []
    for s in stems:
        img = cv2.imread(str(Path(root) / "images" / f"{s}.png"))
        gt = cv2.imread(str(Path(root) / "labels" / f"{s}.png"))[..., 0] > 127
        ip = Path(root) / "ignore" / f"{s}.png"
        keep = ~(cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE) > 127) if ip.exists() else np.ones_like(gt)
        pr = be.predict_proba(img)[0] >= be.threshold
        a, b, c = int((pr & gt & keep).sum()), int((pr & ~gt & keep).sum()), int((~pr & gt & keep).sum())
        tp, fp, fn = tp + a, fp + b, fn + c
        per.append(2 * a / max(2 * a + b + c, 1))
    return {"frames": len(stems), "dice": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            "iou": round(tp / max(tp + fp + fn, 1), 4), "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4), "frame_dice_median": round(float(np.median(per)), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {"note": "DLR VALIDATION ONLY", "results": {}}
    for spec in a.weights:
        name, path = spec.split("=", 1)
        be = UNetBackend(path, tile=a.tile)
        res["results"][name] = {"weights": path, "threshold": be.threshold}
        for s in a.sets:
            r = score(be, SETS[s], val_stems(SETS[s]))
            res["results"][name][s] = r
            print(f"{name:10s} {s:20s} " + "  ".join(f"{k} {v}" for k, v in r.items()))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print("DLR VALIDATION ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
