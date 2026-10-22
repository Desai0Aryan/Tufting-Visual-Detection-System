"""Visual check of a U-Net checkpoint on arbitrary photos (no labels, no numbers).

    python -m tools.predict_sheet --weights runs/<run>/checkpoints/best.pt \
        --images "Images/Car image Static..jpg" "Images/car image moving.jpg" \
        --rois 600,650,1650,650 740,480,2100,620 --out runs/car_check.jpg --open

Per photo: the whole frame with the predicted mask (green), then a zoom on the
ROI: image | predicted mask | probability map. Runs at native resolution.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.unet import UNetBackend  # noqa: E402


def overlay(img, mask):
    o = img.copy()
    o[mask] = (0.3 * o[mask] + 0.7 * np.array((60, 220, 60))).astype(np.uint8)
    return o


def caption(t, text):
    bar = np.full((24, t.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([bar, t])


def fit(t, w):
    return cv2.resize(t, (w, int(round(t.shape[0] * w / t.shape[1]))), interpolation=cv2.INTER_AREA)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--rois", nargs="*", default=None, help="x,y,w,h per image (zoom region)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--title", default="")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    be = UNetBackend(a.weights, threshold=a.threshold)
    blocks = []
    for k, path in enumerate(a.images):
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        prob = be.predict_proba(img)[0]
        mask = prob >= be.threshold
        blocks.append(caption(fit(overlay(img, mask), a.width),
                              f"{Path(path).name}: whole frame, predicted tuft mask (green), threshold {be.threshold}"))
        if a.rois and k < len(a.rois):
            x, y, w, h = (int(v) for v in a.rois[k].split(","))
            m = 40
            sl = (slice(max(y - m, 0), y + h + m), slice(max(x - m, 0), x + w + m))
            heat = cv2.applyColorMap((prob[sl] * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            tiles = [img[sl], overlay(img, mask)[sl], heat]
            tw = (a.width - 2 * 6) // 3
            tiles = [fit(t, tw) for t in tiles]
            sep = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
            row = np.hstack([tiles[0], sep, tiles[1], sep, tiles[2]])
            blocks.append(caption(cv2.copyMakeBorder(row, 0, 0, 0, a.width - row.shape[1], cv2.BORDER_CONSTANT,
                                                     value=(255, 255, 255)),
                                  "zoom on tufts: image | predicted mask | probability (dark = 0, yellow = 1)"))
        blocks.append(np.full((14, a.width, 3), 255, np.uint8))
    head = caption(np.full((2, a.width, 3), 255, np.uint8), a.title or f"checkpoint {a.weights}")
    cv2.imencode(".jpg", np.vstack([head] + blocks), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(a.out)
    print("wrote", a.out)
    if a.open and hasattr(os, "startfile"):
        os.startfile(str(Path(a.out).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
