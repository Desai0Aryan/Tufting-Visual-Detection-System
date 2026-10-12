"""Look for tufts the DLR boxes missed (EC135 by default).

    python -m tools.check_unlabeled --n 20 --out data/external/dlr_tufts/ec135_unlabeled_check

Picks frames spread evenly over the capture sequence, zooms on the Fenestron
duct (circle found with a Hough transform; falls back to the box region), draws
every box (orange, with its ID) and a lettered grid (A-H across, 1-6 down) so a
location can be named consistently from frame to frame.
--candidates also marks dark thin strings outside every box in magenta; on
EC135 that mostly catches lettering, fin edges and stator blades, so it is off
by default and the sheet is meant to be judged by eye.
Writes <out>_1.jpg, <out>_2.jpg (10 frames each) and <out>.csv (one row per mark).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backends import ClassicalBackend  # noqa: E402
from src.mask_to_tufts import extract_tufts  # noqa: E402
from tools.view_dlr import imread, imwrite, index, load  # noqa: E402

MAGENTA, ORANGE = (255, 0, 255), (0, 165, 255)


def candidates_outside(img, boxes, region, pad=6, inside_frac=0.3):
    x0, y0, x1, y1 = region
    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = ClassicalBackend(31, 10, polarity="dark").threshold(gray)
    tufts = extract_tufts(mask, offset=(x0, y0), min_area=40, max_area=6000, min_elongation=3.0,
                          min_length_px=25, max_length_px=300)
    boxed = np.zeros(img.shape[:2], np.uint8)
    for _, bx0, by0, bx1, by1 in boxes:
        cv2.rectangle(boxed, (int(bx0) - pad, int(by0) - pad), (int(bx1) + pad, int(by1) + pad), 255, -1)
    out = []
    for t in tufts:
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(m, [t.contour.astype(np.int32)], -1, 255, -1)
        frac = float(((m > 0) & (boxed > 0)).sum()) / max(int((m > 0).sum()), 1)
        if frac < inside_frac:
            out.append(t)
    return out


def duct_region(img, boxes, margin=1.45):
    """Square around the Fenestron duct: the largest Hough circle near the box cloud."""
    B = np.array([b[1:] for b in boxes], float)
    bx0, by0 = B[:, :2].min(0)
    bx1, by1 = B[:, 2:].max(0)
    H, W = img.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    span = max(bx1 - bx0, by1 - by0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=2, minDist=span, param1=120, param2=60,
                               minRadius=int(span * 0.12), maxRadius=int(span * 0.45))
    if circles is not None:
        cx, cy, r = max(circles[0], key=lambda c: c[2])
        if bx0 <= cx <= bx1 and by0 <= cy <= by1:
            half = r * margin
            return (int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, W)), int(min(cy + half, H))), True
    mx, my = (bx1 - bx0) * 0.25, (by1 - by0) * 0.25
    return (int(max(bx0 - mx, 0)), int(max(by0 - my, 0)), int(min(bx1 + mx, W)), int(min(by1 + my, H))), False


def tile(item, width=760, show_candidates=False):
    img = imread(item.image)
    load(item, img.shape)
    B = np.array([b[1:] for b in item.boxes], float)
    bx0, by0 = B[:, :2].min(0)
    bx1, by1 = B[:, 2:].max(0)
    region, found = duct_region(img, item.boxes)
    cands = candidates_outside(img, item.boxes, region) if show_candidates else []
    out = img.copy()
    for lab, x0, y0, x1, y1 in item.boxes:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), ORANGE, 2)
        cv2.putText(out, lab, (int(x0), int(y0) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv2.LINE_AA)
    for k, t in enumerate(cands, 1):
        c = np.round(t.path[len(t.path) // 2]).astype(int) if t.path is not None else np.round(t.root).astype(int)
        cv2.polylines(out, [np.round(t.path).astype(np.int32).reshape(-1, 1, 2)], False, MAGENTA, 3)
        cv2.circle(out, tuple(c), int(max(18, t.length_px * 0.6)), MAGENTA, 2)
        cv2.putText(out, f"?{k}", (c[0] + 10, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, MAGENTA, 2, cv2.LINE_AA)
    x0, y0, x1, y1 = region
    t = out[y0:y1, x0:x1]
    s = width / t.shape[1]
    t = cv2.resize(t, (width, int(t.shape[0] * s)), interpolation=cv2.INTER_AREA)
    for k in range(1, 8):  # lettered grid: columns A-H, rows 1-6
        x = int(width * k / 8)
        cv2.line(t, (x, 0), (x, t.shape[0]), (200, 200, 200), 1)
    for k in range(1, 6):
        y = int(t.shape[0] * k / 6)
        cv2.line(t, (0, y), (width, y), (200, 200, 200), 1)
    for k in range(8):
        cv2.putText(t, "ABCDEFGH"[k], (int(width * (k + 0.45) / 8), 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    for k in range(6):
        cv2.putText(t, str(k + 1), (3, int(t.shape[0] * (k + 0.55) / 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cap = np.full((24, width, 3), 255, np.uint8)
    note = "duct crop" if found else "box-region crop (no duct circle found)"
    cv2.putText(cap, f"{item.stem}: {len(item.boxes)} boxes, {note}"
                     + (f", {len(cands)} magenta candidates" if show_candidates else ""),
                (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    rows = [{"stem": item.stem, "mark": f"?{k}", "x": round(float(t_.path[len(t_.path) // 2][0]), 1),
             "y": round(float(t_.path[len(t_.path) // 2][1]), 1), "length_px": round(t_.length_px, 1),
             # position relative to the box cloud, so recurring spots can be compared across frames
             "rel_x": round((float(t_.path[len(t_.path) // 2][0]) - bx0) / max(bx1 - bx0, 1), 3),
             "rel_y": round((float(t_.path[len(t_.path) // 2][1]) - by0) / max(by1 - by0, 1), 3)}
            for k, t_ in enumerate(cands, 1)]
    return np.vstack([cap, t]), rows


def grid(tiles, cols=2, gap=8):
    h = max(t.shape[0] for t in tiles)
    w = tiles[0].shape[1]
    rows = []
    for i in range(0, len(tiles), cols):
        r = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + gap, 0, gap, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles[i:i + cols]]
        while len(r) < cols:
            r.append(np.full((h + gap, w + gap, 3), 255, np.uint8))
        rows.append(np.hstack(r))
    return np.vstack(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subset", default="EC135")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/external/dlr_tufts/ec135_unlabeled_check")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--candidates", action="store_true", help="also mark dark strings outside boxes")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    items = [i for i in index("data/external/dlr_tufts/tufts")["detection"] if i.subset.startswith(a.subset)]
    items.sort(key=lambda i: int(re.sub(r"\D", "", i.stem)))
    pick = [items[int(k)] for k in np.linspace(0, len(items) - 1, a.n)]
    tiles, rows = [], []
    for it in pick:
        t, r = tile(it, show_candidates=a.candidates)
        tiles.append(t)
        rows += r
        print(f"{it.stem}: boxes {len(it.boxes)}" + (f", candidates {len(r)}" if a.candidates else ""))
    half = (len(tiles) + 1) // 2
    outs = []
    for k, part in enumerate((tiles[:half], tiles[half:]), 1):
        p = f"{a.out}_{k}.jpg"
        legend = np.full((30, part[0].shape[1] * 2 + 16, 3), 255, np.uint8)
        cv2.putText(legend, "orange = DLR box with its ID (or 'tuft'); grid A-H x 1-6 names locations; "
                            "look for strings with no box on the duct ring, stator blades and hub",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        g = grid(part)
        legend = cv2.resize(legend, (g.shape[1], 30))
        imwrite(p, np.vstack([legend, g]))
        outs.append(p)
    with open(f"{a.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "mark", "x", "y", "length_px", "rel_x", "rel_y"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", ", ".join(outs), "and", f"{a.out}.csv")
    if a.open:
        import os
        for p in outs:
            os.startfile(str(Path(p).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
