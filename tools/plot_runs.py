"""Compare training runs: curves from log.csv and best-checkpoint prediction sheets.

    python -m tools.plot_runs runs/dlr_A_polygons_* runs/dlr_B_polygons_sam2_* --names A B \
        --out runs/dlr_compare --val-data data/dlr/hableg_polygons --open

Writes <out>_curves.png (one panel per measure, one y-axis each) and <out>_preds.jpg
(rows: image | ground truth | prediction of each run; green = hit, red = missed,
blue = false positive) on whole validation frames at native resolution.
Validation numbers are labelled with their domain (e.g. "DLR validation only").
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# reference palette, light mode: categorical slots 1-3 (validated all-pairs in both modes)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_log(run):
    rows = list(csv.DictReader(open(Path(run) / "log.csv", newline="")))
    cols = rows[0].keys()
    return {c: np.array([float(r[c]) if r[c] not in ("", None) else np.nan for r in rows]) for c in cols}


def domain_of(log):
    keys = [k for k in log if k.startswith("val_") and k.endswith("_dice")]
    return keys[0][4:-5] if keys else None


def plot_curves(runs, names, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logs = [read_log(r) for r in runs]
    dom = domain_of(logs[0])
    tag = "" if dom == "own" else f" ({dom.upper()} validation only)"
    panels = [("train_loss", "Training loss", None),
              (f"val_{dom}_loss", f"Validation loss{tag}", None),
              (f"val_{dom}_dice", f"Validation Dice at chosen threshold{tag}", (0, 1)),
              (f"val_{dom}_cldice@0.5", f"Validation clDice at 0.5{tag}", (0, 1))]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURFACE)
    for ax, (key, title, ylim) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        for i, (lg, nm) in enumerate(zip(logs, names)):
            if key not in lg:
                continue
            x, y = lg["epoch"], lg[key]
            ax.plot(x, y, color=SERIES[i], lw=2, label=nm, solid_capstyle="round")
            ax.annotate(nm, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        color=TEXT2, va="center", fontsize=9)
            if key.endswith("_dice") or "cldice" in key:
                b = int(np.nanargmax(y))
                ax.plot(x[b], y[b], "o", ms=8, color=SERIES[i], mec=SURFACE, mew=2)
                ax.annotate(f"{y[b]:.3f}", (x[b], y[b]), xytext=(0, 9), textcoords="offset points",
                            ha="center", color=TEXT2, fontsize=8)
        ax.set_title(title, loc="left", color=TEXT, fontsize=10.5)
        ax.set_xlabel("epoch")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="best", fontsize=9, labelcolor=TEXT2)
    fig.suptitle(" vs ".join(names) + " - " + ", ".join(Path(r).name for r in runs), x=0.01, ha="left",
                 color=TEXT2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def pred_sheet(runs, names, val_data, out, n=4, crop_w=900):
    from src.label_io import load_frame, load_ignore
    from src.labeling import list_samples
    from src.splits import read_meta
    from src.unet import UNetBackend
    meta = read_meta(val_data)
    stems = [s for s in list_samples(val_data) if meta.get(s, {}).get("split") == "val"]
    stems = [stems[int(i)] for i in np.linspace(0, len(stems) - 1, min(n, len(stems)))]
    backends = [UNetBackend(Path(r) / "checkpoints" / "best.pt") for r in runs]
    rows = []
    for stem in stems:
        img, gt, info, _ = load_frame(val_data, stem)
        g = gt > 0
        ys, xs = np.nonzero(g)
        x0, x1 = max(xs.min() - 60, 0), min(xs.max() + 60, img.shape[1])
        y0, y1 = max(ys.min() - 40, 0), min(ys.max() + 40, img.shape[0])
        tiles = [img[y0:y1, x0:x1], np.dstack([gt[y0:y1, x0:x1]] * 3)]
        caps = [stem, "ground truth (human polygons)"]
        for be, nm in zip(backends, names):
            p = be.predict(img).tuft > 0
            o = img.copy()
            for m, c in ((p & g, (60, 200, 60)), (g & ~p, (40, 40, 230)), (p & ~g, (230, 140, 30))):
                o[m] = (0.3 * o[m] + 0.7 * np.array(c)).astype(np.uint8)
            tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (g & ~p).sum()
            tiles.append(o[y0:y1, x0:x1])
            caps.append(f"{nm}: Dice {2 * tp / max(2 * tp + fp + fn, 1):.3f} (thr {be.threshold})")
        s = crop_w / 2 / tiles[0].shape[1]
        tiles = [cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for t in tiles]
        out_tiles = []
        for t, c in zip(tiles, caps):
            bar = np.full((22, t.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, c, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
            out_tiles.append(np.vstack([bar, t]))
        grid = [np.hstack([out_tiles[i], np.full((out_tiles[i].shape[0], 6, 3), 255, np.uint8)])
                for i in range(len(out_tiles))]
        rows.append(np.hstack(grid))
        rows.append(np.full((10, rows[-1].shape[1], 3), 255, np.uint8))
    w = max(r.shape[1] for r in rows)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                       for r in rows])
    legend = np.full((26, w, 3), 255, np.uint8)
    cv2.putText(legend, "green = predicted & labelled   red = labelled, missed   blue = predicted, not labelled"
                        "   (whole validation frames, native resolution; DLR validation only)",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(out, np.vstack([legend, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--val-data", default=None, help="dataset folder with the validation frames")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    runs = [sorted(glob.glob(r))[-1] if any(ch in r for ch in "*?") else r for r in a.runs]
    names = a.names or [Path(r).name for r in runs]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plot_curves(runs, names, a.out + "_curves.png")
    outs = [a.out + "_curves.png"]
    if a.val_data:
        pred_sheet(runs, names, a.val_data, a.out + "_preds.jpg", a.n)
        outs.append(a.out + "_preds.jpg")
    for r, nm in zip(runs, names):
        best = json.loads((Path(r) / "checkpoints" / "best.pt.json").read_text())
        dom = best["selected_on"]
        m = best["best_metrics"][dom]
        tag = "" if dom == "own" else f"  [{dom.upper()} VALIDATION ONLY]"
        print(f"{nm}: best epoch {best['epoch']}  Dice {m['val_dice']:.3f}  IoU {m['val_iou']:.3f}  "
              f"P {m['val_precision']:.3f}  R {m['val_recall']:.3f}  clDice@0.5 {m['val_cldice@0.5']:.3f}  "
              f"thr {m['best_threshold']}{tag}")
    print("wrote", ", ".join(outs))
    if a.open and hasattr(os, "startfile"):
        for o in outs:
            os.startfile(str(Path(o).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
