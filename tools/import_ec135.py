"""EC135 (DLR tuft dataset) -> training data, with a duct-ring ignore band.

    python -m tools.import_ec135 fit                  # fit the duct on every frame -> data/dlr/ec135_fits.json
    python -m tools.import_ec135 sheet                # contact sheets: accepted fits with the band; failed outline frames
    python -m tools.import_ec135 polygons [--keep-failed DSC_3538 ...]
    python -m tools.import_ec135 boxes                # SAM 2 masks from boxes (accepted fits only)
    python -m tools.import_dlr clean --data data/dlr/ec135_sam2/proposed   # glare / width test
    python -m tools.import_ec135 detag                # tape tags in SAM masks -> ignore
    python -m tools.import_dlr promote --src data/dlr/ec135_sam2/proposed --out data/dlr/ec135_sam2/labeled

Why the band: tufts on the inner surface of the duct ring are often not boxed.
Pixels in the band 0.8-1.0 x the fitted duct ellipse get zero loss, except the
boxed / outlined tufts themselves, which stay labelled. Everything else
(stator blades, hub, the pitch-control linkage) stays labelled background.

Frames without a trustworthy duct fit (src/duct_fit.fit_and_check) are dropped.
Split by contiguous DSC numbers: three 49-number val blocks centred on 2765,
3085, 3385, with 20-number gaps on both sides that are dropped; the rest is train.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.duct_fit import band_mask, fit_and_check  # noqa: E402
from tools.import_dlr import rasterize, resolve_duplicates, write_frame  # noqa: E402
from tools.view_dlr import imread, imwrite, index, load, read_labelme, read_voc  # noqa: E402

FITS = Path("data/dlr/ec135_fits.json")
VAL_CENTRES, VAL_HALF, GAP = (2765, 3085, 3385), 24, 20
MIN_OCTANTS = 6
SUBSET = "EC135-ACT⁄FHS"


def number(stem):
    return int(re.sub(r"\D", "", stem))


def split_of(stem):
    n = number(stem)
    for c in VAL_CENTRES:
        if c - VAL_HALF <= n <= c + VAL_HALF:
            return "val"
        if c - VAL_HALF - GAP <= n < c - VAL_HALF or c + VAL_HALF < n <= c + VAL_HALF + GAP:
            return "gap"
    return "train"


def frames():
    """{stem: {"image": Path, "xml": Path|None, "json": Path|None}} for every EC135 frame."""
    ix = index("data/external/dlr_tufts/tufts")
    out = {}
    for it in ix["detection"]:
        if it.subset == SUBSET:
            out[it.stem] = {"image": it.image, "xml": it.xml, "json": None}
    for it in ix["segmentation"]:
        if it.subset == SUBSET:
            out.setdefault(it.stem, {"image": it.image, "xml": None, "json": None})["json"] = it.json
            out[it.stem]["seg_image"] = it.image
    return dict(sorted(out.items(), key=lambda kv: number(kv[0])))


def boxes_for(f):
    """Dataset boxes, or the polygons' bounding boxes when a frame has outlines only."""
    if f["xml"] is not None:
        return read_voc(f["xml"])[1]
    _, _, polys = read_labelme(f["json"])
    return [(l, *p.min(0), *p.max(0)) for l, p in polys]


def ignore_for(shape, ellipse, keep):
    """Band minus the labelled tufts (keep: uint8 mask of boxes / outlines)."""
    band = band_mask(shape, ellipse, 0.8, 1.0)
    band[keep > 0] = 0
    return band


# ---------------------------------------------------------------- fit
def cmd_fit(a):
    fr = frames()
    res = {}
    for k, (stem, f) in enumerate(fr.items()):
        img = imread(f.get("seg_image") or f["image"])
        r = fit_and_check(img, boxes_for(f), MIN_OCTANTS)
        res[stem] = {"ok": r["ok"], "reason": r["reason"], "octants": r["octants"],
                     "ellipse": None if r["ellipse"] is None else
                     [list(map(float, r["ellipse"][0])), list(map(float, r["ellipse"][1])), float(r["ellipse"][2])],
                     "has_boxes": f["xml"] is not None, "has_outlines": f["json"] is not None, "split": split_of(stem)}
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(fr)}")
    FITS.parent.mkdir(parents=True, exist_ok=True)
    FITS.write_text(json.dumps(res, indent=1))
    c = collections.Counter((v["ok"], v["reason"].split(" ")[0]) for v in res.values())
    print(f"{len(res)} EC135 frames: fit accepted {sum(v['ok'] for v in res.values())}, "
          f"dropped {sum(not v['ok'] for v in res.values())} ({dict(c)})")
    return 0


# ---------------------------------------------------------------- sheets
def _thumb(stem, f, rec, w=300):
    img = imread(f.get("seg_image") or f["image"])
    o = img.copy()
    if rec["ellipse"] is not None:
        e = (tuple(rec["ellipse"][0]), tuple(rec["ellipse"][1]), rec["ellipse"][2])
        b = band_mask(img.shape, e) > 0
        o[b] = (0.55 * o[b] + 0.45 * np.array((255, 120, 0))).astype(np.uint8)
        cx, cy = e[0]
        half = max(e[1]) * 1.1
    else:
        B = np.array([b[1:] for b in boxes_for(f)])
        cx, cy = B[:, [0, 2]].mean(), B[:, [1, 3]].mean()
        half = max(np.ptp(B[:, [0, 2]]), np.ptp(B[:, [1, 3]])) * 0.6
    if f["json"] is not None:
        _, _, polys = read_labelme(f["json"])
        for _, p in polys:
            cv2.polylines(o, [np.round(p).astype(np.int32)], True, (60, 220, 60), 3)
    for _, x0, y0, x1, y1 in (read_voc(f["xml"])[1] if f["xml"] else []):
        cv2.rectangle(o, (int(x0), int(y0)), (int(x1), int(y1)), (0, 165, 255), 3)
    H, W = img.shape[:2]
    t = o[int(max(cy - half, 0)):int(min(cy + half, H)), int(max(cx - half, 0)):int(min(cx + half, W))]
    t = cv2.resize(t, (w, w))
    cap = f"{stem} {rec['split']}" + ("" if rec["ok"] else " FAIL")
    cv2.putText(t, cap, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(t, cap, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def _grid(tiles, cols):
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def cmd_sheet(a):
    fits = json.loads(FITS.read_text())
    fr = frames()
    acc = [s for s, v in fits.items() if v["ok"] and v["split"] != "gap"]
    per = a.per_sheet
    outs = []
    for k in range(0, len(acc), per):
        tiles = [_thumb(s, fr[s], fits[s], a.tile) for s in acc[k:k + per]]
        p = f"data/dlr/ec135_fits_accepted_{k // per + 1}.jpg"
        imwrite(p, _grid(tiles, a.cols))
        outs.append(p)
    failed_outline = [s for s, v in fits.items() if not v["ok"] and v["has_outlines"] and v["split"] != "gap"]
    if failed_outline:
        tiles = [_thumb(s, fr[s], fits[s], 420) for s in failed_outline]
        p = "data/dlr/ec135_outline_frames_failed_fit.jpg"
        imwrite(p, _grid(tiles, 4))
        outs.append(p)
    print(f"accepted fits drawn: {len(acc)} -> {outs}")
    print(f"outline frames that failed the fit ({len(failed_outline)}): {failed_outline}")
    return 0


# ---------------------------------------------------------------- labels
def _meta(stem, source, extra=None):
    split = split_of(stem)
    return {"clip_id": f"dlr_ec135_{split}", "frame_idx": number(stem), "split": split, "use": "tufts",
            "domain": "dlr_ec135", "dataset": "dlr_tufts", "subset": "EC135", "source": source,
            **(extra or {})}


def cmd_polygons(a):
    fits, fr = json.loads(FITS.read_text()), frames()
    keep_failed = set(a.keep_failed or [])
    n = collections.Counter()
    for stem, f in fr.items():
        if f["json"] is None:
            continue
        rec = fits[stem]
        if rec["split"] == "gap":
            n["dropped: gap"] += 1
            continue
        if not rec["ok"] and stem not in keep_failed:
            n["dropped: no trustworthy fit, unboxed ring tufts visible"] += 1
            continue
        img = imread(f["seg_image"])
        _, _, polys = read_labelme(f["json"])
        groups, _ = resolve_duplicates(polys, img.shape)
        mask = np.zeros(img.shape[:2], np.uint8)
        ids = np.full(img.shape[:2], "", object)
        for name, ps in groups:
            m = rasterize([(name, q) for q in ps], img.shape)
            mask |= m
            ids[m > 0] = name
        ignore = None
        if rec["ok"]:
            e = (tuple(rec["ellipse"][0]), tuple(rec["ellipse"][1]), rec["ellipse"][2])
            ignore = ignore_for(img.shape, e, cv2.dilate(mask, np.ones((7, 7), np.uint8)))
        write_frame(a.out, "EC135", stem, img, mask,
                    _meta(stem, "dlr_polygon", {"status": "accepted", "reviewed": True,
                                                "band": bool(rec["ok"])}), ignore=ignore, ids=ids)
        n[f"kept: {rec['split']}" + ("" if rec["ok"] else " (fit failed, no unboxed ring tufts)")] += 1
    print(dict(n))
    return 0


def cmd_boxes(a):
    import torch
    from src.sam2_autolabel import load_image_predictor
    from tools.import_dlr import sam_from_boxes
    fits, fr = json.loads(FITS.read_text()), frames()
    pred = load_image_predictor()
    todo = [s for s, f in fr.items() if f["xml"] is not None and f["json"] is None
            and fits[s]["ok"] and fits[s]["split"] != "gap"]
    if a.limit:
        todo = todo[:a.limit]
    n_rej = 0
    for k, stem in enumerate(todo):
        f, rec = fr[stem], fits[stem]
        img = imread(f["image"])
        boxes = read_voc(f["xml"])[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m, ids, recs = sam_from_boxes(img, boxes, pred)
        keep = np.zeros(img.shape[:2], np.uint8)
        for _, x0, y0, x1, y1 in boxes:
            cv2.rectangle(keep, (int(x0) - 4, int(y0) - 4), (int(x1) + 4, int(y1) + 4), 255, -1)
        e = (tuple(rec["ellipse"][0]), tuple(rec["ellipse"][1]), rec["ellipse"][2])
        ignore = ignore_for(img.shape, e, keep)
        for r in recs:  # a box SAM could not segment still holds a tuft: no loss there either
            if not r["accepted"]:
                x0, y0, x1, y1 = (int(round(v)) for v in r["box"])
                ignore[max(y0 - 4, 0):y1 + 5, max(x0 - 4, 0):x1 + 5] = 255
        write_frame(a.out, "EC135", stem, img, m,
                    _meta(stem, "sam2_box", {"status": "proposed", "reviewed": False, "band": True,
                                             "sam_rejected_boxes": sum(not r["accepted"] for r in recs)}),
                    ignore=ignore, ids=ids)
        info_p = Path(a.out) / "json" / f"{stem}.json"
        info = json.loads(info_p.read_text())
        info["guide_boxes"] = recs
        info_p.write_text(json.dumps(info, indent=1))
        n_rej += sum(not r["accepted"] for r in recs)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(todo)}")
    print(f"SAM 2 masks for {len(todo)} frames; boxes SAM could not segment: {n_rej}")
    return 0


def thick_parts(mask, width=13):
    """Mask parts at least `width` px wide (a disc of that diameter fits)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), k) & mask


def cmd_detag(a):
    """On EC135 SAM 2 often includes the dark tape tag at the tuft root. Human outlines
    never do and are never >= 11 px wide; parts >= --width px are cleared and ignored."""
    from src.label_io import tuft_records, tufts_from_mask
    D = Path(a.data)
    meta = json.loads((D / "meta.json").read_text())
    tot = rem = 0
    for stem in sorted(meta):
        lab_p, ign_p = D / "labels" / f"{stem}.png", D / "ignore" / f"{stem}.png"
        lab = cv2.imread(str(lab_p))
        m = (lab[..., 0] > 127).astype(np.uint8)
        t = thick_parts(m, a.width)
        tot, rem = tot + int(m.sum()), rem + int(t.sum())
        if not t.any():
            continue
        t = cv2.dilate(t, np.ones((5, 5), np.uint8)) > 0
        lab[t, 0] = 0
        lab[t, 1] = 0
        ign = cv2.imread(str(ign_p), cv2.IMREAD_GRAYSCALE) if ign_p.exists() else np.zeros(m.shape, np.uint8)
        ign[t] = 255
        cv2.imwrite(str(lab_p), lab)
        cv2.imwrite(str(ign_p), ign)
        info_p = D / "json" / f"{stem}.json"
        info = json.loads(info_p.read_text())
        mask = (lab[..., 0] > 127).astype(np.uint8) * 255
        info["tufts"] = tuft_records(tufts_from_mask(mask) if mask.any() else [])
        info_p.write_text(json.dumps(info, indent=1))
        meta[stem]["detag_px"] = int(t.sum())
    (D / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(meta)} frames: {rem / max(tot, 1):.1%} of tuft pixels were >= {a.width} px wide -> ignored")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit").set_defaults(fn=cmd_fit)
    p = sub.add_parser("sheet")
    p.add_argument("--per-sheet", type=int, default=96)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--tile", type=int, default=240)
    p.set_defaults(fn=cmd_sheet)
    p = sub.add_parser("polygons")
    p.add_argument("--out", default="data/dlr/ec135_polygons")
    p.add_argument("--keep-failed", nargs="*", default=[],
                   help="outline frames whose fit failed but that show no unboxed ring tufts")
    p.set_defaults(fn=cmd_polygons)
    p = sub.add_parser("boxes")
    p.add_argument("--out", default="data/dlr/ec135_sam2/proposed")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_boxes)
    p = sub.add_parser("detag")
    p.add_argument("--data", default="data/dlr/ec135_sam2/proposed")
    p.add_argument("--width", type=int, default=13)
    p.set_defaults(fn=cmd_detag)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
