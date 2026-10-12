"""Draw the DLR tuft dataset's annotations (Zenodo record 10539950).

    python -m tools.view_dlr                                   # 20 random images -> preview.jpg, opened
    python -m tools.view_dlr --n 30 --seed 3 --mode segmentation
    python -m tools.view_dlr --stem GOPR_0090 --out one.jpg     # one image at full resolution

Layout under --root (data/external/dlr_tufts/tufts):
    detection/<subset>/JPEGImages/*.JPG + Annotations/*.xml   Pascal VOC boxes
    segmentation/<subset>/*.JPG + *.json                      LabelMe polygons
Subsets: HABLEG, EC135-ACT⁄FHS (the folder name contains U+2044, a fraction
slash, so images are read with np.fromfile + cv2.imdecode).

Boxes are drawn with their label (tuft ID such as T14 / A5, or "tuft");
polygons are filled as coloured masks with an outline and their ID. When a
segmentation image also has a detection XML, its boxes are drawn too.
Colours are fixed per tuft ID, so the same tuft keeps its colour across images.
Contact-sheet tiles are zoomed to the annotated region (all labels plus a
margin) so thin tufts stay visible; --full shows whole frames.
"""
import argparse
import json
import os
import random
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

DEFAULT_ROOT = "data/external/dlr_tufts/tufts"


@dataclass
class Item:
    subset: str
    stem: str
    image: Path
    xml: Path = None
    json: Path = None
    boxes: list = field(default_factory=list)      # (label, x0, y0, x1, y1)
    polygons: list = field(default_factory=list)   # (label, Nx2 array)
    warnings: list = field(default_factory=list)


def imread(path):
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img, quality=90):
    ok, enc = cv2.imencode(Path(path).suffix or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"could not encode {path}")
    enc.tofile(str(path))


def color_for(label):
    h = zlib.crc32(label.encode("utf-8"))
    hsv = np.uint8([[[h % 180, 200 + (h >> 8) % 56, 230]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def read_voc(path):
    r = ET.parse(path).getroot()
    size = (int(float(r.findtext("size/width") or 0)), int(float(r.findtext("size/height") or 0)))
    boxes = []
    for o in r.findall("object"):
        b = [float(o.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
        boxes.append((o.findtext("name") or "?", *b))
    return size, boxes


def read_labelme(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    polys = [(s.get("label", "?"), np.asarray(s["points"], np.float32))
             for s in d.get("shapes", []) if s.get("shape_type", "polygon") == "polygon" and len(s["points"]) >= 3]
    return (d.get("imageWidth"), d.get("imageHeight")), d.get("imagePath"), polys


def index(root):
    """All annotated images: detection items and segmentation items."""
    root = Path(root)
    items = {"detection": [], "segmentation": []}
    det_xml = {}
    for sub in sorted(p for p in (root / "detection").iterdir() if p.is_dir()):
        imgs = {p.stem: p for p in (sub / "JPEGImages").iterdir()}
        for x in sorted((sub / "Annotations").glob("*.xml")):
            det_xml[(sub.name, x.stem)] = x
            if x.stem in imgs:
                items["detection"].append(Item(sub.name, x.stem, imgs[x.stem], xml=x))
    for sub in sorted(p for p in (root / "segmentation").iterdir() if p.is_dir()):
        for j in sorted(sub.glob("*.json")):
            img = next((p for p in sub.iterdir() if p.stem == j.stem and p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
            if img is not None:
                items["segmentation"].append(Item(sub.name, j.stem, img, xml=det_xml.get((sub.name, j.stem)), json=j))
    return items


def load(item, image_shape):
    H, W = image_shape[:2]
    if item.xml is not None:
        size, item.boxes = read_voc(item.xml)
        if size != (W, H):
            item.warnings.append(f"xml says {size[0]}x{size[1]}, image is {W}x{H}")
    if item.json is not None:
        size, _, item.polygons = read_labelme(item.json)
        if size != (W, H):
            item.warnings.append(f"json says {size[0]}x{size[1]}, image is {W}x{H}")
    return item


def draw(item, img, scale_hint=1.0):
    """Annotate at full resolution; line widths sized so they survive downscaling."""
    out = img.copy()
    lw = max(1, int(round(1.6 / scale_hint)))
    fs = 0.45 / scale_hint
    if item.polygons:
        fill = out.copy()
        for label, pts in item.polygons:
            cv2.fillPoly(fill, [np.round(pts).astype(np.int32)], color_for(label))
        out = cv2.addWeighted(fill, 0.45, out, 0.55, 0)
        for label, pts in item.polygons:
            p = np.round(pts).astype(np.int32)
            cv2.polylines(out, [p], True, color_for(label), lw, cv2.LINE_AA)
            x, y = p[:, 0].max(), p[:, 1].min()
            _text(out, label, (int(x) + lw, int(y)), fs, color_for(label), lw)
    for label, x0, y0, x1, y1 in item.boxes:
        c = color_for(label)
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), c, lw, cv2.LINE_AA)
        if not item.polygons:  # with polygons the ID is already drawn once
            _text(out, label, (int(x0), int(y0) - 3 * lw), fs, c, lw)
    return out


def _text(img, s, org, fs, color, lw):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lw + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, lw, cv2.LINE_AA)


def label_region(item, shape, margin=0.25, aspect=16 / 9):
    """(x0, y0, x1, y1) around every annotation, padded, at a fixed aspect ratio."""
    pts = [np.array([[b[1], b[2]], [b[3], b[4]]]) for b in item.boxes] + [p for _, p in item.polygons]
    if not pts:
        return 0, 0, shape[1], shape[0]
    P = np.vstack(pts)
    (x0, y0), (x1, y1) = P.min(0), P.max(0)
    w, h = (x1 - x0) * (1 + 2 * margin) + 40, (y1 - y0) * (1 + 2 * margin) + 40
    w, h = max(w, h * aspect), max(h, w / aspect)
    w, h = min(w, shape[1]), min(h, shape[0])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0 = int(np.clip(cx - w / 2, 0, shape[1] - w)); Y0 = int(np.clip(cy - h / 2, 0, shape[0] - h))
    return X0, Y0, int(X0 + w), int(Y0 + h)


def tile(item, tile_w, full=False):
    img = imread(item.image)
    if img is None:
        raise IOError(f"cannot read {item.image}")
    load(item, img.shape)
    x0, y0, x1, y1 = (0, 0, img.shape[1], img.shape[0]) if full else label_region(item, img.shape)
    s = tile_w / (x1 - x0)
    drawn = draw(item, img, s)[y0:y1, x0:x1]
    small = cv2.resize(drawn, (tile_w, int(round((y1 - y0) * s))), interpolation=cv2.INTER_AREA)
    kind = "seg" if item.json is not None else "det"
    cap = (f"{kind} {item.subset} {item.stem}  boxes {len(item.boxes)}  polys {len(item.polygons)}"
           + ("  !" + item.warnings[0] if item.warnings else ""))
    bar = np.zeros((22, tile_w, 3), np.uint8)
    cv2.putText(bar, cap, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if not item.warnings else (0, 140, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, small])


def contact_sheet(tiles, cols, pad=6):
    h = max(t.shape[0] for t in tiles)
    w = tiles[0].shape[1]
    rows = []
    for i in range(0, len(tiles), cols):
        row = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + pad, 0, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
               for t in tiles[i:i + cols]]
        while len(row) < cols:
            row.append(np.full((h + pad, w + pad, 3), 255, np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=None, help="default: <root>/../preview.jpg")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["mixed", "detection", "segmentation"], default="mixed",
                    help="mixed = half segmentation (polygons + boxes), half detection (boxes)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile-w", type=int, default=560)
    ap.add_argument("--stem", default=None, help="draw one image at full resolution")
    ap.add_argument("--full", action="store_true", help="whole frames instead of zooming to the labels")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    items = index(a.root)
    out = Path(a.out) if a.out else Path(a.root).parent / "preview.jpg"
    if a.stem:
        hits = [i for i in items["segmentation"] + items["detection"] if i.stem == a.stem]
        if not hits:
            raise SystemExit(f"no annotated image named {a.stem}")
        it = hits[0]
        img = imread(it.image)
        load(it, img.shape)
        imwrite(out, draw(it, img))
        print(f"{it.subset}/{it.stem}: {len(it.boxes)} boxes, {len(it.polygons)} polygons {it.warnings} -> {out}")
    else:
        rng = random.Random(a.seed)
        if a.mode == "mixed":
            k = a.n // 2
            pick = rng.sample(items["segmentation"], k) + rng.sample(items["detection"], a.n - k)
        else:
            pick = rng.sample(items[a.mode], a.n)
        tiles = [tile(it, a.tile_w, a.full) for it in pick]
        imwrite(out, contact_sheet(tiles, a.cols))
        for it in pick:
            print(f"{'seg' if it.json else 'det'} {it.subset:15s} {it.stem:12s} boxes {len(it.boxes):3d} "
                  f"polys {len(it.polygons):3d} {'; '.join(it.warnings)}")
        print(f"contact sheet ({len(pick)} images) -> {out}")
    if not a.no_open and hasattr(os, "startfile"):
        os.startfile(str(out.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
