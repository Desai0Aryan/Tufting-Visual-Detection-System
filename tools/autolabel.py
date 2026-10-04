import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backends import ClassicalBackend  # noqa: E402
from src.image_io import load_image  # noqa: E402
from src.filtering import (DEFAULT_NORM_SIZE, clean_tufts, normalize_scale, polygon_mask,  # noqa: E402
                           scale_points, scale_roi)
from src.labeling import STRICT_LABEL_EXTRACT_KW, render_label, save_sample  # noqa: E402
from src.mask_to_tufts import extract_tufts  # noqa: E402
from src.overlay import draw_tufts  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_roi(v):
    x, y, w, h = [int(t) for t in v.split(",")]
    return x, y, w, h


def parse_polygon(v):
    """'x,y;x,y;x,y;...' -> [(x, y), ...] (original-image pixels)"""
    pts = [tuple(float(t) for t in p.split(",")) for p in v.split(";") if p.strip()]
    if len(pts) < 3 or any(len(p) != 2 for p in pts):
        raise argparse.ArgumentTypeError("--roi-polygon needs >= 3 points as x,y;x,y;x,y")
    return pts


def select_polygon(image, title="Click around the instrumented surface - Enter = done, right-click = undo, Esc = whole image",
                   max_display=1200):
    """Interactive polygon ROI. Returns [(x, y), ...] in image pixels, or None."""
    h, w = image.shape[:2]
    disp_scale = min(1.0, max_display / float(max(h, w)))
    base = cv2.resize(image, None, fx=disp_scale, fy=disp_scale) if disp_scale < 1.0 else image.copy()
    pts = []

    def redraw():
        canvas = base.copy()
        for p in pts:
            cv2.circle(canvas, p, 3, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, np.int32).reshape(-1, 1, 2)], len(pts) > 2, (0, 255, 0), 2)
        cv2.imshow(title, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
        redraw()

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    redraw()
    result = None
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10):  # Enter
            if len(pts) >= 3:
                result = [(x / disp_scale, y / disp_scale) for x, y in pts]
                break
        elif k == 27:  # Esc
            break
    cv2.destroyAllWindows()
    return result


def _propose_one(image, polarity, c, roi, polygon=None):
    mask = ClassicalBackend(polarity=polarity, c=c).predict(image).tuft
    keep = None
    if polygon is not None:  # only label the instrumented surface, not sky/ground/garage floor
        keep = polygon_mask(image.shape, polygon)
    elif roi is not None:
        x, y, w, h = roi
        keep = np.zeros_like(mask)
        keep[y:y + h, x:x + w] = 255
    if keep is not None:
        mask = cv2.bitwise_and(mask, keep)
    return mask, extract_tufts(mask)


def propose(image, polarity="dark", c=5, roi=None, field_filter=True, polygon=None):
    """Strict (Part-1 default) filters for proposals: a bad proposal becomes a
    bad label, so precision matters more than recall here.

    polarity="auto" tries dark and light strings and keeps whichever yields more
    tufts *inside a coherent tuft field* (so a texture-rich photo cannot win
    just by producing lots of clutter)."""
    polarities = ("dark", "light") if polarity == "auto" else (polarity,)
    best = None
    for pol in polarities:
        mask, tufts = _propose_one(image, pol, c, roi, polygon)
        if field_filter:
            tufts = clean_tufts(image, tufts)
        if best is None or len(tufts) > len(best[1]):
            best = (mask, tufts, pol)
    mask, tufts, _ = best
    return mask, tufts


def process_image(path, out, labeler="classical", predictor=None, polarity="dark",
                  c=5, roi=None, tile=1000, min_tufts=3, norm_size=DEFAULT_NORM_SIZE,
                  field_filter=True, strict_sam=True, polygon=None, select_roi=False):
    image = load_image(str(path))
    # canonical scale: the size limits, the SAM prompts and the U-Net crops all assume
    # tufts are ~15-110 px long, which a 4000 px photo violates by 3x
    image, scale = normalize_scale(image, norm_size)
    roi = scale_roi(roi, scale)
    polygon = scale_points(polygon, scale) if polygon is not None else None
    if select_roi:  # per-image polygon, drawn on the (already normalised) image
        polygon = select_polygon(image) or polygon
    h, w = image.shape[:2]
    mask, tufts = propose(image, polarity, c, roi, field_filter, polygon)
    stats = {"found": len(tufts), "scale": round(scale, 3)}
    if labeler == "sam2":
        from src.sam2_labeler import refine_with_sam_detailed
        mask, sam_stats, _ = refine_with_sam_detailed(image, tufts, mask, predictor,
                                                      strict=strict_sam)
        tufts = extract_tufts(mask, **STRICT_LABEL_EXTRACT_KW)
        if field_filter:
            tufts = clean_tufts(image, tufts)
        stats.update(sam_stats)
        stats["labelled"] = len(tufts)
    label = render_label(image.shape, tufts)

    out = Path(out)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "previews" / f"{path.stem}.jpg"), draw_tufts(image, tufts, show_ids=False))

    # split big photos into tiles, drop tiles with (almost) no tufts
    ys = list(range(0, max(h - tile, 0) + 1, tile - 100)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, tile - 100)) or [0]
    if h > tile and ys[-1] + tile < h:
        ys.append(h - tile)
    if w > tile and xs[-1] + tile < w:
        xs.append(w - tile)
    n_saved = 0
    for k, (y0, x0) in enumerate((y, x) for y in ys for x in xs):
        li = label[y0:y0 + tile, x0:x0 + tile]
        if (li[..., 0] > 127).sum() < min_tufts * 15:
            continue
        stem = path.stem if len(ys) * len(xs) == 1 else f"{path.stem}__t{k:02d}"
        save_sample(out, stem, image[y0:y0 + tile, x0:x0 + tile], li,
                    meta={"source": labeler, "origin": [x0, y0], "file": path.name,
                          "scale": round(scale, 4), "reviewed": False})
        n_saved += 1
    stats["tiles_saved"] = n_saved
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="image file or folder")
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--labeler", choices=["classical", "sam2"], default="classical")
    ap.add_argument("--polarity", choices=["dark", "light", "auto"], default="dark",
                    help="dark tufts on light surface (default), light on dark, or auto-pick")
    ap.add_argument("--c", type=float, default=5,
                    help="threshold constant; raise it (10-20) on high-res or textured photos "
                         "to cut texture false positives, e.g. Images/Tufts.jpg works at --c 15")
    ap.add_argument("--roi", type=parse_roi, default=None,
                    help="x,y,w,h: only label tufts inside this box (skip sky/ground)")
    ap.add_argument("--norm-size", type=int, default=DEFAULT_NORM_SIZE,
                    help="downscale photos so the long side is at most this (0 = off). "
                         "Use the same value for training data and main.py inference")
    ap.add_argument("--no-field-filter", action="store_true",
                    help="skip the 'must be inside the tuft field' false-positive filter")
    ap.add_argument("--keep-sam-disagreements", action="store_true",
                    help="old behaviour: if SAM 2 cannot confirm a candidate keep it anyway")
    ap.add_argument("--select-roi", action="store_true",
                    help="draw a polygon around the instrumented surface for each image "
                         "(everything outside it is labelled as background)")
    ap.add_argument("--roi-polygon", type=parse_polygon, default=None,
                    help="non-interactive polygon 'x,y;x,y;x,y;...' in original-image pixels")
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--sam2-config", default=None)
    ap.add_argument("--sam2-checkpoint", default=None)
    a = ap.parse_args(argv)

    src = Path(a.images)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print(f"no images found in {src}")
        return 1
    predictor = None
    if a.labeler == "sam2":
        from src.sam2_labeler import load_sam2_predictor
        predictor = load_sam2_predictor(a.sam2_model, config=a.sam2_config, checkpoint=a.sam2_checkpoint)
    for f in files:
        print(f.name, process_image(f, a.out, a.labeler, predictor, a.polarity, a.c, a.roi,
                                    norm_size=a.norm_size, field_filter=not a.no_field_filter,
                                    strict_sam=not a.keep_sam_disagreements,
                                    polygon=a.roi_polygon, select_roi=a.select_roi))
    print(f"\nnext: python tools/review_labels.py --data {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
