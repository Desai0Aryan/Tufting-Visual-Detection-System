import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backends import ClassicalBackend  # noqa: E402
from src.image_io import load_image  # noqa: E402
from src.labeling import LABEL_EXTRACT_KW, render_label, save_sample  # noqa: E402
from src.mask_to_tufts import extract_tufts  # noqa: E402
from src.overlay import draw_tufts  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_roi(v):
    x, y, w, h = [int(t) for t in v.split(",")]
    return x, y, w, h


def propose(image, polarity="dark", c=5, roi=None):
    """Strict (Part-1 default) filters for proposals: a bad proposal becomes a
    bad label, so precision matters more than recall here. SAM-refined masks
    are re-extracted with the relaxed LABEL_EXTRACT_KW instead."""
    mask = ClassicalBackend(polarity=polarity, c=c).predict(image).tuft
    if roi is not None:  # only label the instrumented surface, not sky/ground
        x, y, w, h = roi
        keep = np.zeros_like(mask)
        keep[y:y + h, x:x + w] = 255
        mask = cv2.bitwise_and(mask, keep)
    return mask, extract_tufts(mask)


def process_image(path, out, labeler="classical", predictor=None, polarity="dark",
                  c=5, roi=None, tile=1000, min_tufts=3):
    image = load_image(str(path))
    h, w = image.shape[:2]
    mask, tufts = propose(image, polarity, c, roi)
    stats = {"found": len(tufts)}
    if labeler == "sam2":
        from src.sam2_labeler import refine_with_sam
        mask, sam_stats = refine_with_sam(image, tufts, mask, predictor)
        tufts = extract_tufts(mask, **LABEL_EXTRACT_KW)
        stats.update(sam_stats)
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
                          "reviewed": False})
        n_saved += 1
    stats["tiles_saved"] = n_saved
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="image file or folder")
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--labeler", choices=["classical", "sam2"], default="classical")
    ap.add_argument("--polarity", choices=["dark", "light"], default="dark",
                    help="dark tufts on light surface (default) or light on dark")
    ap.add_argument("--c", type=float, default=5,
                    help="threshold constant; raise it (10-20) on high-res or textured photos "
                         "to cut texture false positives, e.g. Images/Tufts.jpg works at --c 15")
    ap.add_argument("--roi", type=parse_roi, default=None,
                    help="x,y,w,h: only label tufts inside this box (skip sky/ground)")
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
        print(f.name, process_image(f, a.out, a.labeler, predictor, a.polarity, a.c, a.roi))
    print(f"\nnext: python tools/review_labels.py --data {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
