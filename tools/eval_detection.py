import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector import TuftDetector  # noqa: E402
from src.backends import build_backend  # noqa: E402
from src.filtering import clean_tufts  # noqa: E402
from src.labeling import decode_label, list_samples, load_sample, split_samples  # noqa: E402


def score_image(tufts, gt_mask, min_overlap=0.5, min_gt_px=12, dilate=2):
    """Returns (tp_det, n_det, hit_gt, n_gt)."""
    gt = (gt_mask > 0).astype(np.uint8)
    n_gt, cc = cv2.connectedComponents(gt, connectivity=8)
    gt_sizes = {i: int((cc == i).sum()) for i in range(1, n_gt)}
    gt_ids = [i for i, n in gt_sizes.items() if n >= min_gt_px]
    near = cv2.dilate(gt, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)) > 0
    hit = set()
    tp = 0
    for t in tufts:
        body = np.zeros(gt.shape, np.uint8)
        cv2.drawContours(body, [t.contour.astype(np.int32)], -1, 1, cv2.FILLED)
        area = int(body.sum())
        if area == 0:
            continue
        if (body.astype(bool) & near).sum() / area >= min_overlap:
            tp += 1
        for i in np.unique(cc[body.astype(bool)]):
            if i in gt_sizes and i in gt_ids and ((cc == i) & body.astype(bool)).sum() >= 0.3 * gt_sizes[i]:
                hit.add(int(i))
    return tp, len(tufts), len(hit), len(gt_ids)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="reviewed dataset folder(s)")
    ap.add_argument("--backend", choices=["classical", "unet"], default="classical")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--polarity", choices=["dark", "light"], default="dark")
    ap.add_argument("--no-clean", action="store_true", help="skip the tuft-field filter")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--val-percent", type=int, default=20)
    ap.add_argument("--per-image", action="store_true")
    a = ap.parse_args(argv)

    if a.backend == "unet":
        if not a.weights:
            ap.error("--backend unet needs --weights")
        backend = build_backend("unet", weights=a.weights, threshold=a.threshold)
        print(f"U-Net threshold: {backend.threshold:.2f}")
    else:
        backend = build_backend("classical", polarity=a.polarity)
    detector = TuftDetector(backend=backend)

    tp = n_det = hit = n_gt = 0
    n_images = 0
    for root in a.data:
        stems = list_samples(root)
        split = split_samples(stems, a.val_percent)
        chosen = stems if a.split == "all" else split[a.split]
        for stem in chosen:
            img, lab = load_sample(root, stem)
            tuft_gt, _, _ = decode_label(lab)
            tufts = detector.detect(img)
            if not a.no_clean:
                tufts = clean_tufts(img, tufts)
            t, d, h, g = score_image(tufts, tuft_gt)
            tp, n_det, hit, n_gt = tp + t, n_det + d, hit + h, n_gt + g
            n_images += 1
            if a.per_image:
                print(f"  {Path(root).name}/{stem}: det={d:4d} correct={t:4d}  labelled={g:4d} found={h:4d}")

    if not n_images:
        print("no samples in that split (need >= 2 photos so a held-out split exists, "
              "or use --split all)")
        return 1
    precision = tp / n_det if n_det else 1.0
    recall = hit / n_gt if n_gt else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\n{n_images} tiles ({a.split})  backend={a.backend}  clean={'off' if a.no_clean else 'on'}")
    print(f"  detections {n_det}  correct {tp}  -> precision {precision:.3f}  "
          f"(false detections: {n_det - tp})")
    print(f"  labelled tufts {n_gt}  found {hit}  -> recall {recall:.3f}  (missed: {n_gt - hit})")
    print(f"  F1 {f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
