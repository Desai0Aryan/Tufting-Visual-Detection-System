"""Bootstrap labels with SAM 2 for frames picked by tools.extract_frames.

    python -m tools.auto_label                          # every unlabelled frame in data/frames/frames.csv
    python -m tools.auto_label --clips own_run3 --limit 20

Per frame:
  1. classical detector -> candidate tufts (per-clip settings in configs/autolabel.yaml)
  2. SAM 2 (image predictor, one window per candidate) -> clean masks
  3. same geometry code as detection (tip tangent) -> root / tip / angle per tuft
  4. save to data/real/proposed (image, label PNG, json/<stem>.json, meta.json)
Frames from `use: negative` clips become empty-mask proposals (no detector, no SAM).
Optional: SAM 2 video propagation of each keyframe's masks to the next N frames.

Nothing here is training data until it has been accepted in tools.review_labels.
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector import TuftDetector  # noqa: E402
from src.anchor import tape_distance  # noqa: E402
from src.config import load_config  # noqa: E402
from src.label_io import save_frame, tufts_from_mask  # noqa: E402
from src.labeling import list_samples  # noqa: E402


def detector_settings(cfg, clip_id):
    s = dict(cfg["detector"]["default"])
    s.update((cfg.get("clips") or {}).get(clip_id, {}))
    return s


def candidates(image, s):
    kw = dict(min_area=s["min_area"], max_area=s["max_area"], min_elongation=s["min_elongation"],
              min_length_px=s["min_length_px"], max_length_px=s["max_length_px"],
              block_size=s["block_size"], c=s["c"], roi=tuple(s["roi"]) if s.get("roi") else None,
              tape_root=s.get("tape_root", False), vector_mode="tip")
    if s["polarity"] == "color":
        from src.backends import ColorBackend
        backend = ColorBackend(s.get("color_hue", ((0, 15), (170, 180))), s.get("color_s_min", 80),
                               s.get("color_v_min", 110))
        return TuftDetector(backend=backend, **kw).detect(image)
    pols =("dark", "light") if s["polarity"] == "auto" else (s["polarity"],)
    runs = [TuftDetector(polarity=p, **kw).detect(image) for p in pols]
    if len(runs) == 1:
        return runs[0]
    return pick_polarity(runs)


def pick_polarity(runs, margin=1.2):
    """Clearly more candidates wins; otherwise the thinner candidates win.
    The wrong polarity traces the halo around a string, which is wide."""
    a, b = runs
    if len(a) > margin * len(b):
        return a
    if len(b) > margin * len(a):
        return b
    width = lambda ts: np.median([cv2.contourArea(t.contour) / max(t.length_px, 1) for t in ts]) if ts else np.inf
    return a if width(a) <= width(b) else b


def label_frame(image, s, predictor, cfg):
    """-> (mask, tufts, extra per tuft, rejected candidate records, stats)."""
    from src.sam2_autolabel import refine
    cands = candidates(image, s)
    res = refine(image, cands, predictor, min_side=cfg["sam2"]["min_side"],
                 strict=cfg["sam2"]["strict"], **cfg["accept"])
    anchor = tape_distance(image) if s.get("tape_root") else None
    tufts = tufts_from_mask(res.mask, anchor_dist=anchor,
                            anchor_max_dist=s.get("tape_max_dist", 60)) if res.mask.any() else []
    rejected = []
    for t, st, (_, iou) in zip(cands, res.status, res.per_tuft):
        if st != "accepted":
            rejected.append({"status": st, "sam_iou": round(iou, 3),
                             "contour": t.contour.reshape(-1, 2).astype(int).tolist()})
    stats = {"candidates": len(cands), "accepted": res.status.count("accepted"),
             "rejected": len(rejected), "tufts": len(tufts)}
    return res.mask, tufts, rejected, stats


def color_ignore(image, mask, s):
    """Tuft-coloured pixels outside the proposal mask -> ignore (per-clip `ignore_color`),
    so tufts too small or faint to become candidates are never taught as background."""
    ic = s.get("ignore_color")
    if not ic:
        return None
    from src.backends import ColorBackend
    m = ColorBackend(ic.get("hue", ((0, 18), (165, 180))), ic.get("s_min", 45), ic.get("v_min", 60),
                     close_px=1).predict(image).tuft
    r = int(ic.get("dilate", 3))
    if r > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return (m > 0) & ~(mask > 0)


def propagate(video, start_idx, n, every, masks_by_obj, predictor_video, tmp):
    """SAM 2 video propagation from one keyframe. Yields (frame_idx, frame, mask)."""
    import torch
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frames = []
    for k in range(n + 1):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        cv2.imwrite(str(tmp / f"{k:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 98])
    cap.release()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor_video.init_state(video_path=str(tmp))
        for oid, m in enumerate(masks_by_obj, start=1):
            predictor_video.add_new_mask(state, frame_idx=0, obj_id=oid, mask=m)
        for k, _, logits in predictor_video.propagate_in_video(state):
            if k == 0 or k % every:
                continue
            m = (logits > 0).any(0)[0].cpu().numpy()
            yield start_idx + k, frames[k], (m.astype(np.uint8) * 255)
    shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/autolabel.yaml")
    ap.add_argument("--frames-csv", default="data/frames/frames.csv")
    ap.add_argument("--clips-dir", default="data/clips")
    ap.add_argument("--out", default="data/real/proposed")
    ap.add_argument("--labeled", default="data/real/labeled", help="frames already reviewed are skipped")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stems-file", default=None, help="label only the frames listed (one stem per line)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)

    rows = list(csv.DictReader(open(a.frames_csv, newline="")))
    if a.clips:
        rows = [r for r in rows if r["clip_id"] in set(a.clips)]
    have = set(list_samples(a.out)) | set(list_samples(a.labeled))
    stem_of = lambda r: Path(r["file"]).stem
    rows = [r for r in rows if stem_of(r) not in have]
    if a.stems_file:
        want = set(Path(a.stems_file).read_text().split())
        rows = [r for r in rows if stem_of(r) in want]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        print("nothing to label")
        return 0

    predictor = video_pred = None
    if any(r["use"] != "negative" for r in rows):
        import torch
        from src.sam2_autolabel import load_image_predictor
        predictor = load_image_predictor(cfg["sam2"]["checkpoint"], cfg["sam2"]["config"], a.device)
        if cfg["propagate"]["frames"] > 0:
            from sam2.build_sam import build_sam2_video_predictor
            video_pred = build_sam2_video_predictor(cfg["sam2"]["config"], cfg["sam2"]["checkpoint"],
                                                    device=a.device)
    totals = {"frames": 0, "candidates": 0, "accepted": 0, "tufts": 0, "propagated": 0}
    t0 = time.time()
    for r in rows:
        stem, image = stem_of(r), cv2.imread(r["file"])
        meta = {"clip_id": r["clip_id"], "frame_idx": int(r["frame_idx"]), "split": r["split"],
                "use": r["use"], "reviewed": False, "status": "proposed"}
        if r["use"] == "negative":
            save_frame(a.out, stem, image, np.zeros(image.shape[:2], np.uint8), [],
                       dict(meta, source="negative"))
            print(f"{stem}: hard negative (empty mask)")
            totals["frames"] += 1
            continue
        s = detector_settings(cfg, r["clip_id"])
        import torch
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda")):
            mask, tufts, rejected, st = label_frame(image, s, predictor, cfg)
        extra = [{"source": "sam2"} for _ in tufts]
        save_frame(a.out, stem, image, mask, tufts, dict(meta, source="sam2", **st), rejected, extra,
                   ignore=color_ignore(image, mask, s))
        for k in ("candidates", "accepted", "tufts"):
            totals[k] += st[k]
        totals["frames"] += 1
        print(f"{stem}: candidates {st['candidates']}  SAM accepted {st['accepted']}  "
              f"rejected {st['rejected']}  tufts {st['tufts']}")
        if video_pred is not None and mask.any():
            n, objs = cv2.connectedComponents((mask > 0).astype(np.uint8))
            masks = [(objs == k) for k in range(1, n)]
            video = next(Path(a.clips_dir).glob(f"{r['clip_id']}.*"))
            for fi, frame, m in propagate(video, int(r["frame_idx"]), cfg["propagate"]["frames"],
                                          cfg["propagate"]["every"], masks, video_pred,
                                          Path(a.out) / "_tmp_frames"):
                pstem = f"{r['clip_id']}__f{fi:06d}"
                if pstem in have:
                    continue
                pt = tufts_from_mask(m)
                save_frame(a.out, pstem, frame, m, pt, dict(meta, frame_idx=fi, source="sam2_video",
                                                            keyframe=stem), extra=[{"source": "sam2_video"}] * len(pt))
                have.add(pstem)
                totals["propagated"] += 1
    dt = time.time() - t0
    print(f"\n{totals}  ({dt:.0f}s, {dt / max(totals['frames'], 1):.1f}s/frame)")
    print(f"next: python -m tools.review_labels --data {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
