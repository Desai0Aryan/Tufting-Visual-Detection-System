"""Self-training labels: the tuft-map pipeline labels frames of a video for the U-Net.

    python -m tools.self_train data/videos/flight.mp4 --clip flight1 --val-from 2900 --val-to 4165 --train-to 2780
    python -m tools.train --config configs/unet_resnet34.yaml --set "data.real=[data/selftrain/flight1, ...]"

Teacher = the confirmed tuft map + the current U-Net + the classical skeleton
detector (core/map_pipeline.py). A detection counts only when it is attached
to a confirmed root, so the skeleton detector's clutter never becomes a label.
Per sampled frame:
  tuft      every map-attached detection (its outline)
  ignore    roots with no detection this frame (disk over where the tuft would be);
            tuft-like pixels near the tufts not attached to any root (unmapped tufts,
            e.g. faint far rows: never taught as background)
  background everything else (sky, sea, cockpit, ...)
Frames where fewer than --min-found of the roots are seen (title cards, cuts) are skipped.
Train and val frames come from separate time ranges, registered in sources.csv as two
clips (<clip>_train, <clip>_val); a time split of one video is a weak test, say so.
The map must be confirmed first (tools.process_video ... --edit-map).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.map_pipeline import MapPipeline  # noqa: E402
from core.sources import VideoFileSource  # noqa: E402
from core.tuft_map import TuftMap, apply, default_map_path  # noqa: E402
from src.label_io import save_frame  # noqa: E402


def frame_labels(pipe, frame, index, zone_px, ignore_thr):
    """-> (tuft mask bool, ignore bool, list of Tuft, fraction of roots found)."""
    res = pipe.process_frame(frame, index=index)
    H, W = frame.shape[:2]
    tuft = np.zeros((H, W), np.uint8)
    ignore = np.zeros((H, W), np.uint8)
    found = []
    M = pipe.aligner.ref_to_frame(frame) if pipe.aligner is not None else np.array([[1, 0, 0], [0, 1, 0]], float)
    s = float(np.sqrt(abs(np.linalg.det(M[:, :2])))) or 1.0
    roots = apply(M, [tr.anchor.root for tr in pipe.tracks])
    for tr, r in zip(pipe.tracks, roots):
        d = tr.last
        L = (tr.anchor.length_px or pipe.L) * s
        if d is not None:
            cv2.drawContours(tuft, [np.asarray(d.contour, np.int32)], -1, 1, cv2.FILLED)
            found.append(d)
        else:
            ang = np.radians(tr.anchor.angle_deg)
            c = r + 0.5 * L * np.array([np.cos(ang), -np.sin(ang)])
            cv2.circle(ignore, (int(c[0]), int(c[1])), int(0.75 * L) + 4, 1, -1)
    # near the tufts, anything tuft-like that is not attached to a root: ignore, not background
    zone = np.zeros((H, W), np.uint8)
    if len(roots) >= 3:
        cv2.fillConvexPoly(zone, cv2.convexHull(roots.astype(np.int32)), 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * zone_px + 1, 2 * zone_px + 1))
        zone = cv2.dilate(zone, k)
    # (the model's loose probability only: the skeleton mask marks every dark edge, and
    # wing outlines / panel lines are exactly what the model must learn are not tufts)
    prob = pipe.backend.predict_proba(frame)[0]
    loose = cv2.dilate((prob >= ignore_thr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ignore |= (loose & (zone > 0)).astype(np.uint8)
    tuft_b = tuft > 0
    ignore_b = (ignore > 0) & ~tuft_b
    n = sum(t.detected for t in res.tufts)
    return tuft_b, ignore_b, found, n / max(len(res.tufts), 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--clip", required=True, help="clip id prefix, e.g. flight1 -> flight1_train / flight1_val")
    ap.add_argument("--weights", default="models/tuft_unet.pt")
    ap.add_argument("--map", default=None)
    ap.add_argument("--out", default=None, help="default data/selftrain/<clip>")
    ap.add_argument("--train-to", type=int, required=True, help="train frames: [0, train-to)")
    ap.add_argument("--val-from", type=int, required=True)
    ap.add_argument("--val-to", type=int, default=None)
    ap.add_argument("--train-every", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=15)
    ap.add_argument("--min-found", type=float, default=0.6)
    ap.add_argument("--skeleton", choices=["auto", "dark", "light", "off"], default="auto")
    a = ap.parse_args(argv)

    from src.unet import UNetBackend
    from tools.process_video import pick_skeleton
    map_path = Path(a.map) if a.map else default_map_path(a.video)
    if not map_path.exists():
        print(f"no tuft map at {map_path}: run tools.process_video {a.video} first and confirm the map")
        return 2
    tmap = TuftMap.load(map_path)
    src = VideoFileSource(a.video)
    from core.tuft_map import load_ref
    ref = load_ref(map_path, src, tmap.ref_index)
    backend = UNetBackend(a.weights)
    pipe = MapPipeline(tmap, backend, ref, fps=src.fps)
    pipe.extra = pick_skeleton(pipe, ref, a.skeleton)
    out = Path(a.out or f"data/selftrain/{a.clip}")
    val_to = a.val_to or src.frame_count
    plan = [(i, "train") for i in range(0, a.train_to, a.train_every)] + \
           [(i, "val") for i in range(a.val_from, val_to, a.val_every)]
    L = tmap.median_length()
    kept = {"train": 0, "val": 0}
    for k, (i, split) in enumerate(plan):
        frame = src.read_at(i)
        if frame is None:
            continue
        tuft, ignore, tufts, frac = frame_labels(pipe, frame, i, int(1.5 * L), backend.threshold * 0.4)
        if frac < a.min_found:
            continue
        clip = f"{a.clip}_{split}"
        meta = {"clip_id": clip, "frame_idx": int(i), "split": split, "use": "tufts", "source": "selftrain",
                "domain": "own", "teacher": str(a.weights), "roots_found": round(frac, 3)}
        save_frame(out, f"{clip}__f{i:06d}", frame, tuft.astype(np.uint8) * 255, tufts, meta, ignore=ignore)
        kept[split] += 1
        print(f"\r{k + 1}/{len(plan)}  kept train {kept['train']}  val {kept['val']}", end="", flush=True)
    src.close()
    print(f"\nwrote {out}  (train {kept['train']}, val {kept['val']} frames)")
    print(f"register the clips in sources.csv: {a.clip}_train -> train, {a.clip}_val -> val")
    return 0


if __name__ == "__main__":
    sys.exit(main())
