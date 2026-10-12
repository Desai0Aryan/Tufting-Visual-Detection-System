"""Extract a diverse set of frames from the clips in sources.csv.

    python -m tools.extract_frames --total 500                  # all clips, ~500 frames spread across clips
    python -m tools.extract_frames --clips own_run3 yt_abc --per-clip 40 --stride 10
    python -m tools.extract_frames --use negative --total 80     # tuft-free hard-negative frames

Writes data/frames/<clip_id>/<clip_id>__f<frame>.png at native resolution and
appends to data/frames/frames.csv (clip_id, frame_idx, time_s, brightness,
contrast, motion, file). Frames already extracted are skipped.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.frame_select import farthest_point_sample, feature_matrix, per_clip_quota, scan_video  # noqa: E402
from src.sources import is_dataset, read_sources  # noqa: E402

FIELDS = ["clip_id", "frame_idx", "time_s", "brightness", "contrast", "motion", "use", "split", "file"]


def find_clip(clips_dir, clip_id):
    hits = sorted(Path(clips_dir).glob(f"{clip_id}.*"))
    if not hits:
        raise FileNotFoundError(f"no file for clip {clip_id} in {clips_dir}")
    return hits[0]


def read_frame(path, idx):
    if idx == 0 and Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        return cv2.imread(str(path))
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", default="sources.csv")
    ap.add_argument("--clips-dir", default="data/clips")
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--clips", nargs="*", default=None, help="clip ids (default: all matching --use)")
    ap.add_argument("--use", default="tufts", choices=["tufts", "negative", "background", "any"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--total", type=int, default=500, help="frames across all selected clips")
    g.add_argument("--per-clip", type=int, default=None)
    ap.add_argument("--stride", type=int, default=15, help="scan every N-th frame")
    ap.add_argument("--min-gap-s", type=float, default=0.5, help="min seconds between picked frames")
    ap.add_argument("--dup-thresh", type=float, default=0.02, help="near-duplicate threshold (thumbnail MSE)")
    a = ap.parse_args(argv)

    rows = [r for r in read_sources(a.sources) if not is_dataset(r)]  # image datasets have no video
    if a.clips:
        rows = [r for r in rows if r["clip_id"] in set(a.clips)]
        missing = set(a.clips) - {r["clip_id"] for r in rows}
        if missing:
            raise SystemExit(f"not in {a.sources}: {sorted(missing)} (register with tools.sources first)")
    elif a.use != "any":
        rows = [r for r in rows if r["use"] == a.use]
    if not rows:
        raise SystemExit("no clips selected")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "frames.csv"
    done = set()
    if index.exists():
        done = {(r["clip_id"], int(r["frame_idx"])) for r in csv.DictReader(open(index, newline=""))}

    scans = {}
    for r in rows:
        path = find_clip(a.clips_dir, r["clip_id"])
        cands, fps = scan_video(path, r["clip_id"], a.stride, dup_thresh=a.dup_thresh)
        cands = [c for c in cands if (c.clip_id, c.frame_idx) not in done]
        scans[r["clip_id"]] = (r, path, cands)
        print(f"{r['clip_id']}: {len(cands)} candidate frames after near-duplicate removal")

    if a.per_clip:
        quota = {c: min(a.per_clip, len(v[2])) for c, v in scans.items()}
    else:
        quota = per_clip_quota({c: len(v[2]) for c, v in scans.items()}, a.total)

    new = not index.exists()
    with open(index, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        total = 0
        for cid, (r, path, cands) in scans.items():
            if not cands or quota[cid] == 0:
                continue
            idx = farthest_point_sample(feature_matrix(cands), quota[cid], a.min_gap_s,
                                        [c.time_s for c in cands])
            d = out / cid
            d.mkdir(exist_ok=True)
            for i in sorted(idx, key=lambda i: cands[i].frame_idx):
                c = cands[i]
                frame = read_frame(path, c.frame_idx)
                if frame is None:
                    continue
                f = d / f"{cid}__f{c.frame_idx:06d}.png"
                cv2.imwrite(str(f), frame)
                w.writerow({"clip_id": cid, "frame_idx": c.frame_idx, "time_s": round(c.time_s, 3),
                            "brightness": round(c.brightness, 1), "contrast": round(c.contrast, 1),
                            "motion": round(c.motion, 2), "use": r["use"], "split": r["split"],
                            "file": f.as_posix()})
                total += 1
            sel = [cands[i] for i in idx]
            print(f"{cid}: picked {len(sel)}  brightness {min(c.brightness for c in sel):.0f}-"
                  f"{max(c.brightness for c in sel):.0f}  motion {min(c.motion for c in sel):.1f}-"
                  f"{max(c.motion for c in sel):.1f}")
    print(f"wrote {total} frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
