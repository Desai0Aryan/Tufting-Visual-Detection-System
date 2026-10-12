"""One model-in-the-loop round, one command:

    python -m tools.loop_round --clips yt_new1 own_run7 --weights runs/LATEST

  1. extract diverse frames from the new clips (skips what is already extracted)
  2. run the current U-Net on them -> proposals in data/real/proposed (source=unet);
     tuft-free `negative` clips are predicted too, so false positives show up
  3. open the review tool on exactly those frames (fix, accept, reject)
  4. retrain on all accepted frames (warm start from --weights unless --fresh)

--no-train stops after review. Training refuses to start if no frame of
this round was accepted. The frozen test clips are proposed and reviewed
like any other clip but never trained on (split comes from sources.csv).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.label_io import save_frame, tufts_from_mask  # noqa: E402
from src.labeling import list_samples  # noqa: E402


def resolve_weights(w):
    p = Path(w)
    if p.name == "LATEST" or p.is_dir():
        run = Path((p if p.name == "LATEST" else p / "LATEST").read_text().strip())
        return run / "checkpoints" / "best.pt"
    return p


def propose_with_unet(rows, weights, out, skip, device=None):
    from src.unet import UNetBackend
    be = UNetBackend(weights, device=device)
    made = []
    for r in rows:
        stem = Path(r["file"]).stem
        if stem in skip:
            continue
        img = cv2.imread(r["file"])
        res = be.predict(img)
        mask = res.tuft
        tufts = tufts_from_mask(mask, root_mask=res.root) if mask.any() else []
        meta = {"clip_id": r["clip_id"], "frame_idx": int(r["frame_idx"]), "split": r["split"],
                "use": r["use"], "source": "unet", "weights": str(weights), "threshold": be.threshold,
                "reviewed": False, "status": "proposed"}
        save_frame(out, stem, img, mask, tufts, meta, extra=[{"source": "unet"}] * len(tufts))
        made.append(stem)
        print(f"{stem}: U-Net proposes {len(tufts)} tufts")
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--weights", default="runs/LATEST", help="checkpoint, run dir, or runs/LATEST")
    ap.add_argument("--per-clip", type=int, default=40, help="frames to pick per new clip")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--config", default="configs/unet_resnet34.yaml")
    ap.add_argument("--sources", default="sources.csv")
    ap.add_argument("--clips-dir", default="data/clips")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--proposed", default="data/real/proposed")
    ap.add_argument("--labeled", default="data/real/labeled")
    ap.add_argument("--no-review", action="store_true", help=argparse.SUPPRESS)  # tests only
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="retrain from ImageNet weights")
    ap.add_argument("--set", nargs="*", default=[], help="trainer config overrides")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    from tools import extract_frames, review_labels, train
    weights = resolve_weights(a.weights)
    if not weights.exists():
        raise SystemExit(f"no checkpoint at {weights}")

    # 1. frames
    extract_frames.main(["--sources", a.sources, "--clips-dir", a.clips_dir, "--out", a.frames,
                         "--clips", *a.clips, "--per-clip", str(a.per_clip), "--stride", str(a.stride)])
    rows = [r for r in csv.DictReader(open(Path(a.frames) / "frames.csv", newline=""))
            if r["clip_id"] in set(a.clips)]

    # 2. U-Net proposals
    skip = set(list_samples(a.proposed)) | set(list_samples(a.labeled))
    stems = propose_with_unet(rows, weights, a.proposed, skip, a.device)
    if not stems:
        print("no new frames to propose (all already proposed or labelled)")
        return 0

    # 3. review
    if not a.no_review:
        review_labels.main(["--data", a.proposed, "--out", a.labeled, "--clips", *a.clips])
    meta = json.loads((Path(a.proposed) / "meta.json").read_text())
    accepted = [s for s in stems if meta.get(s, {}).get("status") == "accepted"]
    print(f"round: proposed {len(stems)}, accepted {len(accepted)}")

    # 4. retrain
    if a.no_train:
        return 0
    if not accepted:
        print("nothing accepted this round: not retraining")
        return 0
    args = ["--config", a.config, "--set", f"data.real=[{a.labeled}]", f"data.sources_csv={a.sources}", *a.set]
    if not a.fresh:
        args += ["--init", str(weights)]
    if a.device:
        args += ["--device", a.device]
    return train.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
