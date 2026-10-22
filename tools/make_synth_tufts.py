"""Write a synthetic training set from src/synth_tufts.py (tufted panel in perspective).

    python -m tools.make_synth_tufts --out data/synthetic_panel --count 500
    python -m tools.train --config configs/unet_resnet34.yaml --set "data.synthetic=[data/synthetic_panel]" data.synthetic_share=0.2

Training data only: validation and test stay real footage.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labeling import encode_label, save_sample  # noqa: E402
from src.synth_tufts import make_scene  # noqa: E402

SIZES = [(1280, 720), (1600, 900), (1920, 1080)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/synthetic_panel")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    t0 = time.time()
    for i in range(a.count):
        rng = np.random.default_rng([a.seed, i])          # each frame reproducible on its own
        w, h = SIZES[int(rng.integers(len(SIZES)))]
        scene, meta = make_scene(rng, w, h)
        # truth is exact, so the root label is trusted everywhere (valid = all ones)
        label = encode_label(scene.tuft_mask, scene.root_mask, np.ones_like(scene.tuft_mask))
        save_sample(a.out, f"synp{i:05d}", scene.image, label,
                    meta={"source": "synthetic_panel", "domain": "synthetic", "seed": [a.seed, i], **meta})
        print(f"\r{i + 1}/{a.count}  {time.time() - t0:.0f}s", end="", flush=True)
    print(f"\nwrote {a.count} frames to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
