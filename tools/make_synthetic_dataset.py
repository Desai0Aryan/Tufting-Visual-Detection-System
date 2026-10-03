import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labeling import encode_label, save_sample  # noqa: E402
from src.synth import make_scene  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    for i in range(a.count):
        bg = tuple(int(v) for v in rng.integers(20, 236, 3))
        w, h = [(640, 480), (800, 600), (1024, 768)][int(rng.integers(3))]
        scene = make_scene(
            width=w, height=h, flow_deg=float(rng.uniform(-180, 180)), background=bg,
            seed=int(rng.integers(1 << 30)),
            spacing=(int(rng.integers(60, 110)), int(rng.integers(50, 90))),
            length_range=(int(rng.integers(26, 40)), int(rng.integers(44, 70))),
            hook_prob=float(rng.uniform(0.2, 0.9)), noise=float(rng.uniform(2, 8)),
        )
        # truth is exact, so the root label is trusted everywhere (valid = all ones)
        label = encode_label(scene.tuft_mask, scene.root_mask, np.ones_like(scene.tuft_mask))
        save_sample(a.out, f"syn{i:04d}", scene.image, label,
                    meta={"source": "synthetic", "background_bgr": list(bg)})
    print(f"wrote {a.count} synthetic samples to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
