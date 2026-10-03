import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detector import TuftDetector  # noqa: E402
from src.mask_to_tufts import VECTOR_MODES, extract_tufts  # noqa: E402
from src.synth import BACKGROUNDS, angular_error_deg, make_scene  # noqa: E402


def match(truth, tufts, radius=9.0):
    """Greedy nearest-root matching -> list of (truth, tuft); counts."""
    used, pairs = set(), []
    for tr in truth:
        best, bd = None, radius
        for i, t in enumerate(tufts):
            if i in used:
                continue
            d = np.hypot(t.root[0] - tr["root"][0], t.root[1] - tr["root"][1])
            if d < bd:
                best, bd = i, d
        if best is not None:
            used.add(best)
            pairs.append((tr, tufts[best]))
    return pairs, len(tufts) - len(pairs)


def run(seeds, oracle_root=False):
    rows = {}
    for bg in BACKGROUNDS:
        stats = {m: [] for m in VECTOR_MODES}
        n_truth = n_match = n_fp = n_root_ok = 0
        for seed in range(seeds):
            scene = make_scene(background=bg, seed=seed, flow_deg=15 + 40 * (seed % 3))
            for mode in VECTOR_MODES:
                det = TuftDetector(vector_mode=mode)
                if oracle_root:  # stands in for a model that predicts the root mask
                    class Oracle:
                        name = "oracle"
                        def predict(self, img, s=scene):
                            from src.backends import MaskResult
                            return MaskResult(s.tuft_mask, s.root_mask)
                    det = TuftDetector(backend=Oracle(), vector_mode=mode)
                tufts = det.detect(scene.image)
                pairs, fp = match(scene.truth, tufts)
                for tr, t in pairs:
                    stats[mode].append(angular_error_deg(t.angle_deg, tr["flow_deg"]))
                if mode == "chord":
                    n_truth += len(scene.truth); n_match += len(pairs); n_fp += fp
                    n_root_ok += sum(
                        np.hypot(t.tip[0] - tr["tip"][0], t.tip[1] - tr["tip"][1])
                        < np.hypot(t.root[0] - tr["tip"][0], t.root[1] - tr["tip"][1])
                        for tr, t in pairs)
        rows[bg] = (n_match / n_truth, n_fp, n_root_ok / max(n_match, 1),
                    {m: (np.mean(v) if v else float("nan")) for m, v in stats.items()})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--oracle-root", action="store_true",
                    help="feed the true root mask (simulates a model with a root head)")
    a = ap.parse_args()
    rows = run(a.seeds, a.oracle_root)
    print(f"{'background':<10} {'detect%':>8} {'FP':>4} {'root-ok%':>9}   mean flow error (deg): "
          + "  ".join(f"{m:>6}" for m in VECTOR_MODES))
    for bg, (det, fp, ro, errs) in rows.items():
        print(f"{bg:<10} {100*det:8.1f} {fp:4d} {100*ro:9.1f}   "
              + " " * 24 + "  ".join(f"{errs[m]:6.1f}" for m in VECTOR_MODES))
