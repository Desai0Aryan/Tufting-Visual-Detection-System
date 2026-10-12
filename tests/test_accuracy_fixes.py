import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from main import main as run_main, parse_args
from src.backends import ClassicalBackend
from src.filtering import (clean_tufts, field_filter, normalize_scale, polygon_mask,
                           rescale_tufts, scale_roi)
from src.mask_to_tufts import extract_tufts
from src.sam2_labeler import (_best_sam_mask, refine_with_sam, refine_with_sam_detailed,
                              verify_tufts_with_sam)
from src.synth import make_scene

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


class _Stub:
    """Stand-in for SAM2ImagePredictor (same interface the labeler uses)."""

    def __init__(self, kind):
        self.kind = kind

    def set_image(self, rgb):
        self.rgb, self.shape = rgb, rgb.shape[:2]

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        if self.kind == "raise":
            raise RuntimeError("boom")
        h, w = self.shape
        if self.kind == "whole":
            m = np.ones((h, w), bool)
        else:  # "good": the dark/light string nearest the first positive point, slightly fattened
            gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
            sel = (gray < 90) if gray.mean() > 128 else (gray > 165)
            _, cc = cv2.connectedComponents(sel.astype(np.uint8), connectivity=8)
            x, y = [int(round(v)) for v in point_coords[0]]
            ys, xs = np.nonzero(cc)
            k = np.argmin(np.hypot(xs - x, ys - y))
            m = cv2.dilate((cc == cc[ys[k], xs[k]]).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        # SAM returns several nested guesses when multimask_output=True
        return np.stack([np.ones((h, w), bool), m]) if multimask_output else np.stack([m]), \
            np.array([0.9, 0.8][: (2 if multimask_output else 1)]), None


class ScaleNormalisationTests(unittest.TestCase):
    def test_never_upscales_and_reports_scale(self):
        img = np.zeros((300, 400, 3), np.uint8)
        out, s = normalize_scale(img, 1400)
        self.assertIs(out, img)
        self.assertEqual(s, 1.0)
        big = np.zeros((2000, 4000, 3), np.uint8)
        out, s = normalize_scale(big, 1400)
        self.assertEqual(max(out.shape[:2]), 1400)
        self.assertAlmostEqual(s, 1400 / 4000, places=6)
        self.assertEqual(normalize_scale(big, 0)[1], 1.0)  # 0 = off

    def test_scale_roi(self):
        self.assertEqual(scale_roi((100, 200, 400, 300), 0.5), (50, 100, 200, 150))
        self.assertIsNone(scale_roi(None, 0.5))

    def test_detect_downscaled_then_rescale_lands_on_original_tufts(self):
        sc = make_scene(width=2800, height=2100, seed=3, spacing=(190, 150), length_range=(80, 130),
                        hook_prob=0.0)
        small, s = normalize_scale(sc.image, 1400)
        self.assertLess(s, 1.0)
        mask = ClassicalBackend(polarity="light" if sc.image.mean() < 110 else "dark").predict(small).tuft
        found = rescale_tufts(extract_tufts(mask), 1.0 / s)
        self.assertGreater(len(found), 10)
        roots = np.array([t.root for t in found])
        truth = np.array([g["root"] for g in sc.truth])
        # every truth root should have a detected root/tip end nearby in ORIGINAL pixels
        ends = np.vstack([roots, np.array([t.tip for t in found])])
        d = np.min(np.hypot(*(truth[:, None, :] - ends[None, :, :]).transpose(2, 0, 1)), axis=1)
        self.assertGreater((d < 15).mean(), 0.7)
        for t in found:
            self.assertLess(t.root[0], sc.image.shape[1] + 1)
            self.assertLess(t.root[1], sc.image.shape[0] + 1)


class FieldFilterTests(unittest.TestCase):
    def setUp(self):
        self.sc = make_scene(width=900, height=600, seed=5, spacing=(80, 62), background="white")
        self.img = self.sc.image.copy()
        self.field = extract_tufts(ClassicalBackend().predict(self.img).tuft)

    def test_keeps_the_real_field(self):
        kept = clean_tufts(self.img, self.field)
        self.assertGreaterEqual(len(kept), 0.95 * len(self.field))
        self.assertEqual([t.id for t in kept], list(range(len(kept))))

    def test_drops_a_small_distant_cluster_but_not_the_field(self):
        img = self.img.copy()
        canvas = np.full((600, 1500, 3), 235, np.uint8)
        canvas[:, :900] = img
        rng = np.random.default_rng(0)
        for _ in range(5):  # a little patch of scratches far from the tuft field
            x, y = int(rng.integers(1200, 1400)), int(rng.integers(100, 200))
            cv2.line(canvas, (x, y), (x + 40, y + int(rng.integers(-8, 8))), (20, 20, 20), 3)
        tufts = extract_tufts(ClassicalBackend().predict(canvas).tuft)
        far = [t for t in tufts if t.root[0] > 1150]
        self.assertGreaterEqual(len(far), 3)
        kept = clean_tufts(canvas, tufts)
        self.assertFalse([t for t in kept if t.root[0] > 1150])
        self.assertGreaterEqual(len([t for t in kept if t.root[0] <= 900]),
                                0.95 * len([t for t in tufts if t.root[0] <= 900]))

    def test_tiny_scenes_are_left_alone(self):
        few = self.field[:4]
        self.assertEqual(len(field_filter(self.img, few)), 4)


class PolygonTests(unittest.TestCase):
    def test_polygon_mask(self):
        m = polygon_mask((100, 200), [(10, 10), (150, 10), (150, 80), (10, 80)])
        self.assertEqual(m.shape, (100, 200))
        self.assertEqual(m[40, 60], 255)
        self.assertEqual(m[95, 60], 0)
        self.assertEqual(polygon_mask((50, 50), [(1, 1), (2, 2)]).sum(), 0)  # < 3 points -> empty


class StrictSamTests(unittest.TestCase):
    def setUp(self):
        self.sc = make_scene(width=480, height=360, seed=6, spacing=(110, 80))
        self.mask = self.sc.tuft_mask
        self.tufts = extract_tufts(self.mask, self.sc.root_mask)

    def test_strict_rejects_what_sam_cannot_confirm(self):
        for kind in ("whole", "raise"):
            refined, st, status = refine_with_sam_detailed(
                self.sc.image, self.tufts, self.mask, _Stub(kind), tile=512, strict=True)
            self.assertEqual(st["rejected"], len(self.tufts))
            self.assertEqual(set(status), {"rejected"})
            self.assertEqual(int(refined.sum()), 0)  # nothing the verifier rejected leaks into labels

    def test_non_strict_still_keeps_everything(self):
        _, st, status = refine_with_sam_detailed(
            self.sc.image, self.tufts, self.mask, _Stub("whole"), tile=512, strict=False)
        self.assertEqual(st["fallback"], len(self.tufts))
        self.assertEqual(st["rejected"], 0)

    def test_strict_accepts_confirmed_tufts_and_multimask_picks_the_best(self):
        refined, st, status = refine_with_sam_detailed(
            self.sc.image, self.tufts, self.mask, _Stub("good"), tile=512, strict=True)
        self.assertGreaterEqual(st["sam_accepted"], 0.8 * len(self.tufts))
        self.assertGreater(refined.sum(), 0)
        # stub returns [whole-tile, string]; the tile-sized guess must lose
        prop = self.mask > 0
        whole = np.ones_like(prop)
        string = cv2.dilate(self.mask, np.ones((3, 3), np.uint8)) > 0
        np.testing.assert_array_equal(_best_sam_mask(np.stack([whole, string]), prop), string)

    def test_verify_returns_the_confirmed_subset_with_geometry_untouched(self):
        kept, st = verify_tufts_with_sam(self.sc.image, self.tufts, _Stub("good"), tile=512)
        self.assertGreaterEqual(len(kept), 0.8 * len(self.tufts))
        ids = {id(t) for t in self.tufts}
        self.assertTrue(all(id(t) in ids for t in kept))  # same objects: root/tip/vector unchanged
        kept0, _ = verify_tufts_with_sam(self.sc.image, self.tufts, _Stub("whole"), tile=512)
        self.assertEqual(kept0, [])

    def test_legacy_wrapper_signature(self):
        refined, st = refine_with_sam(self.sc.image, self.tufts, self.mask, _Stub("raise"), tile=512)
        self.assertEqual(st["fallback"] + st["skipped"], len(self.tufts))


class CliTests(unittest.TestCase):
    def test_defaults(self):
        a = parse_args(["--image", str(ROOT / "tests" / "fixtures" / "sample_tufts1.jpg")])
        self.assertIsNone(a.threshold)  # auto from the checkpoint's validation
        # native resolution only; field filter opt-in; SAM 2 is not part of detection
        self.assertFalse(hasattr(a, "norm_size"))
        self.assertFalse(a.clean)
        self.assertFalse(hasattr(a, "verify_sam2"))
        self.assertEqual(a.vector_mode, "tip")
        self.assertFalse(parse_args([str(ROOT / "tests" / "fixtures" / "sample_tufts1.jpg"),
                                     "--no-clean"]).clean)

    def test_end_to_end_on_a_large_image_keeps_original_coordinates(self):
        sc = make_scene(width=2800, height=2100, seed=9, spacing=(190, 150), length_range=(80, 130),
                        hook_prob=0.0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "big.png"
            cv2.imwrite(str(p), sc.image)
            pol = "light" if sc.image.mean() < 110 else "dark"
            # no downscaling any more, so the size limits must match these ~80-130 px tufts
            self.assertEqual(run_main([str(p), "--outdir", d, "--polarity", pol, "--save-mask",
                                       "--max-length-px", "200", "--max-area", "2000"]), 0)
            import pandas as pd
            df = pd.read_csv(next(Path(d).glob("big_*_vectors.csv")))
            self.assertGreater(len(df), 10)
            self.assertTrue((df.root_x.between(0, 2800) & df.root_y.between(0, 2100)).all())
            self.assertGreater(df.length_px.median(), 60)  # original-pixel lengths
            mask = cv2.imread(str(next(Path(d).glob("big_*_mask.png"))), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(mask.shape, (2100, 2800))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class ThresholdFromCheckpointTests(unittest.TestCase):
    def test_reads_selected_threshold_and_falls_back(self):
        from src.unet import suggested_threshold
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "m.pt"
            self.assertEqual(suggested_threshold(w), 0.5)
            Path(str(w) + ".json").write_text(json.dumps(
                {"selected_on": "real", "best_metrics": {"real": {"best_threshold": 0.7}}}))
            self.assertEqual(suggested_threshold(w), 0.7)


if __name__ == "__main__":
    unittest.main()
