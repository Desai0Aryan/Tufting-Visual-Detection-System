import unittest

import numpy as np

from detector import TuftDetector
from src.backends import MaskResult
from src.export import COLUMNS, tufts_dataframe
from src.mask_to_tufts import extract_tufts
from src.synth import angular_error_deg, make_scene


def _match(truth, tufts, radius=9.0):
    pairs, used = [], set()
    for tr in truth:
        for i, t in enumerate(tufts):
            if i not in used and np.hypot(t.root[0] - tr["root"][0], t.root[1] - tr["root"][1]) < radius:
                used.add(i)
                pairs.append((tr, t))
                break
    return pairs


class OracleBackend:
    """Feeds the true masks, standing in for a model with a root head."""
    name = "oracle"

    def __init__(self, scene, with_root=True):
        self.scene, self.with_root = scene, with_root

    def predict(self, image):
        return MaskResult(self.scene.tuft_mask, self.scene.root_mask if self.with_root else None)


class FlowVectorTests(unittest.TestCase):
    def test_classical_recovers_flow_direction_and_root(self):
        scene = make_scene(background="white", seed=1, flow_deg=30)
        # truth here is the bulk flow angle (a chord quantity), not the tip tangent
        tufts = TuftDetector(vector_mode="chord").detect(scene.image)
        pairs = _match(scene.truth, tufts)
        self.assertGreaterEqual(len(pairs), 0.9 * len(scene.truth))
        errs = [angular_error_deg(t.angle_deg, tr["flow_deg"]) for tr, t in pairs]
        self.assertLess(np.mean(errs), 12.0)
        # tip must be farther from the true tip's side than the root: root/tip not swapped
        ok = sum(np.hypot(*np.subtract(t.tip, tr["tip"])) < np.hypot(*np.subtract(t.root, tr["tip"]))
                 for tr, t in pairs)
        self.assertGreaterEqual(ok / len(pairs), 0.95)

    def test_base_mode_beats_old_tip_tangent_on_hooked_tufts(self):
        scene = make_scene(seed=2, flow_deg=20, hook_prob=1.0)
        def mean_err(mode):
            ts = TuftDetector(backend=OracleBackend(scene), vector_mode=mode).detect(scene.image)
            return np.mean([angular_error_deg(t.angle_deg, tr["flow_deg"]) for tr, t in _match(scene.truth, ts)])
        self.assertLess(mean_err("base"), mean_err("tip"))
        self.assertLess(mean_err("chord"), mean_err("tip"))

    def test_root_mask_sets_orientation_with_high_confidence(self):
        scene = make_scene(seed=3, flow_deg=-40)
        ts = extract_tufts(scene.tuft_mask, scene.root_mask)
        pairs = _match(scene.truth, ts)
        self.assertGreater(len(pairs), 0.9 * len(scene.truth))
        self.assertTrue(all(t.confidence >= 0.25 for _, t in pairs))

    def test_flow_prior_flips_ambiguous_tufts_consistently(self):
        # uniform-width strokes: thickness cue is useless, prior must decide
        mask = np.zeros((60, 200), np.uint8)
        import cv2
        cv2.line(mask, (20, 30), (80, 30), 255, 3)
        cv2.line(mask, (120, 30), (180, 30), 255, 3)
        right = extract_tufts(mask, flow_prior=(1, 0))
        left = extract_tufts(mask, flow_prior=(-1, 0))
        self.assertTrue(all(t.vector[0] > 0.9 for t in right))
        self.assertTrue(all(t.vector[0] < -0.9 for t in left))

    def test_roi_offset_is_applied_to_all_geometry(self):
        scene = make_scene(seed=4)
        full = extract_tufts(scene.tuft_mask, scene.root_mask)
        shifted = extract_tufts(scene.tuft_mask, scene.root_mask, offset=(100, 50))
        self.assertEqual(len(full), len(shifted))
        for a, b in zip(full, shifted):
            self.assertAlmostEqual(b.root[0] - a.root[0], 100, places=3)
            self.assertAlmostEqual(b.tip[1] - a.tip[1], 50, places=3)
            self.assertAlmostEqual(float(b.path[0][0] - a.path[0][0]), 100, places=3)

    def test_export_schema_and_unit_vectors(self):
        scene = make_scene(seed=5)
        tufts = TuftDetector().detect(scene.image)
        df = tufts_dataframe(tufts, frame=7, timestamp_s=0.25)
        self.assertEqual(list(df.columns), COLUMNS)
        self.assertTrue((df["frame"] == 7).all() and (df["timestamp_s"] == 0.25).all())
        norms = np.sqrt(df.unit_vector_x ** 2 + df.unit_vector_y ** 2 + df.unit_vector_z ** 2)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class UNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch not installed; U-Net tests skipped")

    def test_forward_shape_and_backend_roundtrip(self):
        import tempfile, os, torch
        from src.unet import UNet, UNetBackend, save_checkpoint
        model = UNet(base=8)
        self.assertEqual(tuple(model(torch.zeros(1, 3, 64, 96)).shape), (1, 2, 64, 96))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            save_checkpoint(model, p, {"base": 8})
            backend = UNetBackend(p, device="cpu", tile=64, overlap=16)  # forces tiling
            res = backend.predict(np.zeros((150, 130, 3), np.uint8))
            self.assertEqual(res.tuft.shape, (150, 130))
            self.assertEqual(res.root.shape, (150, 130))


if __name__ == "__main__":
    unittest.main()
