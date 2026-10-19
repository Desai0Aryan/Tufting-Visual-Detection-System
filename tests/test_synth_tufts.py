import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.labeling import list_samples, load_sample
from src.synth_tufts import make_scene


class SynthTuftsTests(unittest.TestCase):
    def test_scene_labels_match_tufts(self):
        scene, meta = make_scene(np.random.default_rng(3), 640, 360)
        self.assertEqual(scene.image.shape, (360, 640, 3))
        self.assertGreater(meta["tufts"], 10)
        self.assertEqual(len(scene.truth), meta["tufts"])
        self.assertGreater((scene.tuft_mask > 0).mean(), 0.002)
        self.assertLess((scene.tuft_mask > 0).mean(), 0.3)
        for t in scene.truth:                     # every root lies in the frame and is marked
            x, y = (int(round(v)) for v in t["root"])
            self.assertTrue(0 <= x < 640 and 0 <= y < 360)
            self.assertEqual(int(scene.root_mask[y, x]), 255)

    def test_reproducible(self):
        a, _ = make_scene(np.random.default_rng([0, 5]), 320, 180)
        b, _ = make_scene(np.random.default_rng([0, 5]), 320, 180)
        np.testing.assert_array_equal(a.image, b.image)
        np.testing.assert_array_equal(a.tuft_mask, b.tuft_mask)

    def test_tool_writes_dataset(self):
        from tools import make_synth_tufts
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(make_synth_tufts.main(["--out", d, "--count", "2"]), 0)
            self.assertEqual(list_samples(d), ["synp00000", "synp00001"])
            img, lab = load_sample(d, "synp00000")
            self.assertEqual(img.shape[:2], lab.shape[:2])
            self.assertTrue((lab[..., 0] > 127).any())


if __name__ == "__main__":
    unittest.main()
