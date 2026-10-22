import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.transforms import recolor_tuft_background, shuffle_tufts
from tools.draw_tufts import activity_frames


def _crop():
    img = np.full((128, 128, 3), 225, np.uint8)                    # light surface
    cv2.line(img, (20, 64), (100, 66), (30, 30, 30), 3)            # dark yarn
    label = np.zeros((128, 128), bool)
    cv2.line(label.view(np.uint8), (20, 64), (100, 66), 1, 11)     # drawn thicker than the yarn
    return img, label


class RecolorTests(unittest.TestCase):
    def test_yarn_recoloured_band_around_it_not_painted(self):
        img, label = _crop()
        outs = [recolor_tuft_background(img, label, np.random.default_rng(s), tuft_p=1.0, background_p=0.0)
                for s in range(8)]
        change = [np.abs(o[65, 60].astype(int) - img[65, 60]).sum() for o in outs]
        self.assertGreater(max(change), 100)                        # the yarn gets new colours
        for out in outs:
            np.testing.assert_array_equal(out[60, 60], img[60, 60])  # inside the fat label, off the yarn: untouched
            np.testing.assert_array_equal(out[20, 20], img[20, 20])  # surface untouched

    def test_background_recoloured_yarn_kept(self):
        img, label = _crop()
        for seed in range(5):
            out = recolor_tuft_background(img, label, np.random.default_rng(seed), tuft_p=0.0, background_p=1.0)
            self.assertLess(np.abs(out[65, 60].astype(int) - img[65, 60]).sum(), 40)  # yarn about the same
        changed = [np.abs(recolor_tuft_background(img, label, np.random.default_rng(s), 0.0, 1.0)[10, 10].astype(int)
                          - img[10, 10]).sum() for s in range(5)]
        self.assertGreater(max(changed), 20)

    def test_new_yarn_colour_keeps_contrast(self):
        img, label = _crop()
        for seed in range(10):
            out = recolor_tuft_background(img, label, np.random.default_rng(seed), tuft_p=1.0, background_p=0.0)
            g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(int)
            self.assertGreater(abs(g[65, 60] - g[20, 20]), 25)


class ActivityTests(unittest.TestCase):
    def test_busy_stretch_picked_jitter_and_title_card_not(self):
        rows = []
        for fr in range(0, 600):
            n = 0 if 400 <= fr < 430 else 20                      # title card: nothing found
            busy = 12 if 200 <= fr < 260 else (1 if fr == 50 else 0)   # sustained activity vs one jittery frame
            for k in range(n):
                rows.append({"frame": fr, "detected": "True", "stall": "separated" if k < busy else "attached"})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "detected", "stall"])
                w.writeheader()
                w.writerows(rows)
            picked = activity_frames(p, 2, fps=10.0, min_gap_s=2.0)
            self.assertEqual(len(picked), 2)
            self.assertTrue(all(195 <= fr <= 265 for fr in picked), picked)
            self.assertEqual(activity_frames(p, 3, 10.0, between=[0, 15]), sorted(activity_frames(p, 3, 10.0, between=[0, 15])))
            self.assertTrue(all(fr <= 150 for fr in activity_frames(p, 3, 10.0, between=[0, 15])))


class ShuffleTests(unittest.TestCase):
    def _grid(self):
        img = np.full((256, 256, 3), 225, np.uint8)
        tuft, root = np.zeros((256, 256), np.uint8), np.zeros((256, 256), np.uint8)
        for r in range(4):
            for c in range(3):
                x, y = 30 + 70 * c, 30 + 60 * r
                cv2.line(img, (x, y), (x + 45, y + 2), (30, 30, 30), 3)
                cv2.line(tuft, (x, y), (x + 45, y + 2), 255, 7)
                cv2.circle(root, (x, y), 4, 255, -1)
        lab = np.dstack([tuft, root, np.full_like(tuft, 255), np.zeros_like(tuft)])
        return img, lab

    def test_erased_tufts_are_gone_from_image_and_label(self):
        img, lab = self._grid()
        out, ol = shuffle_tufts(img, lab, np.random.default_rng(0), drop_frac=1.0, paste_max=0)
        self.assertEqual(int((ol[..., 0] > 127).sum()), 0)
        self.assertEqual(int((ol[..., 1] > 127).sum()), 0)
        self.assertGreater(int(out.min()), 120)                      # no dark yarn left

    def test_pasted_tufts_carry_their_label(self):
        img, lab = self._grid()
        n0 = cv2.connectedComponents((lab[..., 0] > 127).astype(np.uint8))[0]
        grew = False
        for seed in range(10):
            out, ol = shuffle_tufts(img, lab, np.random.default_rng(seed), drop_frac=0.0, paste_max=6)
            n = cv2.connectedComponents((ol[..., 0] > 127).astype(np.uint8))[0]
            grew |= n > n0
            dark = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) < 100
            self.assertLess((dark & ~(ol[..., 0] > 127)).sum(), 25)   # every pasted yarn pixel is labelled
        self.assertTrue(grew)

    def test_no_tufts_no_change(self):
        img = np.full((64, 64, 3), 200, np.uint8)
        lab = np.zeros((64, 64, 4), np.uint8)
        out, ol = shuffle_tufts(img, lab, np.random.default_rng(0))
        np.testing.assert_array_equal(out, img)


if __name__ == "__main__":
    unittest.main()
