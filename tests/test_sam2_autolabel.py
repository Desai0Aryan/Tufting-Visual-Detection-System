import unittest

import cv2
import numpy as np

from src.mask_to_tufts import extract_tufts
from src.sam2_autolabel import prompt_for, refine, window_for


def _scene(length=400, width=9, n=3):
    """Long, thick strings like native-resolution footage (old tiling dropped these)."""
    img = np.full((1200, 1600, 3), 60, np.uint8)
    mask = np.zeros(img.shape[:2], np.uint8)
    for k in range(n):
        x = 300 + 400 * k
        cv2.line(img, (x, 200), (x + 60, 200 + length), (235, 235, 235), width)
        cv2.line(mask, (x, 200), (x + 60, 200 + length), 255, width)
    tufts = extract_tufts(mask, min_area=50, max_area=50000, max_length_px=2000)
    return img, mask, tufts


class StubPredictor:
    """Returns the true mask (or a variant) inside whatever window was set."""

    def __init__(self, full_mask, mode="exact"):
        self.full, self.mode, self.win, self.sets = full_mask > 0, mode, None, 0

    def set_image(self, rgb):
        self.shape = rgb.shape[:2]
        self.sets += 1

    def locate(self, x0, y0):
        self.win = (x0, y0)

    def predict(self, point_coords, point_labels, box, multimask_output):
        x0, y0 = self.win
        h, w = self.shape
        m = self.full[y0:y0 + h, x0:x0 + w].copy()
        if self.mode == "empty":
            m[:] = False
        elif self.mode == "speck":
            m[5:9, 5:9] = True
        return np.stack([m, np.zeros_like(m), m]), np.array([0.9, 0.1, 0.8]), None


class _Locating:
    """Wrap refine's set_image so the stub knows the window origin."""

    def __init__(self, stub, image):
        self.stub, self.image = stub, image

    def set_image(self, rgb):
        H, W = self.image.shape[:2]
        h, w = rgb.shape[:2]
        rgb_full = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        for y0 in range(H - h + 1):
            row = rgb_full[y0, :, :]
            for x0 in range(W - w + 1):
                if np.array_equal(row[x0:x0 + w], rgb[0]) and np.array_equal(rgb_full[y0:y0 + h, x0:x0 + w], rgb):
                    self.stub.locate(x0, y0)
                    self.stub.set_image(rgb)
                    return
        raise AssertionError("window not found")

    def predict(self, **kw):
        return self.stub.predict(**kw)


class Sam2AutolabelTests(unittest.TestCase):
    def test_window_always_contains_long_candidates(self):
        img, _, tufts = _scene(length=700)
        for t in tufts:
            x0, y0, x1, y1 = window_for(t, img.shape)
            bx, by, bw, bh = cv2.boundingRect(t.contour)
            self.assertTrue(x0 <= bx and y0 <= by and bx + bw <= x1 and by + bh <= y1)
            self.assertGreaterEqual(min(x1 - x0, y1 - y0), 512)

    def test_negative_points_clear_thick_yarn(self):
        img, mask, tufts = _scene(width=15)
        for t in tufts:
            pts, labels, box = prompt_for(t, (0, 0))
            for (x, y), l in zip(pts, labels):
                on = mask[int(round(y)), int(round(x))] > 0
                self.assertEqual(on, bool(l), (x, y, l))

    def _run(self, mode):
        # random noise background so window crops are unique and can be located
        img, mask, tufts = _scene()
        img = (img.astype(int) + np.random.default_rng(0).integers(0, 20, img.shape)).clip(0, 255).astype(np.uint8)
        stub = StubPredictor(mask, mode)
        return refine(img, tufts, _Locating(stub, img)), mask, tufts

    def test_long_tufts_are_refined_not_dropped(self):
        r, mask, tufts = self._run("exact")
        self.assertEqual(r.status, ["accepted"] * len(tufts))
        self.assertGreater(((r.mask > 0) & (mask > 0)).sum() / (mask > 0).sum(), 0.95)

    def test_disagreement_is_rejected_in_strict_mode(self):
        r, _, tufts = self._run("empty")
        self.assertEqual(r.status, ["rejected"] * len(tufts))
        self.assertEqual(int(r.mask.sum()), 0)

    def test_stray_components_are_dropped(self):
        r, mask, _ = self._run("speck")
        self.assertEqual(int(((r.mask > 0) & ~(mask > 0)).sum()), 0)


if __name__ == "__main__":
    unittest.main()
