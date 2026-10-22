import unittest

import cv2
import numpy as np

from src.card_groups import GridPage, combine, crop_for_embedding, kmeans, make_groups, shape_features
from src.tuft_cards import frame_cards


def _cards():
    """A thin line and a fat blob, both bright on a dark frame."""
    img = np.full((300, 400, 3), 40, np.uint8)
    mask = np.zeros((300, 400), np.uint8)
    cv2.line(img, (50, 50), (60, 250), (230, 230, 230), 3)
    cv2.line(mask, (50, 50), (60, 250), 255, 3)
    cv2.ellipse(img, (280, 150), (60, 35), 0, 0, 360, (230, 230, 230), -1)
    cv2.ellipse(mask, (280, 150), (60, 35), 0, 0, 360, 255, -1)
    return img, frame_cards("f", mask, {})


class FeatureTests(unittest.TestCase):
    def test_shape_tells_thin_line_from_blob(self):
        img, (line, blob) = _cards()
        fl, fb = shape_features(img, line), shape_features(img, blob)
        self.assertGreater(fl[2], fb[2] + 1.0)     # elongation
        self.assertLess(fl[1], fb[1])              # width
        self.assertGreater(fl[3], 0.5)             # bright against a dark ring

    def test_crop_dims_outside_candidate(self):
        img, (line, _) = _cards()
        crop = crop_for_embedding(img, line)
        self.assertEqual(crop.shape, (128, 128, 3))
        self.assertLess(crop[5, 5].max(), 40 / 255 * 0.41 + 1e-3)

    def test_combine_keeps_cnn_part(self):
        cnn = np.eye(3, dtype=np.float32)
        out = combine(cnn, np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], np.float32))
        self.assertEqual(out.shape, (3, 7))
        np.testing.assert_array_equal(out[:, :3], cnn)


class ClusterTests(unittest.TestCase):
    def test_kmeans_separates_blobs(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, 0.1, (50, 4)), rng.normal(5, 0.1, (50, 4))])
        lab = kmeans(x, 2, device="cpu")
        self.assertEqual(len(set(lab[:50])), 1)
        self.assertEqual(len(set(lab[50:])), 1)
        self.assertNotEqual(lab[0], lab[-1])

    def test_groups_capped_centre_first_largest_first(self):
        rng = np.random.default_rng(1)
        x = np.vstack([rng.normal(0, 1, (300, 8)), rng.normal(20, 1, (30, 8))])
        groups = make_groups(x, per_group=100, max_size=48)
        self.assertEqual(sorted(np.concatenate(groups).tolist()), list(range(330)))
        self.assertTrue(all(len(g) <= 48 for g in groups))
        self.assertEqual([len(g) for g in groups], sorted((len(g) for g in groups), reverse=True))
        g = groups[0]
        d = np.linalg.norm(x[g] - x[g].mean(0), axis=1)
        self.assertTrue(np.all(np.diff(d) >= 0))

    def test_identical_points_are_chunked(self):
        groups = make_groups(np.zeros((100, 3)), per_group=100, max_size=48)
        self.assertTrue(all(len(g) <= 48 for g in groups))
        self.assertEqual(sum(len(g) for g in groups), 100)


class GridPageTests(unittest.TestCase):
    def test_key_fills_only_cells_without_own_vote(self):
        p = GridPage(["a", "b", "c"])
        p.click(1)                 # n
        p.click(2); p.click(2)     # y
        self.assertEqual(p.commit("u"), {"a": "u", "b": "n", "c": "y"})
        p.click(1); p.click(1); p.click(1)   # back to none
        self.assertEqual(p.commit("n")["b"], "n")
        p.click(99)                # outside the page: nothing
        self.assertEqual(len(p.commit("y")), 3)

    def test_page_renders(self):
        from tools.vote_groups import render_page
        img, cards = _cards()
        p = GridPage([c.id for c in cards])
        p.click(0)
        canvas = render_page(cards, p, lambda stem: img, "status")
        self.assertEqual(canvas.ndim, 3)


if __name__ == "__main__":
    unittest.main()
