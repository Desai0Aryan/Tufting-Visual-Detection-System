import unittest

import cv2
import numpy as np

from src.sam_click import best_tuft, click_window, tuft_from_click


def _masks():
    """SAM-like nested guesses: the string; string + a long panel seam; a big blob."""
    string = np.zeros((200, 300), np.uint8)
    cv2.line(string, (200, 100), (130, 104), 1, 3)
    seam = string.copy()
    cv2.line(seam, (130, 104), (130, 10), 1, 2)
    cv2.line(seam, (130, 104), (130, 195), 1, 2)
    blob = np.zeros_like(string)
    cv2.circle(blob, (195, 100), 40, 1, -1)
    return np.stack([seam, string, blob])


class FakePredictor:
    def __init__(self, masks):
        self.masks, self.image = masks, None

    def set_image(self, rgb):
        self.image = rgb

    def predict(self, **kw):
        h, w = self.image.shape[:2]
        return self.masks[:, :h, :w], None, None


class SamClickTests(unittest.TestCase):
    def test_smallest_string_wins_root_at_click(self):
        t = best_tuft(_masks(), (196, 100), max_len=150)
        self.assertIsNotNone(t)
        self.assertLess(np.hypot(t.root[0] - 196, t.root[1] - 100), 8)      # root at the click end
        self.assertLess(abs(t.length_px - 70), 12)                            # the string, not string + seam
        self.assertGreater(abs(t.angle_deg), 150)                             # pointing left

    def test_seam_rejected_by_length_blob_by_shape(self):
        seam, _, blob = _masks()
        self.assertIsNone(best_tuft(np.stack([seam]), (196, 100), max_len=150))
        self.assertIsNone(best_tuft(np.stack([blob]), (196, 100)))

    def test_click_far_from_any_mask(self):
        self.assertIsNone(best_tuft(_masks()[1:2], (20, 180)))

    def test_window_coords_mapped_back(self):
        img = np.zeros((600, 900, 3), np.uint8)
        x0, y0, x1, y1 = click_window((700, 500), img.shape, 300)
        self.assertEqual((x1 - x0, y1 - y0), (300, 300))
        self.assertTrue(x1 <= 900 and y1 <= 600)
        m = np.zeros((1, 300, 300), np.uint8)
        cv2.line(m[0], (700 - x0, 500 - y0), (700 - x0 - 80, 500 - y0 + 4), 1, 3)
        st = tuft_from_click(img, (700, 500), FakePredictor(m), window=300)
        self.assertIsNotNone(st)
        self.assertLess(np.hypot(*(st[0] - [700, 500])), 8)


if __name__ == "__main__":
    unittest.main()
