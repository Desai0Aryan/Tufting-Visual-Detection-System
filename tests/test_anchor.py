import unittest

import cv2
import numpy as np

from detector import TuftDetector
from src.anchor import one_per_anchor, tape_distance
from src.backends import MaskResult

YELLOW = (0, 220, 240)  # BGR


def _scene():
    """Dark surface, white uniform-width strings (thickness can't tell root from
    tip), yellow tape at the RIGHT end of each anchored string, one stray."""
    img = np.full((200, 400, 3), 40, np.uint8)
    mask = np.zeros((200, 400), np.uint8)
    for y in (40, 100):
        cv2.line(img, (60, y), (160, y), (235, 235, 235), 3)
        cv2.line(mask, (60, y), (160, y), 255, 3)
        cv2.rectangle(img, (162, y - 6), (174, y + 6), YELLOW, -1)
    cv2.line(img, (230, 160), (340, 160), (235, 235, 235), 3)  # stray: no tape
    cv2.line(mask, (230, 160), (340, 160), 255, 3)
    return img, mask


class _MaskBackend:
    name = "oracle"

    def __init__(self, mask):
        self.mask = mask

    def predict(self, image):
        return MaskResult(self.mask)


class TapeAnchorTests(unittest.TestCase):
    def test_tape_distance_none_without_tape_or_color(self):
        self.assertIsNone(tape_distance(np.full((50, 50, 3), 40, np.uint8)))
        self.assertIsNone(tape_distance(np.full((50, 50), 40, np.uint8)))

    def test_root_is_the_taped_end_and_strays_are_rejected(self):
        img, mask = _scene()
        det = TuftDetector(backend=_MaskBackend(mask), tape_root=True, tape_max_dist=20)
        tufts = det.detect(img)
        self.assertEqual(len(tufts), 2)
        for t in tufts:
            self.assertGreater(t.root[0], t.tip[0])  # tape is on the right
            self.assertEqual(t.confidence, 1.0)
        self.assertEqual([t.id for t in tufts], [0, 1])

    def test_without_tape_root_the_stray_is_kept(self):
        img, mask = _scene()
        self.assertEqual(len(TuftDetector(backend=_MaskBackend(mask)).detect(img)), 3)

    def test_tape_works_inside_an_roi(self):
        img, mask = _scene()
        det = TuftDetector(backend=_MaskBackend(mask[20:130, 40:200]), tape_root=True,
                           tape_max_dist=20, roi=(40, 20, 160, 110))
        tufts = det.detect(img)
        self.assertEqual(len(tufts), 2)
        self.assertTrue(all(150 < t.root[0] < 175 for t in tufts))

    def test_one_per_anchor_keeps_longest(self):
        from src.tuft import Tuft
        mk = lambda root, length: Tuft(0, root, (0, 0), (1, 0), 0.0, length, np.zeros((1, 1, 2), np.int32))
        kept = one_per_anchor([mk((100, 50), 40), mk((105, 52), 80), mk((300, 50), 30)], max_dist=20)
        self.assertEqual(sorted(t.length_px for t in kept), [30, 80])

    def test_light_polarity_threads_through_to_classical_backend(self):
        img, _ = _scene()
        self.assertEqual(TuftDetector(polarity="light").backend.polarity, "light")
        with self.assertRaises(ValueError):
            TuftDetector(polarity="sideways")


if __name__ == "__main__":
    unittest.main()
