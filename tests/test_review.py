import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.label_io import load_frame, save_frame, tufts_from_mask
from src.review_core import ReviewState


def _frame():
    img = np.full((400, 600, 3), 50, np.uint8)
    mask = np.zeros((400, 600), np.uint8)
    for x in (100, 250):
        cv2.line(img, (x, 50), (x + 20, 350), (240, 240, 240), 5)   # 300 px: longer than the old 260 px cap
        cv2.line(mask, (x, 50), (x + 20, 350), 255, 5)
    rejected = np.array([[430, 60], [440, 60], [460, 340], [450, 340]])
    info = {"tufts": [], "rejected_candidates": [{"status": "rejected", "contour": rejected.tolist()}]}
    return img, mask, info


class ReviewStateTests(unittest.TestCase):
    def test_click_deletes_only_that_tuft_and_undo_restores(self):
        img, mask, info = _frame()
        s = ReviewState(img, mask, info)
        self.assertEqual(len(s.tufts), 2)
        self.assertEqual(s.click(103, 120), "deleted")
        self.assertEqual(len(s.tufts), 1)
        self.assertEqual(int(s.mask[200, 260] > 0), 1)
        s.undo()
        np.testing.assert_array_equal(s.mask, mask)

    def test_restore_rejected_candidate(self):
        img, mask, info = _frame()
        s = ReviewState(img, mask, info)
        self.assertEqual(s.click(445, 200), "restored")
        self.assertEqual(len(s.tufts), 3)
        self.assertIsNone(s.click(560, 380))  # empty space: nothing happens

    def test_brush_add_and_erase(self):
        img, mask, info = _frame()
        s = ReviewState(img, mask, info)
        s.begin_stroke(); s.paint((500, 50), (520, 300), 2, add=True); s.end_stroke(True, 2)
        self.assertEqual(len(s.tufts), 3)
        s.begin_stroke(); s.paint((90, 200), (140, 200), 6, add=False); s.end_stroke(False, 6)
        self.assertEqual(int(s.mask[200, 105]), 0)  # cut through the first tuft
        summ = s.summary("accept")
        self.assertGreater(summ["pixels_added"], 0)
        self.assertGreater(summ["pixels_removed"], 0)
        self.assertEqual([e["action"] for e in summ["edits"]], ["brush_add", "brush_erase"])


class ReviewAppTests(unittest.TestCase):
    def _data(self, d):
        img, mask, info = _frame()
        tufts = tufts_from_mask(mask)
        for k in range(2):
            save_frame(f"{d}/proposed", f"clipA__f{k:06d}", img, mask, tufts,
                       {"clip_id": "clipA", "frame_idx": k, "split": "train", "use": "tufts",
                        "source": "sam2", "status": "proposed"},
                       rejected=info["rejected_candidates"])
        return mask

    def test_accept_writes_exact_mask_and_reject_is_logged(self):
        from tools.review_labels import ReviewApp
        with tempfile.TemporaryDirectory() as d:
            mask = self._data(d)
            app = ReviewApp(f"{d}/proposed", f"{d}/labeled", ["clipA__f000000", "clipA__f000001"],
                            f"{d}/review_log.jsonl")
            self.assertEqual(app.render().shape[2], 3)
            self.assertTrue(app.on_key(ord("a")))            # accept first, move on
            self.assertFalse(app.on_key(ord("r")))           # reject last -> done
            _, saved, info, meta = load_frame(f"{d}/labeled", "clipA__f000000")
            np.testing.assert_array_equal(saved > 0, mask > 0)  # no pixels lost on save
            self.assertEqual(len(info["tufts"]), 2)
            self.assertEqual((meta["status"], meta["proposed_by"], meta["clip_id"]), ("accepted", "sam2", "clipA"))
            self.assertFalse(Path(d, "labeled", "images", "clipA__f000001.png").exists())
            prop_meta = json.loads(Path(d, "proposed", "meta.json").read_text())
            self.assertEqual(prop_meta["clipA__f000001"]["status"], "rejected")
            log = [json.loads(l) for l in open(f"{d}/review_log.jsonl")]
            self.assertEqual([r["decision"] for r in log], ["accept", "reject"])

    def test_mouse_click_in_view_coordinates_and_zoom(self):
        from tools.review_labels import ReviewApp
        import cv2 as _cv
        with tempfile.TemporaryDirectory() as d:
            self._data(d)
            app = ReviewApp(f"{d}/proposed", f"{d}/labeled", ["clipA__f000000"], f"{d}/log.jsonl",
                            max_w=300, max_h=200)  # display at half size
            app.on_mouse(_cv.EVENT_LBUTTONDOWN, 103 * 0.5, 120 * 0.5, 0)
            self.assertEqual(len(app.state.tufts), 1)
            before = app.to_image(100, 60)
            app.on_mouse(_cv.EVENT_MOUSEWHEEL, 100, 60, 1)
            after = app.to_image(100, 60)
            np.testing.assert_allclose(before, after, atol=1e-6)  # zoom keeps the cursor point fixed
            app.on_key(ord("b")); app.on_key(ord("]"))
            self.assertEqual((app.mode, app.radius), ("add", 4))


if __name__ == "__main__":
    unittest.main()
