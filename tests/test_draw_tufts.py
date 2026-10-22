import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.label_io import load_frame, load_ignore
from tools.draw_tufts import load_strokes, save_strokes, stroke_near
from tools.drawn_to_labels import frame_labels, main as to_labels


class DrawTests(unittest.TestCase):
    def test_strokes_roundtrip_and_hit_test(self):
        strokes = [np.array([[10, 10], [60, 12], [100, 20]], float), np.array([[10, 100], [80, 110]], float)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "json" / "a.json"
            save_strokes(p, "a", 7, strokes, (320, 200))
            back = load_strokes(p)
            self.assertEqual(json.loads(p.read_text())["frame_idx"], 7)
        self.assertEqual(len(back), 2)
        np.testing.assert_allclose(back[0], strokes[0])
        self.assertEqual(stroke_near(strokes, (62, 14), 5), 0)
        self.assertIsNone(stroke_near(strokes, (200, 180), 5))


class LabelTests(unittest.TestCase):
    def test_line_becomes_tuft_with_root_at_start(self):
        img = np.full((200, 300, 3), 220, np.uint8)
        st = np.array([[200, 100], [150, 102], [100, 110]], float)       # root right, tip left
        model = np.zeros((200, 300), bool)
        model[20:24, 20:80] = True                                       # a tuft the user skipped
        tuft, ignore, tufts = frame_labels(img, [st], model_mask=model)
        self.assertTrue(tuft[101, 175])
        self.assertFalse(tuft[50, 175])
        self.assertTrue(ignore[22, 50])                                  # skipped tuft: ignore, not background
        self.assertFalse(ignore[180, 280])                               # far away: background
        self.assertFalse((tuft & ignore).any())
        self.assertEqual(tufts[0].root, (200.0, 100.0))
        self.assertGreater(abs(tufts[0].angle_deg), 150)                 # pointing left

    def test_tool_writes_train_and_val(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            img = np.full((120, 160, 3), 220, np.uint8)
            for fi in (10, 500):
                stem = f"v__f{fi:06d}"
                (d / "drawn" / "images").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / "drawn" / "images" / f"{stem}.png"), img)
                save_strokes(d / "drawn" / "json" / f"{stem}.json", stem, fi, [np.array([[100, 60], [40, 62]])], (160, 120),
                             checked=True)
            save_strokes(d / "drawn" / "json" / "v__f000700.json", "v__f000700", 700,     # pre-filled, never checked
                         [np.array([[100, 60], [40, 62]])], (160, 120))
            cv2.imwrite(str(d / "drawn" / "images" / "v__f000700.png"), img)
            rc = to_labels([str(d / "drawn"), "--clip", "v", "--val-from", "400", "--out", str(d / "lab"),
                            "--no-model-ignore"])
            self.assertEqual(rc, 0)
            _, m, info, meta = load_frame(d / "lab", "v_train__f000010")
            self.assertEqual(meta["split"], "train")
            self.assertEqual(int(m[61, 70]), 255)
            self.assertEqual(len(info["tufts"]), 1)
            _, _, _, meta = load_frame(d / "lab", "v_val__f000500")
            self.assertEqual(meta["clip_id"], "v_val")
            self.assertIsNotNone(load_ignore(d / "lab", "v_val__f000500"))   # the thin border
            self.assertFalse((d / "lab" / "images" / "v_val__f000700.png").exists())


if __name__ == "__main__":
    unittest.main()
