import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.label_io import load_frame


def _frames_csv(d, use):
    Path(d, "frames", "c1").mkdir(parents=True)
    img = np.full((360, 640, 3), 90, np.uint8)
    for x in (100, 250, 400):
        cv2.line(img, (x, 60), (x + 15, 260), (245, 245, 245), 5)
    f = Path(d, "frames", "c1", "c1__f000010.png")
    cv2.imwrite(str(f), img)
    with open(f"{d}/frames/frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "frame_idx", "time_s", "brightness", "contrast",
                                           "motion", "use", "split", "file"])
        w.writeheader()
        w.writerow({"clip_id": "c1", "frame_idx": 10, "time_s": 0.3, "use": use, "split": "train",
                    "file": f.as_posix()})
    return img


class _ExactSam:
    """Stub image predictor: returns the candidate region (what a perfect SAM would)."""

    def set_image(self, rgb):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.m = g > 200

    def predict(self, point_coords, point_labels, box, multimask_output):
        x0, y0, x1, y1 = [int(v) for v in box]
        m = np.zeros_like(self.m)
        m[max(y0, 0):y1, max(x0, 0):x1] = self.m[max(y0, 0):y1, max(x0, 0):x1]
        return np.stack([m, m, m]), np.array([0.9, 0.9, 0.9]), None


class AutoLabelTests(unittest.TestCase):
    def test_negative_clip_frames_become_empty_proposals(self):
        from tools import auto_label
        with tempfile.TemporaryDirectory() as d:
            _frames_csv(d, "negative")
            rc = auto_label.main(["--frames-csv", f"{d}/frames/frames.csv", "--out", f"{d}/proposed",
                                  "--labeled", f"{d}/labeled"])
            self.assertEqual(rc, 0)
            img, mask, info, meta = load_frame(f"{d}/proposed", "c1__f000010")
            self.assertEqual(int(mask.sum()), 0)
            self.assertEqual((meta["source"], meta["status"], meta["clip_id"]), ("negative", "proposed", "c1"))
            # rerun: already proposed -> skipped
            self.assertEqual(auto_label.main(["--frames-csv", f"{d}/frames/frames.csv", "--out",
                                              f"{d}/proposed", "--labeled", f"{d}/labeled"]), 0)

    def test_label_frame_with_stub_sam_gives_geometry(self):
        from src.config import load_config
        from tools.auto_label import detector_settings, label_frame
        with tempfile.TemporaryDirectory() as d:
            img = _frames_csv(d, "tufts")
        cfg = load_config("configs/autolabel.yaml")
        s = detector_settings(cfg, "c1")
        self.assertEqual(s["polarity"], "auto")
        mask, tufts, rejected, st = label_frame(img, s, _ExactSam(), cfg)
        self.assertEqual(st["candidates"], 3)
        self.assertEqual(st["accepted"], 3)
        self.assertEqual(len(tufts), 3)
        for t in tufts:  # tip tangent points down the strings (towards larger y)
            self.assertLess(t.vector[1], 0) if t.root[1] > t.tip[1] else self.assertGreater(t.vector[1], 0)

    def test_per_clip_settings_override_default(self):
        from src.config import load_config
        from tools.auto_label import detector_settings
        cfg = load_config("configs/autolabel.yaml")
        s = detector_settings(cfg, "own_car_static")
        self.assertEqual((s["polarity"], s["tape_root"], s["c"]), ("light", True, 30))
        self.assertEqual(detector_settings(cfg, "unknown")["polarity"], "auto")


if __name__ == "__main__":
    unittest.main()
