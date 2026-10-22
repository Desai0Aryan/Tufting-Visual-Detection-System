import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.frame_select import farthest_point_sample, per_clip_quota, scan_video
from src.sources import add_row, make_row


def segmented_video(path, fps=30):
    """60 frames calm/dim, 60 frames bright, 60 frames fluttering tufts."""
    h, w = 360, 640
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(60, 140, (h, w, 3)).astype(np.uint8), (0, 0), 5)
    for k in range(180):
        f = base.copy()
        if 60 <= k < 120:
            f = cv2.convertScaleAbs(f, alpha=1.0, beta=90)
        for j in range(5):
            x = 80 + 110 * j
            wob = int(40 * np.sin(k * 1.3 + j)) if k >= 120 else 0
            cv2.line(f, (x, 120), (x + 20 + wob, 240), (245, 245, 245), 3)
        vw.write(f)
    vw.release()


class FrameSelectTests(unittest.TestCase):
    def test_fps_covers_all_clusters(self):
        rng = np.random.default_rng(0)
        centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], float)
        X = np.vstack([c + rng.normal(0, 0.3, (25, 2)) for c in centers])
        idx = farthest_point_sample(X, 4)
        self.assertEqual(sorted({int(i) // 25 for i in idx}), [0, 1, 2, 3])

    def test_min_time_gap(self):
        X = np.random.default_rng(1).random((50, 3))
        times = np.arange(50) * 0.1
        idx = farthest_point_sample(X, 50, min_gap_s=1.0, times=times)
        t = np.sort(times[idx])
        self.assertTrue((np.diff(t) >= 1.0 - 1e-9).all())
        self.assertLessEqual(len(idx), 5)

    def test_quota_is_balanced_and_capped(self):
        q = per_clip_quota({"a": 100, "b": 5, "c": 100}, 60)
        self.assertEqual(q["b"], 5)
        self.assertEqual(sum(q.values()), 60)
        self.assertLessEqual(abs(q["a"] - q["c"]), 1)

    def test_near_duplicates_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            segmented_video(f"{d}/v.mp4")
            cands, fps = scan_video(f"{d}/v.mp4", "v", stride=5)
            calm = [c for c in cands if c.frame_idx < 60]
            self.assertLessEqual(len(calm), 2)          # 12 identical calm frames -> ~1 kept
            flutter = [c for c in cands if c.frame_idx >= 120]
            self.assertGreater(len(flutter), 5)          # moving tufts are not duplicates
            self.assertGreater(np.mean([c.motion for c in flutter]), 2 * max(c.motion for c in calm) + 0.1)

    def test_cli_picks_bright_and_fluttering_frames(self):
        from tools import extract_frames
        with tempfile.TemporaryDirectory() as d:
            Path(d, "clips").mkdir()
            segmented_video(f"{d}/clips/own_v.mp4")
            add_row(f"{d}/s.csv", make_row("own_v", "train", "tufts", "own"))
            rc = extract_frames.main(["--sources", f"{d}/s.csv", "--clips-dir", f"{d}/clips",
                                      "--out", f"{d}/frames", "--total", "3", "--stride", "5",
                                      "--min-gap-s", "0.3"])
            self.assertEqual(rc, 0)
            rows = list(csv.DictReader(open(f"{d}/frames/frames.csv", newline="")))
            self.assertEqual(len(rows), 3)
            segs = {int(r["frame_idx"]) // 60 for r in rows}
            self.assertEqual(segs, {0, 1, 2})            # one from each regime
            img = cv2.imread(rows[0]["file"])
            self.assertEqual(img.shape[:2], (360, 640))  # native resolution
            # rerun: nothing new is duplicated
            extract_frames.main(["--sources", f"{d}/s.csv", "--clips-dir", f"{d}/clips",
                                 "--out", f"{d}/frames", "--total", "3", "--stride", "5"])
            rows2 = list(csv.DictReader(open(f"{d}/frames/frames.csv", newline="")))
            self.assertEqual(len({(r["clip_id"], r["frame_idx"]) for r in rows2}), len(rows2))


if __name__ == "__main__":
    unittest.main()
