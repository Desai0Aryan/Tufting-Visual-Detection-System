import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from main import main as run_main, parse_args
from src.synth import make_scene
from src.tuft import Tuft
from src.video_io import VideoReader, VideoWriter, is_video_path
from tracker import TuftTracker


def _write_video(path, n=8, seed=3, shift_per_frame=1):
    """A short clip: the same synthetic tuft field, nudged a pixel per frame (a stand-in for
    a camera that is not perfectly still)."""
    sc = make_scene(seed=seed)
    h, w = sc.image.shape[:2]
    with VideoWriter(path, 10, (w, h)) as vw:
        for i in range(n):
            m = np.float32([[1, 0, i * shift_per_frame], [0, 1, 0]])
            vw.write(cv2.warpAffine(sc.image, m, (w, h), borderMode=cv2.BORDER_REPLICATE))
    return sc, sc.image.mean() < 110


def _tuft(i, x, y):
    return Tuft(id=i, root=(x, y), tip=(x + 20, y), vector=(1.0, 0.0), angle_deg=0.0,
                length_px=20.0, contour=np.zeros((3, 1, 2), np.int32))


class VideoCliTests(unittest.TestCase):
    def test_video_detected_by_extension_and_flag(self):
        self.assertTrue(is_video_path("a/b.MP4"))
        self.assertFalse(is_video_path("a/b.jpg"))
        self.assertTrue(parse_args(["clip.mp4"]).is_video)
        self.assertTrue(parse_args(["--video", "clip"]).is_video)
        self.assertFalse(parse_args(["--image", "x.jpg"]).is_video)
        with self.assertRaises(SystemExit):
            parse_args(["a.mp4", "--video", "b.mp4"])

    def test_end_to_end_video_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "clip.mp4"
            sc, light = _write_video(src, n=8)
            out = Path(d) / "out"
            args = [str(src), "--outdir", str(out), "--polarity", "light" if light else "dark"]
            self.assertEqual(run_main(args + ["--frame-step", "2", "--save-frames"]), 0)

            video = next(out.glob("clip_*_annotated.mp4"))
            with VideoReader(video) as r:
                self.assertEqual((r.width, r.height), (sc.image.shape[1], sc.image.shape[0]))
                self.assertEqual(sum(1 for _ in r.frames()), 4)   # 8 frames, every 2nd
                self.assertAlmostEqual(r.fps, 5.0, places=1)       # fps / step

            df = pd.read_csv(next(out.glob("clip_*_vectors.csv")))
            self.assertEqual(sorted(df.frame.unique()), [0, 2, 4, 6])
            self.assertAlmostEqual(df.timestamp_s.max(), 0.6, places=2)
            per_frame = df.groupby("frame").size()
            self.assertTrue((per_frame > 10).all())
            # tracking: most tufts exist in every processed frame under one ID
            persistent = (df.groupby("tuft_id").frame.nunique() == 4).sum()
            self.assertGreater(persistent / df.tuft_id.nunique(), 0.8)
            self.assertEqual(len(list(next(out.glob("clip_*_frames")).glob("*.png"))), 4)

    def test_bad_video_and_roi_report_errors(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_main([str(Path(d) / "missing.mp4"), "--outdir", d]), 1)
            bad = Path(d) / "bad.mp4"
            bad.write_bytes(b"not a video")
            self.assertEqual(run_main([str(bad), "--outdir", d]), 1)
            src = Path(d) / "clip.mp4"
            _write_video(src, n=2)
            self.assertEqual(run_main([str(src), "--outdir", d, "--roi", "0,0,99999,99999"]), 1)


class TrackerTests(unittest.TestCase):
    def test_ids_persist_when_roots_drift_and_order_changes(self):
        tr = TuftTracker(max_dist=10)
        a = tr.update([_tuft(0, 50, 50), _tuft(1, 200, 50)])
        b = tr.update([_tuft(0, 203, 52), _tuft(1, 52, 51)])   # order swapped, small drift
        self.assertEqual({t.id for t in a}, {0, 1})
        self.assertEqual([t.id for t in b], [1, 0])

    def test_new_tuft_gets_new_id_and_missing_tuft_is_retired(self):
        tr = TuftTracker(max_dist=10, max_missing=1)
        tr.update([_tuft(0, 50, 50)])
        out = tr.update([_tuft(0, 50, 50), _tuft(1, 300, 300)])
        self.assertEqual([t.id for t in out], [0, 1])
        tr.update([_tuft(0, 300, 300)])   # tuft 0 missing 1 frame (still remembered)
        tr.update([_tuft(0, 300, 300)])   # missing 2 > max_missing -> retired
        again = tr.update([_tuft(0, 50, 50)])
        self.assertEqual(again[0].id, 2)  # fresh ID, not recycled

    def test_input_tufts_are_not_mutated(self):
        tr = TuftTracker()
        t = _tuft(7, 10, 10)
        tr.update([_tuft(0, 10, 10)])
        out = tr.update([t])
        self.assertEqual(t.id, 7)
        self.assertEqual(out[0].id, 0)


if __name__ == "__main__":
    unittest.main()
