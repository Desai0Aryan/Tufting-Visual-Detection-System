import tempfile
import unittest
from dataclasses import dataclass

import cv2
import numpy as np

from core.tracker import (ATTACHED, SEPARATED, UNKNOWN, UNSTEADY, StallConfig, StaticPredictor, Tracker,
                          TrackerConfig, circ_mean_deg, circ_std_deg)


@dataclass
class Det:
    root: tuple
    tip: tuple
    angle_deg: float
    length_px: float
    vector: tuple = (1.0, 0.0)
    confidence: float = 1.0
    path: object = None


def det(x, y, angle=0.0, length=60.0):
    a = np.radians(angle)
    return Det((x, y), (x + length * np.cos(a), y - length * np.sin(a)), angle, length,
               (float(np.cos(a)), float(-np.sin(a))))


def tracker(**kw):
    stall = StallConfig(window_s=1.0, min_samples=5)
    return Tracker(TrackerConfig(max_dist_px=25, n_confirm=3, max_missing=5, stall=stall, **kw),
                   predictor=StaticPredictor(), fps=10)


GRAY = np.zeros((10, 10), np.uint8)


class CircularStatsTests(unittest.TestCase):
    def test_wraparound(self):
        self.assertAlmostEqual(abs(circ_mean_deg([179, -179])), 180, places=4)
        self.assertLess(circ_std_deg([179, -179, 180]), 2)
        self.assertGreater(circ_std_deg([0, 90, 180, -90]), 100)


class TrackerTests(unittest.TestCase):
    def test_ids_persist_and_confirm(self):
        tr = tracker()
        for f in range(10):
            out = tr.update(GRAY, [det(100 + 2 * f, 50), det(300 + 2 * f, 50)], f)
        self.assertEqual(sorted(t.track_id for t in out), [1, 2])
        self.assertTrue(all(t.state == "confirmed" for t in out))

    def test_tentative_until_confirmed_and_dropped_on_miss(self):
        tr = tracker()
        out = tr.update(GRAY, [det(100, 50)], 0)
        self.assertEqual(out[0].state, "tentative")
        out = tr.update(GRAY, [], 1)
        self.assertEqual(out, [])  # tentative + miss -> gone

    def test_missing_keeps_id_then_drops(self):
        tr = tracker()
        for f in range(4):
            tr.update(GRAY, [det(100, 50)], f)
        for f in range(4, 7):  # 3 misses <= max_missing
            out = tr.update(GRAY, [], f)
            self.assertEqual(out[0].state, "missing")
            self.assertFalse(out[0].tuft)
        out = tr.update(GRAY, [det(102, 51)], 7)
        self.assertEqual((out[0].track_id, out[0].state), (1, "confirmed"))
        for f in range(8, 15):
            out = tr.update(GRAY, [], f)
        self.assertEqual(out, [])  # dropped after max_missing

    def test_gate_prevents_long_jumps(self):
        tr = tracker()
        for f in range(4):
            tr.update(GRAY, [det(100, 50)], f)
        out = tr.update(GRAY, [det(200, 50)], 4)  # 100 px away: new track, old one missing
        self.assertEqual(sorted((t.track_id, t.state) for t in out), [(1, "missing"), (2, "tentative")])

    def test_stall_labels(self):
        tr = tracker()
        rng = np.random.default_rng(0)
        for f in range(12):
            dets = [det(100, 50, 0 + rng.normal(0, 2)),                       # steady
                    det(200, 50, rng.uniform(-60, 60)),                       # fluttering
                    det(300, 50, 0, 60 if f < 6 else 35),                     # shortening (curling)
                    det(400, 50, 180 + rng.normal(0, 3)),                     # reversed vs bulk flow
                    det(500, 50, 2), det(600, 50, -2)]                        # more steady ones
            out = {t.track_id: t for t in tr.update(GRAY, dets, f)}
        self.assertEqual(out[1].stall, ATTACHED)
        self.assertIn(out[2].stall, (UNSTEADY, SEPARATED))
        self.assertGreater(out[2].circ_std_deg, 15)
        self.assertIn(out[3].stall, (UNSTEADY, SEPARATED))
        self.assertLess(out[3].length_ratio, 0.8)
        self.assertEqual(out[4].stall, SEPARATED)
        self.assertTrue(out[4].reversed)

    def test_root_tip_flips_are_corrected_and_keep_the_id(self):
        tr = tracker()
        for f in range(6):
            d = det(100, 50, 0, 60)
            if f % 2:  # detector swaps the ends every other frame
                from core.tracker import flip_detection
                d = flip_detection(d)
            out = tr.update(GRAY, [d], f)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].track_id, 1)
        self.assertAlmostEqual(out[0].root[0], 100, places=4)
        self.assertLess(abs(out[0].angle_deg), 1e-6 + 1.0)

    def test_unknown_until_enough_history(self):
        tr = tracker()
        out = tr.update(GRAY, [det(100, 50)], 0)
        self.assertEqual(out[0].stall, UNKNOWN)


def write_tuft_video(path, n=30, w=320, h=240, fps=10):
    """White strings on a dark surface; one sways, the scene drifts 1 px/frame."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        for k, x in enumerate((60, 140, 220)):
            sway = 10 * np.sin(f / 2) if k == 1 else 0
            x0 = x + f // 3
            cv2.line(img, (x0, 60), (int(x0 + 50 + sway), int(150 + sway)), (235, 235, 235), 3)
        vw.write(img)
    vw.release()


class PipelineTests(unittest.TestCase):
    def test_process_frame_contract_and_tracking(self):
        from core import Pipeline, PipelineConfig, VideoFileSource
        with tempfile.TemporaryDirectory() as d:
            write_tuft_video(f"{d}/v.mp4")
            src = VideoFileSource(f"{d}/v.mp4")
            pipe = Pipeline(PipelineConfig(backend="classical", polarity="light", block_size=31, c=15,
                                           min_length_px=40), fps=src.fps)
            results = [pipe.process_frame(fr, i) for i, fr in src]
            src.close()
        self.assertEqual(len(results), 30)
        last = results[-1]
        self.assertEqual(last.index, 29)
        self.assertEqual(last.shape, (240, 320))
        vis = last.visible()
        self.assertEqual(len(vis), 3)
        self.assertEqual(sorted(t.track_id for t in vis), [1, 2, 3])  # same IDs from frame 0
        m = last.mask()
        self.assertEqual(m.shape, (240, 320))
        self.assertGreater(int((m > 0).sum()), 100)
        for t in vis:
            self.assertAlmostEqual(float(np.hypot(*t.vector)), 1.0, places=4)
        self.assertIsNotNone(pipe.history(1))

    def test_unet_without_checkpoint_is_a_clear_error(self):
        from core import Pipeline, PipelineConfig
        with self.assertRaises(FileNotFoundError):
            Pipeline(PipelineConfig(backend="unet", checkpoint="nope.pt"))


if __name__ == "__main__":
    unittest.main()
