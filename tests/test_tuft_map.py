import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.map_pipeline import MapPipeline
from core.tuft_map import Anchor, TuftMap, anchors_from_detections, drop_tip_duplicates

ROOTS = [(100 + 150 * c, 80 + 90 * r) for r in range(3) for c in range(4)]   # 12 tufts, flow along -x
L = 70.0


def _scan(n_frames=40, seed=0, flow=(-1.0, 0.0)):
    """Detections over a video: fixed roots, tips wandering around root + L*flow,
    the end order a coin flip (like the model)."""
    rng = np.random.default_rng(seed)
    f = np.array(flow)
    dets = []
    for fr in range(n_frames):
        for r in ROOTS:
            ang = rng.normal(0, 0.35)
            c, s = np.cos(ang), np.sin(ang)
            d = np.array([c * f[0] - s * f[1], s * f[0] + c * f[1]])
            root = np.array(r) + rng.normal(0, 1.0, 2)
            tip = root + L * d
            ends = (tuple(root), tuple(tip)) if rng.random() < 0.5 else (tuple(tip), tuple(root))
            dets.append((fr, ends[0], ends[1], L))
    return dets


class MapBuildTests(unittest.TestCase):
    def test_roots_found_and_flow_estimated_despite_coin_flip_ends(self):
        for flow in [(-1.0, 0.0), (0.0, 1.0)]:
            anchors, f = anchors_from_detections(_scan(flow=flow), 40)
            self.assertGreater(float(np.dot(f, flow)), 0.95)
            self.assertEqual(len(anchors), len(ROOTS))
            for a in anchors:
                d = min(np.hypot(a.root[0] - r[0], a.root[1] - r[1]) for r in ROOTS)
                self.assertLess(d, 6.0)
                v = np.array([np.cos(np.radians(a.angle_deg)), -np.sin(np.radians(a.angle_deg))])
                self.assertGreater(float(v @ flow), 0.8)

    def test_given_flow_puts_roots_at_the_upstream_end(self):
        anchors, _ = anchors_from_detections(_scan(), 40, flow=(1.0, 0.0))   # deliberately wrong
        tips = [(r[0] - L, r[1]) for r in ROOTS]
        self.assertTrue(anchors)
        for a in anchors:                      # every root lands on the (wandering) tip end, not the tape
            to_tip = min(np.hypot(a.root[0] - t[0], a.root[1] - t[1]) for t in tips)
            to_root = min(np.hypot(a.root[0] - r[0], a.root[1] - r[1]) for r in ROOTS)
            self.assertLess(to_tip, to_root)

    def test_root_seen_with_and_without_tape_merges(self):
        a = Anchor(1, (200.0, 100.0), L, 180.0, 0.9)        # tape end, pointing -x
        b = Anchor(2, (175.0, 101.0), L, 0.0, 0.95)         # where the string leaves the tape
        c = Anchor(3, (200.0, 190.0), L, 180.0, 0.9)        # next row: not merged
        kept = drop_tip_duplicates([a, b, c], L, np.array([-1.0, 0.0]))
        self.assertEqual([k.id for k in kept], [1, 3])
        self.assertEqual(kept[0].root, (200.0, 100.0))

    def test_map_roundtrip_and_editing(self):
        m = TuftMap("v.mp4", 5, (640, 360), [Anchor(1, (10.0, 20.0), 30.0, 90.0, 0.8)], (-1.0, 0.0))
        nid = m.add((100, 100), 30.0, 180.0)
        self.assertEqual(nid, 2)
        self.assertIsNone(m.remove_near((300, 300), 10))
        self.assertEqual(m.remove_near((12, 21), 10).id, 1)
        with tempfile.TemporaryDirectory() as d:
            m.save(Path(d) / "m.json")
            m2 = TuftMap.load(Path(d) / "m.json")
        self.assertEqual(m2.size, (640, 360))
        self.assertEqual(m2.flow, (-1.0, 0.0))
        self.assertEqual([(a.id, a.root, a.manual) for a in m2.anchors], [(2, (100.0, 100.0), True)])


class _Backend:
    """Probability = dark pixels (strings drawn dark on light)."""
    threshold = 0.5
    device = "cpu"

    def predict_proba(self, bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return (1.0 - g / 255.0)[None]


def _frame(angles, missing=()):
    img = np.full((360, 720, 3), 230, np.uint8)
    for k, (r, ang) in enumerate(zip(ROOTS, angles)):
        if k in missing:
            continue
        d = np.array([np.cos(np.radians(ang)), -np.sin(np.radians(ang))])
        cv2.line(img, tuple(int(v) for v in r), tuple(int(v) for v in np.array(r) + L * d), (20, 20, 20), 3)
    return img


class MapPipelineTests(unittest.TestCase):
    def _map(self):
        return TuftMap("v", 0, (720, 360), [Anchor(i + 1, r, L, 180.0, 1.0) for i, r in enumerate(ROOTS)], (-1.0, 0.0))

    def test_ids_fixed_arrows_from_root_and_missing_kept(self):
        tmap = self._map()
        ref = _frame([180.0] * len(ROOTS))
        pipe = MapPipeline(tmap, _Backend(), ref, fps=30, align=False)
        res = pipe.process_frame(_frame([200.0] * len(ROOTS), missing={3}), index=0)
        self.assertEqual([t.track_id for t in res.tufts], list(range(1, len(ROOTS) + 1)))
        for t in res.tufts:
            if t.track_id == 4:
                self.assertFalse(t.detected)
                self.assertEqual(t.state, "missing")
                continue
            self.assertTrue(t.detected)
            r = ROOTS[t.track_id - 1]
            self.assertLess(np.hypot(t.root[0] - r[0], t.root[1] - r[1]), 8.0)   # root end at the tape
            self.assertLess(abs((t.angle_deg - 200.0 + 180) % 360 - 180), 20.0)

    def test_clutter_away_from_roots_is_ignored(self):
        tmap = self._map()
        frame = _frame([180.0] * len(ROOTS))
        cv2.line(frame, (650, 20), (700, 340), (20, 20, 20), 3)   # a long dark line nowhere near a root
        pipe = MapPipeline(tmap, _Backend(), frame, fps=30, align=False)
        res = pipe.process_frame(frame, index=0)
        self.assertEqual(len(res.tufts), len(ROOTS))
        self.assertTrue(all(t.detected for t in res.tufts))


class MapReuseTests(unittest.TestCase):
    def test_reference_frame_saved_and_reloaded(self):
        from core.tuft_map import load_ref, ref_path
        m = TuftMap("a.mp4", 3, (64, 48), [Anchor(1, (10.0, 20.0), 30.0, 180.0, 1.0)], (-1.0, 0.0))
        ref = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "aircraft.tuftmap.json"
            m.save(path, ref)
            self.assertTrue(ref_path(path).exists())
            np.testing.assert_array_equal(load_ref(path), ref)          # usable on any later video
            self.assertIsNone(load_ref(Path(d) / "other.json"))

    def test_aligner_reports_inliers(self):
        from core.tuft_map import Aligner
        rng = np.random.default_rng(1)
        ref = cv2.GaussianBlur(rng.integers(0, 255, (360, 480, 3), dtype=np.uint8), (3, 3), 0)
        al = Aligner(ref)
        M = al.ref_to_frame(np.roll(ref, (5, 8), axis=(0, 1)))
        self.assertGreater(al.inliers, 40)
        self.assertAlmostEqual(M[0, 2], 8, delta=1.5)
        self.assertAlmostEqual(M[1, 2], 5, delta=1.5)
        al2 = Aligner(ref)
        al2.ref_to_frame(np.zeros_like(ref))                              # nothing in common
        self.assertEqual(al2.inliers, 0)


if __name__ == "__main__":
    unittest.main()
