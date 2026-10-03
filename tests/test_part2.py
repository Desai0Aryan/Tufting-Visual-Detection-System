import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from detector import TuftDetector
from src.augment import augment, random_geometric, random_photometric
from src.backends import ClassicalBackend
from src.dataset import CropDataset, crop_or_pad, make_training_crop, tile_coords, to_arrays
from src.labeling import (decode_label, encode_label, list_samples, load_sample, render_label,
                          save_sample, split_of, split_samples)
from src.mask_to_tufts import extract_tufts
from src.metrics import SegMeter
from src.review import ReviewSession
from src.sam2_labeler import accept_mask, build_prompt, iter_tiles, refine_with_sam
from src.synth import make_scene

ROOT = Path(__file__).resolve().parents[1]


def _scene_label(seed=0, **kw):
    sc = make_scene(seed=seed, **kw)
    return sc, encode_label(sc.tuft_mask, sc.root_mask, np.ones_like(sc.tuft_mask))


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1)


class AugmentTests(unittest.TestCase):
    def test_geometric_keeps_image_and_label_aligned_and_binary(self):
        rng = np.random.default_rng(0)
        mask = np.zeros((120, 160), np.uint8)
        cv2.line(mask, (20, 30), (140, 90), 255, 5)
        img = np.dstack([mask] * 3)
        lab = encode_label(mask, mask, mask)
        for _ in range(25):
            i2, l2 = random_geometric(img, lab, rng)
            self.assertEqual(l2.shape, lab.shape)
            self.assertTrue(set(np.unique(l2)) <= {0, 255})
            self.assertGreater(_iou(i2[..., 0] > 127, l2[..., 0] > 127), 0.85)

    def test_photometric_changes_pixels_not_shape_and_is_seeded(self):
        sc = make_scene(seed=1)
        a = random_photometric(sc.image, np.random.default_rng(5))
        b = random_photometric(sc.image, np.random.default_rng(5))
        self.assertEqual(a.shape, sc.image.shape)
        self.assertEqual(a.dtype, np.uint8)
        np.testing.assert_array_equal(a, b)

    def test_augment_pair(self):
        sc, lab = _scene_label(2)
        i2, l2 = augment(sc.image, lab, np.random.default_rng(1))
        self.assertEqual(i2.shape, sc.image.shape)
        self.assertEqual(l2.shape, lab.shape)


class LabelingTests(unittest.TestCase):
    def test_render_then_extract_round_trip_keeps_orientation(self):
        sc = make_scene(seed=3, flow_deg=25)
        tufts = extract_tufts(sc.tuft_mask, sc.root_mask)
        label = render_label(sc.image.shape, tufts)
        tuft, root, valid = decode_label(label)
        self.assertGreater(_iou(tuft, sc.tuft_mask > 0), 0.8)
        self.assertGreater(valid.mean(), 0.5)
        # a model that predicted exactly this root mask must recover the same orientation
        again = extract_tufts(tuft.astype(np.uint8) * 255, root.astype(np.uint8) * 255)
        self.assertGreaterEqual(len(again), 0.9 * len(tufts))
        ok = 0
        for t in again:
            near = min(tufts, key=lambda u: np.hypot(u.root[0] - t.root[0], u.root[1] - t.root[1]))
            ok += np.hypot(near.root[0] - t.root[0], near.root[1] - t.root[1]) < 6
        self.assertGreaterEqual(ok / len(again), 0.95)

    def test_untrusted_roots_are_masked_out_of_valid(self):
        mask = np.zeros((80, 200), np.uint8)
        cv2.line(mask, (20, 40), (100, 40), 255, 3)   # uniform width: orientation is a coin flip
        tufts = extract_tufts(mask)
        self.assertLess(tufts[0].confidence, 0.4)
        _, root, valid = decode_label(render_label(mask.shape, tufts))
        self.assertEqual(root.sum(), 0)
        self.assertFalse(valid[40, 60])             # on the untrusted tuft: ignored
        self.assertTrue(valid[5, 190])               # far from any tuft: certain background
        _, root2, valid2 = decode_label(render_label(mask.shape, tufts, trusted=[True]))
        self.assertGreater(root2.sum(), 0)
        self.assertTrue(valid2[40, 60])

    def test_dataset_io_and_splits(self):
        sc, lab = _scene_label(4)
        with tempfile.TemporaryDirectory() as d:
            for stem in ("a", "b__t00", "b__t01", "c"):
                save_sample(d, stem, sc.image, lab, meta={"source": "test"})
            self.assertEqual(list_samples(d), ["a", "b__t00", "b__t01", "c"])
            img, l2 = load_sample(d, "a")
            np.testing.assert_array_equal(l2, lab)
            self.assertEqual(json.loads((Path(d) / "meta.json").read_text())["a"]["source"], "test")
        self.assertEqual(split_of("b__t00"), split_of("b__t01"))   # tiles of one photo stay together
        s = split_samples(["x", "y"])
        self.assertTrue(s["train"] and s["val"])
        self.assertTrue(split_samples(["only"])["train"])
        s = split_samples(["a__t00", "a__t01", "b__t00"], val_percent=0)
        self.assertEqual(set(s["train"] + s["val"]), {"a__t00", "a__t01", "b__t00"})
        self.assertFalse(
            {name.split("__")[0] for name in s["train"]}
            & {name.split("__")[0] for name in s["val"]}
        )

    def test_classical_light_polarity_finds_tufts_on_dark_surface(self):
        sc = make_scene(background="dark", seed=1)
        dark = TuftDetector(backend=ClassicalBackend(polarity="dark")).detect(sc.image)
        light = TuftDetector(backend=ClassicalBackend(polarity="light")).detect(sc.image)
        self.assertLess(len(dark), 0.2 * len(sc.truth))
        self.assertGreater(len(light), 0.9 * len(sc.truth))
        with self.assertRaises(ValueError):
            ClassicalBackend(polarity="sideways")


class Sam2LabelerTests(unittest.TestCase):
    def setUp(self):
        self.sc = make_scene(width=480, height=360, seed=6, spacing=(110, 80))
        self.mask = self.sc.tuft_mask
        self.tufts = extract_tufts(self.mask, self.sc.root_mask)

    def test_prompt_points_on_string_and_negatives_off(self):
        t = self.tufts[0]
        p = build_prompt(t)
        pos = p.points[p.labels == 1]
        neg = p.points[p.labels == 0]
        dil = cv2.dilate(self.mask, np.ones((5, 5), np.uint8)) > 0
        self.assertTrue(all(dil[int(round(y)), int(round(x))] for x, y in pos))
        self.assertFalse(any(self.mask[int(round(y)), int(round(x))] for x, y in neg))
        self.assertTrue(p.box[0] <= t.path[:, 0].min() and p.box[2] >= t.path[:, 0].max())
        shifted = build_prompt(t, offset=(100, 50))
        np.testing.assert_allclose(shifted.points, p.points - [100, 50], atol=1e-4)

    def test_accept_mask(self):
        prop = np.zeros((50, 50), bool); prop[20:23, 5:40] = True
        self.assertTrue(accept_mask(prop.copy(), prop))
        bigger = np.zeros_like(prop); bigger[18:26, 3:42] = True
        self.assertTrue(accept_mask(bigger, prop))
        self.assertFalse(accept_mask(np.ones_like(prop), prop))          # whole tile: rejected
        other = np.zeros_like(prop); other[40:43, 5:40] = True
        self.assertFalse(accept_mask(other, prop))                        # different object
        self.assertFalse(accept_mask(np.zeros_like(prop), prop))

    def test_iter_tiles_cover_image(self):
        for h, w in [(300, 300), (769, 1200), (1000, 1000), (513, 700)]:
            cov = np.zeros((h, w), bool)
            for x0, y0, x1, y1 in iter_tiles(h, w, 512, 128):
                self.assertTrue(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h)
                cov[y0:y1, x0:x1] = True
            self.assertTrue(cov.all(), (h, w))

    def _stub(self, kind):
        mask_full = self.mask

        class Stub:
            def set_image(self, rgb):
                self.shape = rgb.shape[:2]
                self.rgb = rgb
            def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
                if kind == "raise":
                    raise RuntimeError("boom")
                h, w = self.shape
                m = np.zeros((h, w), bool)
                if kind == "whole":
                    m[:] = True
                else:  # "good": behave like SAM - the dark string nearest the first positive point, slightly fattened
                    gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
                    dark = (gray < 90).astype(np.uint8) if gray.mean() > 128 else (gray > 165).astype(np.uint8)
                    n_cc, cc = cv2.connectedComponents(dark, connectivity=8)
                    x, y = [int(round(v)) for v in point_coords[0]]
                    ys, xs = np.nonzero(cc)
                    k = np.argmin(np.hypot(xs - x, ys - y))
                    m = cv2.dilate((cc == cc[ys[k], xs[k]]).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                return np.stack([m]), np.array([0.9]), None
        return Stub()

    def test_refine_with_stub_predictors(self):
        img, n = self.sc.image, len(self.tufts)
        for kind in ("whole", "raise"):
            refined, st = refine_with_sam(img, self.tufts, self.mask, self._stub(kind), tile=256, overlap=128)
            self.assertEqual(st["sam_accepted"], 0)
            self.assertEqual(st["fallback"] + st["skipped"], n)
            # fallback means the proposals' own pixels survive
            self.assertGreater(_iou(refined > 0, self.mask > 0), 0.8)
        refined, st = refine_with_sam(img, self.tufts, self.mask, self._stub("good"), tile=512)
        self.assertEqual(st["sam_accepted"] + st["fallback"], n)
        self.assertGreaterEqual(st["sam_accepted"], 0.8 * n)      # the accept path really runs
        self.assertGreater(_iou(refined > 0, cv2.dilate(self.mask, np.ones((3, 3), np.uint8)) > 0), 0.8)

    def test_every_proposal_is_accounted_for_across_tiles(self):
        # tiny tiles force proposals to straddle tile borders -> they must fall back, not vanish
        _, st = refine_with_sam(self.sc.image, self.tufts, self.mask, self._stub("raise"), tile=160, overlap=40)
        self.assertEqual(st["fallback"] + st["skipped"], len(self.tufts))


class ReviewTests(unittest.TestCase):
    def setUp(self):
        sc, lab = _scene_label(7)
        self.sess = ReviewSession(sc.image, lab)

    def test_reject_flip_undo_render(self):
        s = self.sess
        n = len(s.tufts)
        self.assertGreater(n, 10)
        t0 = s.tufts[0]
        i = s.nearest(*t0.path[len(t0.path) // 2])
        self.assertEqual(i, 0)
        s.toggle(0)
        self.assertEqual(s.n_kept, n - 1)
        s.undo()
        self.assertEqual(s.n_kept, n)
        before_root = s.tufts[0].root
        s.flip(0)
        self.assertEqual(s.tufts[0].tip, before_root)
        s.undo()
        self.assertEqual(s.tufts[0].root, before_root)
        self.assertEqual(s.reject_box(0, 0, 10 ** 5, 10 ** 5), n)
        tuft, root, _ = decode_label(s.render())
        self.assertEqual(tuft.sum(), 0)
        self.assertIsNone(s.nearest(-500, -500))

    def test_flipped_root_is_written_to_the_label(self):
        s = self.sess
        old_tip = s.tufts[0].tip
        s.flip(0)
        _, root, valid = decode_label(s.render())
        self.assertTrue(root[int(round(old_tip[1])), int(round(old_tip[0]))])
        self.assertTrue(valid[int(round(old_tip[1])), int(round(old_tip[0]))])


class DatasetTests(unittest.TestCase):
    def test_tile_coords_cover(self):
        for h, w in [(256, 256), (300, 700), (1000, 1000), (100, 100)]:
            cov = np.zeros((h, w), bool)
            for y, x in tile_coords(h, w, 256):
                cov[y:y + 256, x:x + 256] = True
            self.assertTrue(cov.all(), (h, w))

    def test_crop_pad_and_arrays(self):
        sc, lab = _scene_label(8)
        c, l = crop_or_pad(sc.image, lab, 400, 600, 256)       # runs off the edge -> padded
        self.assertEqual(c.shape, (256, 256, 3))
        self.assertEqual(l.shape, (256, 256, 3))
        rgb, tgt, wgt = to_arrays(c, l)
        self.assertEqual((rgb.shape, tgt.shape, wgt.shape), ((3, 256, 256), (2, 256, 256), (1, 256, 256)))
        self.assertEqual(rgb.dtype, np.float32)
        self.assertTrue(0.0 <= rgb.min() and rgb.max() <= 1.0)
        self.assertTrue(set(np.unique(tgt)) <= {0.0, 1.0})

    def test_training_crops_favour_tufts(self):
        sc, lab = _scene_label(9, width=900, height=700)
        rng = np.random.default_rng(0)
        frac = [(make_training_crop(sc.image, lab, 128, rng, 0.9)[1][..., 0] > 127).mean() for _ in range(30)]
        uniform = [(make_training_crop(sc.image, lab, 128, rng, 0.0)[1][..., 0] > 127).mean() for _ in range(30)]
        self.assertGreater(np.mean(frac), np.mean(uniform))

    def test_crop_dataset_train_val_and_epochs(self):
        with tempfile.TemporaryDirectory() as d_real, tempfile.TemporaryDirectory() as d_syn:
            for i in range(6):
                sc, lab = _scene_label(i)
                save_sample(d_real, f"p{i}", sc.image, lab)
                save_sample(d_syn, f"s{i}", sc.image, lab)
            ds = CropDataset([d_real, d_syn], "train", size=96, epoch_size=10, seed=3)
            ds.set_root_shares({d_real: 0.8, d_syn: 0.2})
            self.assertAlmostEqual(float(ds.p.sum()), 1.0)
            share_real = sum(p for p, (r, _) in zip(ds.p, ds.items) if r == d_real)
            self.assertAlmostEqual(share_real, 0.8, places=6)
            self.assertEqual(len(ds), 10)
            a0 = ds[2][0]; a0b = ds[2][0]
            np.testing.assert_array_equal(a0, a0b)            # same epoch + index -> same crop
            ds.set_epoch(1)
            self.assertFalse(np.array_equal(a0, ds[2][0]))    # new epoch -> new crop
            val = CropDataset([d_real], "val", size=96)
            self.assertGreater(len(val), 0)
            self.assertEqual(val[0][0].shape, (3, 96, 96))
            with self.assertRaises(ValueError):
                ds.set_root_shares({"nowhere": 1.0})


class MetricsTests(unittest.TestCase):
    def test_dice_iou_and_weight(self):
        m = SegMeter()
        pred = np.array([[1, 1, 0, 0]]); tgt = np.array([[1, 0, 0, 1]])
        m.update(pred, tgt)
        self.assertAlmostEqual(m.dice, 0.5)
        self.assertAlmostEqual(m.iou, 1 / 3)
        w = SegMeter(); w.update(pred, tgt, weight=np.array([[1, 0, 0, 0]]))
        self.assertAlmostEqual(w.dice, 1.0)
        self.assertEqual(SegMeter().dice, 1.0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed; training smoke test skipped")
class TrainSmokeTest(unittest.TestCase):
    def test_two_epoch_run_produces_loadable_checkpoint(self):
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        import make_synthetic_dataset
        import train_unet
        from src.unet import UNetBackend
        with tempfile.TemporaryDirectory() as d:
            make_synthetic_dataset.main(["--out", f"{d}/syn", "--count", "5", "--seed", "1"])
            ck = f"{d}/m.pt"
            rc = train_unet.main(["--synthetic", f"{d}/syn", "--out", ck, "--epochs", "2", "--epoch-size", "8",
                                  "--batch", "4", "--size", "64", "--base", "4", "--device", "cpu"])
            self.assertEqual(rc, 0)
            self.assertTrue(Path(ck).exists() and Path(ck + ".json").exists())
            res = UNetBackend(ck, device="cpu").predict(np.zeros((100, 120, 3), np.uint8))
            self.assertEqual(res.tuft.shape, (100, 120))


if __name__ == "__main__":
    unittest.main()
