"""Split rules and an end-to-end CPU run of tools/train.py on toy line images.

The toy images are random straight lines on noise (not the tuft generator):
they only exercise the plumbing - splits, crops, loss, checkpoints, logs,
sample sheets and resume.
"""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.labeling import encode_label, save_sample
from src.splits import SOURCES_COLUMNS, load_split_map, split_real

HAS = all(importlib.util.find_spec(m) for m in ("torch", "segmentation_models_pytorch", "albumentations"))


def write_sources(path, clips):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SOURCES_COLUMNS)
        w.writeheader()
        for cid, split in clips.items():
            w.writerow({"clip_id": cid, "split": split})


def toy_dataset(root, clips_frames, size=(96, 128), seed=0):
    rng = np.random.default_rng(seed)
    for cid, n in clips_frames.items():
        for f in range(n):
            img = rng.integers(60, 120, (*size, 3)).astype(np.uint8)
            mask = np.zeros(size, np.uint8)
            for _ in range(3):
                p = tuple(int(v) for v in rng.integers(0, size[1], 2))
                q = (int(rng.integers(0, size[1])), int(rng.integers(0, size[0])))
                cv2.line(img, (p[0], p[1] % size[0]), q, (240, 240, 240), 3)
                cv2.line(mask, (p[0], p[1] % size[0]), q, 255, 3)
            save_sample(root, f"{cid}__f{f:03d}", img,
                        encode_label(mask, np.zeros_like(mask), np.ones_like(mask)),
                        meta={"clip_id": cid})


class SplitTests(unittest.TestCase):
    def test_whole_clips_go_to_one_split(self):
        with tempfile.TemporaryDirectory() as d:
            write_sources(f"{d}/sources.csv", {"a": "train", "b": "val", "c": "test"})
            toy_dataset(f"{d}/real", {"a": 3, "b": 2, "c": 2})
            from src.labeling import list_samples
            parts = split_real(f"{d}/real", list_samples(f"{d}/real"), load_split_map(f"{d}/sources.csv"))
            self.assertEqual({s.split("__")[0] for s in parts["train"]}, {"a"})
            self.assertEqual({s.split("__")[0] for s in parts["val"]}, {"b"})
            self.assertEqual({s.split("__")[0] for s in parts["test"]}, {"c"})

    def test_missing_clip_id_or_unknown_clip_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            write_sources(f"{d}/sources.csv", {"a": "train"})
            toy_dataset(f"{d}/real", {"a": 1, "zzz": 1})
            from src.labeling import list_samples
            with self.assertRaises(ValueError):
                split_real(f"{d}/real", list_samples(f"{d}/real"), load_split_map(f"{d}/sources.csv"))
            meta = json.loads(Path(d, "real", "meta.json").read_text())
            meta["a__f000"].pop("clip_id")
            Path(d, "real", "meta.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                split_real(f"{d}/real", ["a__f000"], load_split_map(f"{d}/sources.csv"))

    def test_sources_csv_rejects_duplicates_and_bad_splits(self):
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/s.csv"
            Path(p).write_text("clip_id,split\na,train\na,val\n")
            with self.assertRaises(ValueError):
                load_split_map(p)
            Path(p).write_text("clip_id,split\na,holdout\n")
            with self.assertRaises(ValueError):
                load_split_map(p)


@unittest.skipUnless(HAS, "torch / smp / albumentations not installed")
class TrainLoopTests(unittest.TestCase):
    def test_cpu_run_writes_checkpoints_logs_samples_and_resumes(self):
        from tools import train
        with tempfile.TemporaryDirectory() as d:
            write_sources(f"{d}/sources.csv", {"a": "train", "b": "train", "v": "val", "t": "test"})
            toy_dataset(f"{d}/real", {"a": 3, "b": 3, "v": 2, "t": 2})
            sets = [f"run.out_dir={d}/runs", f"data.sources_csv={d}/sources.csv",
                    f"data.real=[{d}/real]", "data.synthetic=[]", "data.crop=64", "data.val_tile=64",
                    "data.epoch_size=8", "data.batch_size=4", "data.workers=0",
                    "model.encoder=resnet18", "model.encoder_weights=null",
                    "train.epochs=2", "train.amp=off", "train.channels_last=false",
                    "train.sample_every=1", "train.sample_count=2", "optim.warmup_epochs=1",
                    "loss.w_cldice=0.3"]
            rc = train.main(["--config", "configs/unet_resnet34.yaml", "--device", "cpu", "--set", *sets])
            self.assertEqual(rc, 0)
            run = Path(Path(d, "runs", "LATEST").read_text())
            for f in ("config.yaml", "split.json", "log.csv", "steps.csv",
                      "checkpoints/best.pt", "checkpoints/best.pt.json", "checkpoints/last.pt",
                      "samples/epoch_001.png", "samples/epoch_002.png"):
                self.assertTrue((run / f).exists(), f)
            rows = list(csv.DictReader(open(run / "log.csv")))
            self.assertEqual(len(rows), 2)
            self.assertNotEqual(rows[0]["val_dice"], "")
            split = json.loads((run / "split.json").read_text())
            used = split["real"][f"{d}/real"]["train"] + split["real"][f"{d}/real"]["val"]
            self.assertFalse([s for s in used if s.startswith("t__")])  # test clip never loaded
            self.assertEqual(split["test_samples"], 2)
            from src.unet import UNetBackend, suggested_threshold
            ck = run / "checkpoints" / "best.pt"
            self.assertIn(suggested_threshold(ck), (0.3, 0.4, 0.5, 0.6, 0.7))
            res = UNetBackend(ck, device="cpu").predict(np.zeros((70, 90, 3), np.uint8))
            self.assertEqual(res.tuft.shape, (70, 90))
            # resume for one more epoch
            rc = train.main(["--config", "configs/unet_resnet34.yaml", "--device", "cpu",
                             "--resume", str(run / "checkpoints" / "last.pt"),
                             "--set", *[s for s in sets if not s.startswith("train.epochs")], "train.epochs=3"])
            self.assertEqual(rc, 0)

    def test_crops_are_reproducible_and_differ_between_epochs(self):
        from src.seg_dataset import EpochSampler, Item, TrainCrops
        with tempfile.TemporaryDirectory() as d:
            toy_dataset(f"{d}/real", {"a": 2})
            items = [Item(f"{d}/real", s, "real") for s in ("a__f000", "a__f001")]
            ds = TrainCrops(items, size=64, epoch_size=4, seed=3)
            s = EpochSampler(4)
            e0 = list(s); s.set_epoch(1); e1 = list(s)
            self.assertEqual(e0, [0, 1, 2, 3]); self.assertEqual(e1, [4, 5, 6, 7])
            a, b = ds[5][0], ds[5][0]
            self.assertTrue(bool((a == b).all()))
            self.assertFalse(bool((ds[0][0] == ds[4][0]).all()))
            img, tgt, valid = ds[0]
            self.assertEqual(tuple(img.shape), (3, 64, 64))
            self.assertEqual(tuple(tgt.shape), (1, 64, 64))
            self.assertTrue(set(tgt.unique().tolist()) <= {0.0, 1.0})

    def test_small_frames_pad_label_with_the_image(self):
        from src.seg_dataset import crop_pad
        img = np.zeros((20, 30, 3), np.uint8); lab = np.zeros((20, 30, 3), np.uint8)
        img[:, -3:] = 255; lab[:, -3:] = 255  # a tuft at the right edge
        ci, cl = crop_pad(img, lab, 0, 0, 40)
        np.testing.assert_array_equal(ci[..., 0] > 127, cl[..., 0] > 127)



class IgnoreDatasetTests(unittest.TestCase):
    def test_ignore_png_becomes_zero_weight(self):
        from src.label_io import save_frame
        from src.seg_dataset import Item, ValTiles
        with tempfile.TemporaryDirectory() as d:
            img = np.zeros((64, 64, 3), np.uint8); m = np.zeros((64, 64), np.uint8)
            ign = np.zeros((64, 64), np.uint8); ign[:16, :16] = 255
            save_frame(f"{d}/r", "c__f0", img, m, [], {"clip_id": "c"}, ignore=ign)
            _, _, valid = ValTiles([Item(f"{d}/r", "c__f0", "real")], size=64)[0]
            self.assertEqual(tuple(valid.shape), (2, 64, 64))
            self.assertEqual(float(valid[1, :16, :16].sum()), 0.0)
            self.assertEqual(float(valid[1, 16:, 16:].min()), 1.0)


if __name__ == "__main__":
    unittest.main()
