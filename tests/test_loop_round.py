import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.sources import add_row, make_row

HAS = all(importlib.util.find_spec(m) for m in ("torch", "segmentation_models_pytorch", "albumentations"))


@unittest.skipUnless(HAS, "torch / smp / albumentations not installed")
class LoopRoundTests(unittest.TestCase):
    def _setup(self, d):
        import torch
        from src.unet import build_model, save_checkpoint
        from tests.test_sources import write_video
        Path(d, "clips").mkdir()
        write_video(f"{d}/clips/own_a.mp4", w=320, h=192, n=120)
        write_video(f"{d}/clips/own_v.mp4", w=320, h=192, n=120, seed=1)
        add_row(f"{d}/sources.csv", make_row("own_a", "train", "tufts", "own"))
        add_row(f"{d}/sources.csv", make_row("own_v", "val", "tufts", "own"))
        cfg = {"arch": "smp_unet", "encoder": "resnet18", "encoder_weights": None, "classes": 1}
        torch.manual_seed(0)
        save_checkpoint(build_model(cfg), f"{d}/w.pt", cfg)
        return ["--sources", f"{d}/sources.csv", "--clips-dir", f"{d}/clips", "--frames", f"{d}/frames",
                "--proposed", f"{d}/proposed", "--labeled", f"{d}/labeled", "--weights", f"{d}/w.pt",
                "--per-clip", "2", "--stride", "5", "--device", "cpu"]

    def test_proposals_then_no_training_without_acceptance(self):
        from tools import loop_round
        with tempfile.TemporaryDirectory() as d:
            args = self._setup(d)
            rc = loop_round.main(["--clips", "own_a", *args, "--no-review"])
            self.assertEqual(rc, 0)
            meta = json.loads(Path(d, "proposed", "meta.json").read_text())
            self.assertEqual(len(meta), 2)
            self.assertTrue(all(m["source"] == "unet" and m["status"] == "proposed" for m in meta.values()))
            self.assertFalse(Path(d, "runs").exists())

    def test_accepted_frames_trigger_a_warm_started_retrain(self):
        from tools import loop_round, review_labels
        from tools.review_labels import ReviewApp

        def accept_all(argv):
            a = dict(zip(argv[::2], argv[1::2]))
            meta = json.loads(Path(a["--data"], "meta.json").read_text())
            stems = sorted(s for s, m in meta.items() if m["status"] == "proposed")
            app = ReviewApp(a["--data"], a["--out"], stems, Path(a["--out"]).parent / "log.jsonl")
            while app.on_key(ord("a")):
                pass
            return 0

        with tempfile.TemporaryDirectory() as d:
            args = self._setup(d)
            sets = [f"run.out_dir={d}/runs", "data.crop=64", "data.val_tile=64", "data.epoch_size=4",
                    "data.batch_size=2", "data.workers=0", "model.encoder=resnet18",
                    "model.encoder_weights=null", "train.epochs=1", "train.amp=off",
                    "train.channels_last=false", "train.sample_count=1", "optim.warmup_epochs=0"]
            with mock.patch.object(review_labels, "main", accept_all):
                rc = loop_round.main(["--clips", "own_a", "own_v", *args, "--set", *sets])
            self.assertEqual(rc, 0)
            run = Path(Path(d, "runs", "LATEST").read_text())
            split = json.loads((run / "split.json").read_text())
            self.assertEqual(Path(split["init"]), Path(d, "w.pt"))
            real = split["real"][f"{d}/labeled"]
            self.assertEqual({s.split("__")[0] for s in real["train"]}, {"own_a"})
            self.assertEqual({s.split("__")[0] for s in real["val"]}, {"own_v"})
            self.assertTrue((run / "checkpoints" / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
