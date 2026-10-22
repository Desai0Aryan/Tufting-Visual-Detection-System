import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.sources import (blockiness, check, is_creative_commons, make_row, add_row, quality_problems,
                         read_sources, video_stats)


def write_video(path, w=1280, h=720, n=20, fps=30, seed=0):
    rng = np.random.default_rng(seed)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    base = cv2.GaussianBlur(rng.integers(90, 170, (h, w, 3)).astype(np.uint8), (0, 0), 6)
    for k in range(n):
        f = base.copy()
        for j in range(6):
            x = 150 + 180 * j
            cv2.line(f, (x, 200), (x + 10 + 3 * k % 20, 330), (240, 240, 240), 3)
        vw.write(f)
    vw.release()


class SourcesTests(unittest.TestCase):
    def test_license_detection(self):
        self.assertTrue(is_creative_commons("Creative Commons Attribution license (reuse allowed)"))
        self.assertTrue(is_creative_commons("CC BY 4.0"))
        self.assertFalse(is_creative_commons(""))
        self.assertFalse(is_creative_commons("Standard YouTube License"))

    def test_blockiness_flags_heavy_jpeg(self):
        rng = np.random.default_rng(1)
        img = cv2.GaussianBlur(rng.integers(0, 255, (256, 256)).astype(np.uint8), (0, 0), 3)
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 3])
        bad = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        self.assertLess(blockiness(img), 1.15)
        self.assertGreater(blockiness(bad), blockiness(img) + 0.2)

    def test_quality_rules(self):
        ok = {"width": 1280, "height": 720, "bpp": 0.1, "blockiness": 1.0}
        self.assertEqual(quality_problems(ok), [])
        self.assertTrue(quality_problems(dict(ok, height=480)))
        self.assertTrue(quality_problems(dict(ok, bpp=0.005)))
        self.assertEqual(quality_problems(dict(ok, bpp=0.005), force=True), [])
        self.assertTrue(quality_problems(dict(ok, height=480), force=True))  # never below 720p

    def test_video_stats(self):
        with tempfile.TemporaryDirectory() as d:
            write_video(f"{d}/a.mp4")
            s = video_stats(f"{d}/a.mp4")
            self.assertEqual((s["width"], s["height"], s["frames"]), (1280, 720, 20))

    def test_registry_and_check(self):
        with tempfile.TemporaryDirectory() as d:
            csv_p, clips = f"{d}/sources.csv", Path(d, "clips")
            clips.mkdir()
            add_row(csv_p, make_row("yt_a", "train", "tufts", "youtube", license_text="Creative Commons Attribution"))
            with self.assertRaises(ValueError):
                add_row(csv_p, make_row("yt_a", "val", "tufts", "youtube"))  # one clip, one split
            add_row(csv_p, make_row("yt_b", "val", "tufts", "youtube", license_text="Standard"))
            (clips / "yt_a.mp4").write_bytes(b"x")
            errors, _ = check(csv_p, clips)
            joined = " ".join(errors)
            self.assertIn("yt_b: external clip without", joined)
            self.assertIn("yt_b: no video file", joined)
            self.assertIn("own footage", joined)
            add_row(csv_p, make_row("own_run1", "test", "tufts", "own", license_text="own footage"))
            (clips / "own_run1.mp4").write_bytes(b"x")
            errors, summary = check(csv_p, clips)
            self.assertNotIn("own footage", " ".join(errors))
            self.assertEqual(summary["test"]["tufts"], 1)
            self.assertEqual(len(read_sources(csv_p)), 3)

    def test_add_local_cli(self):
        from tools import sources as cli
        with tempfile.TemporaryDirectory() as d:
            write_video(f"{d}/run 1.mp4")
            rc = cli.main(["--sources", f"{d}/s.csv", "--clips-dir", f"{d}/clips", "add-local",
                           f"{d}/run 1.mp4", "--split", "test", "--use", "tufts", "--force"])
            self.assertEqual(rc, 0)
            row = read_sources(f"{d}/s.csv")[0]
            self.assertEqual((row["clip_id"], row["origin"], row["split"]), ("own_run_1", "own", "test"))
            self.assertTrue(Path(d, "clips", "own_run_1.mp4").exists())


if __name__ == "__main__":
    unittest.main()
