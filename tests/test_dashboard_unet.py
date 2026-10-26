"""The dashboard's trained-model path (U-Net + tuft map). The real model needs torch and trained weights, so these
tests plug in a stand-in `src.unet.UNetBackend` that turns the classical mask into probabilities; everything else
(map building, map pipeline, review edits, saved maps, warnings) is the real code."""
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import io

import cv2
import numpy as np

from dashboard import analysis
from dashboard.demo import make_demo_video
from dashboard.server import create_app
from src.backends import ClassicalBackend


class FakeUNet:
    def __init__(self, weights, device=None, threshold=None):
        self.weights, self.device, self.threshold = weights, "cpu", threshold or 0.5
        self.be = {p: ClassicalBackend(51, 10, polarity=p) for p in ("dark", "light")}
        self.pol = "dark"

    def predict_proba(self, bgr):
        m = self.be[self.pol].predict(bgr).tuft
        return (m.astype(np.float32) / 255.0 * 0.9)[None]

    def predict(self, image):
        from src.backends import MaskResult
        return MaskResult(tuft=(self.predict_proba(image)[0] >= self.threshold).astype(np.uint8) * 255)


def _wait(c, jid, timeout=180, also=()):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = c.get(f"/api/jobs/{jid}").json
        if m["status"] not in ("queued", "running") or m["status"] in also:
            return m
        time.sleep(0.3)
    raise TimeoutError


class UNetPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.weights = t / "fake.pt"
        cls.weights.write_bytes(b"x")
        cls.patches = [
            mock.patch.dict(sys.modules, {"src.unet": types.SimpleNamespace(UNetBackend=FakeUNet)}),
            mock.patch.object(analysis, "torch_available", lambda: True),
            mock.patch.dict("os.environ", {"TUFT_WEIGHTS": str(cls.weights)}),
        ]
        for p in cls.patches:
            p.start()
        cls.clip = str(t / "clip.mp4")
        make_demo_video(cls.clip, seconds=2.0, fps=12, size=(960, 540))
        cls.app = create_app(str(t / "data"))
        cls.c = cls.app.test_client()
        with open(cls.clip, "rb") as fh:
            r = cls.c.post("/api/upload", data={"files": (fh, "clip.mp4"), "review": "0"}, content_type="multipart/form-data")
        cls.jid = r.json["jobs"][0]["id"]
        cls.meta = _wait(cls.c, cls.jid)

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()

    def test_default_engine_is_the_trained_model_with_a_map(self):
        self.assertEqual(self.meta["status"], "done", self.meta.get("error"))
        st = self.meta["stats"]
        self.assertEqual(st["engine_kind"], "unet")
        self.assertEqual(st["tracking"], "map")
        self.assertIn("U-Net", st["engine"])
        self.assertEqual(st["warnings"], [])
        self.assertGreater(st["tufts_total"], 10)
        s = self.c.get("/api/status").json
        self.assertTrue(s["unet"])
        self.assertEqual(s["warning"], "")

    def test_ids_are_stable_one_per_anchor(self):
        import gzip, json
        self.c.post(f"/api/jobs/{self.jid}/reanalyze", json={"tracking": "map", "map_mode": "keep"})
        _wait(self.c, self.jid)
        r = self.c.get(f"/api/jobs/{self.jid}/results.json")
        res = json.loads(gzip.decompress(r.data) if r.data[:2] == b"\x1f\x8b" else r.data)
        ids = {r[0] for f in res["frames"] for r in f}
        m = self.c.get(f"/api/jobs/{self.jid}/map").json["map"]
        self.assertLessEqual(len(ids), len(m["anchors"]) + 1)
        self.assertEqual(res["tracking"], "map")

    def test_map_review_edit_and_save(self):
        m = self.c.get(f"/api/jobs/{self.jid}/map").json["map"]
        n = len(m["anchors"])
        r = self.c.put(f"/api/jobs/{self.jid}/map", json={"remove": [m["anchors"][0]["id"]], "add": [[50, 60]]})
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(len(r.json["map"]["anchors"]), n)
        self.assertTrue(any(a["manual"] for a in r.json["map"]["anchors"]))
        r = self.c.post(f"/api/jobs/{self.jid}/map/save", json={"name": "Wing cam 1"})
        self.assertEqual(r.json["name"], "Wing cam 1")
        self.assertIn("Wing cam 1", self.c.get("/api/status").json["maps"])
        self.assertEqual(self.c.post(f"/api/jobs/{self.jid}/map/save", json={"name": " "}).status_code, 400)
        self.assertEqual(self.c.get(f"/api/jobs/{self.jid}/map_ref.jpg").status_code, 200)

    def test_reanalyze_keeping_the_reviewed_map_and_with_a_saved_map(self):
        self.c.post(f"/api/jobs/{self.jid}/map/save", json={"name": "keepme"})
        for opts in ({"map_mode": "keep", "tracking": "map"}, {"map_mode": "build", "map_name": "keepme", "tracking": "map"}):
            r = self.c.post(f"/api/jobs/{self.jid}/reanalyze", json=opts)
            self.assertEqual(r.status_code, 200, r.json)
            m = _wait(self.c, self.jid)
            self.assertEqual(m["status"], "done", m.get("error"))
            self.assertIn("tuft map", m["stats"]["engine"])

    def test_free_camera_uses_the_model_without_a_map(self):
        r = self.c.post(f"/api/jobs/{self.jid}/reanalyze", json={"tracking": "free", "map_mode": "build", "map_name": ""})
        self.assertEqual(r.status_code, 200)
        m = _wait(self.c, self.jid)
        self.assertEqual(m["status"], "done", m.get("error"))
        self.assertEqual(m["stats"]["tracking"], "free")


class ReviewPopupTests(unittest.TestCase):
    """Upload -> tufts found -> pop-up (add / remove) -> process every frame."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        w = t / "fake.pt"
        w.write_bytes(b"x")
        cls.patches = [
            mock.patch.dict(sys.modules, {"src.unet": types.SimpleNamespace(UNetBackend=FakeUNet)}),
            mock.patch.object(analysis, "torch_available", lambda: True),
            mock.patch.dict("os.environ", {"TUFT_WEIGHTS": str(w)}),
        ]
        for p in cls.patches:
            p.start()
        cls.clip = str(t / "clip.mp4")
        make_demo_video(cls.clip, seconds=2.0, fps=12, size=(960, 540))
        cls.c = create_app(str(t / "data")).test_client()

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()

    def _upload(self):
        with open(self.clip, "rb") as fh:
            r = self.c.post("/api/upload", data={"files": (fh, "clip.mp4")}, content_type="multipart/form-data")
        return r.json["jobs"][0]["id"]

    def test_video_waits_for_the_popup_then_processes_every_frame(self):
        jid = self._upload()
        m = _wait(self.c, jid, also=("review",))
        self.assertEqual(m["status"], "review", m.get("error"))
        self.assertEqual(m["review_kind"], "map")
        self.assertGreater(m["tufts"], 5)
        # nothing was processed yet
        self.assertEqual(self.c.get(f"/api/jobs/{jid}/results.json").status_code, 409)
        mp = self.c.get(f"/api/jobs/{jid}/map").json["map"]
        n = len(mp["anchors"])
        self.assertIn("length", mp["anchors"][0])
        self.assertGreater(mp["median_length"], 0)
        # the person removes two tufts and adds one
        gone = [a["id"] for a in mp["anchors"][:2]]
        r = self.c.put(f"/api/jobs/{jid}/map", json={"remove": gone, "add": [[40, 40]]})
        self.assertEqual(len(r.json["map"]["anchors"]), n - 1)
        r = self.c.post(f"/api/jobs/{jid}/reanalyze", json={"map_mode": "keep", "review": False, "tracking": "map", "step": 1})
        self.assertEqual(r.status_code, 200, r.json)
        m = _wait(self.c, jid)
        self.assertEqual(m["status"], "done", m.get("error"))
        self.assertEqual(m["stats"]["step"], 1)
        self.assertEqual(m["stats"]["frames"], m["info"]["frames"])          # every frame
        self.assertIn("reviewed", m["stats"]["engine"])
        self.assertLessEqual(m["stats"]["tufts_total"], n - 1 + 1)

    def test_closing_the_popup_keeps_the_job_waiting(self):
        jid = self._upload()
        self.assertEqual(_wait(self.c, jid, also=("review",))["status"], "review")
        self.assertEqual(self.c.get(f"/api/jobs/{jid}/map").status_code, 200)    # can be reopened
        self.assertEqual(self.c.delete(f"/api/jobs/{jid}").status_code, 200)

    def test_picture_popup_remove_and_add(self):
        img = np.full((300, 400, 3), 225, np.uint8)
        for k in range(6):
            x, y = 40 + 55 * k, 150
            cv2.line(img, (x, y), (x + 8, y - 36), (30, 30, 30), 3)
        ok, buf = cv2.imencode(".png", img)
        r = self.c.post("/api/upload", data={"files": (io.BytesIO(buf.tobytes()), "pic.png")}, content_type="multipart/form-data")
        jid = r.json["jobs"][0]["id"]
        m = _wait(self.c, jid, also=("review",))
        self.assertEqual(m["status"], "review", m.get("error"))
        self.assertEqual(m["review_kind"], "image")
        d = self.c.get(f"/api/jobs/{jid}/image_tufts").json
        n = len(d["tufts"])
        self.assertGreater(n, 2)
        r = self.c.put(f"/api/jobs/{jid}/image_tufts", json={"remove": [0], "add": [[200, 250], {"root": [20, 250], "tip": [20, 210]}]})
        self.assertEqual(r.json["count"], n + 1)
        self.c.post(f"/api/jobs/{jid}/reanalyze", json={"map_mode": "keep", "review": False})
        m = _wait(self.c, jid)
        self.assertEqual(m["status"], "done", m.get("error"))
        self.assertEqual(m["stats"]["tufts_total"], n + 1)
        self.assertEqual(self.c.get(f"/api/jobs/{jid}/media").status_code, 200)
        tufts = self.c.get(f"/api/jobs/{jid}/summary.json").json["rows"]
        self.assertEqual(len(tufts), n + 1)


class FallbackTests(unittest.TestCase):
    def test_warning_when_model_is_missing(self):
        with mock.patch.dict("os.environ", {"TUFT_WEIGHTS": ""}), \
             mock.patch.object(analysis, "find_weights", lambda explicit="": ""):
            st = analysis.engine_status(analysis.Options())
            self.assertEqual(st["engine"], "classical")
            self.assertIn("No trained model", st["warning"])
            with self.assertRaises(ValueError):
                analysis.engine_status(analysis.Options(engine="unet"))

    def test_warning_when_torch_is_missing(self):
        with mock.patch.object(analysis, "find_weights", lambda explicit="": "/x/y.pt"), \
             mock.patch.object(analysis, "torch_available", lambda: False):
            st = analysis.engine_status(analysis.Options())
            self.assertEqual(st["engine"], "classical")
            self.assertIn("PyTorch", st["warning"])

    def test_chosen_classical_has_no_warning(self):
        self.assertEqual(analysis.engine_status(analysis.Options(engine="classical"))["warning"], "")


if __name__ == "__main__":
    unittest.main()
