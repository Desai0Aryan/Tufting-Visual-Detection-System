import io
import tempfile
import time
import unittest
import zipfile

import cv2
import numpy as np
import pandas as pd

from dashboard import analysis, tables
from dashboard.demo import make_demo_video
from dashboard.server import create_app


def _wait(client, jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = client.get(f"/api/jobs/{jid}").json
        if m["status"] not in ("queued", "running"):
            return m
        time.sleep(0.3)
    raise TimeoutError("analysis did not finish")


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.app = create_app(cls.tmp.name + "/data")
        cls.c = cls.app.test_client()
        cls.clip = cls.tmp.name + "/clip.mp4"
        make_demo_video(cls.clip, seconds=2.0, fps=12, size=(960, 540))
        with open(cls.clip, "rb") as fh:
            r = cls.c.post("/api/upload", data={"files": (fh, "clip.mp4"), "review": "0"}, content_type="multipart/form-data")
        assert r.status_code == 200, r.json
        cls.jid = r.json["jobs"][0]["id"]
        cls.meta = _wait(cls.c, cls.jid)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_video_analysis_finishes_with_every_tuft(self):
        self.assertEqual(self.meta["status"], "done", self.meta.get("error"))
        self.assertEqual(self.meta["tufts"], 45)

    def test_annotated_video_is_playable_and_same_length(self):
        r = self.c.get(f"/api/jobs/{self.jid}/media")
        self.assertEqual((r.status_code, r.mimetype), (200, "video/mp4"))
        path = self.app.store.dir(self.jid) / "annotated.mp4"
        cap = cv2.VideoCapture(str(path))
        self.assertEqual((int(cap.get(3)), int(cap.get(4))), (960, 540))
        self.assertGreater(cap.get(cv2.CAP_PROP_FRAME_COUNT), 10)
        cap.release()

    def test_clean_vector_time_history_columns(self):
        df = pd.read_csv(io.StringIO(self.c.get(f"/api/jobs/{self.jid}/download/vector_time_history.csv").data.decode()))
        for col in ("time_s", "tuft_id", "x", "y", "z", "vector_x", "vector_y", "vector_z"):
            self.assertIn(col, df.columns)
        n = np.sqrt(df.vector_x ** 2 + df.vector_y ** 2 + df.vector_z ** 2)
        self.assertTrue(np.allclose(n[n > 0], 1.0, atol=1e-3))
        self.assertTrue((df.z == 0).all())

    def test_coordinate_system_changes_positions_and_vectors(self):
        before = self.c.get(f"/api/jobs/{self.jid}/summary.json").json
        self.assertEqual(before["units"], "px")
        r = self.c.put(f"/api/jobs/{self.jid}/coords", json={
            "method": "origin_axis", "origin": [60, 300], "axis": [560, 300], "distance": 1000, "units": "mm"})
        self.assertEqual(r.status_code, 200, r.json)
        after = self.c.get(f"/api/jobs/{self.jid}/summary.json").json
        self.assertEqual(after["units"], "mm")
        self.assertNotEqual(before["rows"][0]["x"], after["rows"][0]["x"])
        bad = self.c.put(f"/api/jobs/{self.jid}/coords", json={"method": "points", "points": [[0, 0, 0, 0]]})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("reference points", bad.json["error"])

    def test_user_positions_override_per_component(self):
        self.c.put(f"/api/jobs/{self.jid}/locations/5", json={"x": "11.5", "z": "3"})
        row = [r for r in self.c.get(f"/api/jobs/{self.jid}/summary.json").json["rows"] if r["tuft_id"] == 5][0]
        self.assertEqual((row["x"], row["z"], row["position_source"]), (11.5, 3.0, "partly entered"))
        h = pd.read_csv(io.StringIO(self.c.get(f"/api/jobs/{self.jid}/download/vector_time_history.csv").data.decode()))
        g = h[h.tuft_id == 5]
        self.assertTrue((g.x == 11.5).all() and (g.z == 3.0).all())
        self.assertEqual(self.c.put(f"/api/jobs/{self.jid}/locations/5", json={"x": "abc"}).status_code, 400)
        r = self.c.post(f"/api/jobs/{self.jid}/locations.csv", data=b"tuft_id,x,y,z\n6,1,2,3\n", content_type="text/csv")
        self.assertEqual(r.json["imported"], 1)
        self.c.delete(f"/api/jobs/{self.jid}/locations")

    def test_lift_estimate_keeps_unit_length(self):
        self.c.put(f"/api/jobs/{self.jid}/lift", json={"on": True})
        h = pd.read_csv(io.StringIO(self.c.get(f"/api/jobs/{self.jid}/download/vector_time_history.csv").data.decode()))
        self.c.put(f"/api/jobs/{self.jid}/lift", json={"on": False})
        n = np.sqrt(h.vector_x ** 2 + h.vector_y ** 2 + h.vector_z ** 2)
        self.assertTrue(np.allclose(n[n > 0], 1.0, atol=1e-3))
        self.assertGreater(h.vector_z.max(), 0)          # the flapping tufts shorten, so some lift is estimated

    def test_downloads(self):
        z = zipfile.ZipFile(io.BytesIO(self.c.get(f"/api/jobs/{self.jid}/download/all.zip").data))
        names = " ".join(z.namelist())
        for part in ("vector_time_history.csv", "tuft_summary.csv", "results.xlsx", "annotated.mp4"):
            self.assertIn(part, names)
        self.assertEqual(self.c.get(f"/api/jobs/{self.jid}/download/results.xlsx").status_code, 200)

    def test_flapping_tufts_are_flagged(self):
        rows = self.c.get(f"/api/jobs/{self.jid}/summary.json").json["rows"]
        states = {r["main_state"] for r in rows}
        self.assertIn("steady", states)
        self.assertTrue(states & {"stalled", "fluttering"})

    def test_rejects_non_media_and_broken_video(self):
        r = self.c.post("/api/upload", data={"files": (io.BytesIO(b"x"), "notes.txt")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        r = self.c.post("/api/upload", data={"files": (io.BytesIO(b"nope"), "broken.mp4")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertIn("could not be opened", r.json["errors"][0]["error"])

    def test_image_is_analysed(self):
        img = cv2.imread("tests/fixtures/Nasa_High_Alpha.jpg")
        ok, buf = cv2.imencode(".png", img)
        r = self.c.post("/api/upload", data={"files": (io.BytesIO(buf.tobytes()), "pic.png"), "review": "0"}, content_type="multipart/form-data")
        m = _wait(self.c, r.json["jobs"][0]["id"])
        self.assertEqual(m["status"], "done")
        self.assertGreater(m["tufts"], 20)
        self.assertEqual(self.c.get(f"/api/jobs/{m['id']}/media").mimetype, "image/jpeg")

    def test_missing_job_is_404_with_message(self):
        r = self.c.get("/api/jobs/nope/summary.json")
        self.assertEqual(r.status_code, 404)
        self.assertIn("no longer exists", r.json["error"])


class OptionTests(unittest.TestCase):
    def test_options_are_sanitised(self):
        o = analysis.Options.from_dict({"engine": "x", "polarity": "y", "step": "-3", "roi": [1, 2, 3, 4]})
        self.assertEqual((o.engine, o.polarity, o.step, o.roi), ("auto", "auto", 0, None))
        self.assertEqual(analysis.Options.from_dict({"roi": [10, 10, 200, 100]}).roi, [10, 10, 200, 100])

    def test_classical_settings_scale_with_picture_size(self):
        a = analysis.classical_params(640, 480, analysis.Options())
        b = analysis.classical_params(1920, 1080, analysis.Options())
        self.assertAlmostEqual(b["max_length_px"] / a["max_length_px"], 3.0)

    def test_summary_of_empty_results_does_not_crash(self):
        res = {"kind": "video", "fps": 10, "step": 1, "width": 100, "height": 80, "times": [0.0], "frames": [[]],
               "fields": analysis.FIELDS, "engine": ""}
        self.assertTrue(tables.history_frame(res, None).empty)
        self.assertTrue(tables.summary_frame(res, None).empty)
        self.assertEqual(tables.overview(res, None)["tufts"], 0)


if __name__ == "__main__":
    unittest.main()
