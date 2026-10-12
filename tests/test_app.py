import csv
import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.selection import distance_to_tuft, nearest_tuft, update_selection
from core.results import TuftResult
from tests.test_core import write_tuft_video

HAS_QT = importlib.util.find_spec("PySide6") is not None


def tuft(tid, root, tip):
    path = np.linspace(root, tip, 10).astype(np.float32)
    return TuftResult(tid, "confirmed", root, tip, (1.0, 0.0), 0.0, 50.0, 1.0, 2.0, False, "attached", path=path)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tufts = [tuft(1, (100, 100), (150, 100)), tuft(2, (100, 200), (150, 220))]

    def test_nearest_uses_the_centre_line_not_just_the_root(self):
        self.assertEqual(nearest_tuft(self.tufts, 148, 103), 1)       # near the tip end
        self.assertEqual(nearest_tuft(self.tufts, 125, 212), 2)
        self.assertIsNone(nearest_tuft(self.tufts, 400, 400))          # empty space
        self.assertAlmostEqual(distance_to_tuft(self.tufts[0], 120, 110), 10.0, places=4)

    def test_click_and_ctrl_click(self):
        s = update_selection(set(), 1, additive=False)
        self.assertEqual(s, {1})
        s = update_selection(s, 2, additive=True)
        self.assertEqual(s, {1, 2})
        s = update_selection(s, 1, additive=True)                      # ctrl toggles off
        self.assertEqual(s, {2})
        self.assertEqual(update_selection(s, None, additive=True), {2})  # ctrl on empty: keep
        self.assertEqual(update_selection(s, None, additive=False), set())  # click on empty: clear


class HeadlessSmokeTest(unittest.TestCase):
    """Processes 30 frames through the pipeline API and checks every export file."""

    def test_process_30_frames_and_export(self):
        from app.exporter import export_all
        from app.store import ResultStore
        from core import Pipeline, PipelineConfig, VideoFileSource
        with tempfile.TemporaryDirectory() as d:
            write_tuft_video(f"{d}/clip.mp4", n=30)
            store = ResultStore()
            with VideoFileSource(f"{d}/clip.mp4") as src:
                pipe = Pipeline(PipelineConfig(backend="classical", polarity="light", block_size=31, c=15,
                                               min_length_px=40), fps=src.fps)
                for i, fr in src:
                    store.put(pipe.process_frame(fr, i))
            self.assertEqual(store.frontier, 29)
            results = store.ordered()
            sel = [results[-1].visible()[0].track_id]
            paths = export_all(f"{d}/clip.mp4", results, f"{d}/out", ["arrows", "ids", "stall", "mask"], sel)
            cap = cv2.VideoCapture(str(paths["video"]))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 30)
            ok, fr = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(fr.shape[:2], (240, 320))
            rows = list(csv.DictReader(open(paths["csv_all"], newline="")))
            self.assertGreaterEqual(len(rows), 3 * 25)
            self.assertEqual({r["frame"] for r in rows} >= {"0", "29"}, True)
            self.assertTrue({"track_id", "stall", "angle_deg", "length_ratio"} <= set(rows[0]))
            sel_rows = list(csv.DictReader(open(paths["csv_selected"], newline="")))
            self.assertTrue(sel_rows and {r["track_id"] for r in sel_rows} == {str(sel[0])})


@unittest.skipUnless(HAS_QT, "PySide6 not installed")
class MainWindowOffscreenTest(unittest.TestCase):
    def test_open_process_select_and_overlay_toggle(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from app import settings as settings_mod
        from app.main_window import MainWindow
        qapp = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as d:
            write_tuft_video(f"{d}/clip.mp4", n=30)
            qs = QSettings(f"{d}/settings.ini", QSettings.IniFormat)
            cfg = settings_mod.AppSettings(backend="classical", polarity="light", min_length_px=40)
            settings_mod.save(cfg, qs)
            win = MainWindow(qsettings=qs)
            win.open_video(f"{d}/clip.mp4")
            t0 = time.time()
            while win.state.store.frontier < 29 and time.time() - t0 < 60:
                qapp.processEvents()   # the UI keeps running while the worker processes
                time.sleep(0.02)
            self.assertEqual(win.state.store.frontier, 29)
            win.show_frame(20)
            r = win.state.store.get(20)
            target = r.visible()[0]
            mid = target.path[len(target.path) // 2]
            win.on_click(float(mid[0]), float(mid[1]), False)
            self.assertEqual(win.state.selected, {target.track_id})
            table = win.panels[0].table
            self.assertEqual(table.rowCount(), 1)
            self.assertEqual(table.item(0, 0).text(), str(target.track_id))
            win.ov_actions["mask"].setChecked(True)
            self.assertIn("mask", settings_mod.load(qs).overlays)   # saved between sessions
            win.close()


if __name__ == "__main__":
    unittest.main()
