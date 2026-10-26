"""Job storage and the background worker. One analysis runs at a time; the rest wait in a queue.

Layout on disk (DATA_DIR/jobs/<id>/):
    meta.json  input.<ext>  first_frame.jpg  preview.jpg  results.json.gz
    annotated.mp4 | annotated.jpg   setup.json
"""
import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2

from . import analysis
from .analysis import Cancelled, Options, ReviewNeeded

VIDEO_EXT = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".avif", ".heic"}


class JobStore:
    def __init__(self, data_dir):
        self.root = Path(data_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.meta = {}
        self.cancel = set()
        self.q = queue.Queue()
        for d in sorted(self.root.iterdir()):
            m = d / "meta.json"
            if m.is_file():
                try:
                    meta = json.loads(m.read_text("utf8"))
                except ValueError:
                    continue
                if meta.get("status") in ("queued", "running"):
                    meta["status"] = "error"
                    meta["error"] = "The analysis was interrupted. Press Analyze again to restart it."
                    self._write(meta)
                self.meta[meta["id"]] = meta
        threading.Thread(target=self._worker, daemon=True, name="tuft-worker").start()

    # ---------------------------------------------------------------- helpers
    def dir(self, jid):
        return self.root / jid

    def _write(self, meta):
        d = self.dir(meta["id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta), "utf8")
        tmp.replace(d / "meta.json")

    def update(self, jid, **kw):
        with self.lock:
            m = self.meta.get(jid)
            if m is None:
                return None
            m.update(kw)
            self._write(m)
            return dict(m)

    def get(self, jid):
        with self.lock:
            m = self.meta.get(jid)
            return dict(m) if m else None

    def list(self):
        with self.lock:
            return sorted((dict(m) for m in self.meta.values()), key=lambda m: m["created"], reverse=True)

    # ---------------------------------------------------------------- create / delete
    def create(self, filename, data_writer, options=None):
        """data_writer(path) saves the uploaded bytes. -> meta dict (already queued)."""
        ext = Path(filename).suffix.lower()
        if ext in VIDEO_EXT:
            kind = "video"
        elif ext in IMAGE_EXT:
            kind = "image"
        else:
            raise ValueError(f"'{filename}' is not a video or picture. Use MP4 for video, or JPG / PNG for pictures.")
        jid = uuid.uuid4().hex[:10]
        d = self.dir(jid)
        d.mkdir(parents=True)
        src = d / f"input{ext}"
        try:
            data_writer(src)
            info, first = self._probe(kind, src)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            raise
        cv2.imencode(".jpg", first, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(d / "first_frame.jpg"))
        meta = {"id": jid, "name": Path(filename).name, "kind": kind, "input": src.name, "created": time.time(),
                "status": "queued", "progress": 0.0, "message": "Waiting to start", "error": "",
                "info": info, "options": Options.from_dict(options).__dict__, "stats": {}, "tufts": 0,
                "eta": None, "version": 1}
        with self.lock:
            self.meta[jid] = meta
            self._write(meta)
        (d / "setup.json").write_text(json.dumps({"coords": {"defined": False}, "locations": {},
                                                  "estimate_lift": False}), "utf8")
        self.q.put(jid)
        return dict(meta)

    @staticmethod
    def _probe(kind, src):
        if kind == "video":
            info, first = analysis.video_info(src)
            return {k: info[k] for k in ("fps", "frames", "width", "height", "duration")}, first
        data = cv2.imread(str(src))
        if data is None:
            import numpy as np
            data = cv2.imdecode(np.fromfile(str(src), dtype="uint8"), cv2.IMREAD_COLOR)
        if data is None:
            try:
                from PIL import Image
                import numpy as np
                data = cv2.cvtColor(np.asarray(Image.open(src).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError("This picture could not be read. Use a JPG or PNG file.")
        h, w = data.shape[:2]
        return {"fps": 1.0, "frames": 1, "width": w, "height": h, "duration": 0.0}, data

    def delete(self, jid):
        with self.lock:
            self.cancel.add(jid)
            self.meta.pop(jid, None)
        shutil.rmtree(self.dir(jid), ignore_errors=True)

    def reanalyze(self, jid, options):
        with self.lock:
            m = self.meta.get(jid)
            if m is None:
                raise KeyError(jid)
            if m["status"] in ("queued", "running"):
                raise ValueError("This analysis is already running.")
            self.cancel.discard(jid)
        opts = Options.from_dict({**m["options"], **(options or {})})
        out = self.update(jid, options=opts.__dict__, status="queued", progress=0.0, error="",
                          message="Waiting to start", version=m.get("version", 1) + 1)
        self.q.put(jid)
        return out

    def request_cancel(self, jid):
        with self.lock:
            if jid in self.meta and self.meta[jid]["status"] in ("queued", "running"):
                self.cancel.add(jid)
                return True
        return False

    # ---------------------------------------------------------------- worker
    def _worker(self):
        while True:
            jid = self.q.get()
            m = self.get(jid)
            if m is None or jid in self.cancel and m["status"] == "queued":
                if m is not None:
                    self.update(jid, status="cancelled", message="Cancelled")
                self.cancel.discard(jid)
                continue
            try:
                self._run(jid, m)
            except Exception:                    # never let the worker thread die
                traceback.print_exc()

    def _run(self, jid, m):
        d = self.dir(jid)
        opts = Options.from_dict(m["options"])
        self.update(jid, status="running", progress=0.0, message="Starting", error="")
        t_last = [0.0]

        def progress(done, total, tufts, eta, fps, preview=None, frac=None, message=None):
            frac = (done / total if total else 0.0) if frac is None else frac
            msg = message or (f"Frame {done} of {total}" if total > 1 else "Looking for tufts")
            self.update(jid, progress=round(frac, 4), tufts=int(tufts), eta=eta, message=msg,
                        fps=round(fps, 2))
            if preview is not None and time.time() - t_last[0] > 0.45:
                t_last[0] = time.time()
                p = preview
                if p.shape[1] > 960:
                    p = cv2.resize(p, (960, int(p.shape[0] * 960 / p.shape[1])))
                cv2.imencode(".jpg", p, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tofile(str(d / "preview.tmp.jpg"))
                (d / "preview.tmp.jpg").replace(d / "preview.jpg")

        def cancelled():
            return jid in self.cancel or jid not in self.meta

        src = d / m["input"]
        try:
            fn = analysis.analyze_video if m["kind"] == "video" else analysis.analyze_image
            stats = fn(src, d, opts, progress, cancelled)
        except ReviewNeeded as r:
            if jid in self.meta:
                self.update(jid, status="review", progress=0.0, eta=None, tufts=r.count, review_kind=r.kind, scan_spf=r.spf,
                            message="Check the tufts, then process all frames")
            return
        except Cancelled:
            self.cancel.discard(jid)
            if jid in self.meta:
                self.update(jid, status="cancelled", message="Cancelled", eta=None)
            return
        except Exception as exc:
            traceback.print_exc()
            if jid in self.meta:
                self.update(jid, status="error", error=str(exc) or exc.__class__.__name__, eta=None,
                            message="Failed")
            return
        if jid in self.meta:
            self.update(jid, status="done", progress=1.0, message="Done", stats=stats, eta=None,
                        tufts=stats["tufts_total"])
