"""Run the existing detector / tracker over a video or an image and store plain results.

The detection, geometry and tracking code is the project's own (core.Pipeline, detector.TuftDetector).
This module only: picks the engine, scales the classical settings to the picture size, loops over
the frames, draws the annotated output and saves a compact results file.

Raw record per tuft per frame (all pixels, y down):
    [id, root_x, root_y, tip_x, tip_y, vec_x, vec_y, angle_deg, length_px, length_ratio, state, confidence]
state: 0 learning, 1 steady (attached), 2 fluttering (unsteady), 3 stalled (separated)
"""
import glob
import gzip
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import media

STATE_CODE = {"unknown": 0, "attached": 1, "unsteady": 2, "separated": 3}
FIELDS = ["id", "root_x", "root_y", "tip_x", "tip_y", "vec_x", "vec_y", "angle_deg", "length_px",
          "length_ratio", "state", "confidence"]
TARGET_FRAMES = 600             # auto frame skipping aims for about this many processed frames


class Cancelled(Exception):
    pass


class ReviewNeeded(Exception):
    """Raised after the first stage (tuft map / picture detections) so the person can check the tufts in the
    pop-up before every frame is processed. `kind` is "map" (video) or "image"; `count` the tufts proposed."""

    def __init__(self, kind, count, spf=None):
        super().__init__(f"{count} tufts are waiting to be checked")
        self.kind, self.count = kind, int(count)
        self.spf = spf          # seconds the scan took per frame, used for the "about N min" estimate


@dataclass
class Options:
    engine: str = "auto"        # auto / classical / unet
    polarity: str = "auto"      # auto / light / dark  (colour of the tufts against the surface)
    size: str = "auto"          # auto / small / large (multiplies the tuft length limits)
    step: int = 0               # analyse every Nth frame; 0 = automatic
    roi: Optional[list] = None  # [x, y, w, h] in video pixels, or None for the whole picture
    weights: str = ""           # path to a model (.pt); empty = find one automatically
    tracking: str = "map"       # map = camera fixed on the surface (tuft map, most accurate); free = any camera
    map_mode: str = "build"     # build a tuft map from the video, or keep the one already saved for this analysis
    map_name: str = ""          # a tuft map saved earlier for the same aircraft / camera mount
    review: bool = False        # stop after finding the tufts so the person can add / remove them in the pop-up

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        o = cls(**known)
        if o.engine not in ("auto", "classical", "unet"):
            o.engine = "auto"
        if o.polarity not in ("auto", "light", "dark"):
            o.polarity = "auto"
        if o.size not in ("auto", "small", "large"):
            o.size = "auto"
        if o.tracking not in ("map", "free"):
            o.tracking = "map"
        if o.map_mode not in ("build", "keep"):
            o.map_mode = "build"
        o.review = bool(o.review)
        o.weights = str(o.weights or "").strip().strip('"')
        o.map_name = "".join(ch for ch in str(o.map_name or "") if ch.isalnum() or ch in "-_ .")[:60].strip()
        try:
            o.step = max(0, int(o.step))
        except (TypeError, ValueError):
            o.step = 0
        if o.roi is not None:
            try:
                r = [int(round(float(v))) for v in o.roi]
                o.roi = r if len(r) == 4 and r[2] > 8 and r[3] > 8 else None
            except (TypeError, ValueError):
                o.roi = None
        return o


# ---------------------------------------------------------------- engine selection
def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def find_weights(explicit=""):
    """The trained model to use: the path typed in Setup > $TUFT_WEIGHTS > models/tuft_unet.pt (the default of
    tools.process_video) > the newest runs/*/checkpoints/best.pt."""
    from . import ROOT
    cands = [explicit, os.environ.get("TUFT_WEIGHTS", ""), str(ROOT / "models" / "tuft_unet.pt")]
    runs = sorted(glob.glob(str(ROOT / "runs" / "*" / "checkpoints" / "best.pt")), key=os.path.getmtime, reverse=True)
    for c in cands + runs:
        if c and Path(c).is_file():
            return str(c)
    return ""


def _fix_for(why):
    if "PyTorch" in why:
        return "Close TuftView and open it with Start-TuftView again; it installs PyTorch automatically."
    if "not found:" in why:
        return "Check the path in Setup > Model file, then analyze again."
    return "Copy the model file to models/tuft_unet.pt (or finish a training run), then analyze again."


def engine_status(opts=None):
    """What would run right now: {engine, weights, note, warning}. `warning` is set when the trained model
    could not be used and the less accurate classical detector takes over (the dashboard shows it)."""
    opts = opts or Options()
    if opts.engine == "classical":
        return {"engine": "classical", "weights": "", "note": "Classical detector (chosen)", "warning": ""}
    w = find_weights(opts.weights)
    why = ""
    if opts.weights and not Path(opts.weights).is_file():
        why = f"The model file you entered was not found: {opts.weights}"
    elif not w:
        why = "No trained model found."
    elif not torch_available():
        why = "PyTorch is not installed, so the trained model cannot run."
    if why:
        if opts.engine == "unet":
            raise ValueError(why)
        return {"engine": "classical", "weights": "", "note": "Classical detector", "warning": why, "fix": _fix_for(why)}
    return {"engine": "unet", "weights": w, "note": f"U-Net ({Path(w).name})", "warning": ""}


def pick_engine(opts: Options):
    """-> (engine, weights, note). Kept for callers that do not need the warning."""
    e = engine_status(opts)
    return e["engine"], e["weights"], e["note"]


def classical_params(width, height, opts: Options):
    """The README's classical settings, scaled to the picture (they were tuned for ~640 px images)."""
    L = max(width, height) / 640.0
    m = {"auto": 1.0, "small": 0.55, "large": 2.0}[opts.size]
    return dict(block_size=int(25 * L) | 1, c=5.0, min_area=15 * L * L * m * m, max_area=600 * L * L * m * m,
                min_elongation=2.0, min_length_px=12 * L * m, max_length_px=110 * L * m)


def _pipeline_config(engine, weights, polarity, w, h, opts):
    from core import PipelineConfig
    if engine == "unet":
        return PipelineConfig(backend="unet", checkpoint=weights, max_length_px=5000, min_length_px=10,
                              min_area=10, max_area=1e7)
    p = classical_params(w, h, opts)
    return PipelineConfig(backend="classical", polarity=polarity, block_size=p["block_size"], c=p["c"],
                          min_area=p["min_area"], max_area=p["max_area"], min_elongation=p["min_elongation"],
                          min_length_px=p["min_length_px"], max_length_px=p["max_length_px"],
                          keep_mask=False)


def _count_tufts(frame, polarity, w, h, opts, roi):
    from detector import TuftDetector
    from src.backends import build_backend
    p = classical_params(w, h, opts)
    be = build_backend("classical", block_size=p["block_size"], c=p["c"], clahe_clip=3.0, clahe_tile=8,
                       polarity=polarity)
    det = TuftDetector(min_area=p["min_area"], max_area=p["max_area"], min_elongation=p["min_elongation"],
                       min_length_px=p["min_length_px"], max_length_px=p["max_length_px"], backend=be, roi=roi)
    return len(det.detect(frame))


def resolve_polarity(frames, w, h, opts, roi):
    """Tuft colour for the whole video: whichever polarity finds more tufts on a few sample frames."""
    if opts.polarity in ("light", "dark"):
        return opts.polarity
    score = {"light": 0, "dark": 0}
    for f in frames:
        for pol in score:
            score[pol] += _count_tufts(f, pol, w, h, opts, roi)
    return "light" if score["light"] > score["dark"] else "dark"


def _roi(opts, w, h):
    if not opts.roi:
        return None
    x, y, rw, rh = opts.roi
    x, y = max(0, min(x, w - 2)), max(0, min(y, h - 2))
    rw, rh = min(rw, w - x), min(rh, h - y)
    return (x, y, rw, rh) if rw > 8 and rh > 8 else None


def _round(v, n=2):
    return None if v is None or v != v else round(float(v), n)


def _rec(t):
    st = STATE_CODE.get(t.stall, 0)
    ratio = t.length_ratio if t.length_ratio == t.length_ratio else 1.0
    return [int(t.track_id), _round(t.root[0]), _round(t.root[1]), _round(t.tip[0]), _round(t.tip[1]),
            _round(t.vector[0], 4), _round(t.vector[1], 4), _round(t.angle_deg, 2), _round(t.length_px),
            _round(ratio, 3), st, _round(t.confidence, 2)]


def draw_records(frame, recs, banner=None, cs=None):
    rows = [{"id": r[0], "root": (r[1], r[2]), "tip": (r[3], r[4]), "vector": (r[5], r[6]),
             "length": r[8], "state": r[10]} for r in recs]
    return media.draw_frame(frame, rows, banner=banner, cs=cs)


# ---------------------------------------------------------------- results file
def save_results(path, results):
    with gzip.open(path, "wt", encoding="utf8", compresslevel=5) as fh:
        json.dump(results, fh, separators=(",", ":"))


def load_results(path):
    with gzip.open(path, "rt", encoding="utf8") as fh:
        return json.load(fh)


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("This file could not be opened as a video. Use a standard MP4 (H.264) file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps == fps and fps > 0 else 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first = cap.read()
    cap.release()
    if not ok or w <= 0:
        raise ValueError("No picture could be read from this video. Try re-exporting it as an MP4 (H.264).")
    return {"fps": fps, "frames": n, "width": w, "height": h, "duration": n / fps if n else 0.0}, first


# ---------------------------------------------------------------- video
def analyze_video(src, out_dir, opts: Options, progress: Callable, cancelled: Callable):
    """Detect + track tufts through the whole video. Writes results.json.gz and annotated.mp4."""
    out_dir = Path(out_dir)
    info, first = video_info(src)
    w, h, fps, n = info["width"], info["height"], info["fps"], info["frames"]
    step = opts.step or max(1, math.ceil(n / TARGET_FRAMES)) if n else max(1, opts.step)
    roi = _roi(opts, w, h)
    status = engine_status(opts)
    engine, weights, note = status["engine"], status["weights"], status["note"]
    warnings = [status["warning"]] if status["warning"] else []

    runner = None
    pipe = None
    polarity = "light"
    if engine == "unet" and opts.tracking == "map":
        from . import unet_video
        stop_at_map = opts.review and opts.map_mode != "keep"
        runner = unet_video.MapRunner(src, out_dir, opts, weights, fps / step, progress, cancelled,
                                      build_only=stop_at_map)
        if stop_at_map:
            raise ReviewNeeded("map", len(runner.map.anchors), getattr(runner, "scan_spf", None))
        note = runner.note
        warnings += runner.warnings
        if roi is not None:
            warnings.append("The analysis area is ignored with the tuft map (the map already fixes where the tufts are).")
            roi = None
        process = lambda frame, k, t: runner.process(frame, k, t)
    else:
        if engine == "classical":
            cap = cv2.VideoCapture(str(src))
            samples = []
            for pos in ([0] if n < 4 else [0, n // 3, (2 * n) // 3]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, f = cap.read()
                if ok:
                    samples.append(f)
            cap.release()
            polarity = resolve_polarity(samples or [first], w, h, opts, roi)
            note += f", {polarity} tufts"
        from core import Pipeline
        cfg = _pipeline_config(engine, weights, polarity, w, h, opts)
        pipe = Pipeline(cfg, fps=fps / step)
        if roi is not None:
            pipe.detector.roi = roi
        process = lambda frame, k, t: pipe.process_frame(frame, index=k, time_s=t).visible()

    out_fps = fps / step
    times, frames_out = [], []
    video_path = Path(out_dir) / "annotated.mp4"
    cap = cv2.VideoCapture(str(src))
    t0 = time.time()
    total_proc = math.ceil(n / step) if n else 0
    writer = media.H264Writer(video_path, out_fps, (w, h))
    tuft_total = 0

    # Three stages run side by side so the model never waits for the disk or the encoder:
    #   reader thread (decode, skipping unused frames cheaply) -> this thread (model + tuft map) -> writer thread
    #   (draw arrows, H.264 encode, live preview). Same frames, same order, same numbers as a plain loop.
    rq, wq = queue.Queue(maxsize=6), queue.Queue(maxsize=6)
    stop, errors = threading.Event(), []

    def put(q, item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def read_loop():
        try:
            idx = -1
            while not stop.is_set():
                if (idx + 1) % step:                    # a frame we skip: decode it, but do not convert it
                    if not cap.grab():
                        break
                    idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if not put(rq, (idx, frame)):
                    return
        except Exception as exc:
            errors.append(exc)
        finally:
            put(rq, None)

    def write_loop():
        done = 0
        try:
            while True:
                item = wq.get()
                if item is None:
                    return
                frame, recs, idx = item
                ann = draw_records(frame, recs, banner=f"t = {idx / fps:6.2f} s    tufts: {len(recs)}")
                writer.write(ann)
                done += 1
                if done == 1 or done % 3 == 0:
                    el = time.time() - t0
                    eta = (el / done) * (total_proc - done) if total_proc else None
                    progress(done=done, total=total_proc, tufts=len(recs), eta=eta, fps=done / max(el, 1e-6),
                             preview=ann)
        except Exception as exc:
            errors.append(exc)
            stop.set()
        finally:
            while True:                                 # never leave the model thread blocked on a full queue
                try:
                    wq.get_nowait()
                except queue.Empty:
                    break

    rt = threading.Thread(target=read_loop, daemon=True, name="tuft-reader")
    wt = threading.Thread(target=write_loop, daemon=True, name="tuft-writer")
    rt.start()
    wt.start()
    k = 0
    try:
        while True:
            try:
                item = rq.get(timeout=0.25)
            except queue.Empty:
                if cancelled():
                    raise Cancelled()
                if errors:
                    raise errors[0]
                continue
            if item is None:
                break
            if cancelled():
                raise Cancelled()
            if errors:
                raise errors[0]
            idx, frame = item
            recs = [_rec(t) for t in process(frame, k, idx / fps)]
            times.append(round(idx / fps, 4))
            frames_out.append(recs)
            tuft_total += len(recs)
            k += 1
            if not put(wq, (frame, recs, idx)):
                break
        if errors:
            raise errors[0]
        put(wq, None)
        wt.join()
        if errors:
            raise errors[0]
    finally:
        stop.set()
        rt.join(timeout=5)
        wt.join(timeout=5)
        cap.release()
        writer.close()
    done = k
    if not frames_out:
        raise ValueError("No frames could be read from this video.")

    results = {"kind": "video", "fps": fps, "step": step, "width": w, "height": h, "engine": note,
               "browser_video": writer.browser_ok, "fields": FIELDS, "times": times, "frames": frames_out,
               "roi": list(roi) if roi else None, "polarity": polarity if engine == "classical" else None,
               "tracking": "map" if runner else "free", "warnings": warnings}
    save_results(out_dir / "results.json.gz", results)
    ids = {r[0] for f in frames_out for r in f}
    return {"warnings": warnings, "tracking": results["tracking"], "engine_kind": engine, "frames": done, "tufts_total": len(ids), "mean_tufts": tuft_total / max(done, 1),
            "seconds": time.time() - t0, "engine": note, "step": step, "browser_video": writer.browser_ok}


# ---------------------------------------------------------------- pictures: detect, then (optionally) let the person edit
IMAGE_TUFTS = "image_tufts.json"


def _detect_image(img, opts, roi, engine, weights, note):
    """The same detector call as before. -> (list of {root, tip, conf}, polarity, note)."""
    from detector import TuftDetector
    from src.backends import build_backend
    h, w = img.shape[:2]

    def run(backend):
        p = classical_params(w, h, opts)
        if engine == "unet":
            p.update(min_area=10, max_area=1e7, min_length_px=10, max_length_px=5000)
        det = TuftDetector(min_area=p["min_area"], max_area=p["max_area"], min_elongation=p["min_elongation"],
                           min_length_px=p["min_length_px"], max_length_px=p["max_length_px"],
                           backend=backend, roi=roi)
        return det.detect(img)

    polarity = None
    if engine == "unet":
        found = run(build_backend("unet", weights=weights, threshold=None))
    else:
        p = classical_params(w, h, opts)
        pols = ("light", "dark") if opts.polarity == "auto" else (opts.polarity,)
        runs = []
        for pol in pols:
            be = build_backend("classical", block_size=p["block_size"], c=p["c"], clahe_clip=3.0,
                               clahe_tile=8, polarity=pol)
            runs.append((pol, run(be)))
        polarity, found = max(runs, key=lambda r: len(r[1]))
        note += f", {polarity} tufts"
    return [{"root": [float(t.root[0]), float(t.root[1])], "tip": [float(t.tip[0]), float(t.tip[1])],
             "vec": [float(t.vector[0]), float(t.vector[1])], "angle": float(t.angle_deg),
             "length": float(t.length_px), "conf": float(t.confidence)} for t in found], polarity, note


def save_image_tufts(out_dir, tufts):
    p = Path(out_dir) / IMAGE_TUFTS
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"tufts": tufts}), "utf8")
    tmp.replace(p)


def load_image_tufts(out_dir):
    return json.loads((Path(out_dir) / IMAGE_TUFTS).read_text("utf8"))["tufts"]


def image_rec(i, t):
    """Raw record for one picture tuft ({root, tip, conf}); the angle is the math angle, y flipped, as in Tuft."""
    (rx, ry), (tx, ty) = t["root"], t["tip"]
    dx, dy = tx - rx, ty - ry
    n = math.hypot(dx, dy) or 1.0
    vec = t.get("vec") or [dx / n, dy / n]                       # detected tufts keep the detector's tip direction
    ang = t["angle"] if "angle" in t else math.degrees(math.atan2(-vec[1], vec[0]))
    length = t.get("length") or n
    return [i, _round(rx), _round(ry), _round(tx), _round(ty), _round(vec[0], 4), _round(vec[1], 4),
            _round(ang, 2), _round(length), 1.0, 0, _round(t.get("conf", 1.0), 2)]


def edit_image_tufts(out_dir, remove=(), add=()):
    """Apply the pop-up's edits to the detected picture tufts and save. `add` items are {root, tip} or [x, y]
    (a click: it gets the typical length and direction of the tufts found). -> the new list."""
    tufts = load_image_tufts(out_dir)
    gone = {int(i) for i in remove}
    tufts = [t for k, t in enumerate(tufts) if k not in gone]
    lens = [math.dist(t["root"], t["tip"]) for t in tufts]
    typ_len = float(np.median(lens)) if lens else 40.0
    vx = float(np.mean([t["tip"][0] - t["root"][0] for t in tufts])) if tufts else typ_len
    vy = float(np.mean([t["tip"][1] - t["root"][1] for t in tufts])) if tufts else 0.0
    vn = math.hypot(vx, vy) or 1.0
    for a in add:
        if isinstance(a, dict):
            root, tip = [float(v) for v in a["root"]], [float(v) for v in a["tip"]]
            if math.dist(root, tip) < 3:
                continue
        else:
            root = [float(a[0]), float(a[1])]
            tip = [root[0] + vx / vn * typ_len, root[1] + vy / vn * typ_len]
        tufts.append({"root": root, "tip": tip, "conf": 1.0, "manual": True})
    save_image_tufts(out_dir, tufts)
    return tufts


# ---------------------------------------------------------------- image
def analyze_image(src, out_dir, opts: Options, progress: Callable, cancelled: Callable):
    """Same detector on one still picture (the pipeline main.py uses for --image).
    With opts.review it stops after detecting so the person can add / remove tufts (see edit_image_tufts)."""
    out_dir = Path(out_dir)
    data = np.fromfile(str(src), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        try:                                    # AVIF/HEIC etc. renamed to .jpg: Pillow can sometimes read them
            from PIL import Image
            img = cv2.cvtColor(np.asarray(Image.open(src).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError("This picture could not be read. Use a JPG or PNG file.")
    h, w = img.shape[:2]
    roi = _roi(opts, w, h)
    status = engine_status(opts)
    engine, weights, note = status["engine"], status["weights"], status["note"]
    t0 = time.time()
    progress(done=0, total=1, tufts=0, eta=None, fps=0, preview=img)
    polarity = None
    reviewed = Path(out_dir) / IMAGE_TUFTS
    if opts.map_mode == "keep" and reviewed.is_file():       # the person already checked the tufts in the pop-up
        tufts = load_image_tufts(out_dir)
        note += ", tufts checked by you"
    else:
        tufts, polarity, note = _detect_image(img, opts, roi, engine, weights, note)
        if cancelled():
            raise Cancelled()
        if opts.review:
            save_image_tufts(out_dir, tufts)
            raise ReviewNeeded("image", len(tufts))
    if cancelled():
        raise Cancelled()

    recs = [image_rec(i, t) for i, t in enumerate(tufts)]
    ann = draw_records(img, recs)
    cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tofile(str(out_dir / "annotated.jpg"))
    results = {"kind": "image", "fps": 1.0, "step": 1, "width": w, "height": h, "engine": note,
               "browser_video": False, "fields": FIELDS, "times": [0.0], "frames": [recs],
               "roi": list(roi) if roi else None, "polarity": polarity, "tracking": "free",
               "warnings": [status["warning"]] if status["warning"] else []}
    save_results(out_dir / "results.json.gz", results)
    progress(done=1, total=1, tufts=len(recs), eta=0, fps=0, preview=ann)
    return {"warnings": results["warnings"], "tracking": "free", "engine_kind": engine,
            "frames": 1, "tufts_total": len(recs), "mean_tufts": float(len(recs)),
            "seconds": time.time() - t0, "engine": note, "step": 1, "browser_video": False}


# ---------------------------------------------------------------- re-render with axes
def rerender_video(src, out_dir, results, cs, progress=None):
    """Draw the saved results onto the original video again (no detection), e.g. with the coordinate axes."""
    fps, step = results["fps"], results["step"]
    cap = cv2.VideoCapture(str(src))
    tmp = Path(out_dir) / "annotated_new.mp4"
    w, h = results["width"], results["height"]
    writer = media.H264Writer(tmp, fps / step, (w, h))
    n_out = len(results["frames"])
    try:
        idx = -1
        k = 0
        while k < n_out:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            recs = results["frames"][k]
            ann = draw_records(frame, recs, banner=f"t = {idx / fps:6.2f} s    tufts: {len(recs)}", cs=cs)
            writer.write(ann)
            k += 1
            if progress and k % 10 == 0:
                progress(k / n_out)
    finally:
        cap.release()
        writer.close()
    os.replace(tmp, Path(out_dir) / "annotated.mp4")
