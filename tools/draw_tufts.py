"""Check and fix the tufts: one line per tuft, from the root (tape) to the tip.

    python -m tools.draw_tufts data/videos/tufts_flight1.mp4                # 20 frames spread over the video
    python -m tools.draw_tufts data/videos/tufts_flight1.mp4 --frames 40
    python -m tools.draw_tufts data/videos/tufts_flight1.mp4 --at 100 2479 3300
    python -m tools.draw_tufts data/videos/tufts_flight1.mp4 --prefill none  # empty frames, draw everything
    python -m tools.draw_tufts data/videos/tufts_flight1.mp4 --between 75 95 120 139   # only these seconds

Frames are picked where the flow is doing something (--pick activity, the default
when results/<video>/<video>_tufts.csv exists from tools.process_video): the share of
unsteady / separated tufts, spread out in time and away from frames already done.
Calm stretches look the same frame after frame, so they get few frames.

Each new frame opens pre-filled with the detected tufts (--prefill map: skeleton
detector + model, kept only where attached to the video's tuft map, so no clutter;
--prefill skeleton: the raw skeleton detector). Fix what is wrong:

  right-click    delete the line under the cursor (wrong, or a bad shape: delete and redraw)
  left-click     SAM 2 outlines the tuft for you: click ON THE STRING NEXT TO THE TAPE (root end)
  left-drag      draw a missed tuft by hand: start ON THE ROOT (tape), drag along the string to the TIP
  f              flip the line under the cursor (root and tip swapped)
  wheel / middle-drag   zoom at cursor / pan        z  undo
  h              show / hide the model's guess (faint blue), to spot faint tufts
  Enter          accept this frame as it is (when the pre-fill is already right)
  n / p          next / previous frame (just moves)  q  save and quit

A frame counts as checked once you have edited it (add / delete / flip) or pressed
Enter; frames you only skipped past stay unchecked. Only checked frames become
training labels. Everything is saved as you go to data/drawn/<video>/json/<stem>.json
(frames in images/), so you can stop and continue later.
tools.drawn_to_labels turns checked frames into training labels.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WIN = "check tufts: right-click delete, drag ROOT->TIP add, f flip"
LINE, ROOT, HINT = (0, 255, 255), (0, 0, 255), (255, 140, 0)


def pick_frames(src, n, at=None, min_brightness=30):
    """Evenly spread frame indices, skipping black / title frames."""
    if at:
        return [int(i) for i in at]
    out = []
    for i in np.linspace(0, src.frame_count - 1, n * 2).astype(int):
        f = src.read_at(int(i))
        if f is not None and f.mean() >= min_brightness and (not out or i - out[-1] > src.frame_count / (n * 3)):
            out.append(int(i))
    return out[:n] if len(out) <= n else [out[k] for k in np.linspace(0, len(out) - 1, n).astype(int)]


def activity_frames(csv_path, n, fps, done=(), between=None, min_gap_s=2.0, skip_found_below=0.5):
    """Frames where the flow is doing something, from a tools.process_video CSV:
    score = share of tufts unsteady or separated, averaged over about a second. Greedy best-first with a minimum
    gap in time; frames near ones already done are skipped; title cards (few tufts
    found) are never picked."""
    import csv
    from collections import defaultdict
    found, busy = defaultdict(int), defaultdict(int)
    total = 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"])
            if r["detected"] in ("True", "1"):
                found[fr] += 1
                busy[fr] += r["stall"] in ("unsteady", "separated")
    if not found:
        return []
    total = max(found.values())
    def ok(fr):
        if found[fr] < skip_found_below * total:
            return False
        if between:
            s = fr / fps
            return any(lo <= s <= hi for lo, hi in zip(between[::2], between[1::2]))
        return True
    # sustained activity only: the share of busy tufts averaged over about a second,
    # so a few tufts flickering to "unsteady" from measurement jitter do not count
    frames = np.array(sorted(found))
    share = np.array([busy[fr] / found[fr] for fr in frames])
    w = max(1, int(round(fps)))
    smooth = np.convolve(share, np.ones(2 * w + 1) / (2 * w + 1), mode="same")
    score = dict(zip(frames.tolist(), smooth.tolist()))
    cands = sorted((fr for fr in found if ok(fr)), key=lambda fr: -score[fr])
    gap = min_gap_s * fps
    picked = []
    for fr in cands:
        if all(abs(fr - q) >= gap for q in list(done) + picked):
            picked.append(fr)
        if len(picked) >= n:
            break
    return sorted(picked)


def load_strokes(path):
    p = Path(path)
    return [np.array(s, float) for s in json.loads(p.read_text())["strokes"]] if p.exists() else []


def is_checked(path):
    p = Path(path)
    return p.exists() and bool(json.loads(p.read_text()).get("checked"))


def save_strokes(path, stem, frame_idx, strokes, size, checked=False, prefill=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"stem": stem, "frame_idx": frame_idx, "size": list(size),
                                      "checked": bool(checked), "prefill": prefill,
                                      "strokes": [np.round(s, 1).tolist() for s in strokes]}))


def stroke_near(strokes, xy, radius):
    """Index of the stroke with a point within radius of xy, or None."""
    best, best_d = None, radius
    for i, s in enumerate(strokes):
        d = float(np.min(np.linalg.norm(s - np.asarray(xy, float), axis=1)))
        if d <= best_d:
            best, best_d = i, d
    return best


class DrawApp:
    def __init__(self, frames, out, hint_fn=None, prefill_fn=None, prefill_name=None, sam_fn=None,
                 max_w=1600, max_h=900):
        self.frames, self.out, self.hint_fn, self.sam_fn = frames, Path(out), hint_fn, sam_fn
        self.message = ""
        self.prefill_fn, self.prefill_name = prefill_fn, prefill_name
        self.max_w, self.max_h = max_w, max_h
        self.k, self.show_hint = 0, False
        self.drawing, self.pan, self.mouse = None, None, (0, 0)
        self.load(0)

    def json_path(self, stem):
        return self.out / "json" / f"{stem}.json"

    def load(self, k):
        self.k = k
        self.idx, self.stem, self.img = self.frames[k]
        jp = self.json_path(self.stem)
        self.checked = is_checked(jp)
        self.undo, self.hint = [], None
        h, w = self.img.shape[:2]
        self.base = min(self.max_w / w, self.max_h / h, 1.0)
        self.zoom, self.cx, self.cy = 1.0, w / 2, h / 2
        self.vw, self.vh = int(w * self.base), int(h * self.base)
        img_path = self.out / "images" / f"{self.stem}.png"
        if not img_path.exists():
            img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_path), self.img)
        if jp.exists():
            self.strokes = load_strokes(jp)
        else:
            self.strokes = [np.asarray(s, float) for s in self.prefill_fn(self.idx, self.img)] if self.prefill_fn else []
            self.save()

    def save(self):
        save_strokes(self.json_path(self.stem), self.stem, self.idx, self.strokes, self.img.shape[1::-1],
                     self.checked, self.prefill_name)

    def mark_checked(self):
        self.checked = True
        self.save()

    @property
    def scale(self):
        return self.base * self.zoom

    def to_image(self, x, y):
        return self.cx + (x - self.vw / 2) / self.scale, self.cy + (y - self.vh / 2) / self.scale

    def render(self):
        s = self.scale
        M = np.float32([[s, 0, self.vw / 2 - self.cx * s], [0, s, self.vh / 2 - self.cy * s]])
        base = self.img
        if self.show_hint:
            if self.hint is None and self.hint_fn is not None:
                self.hint = self.hint_fn(self.img)
            if self.hint is not None:
                base = self.img.copy()
                base[self.hint] = (0.55 * base[self.hint] + 0.45 * np.array(HINT)).astype(np.uint8)
        view = cv2.warpAffine(base, M, (self.vw, self.vh), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        lw = 1 if s < 1.5 else 2
        for st in self.strokes + ([np.array(self.drawing)] if self.drawing and len(self.drawing) > 1 else []):
            pts = np.round(st * s + M[:, 2]).astype(np.int32)
            cv2.polylines(view, [pts], False, LINE, lw, cv2.LINE_AA)
            cv2.circle(view, tuple(pts[0]), lw + 2, ROOT, -1, cv2.LINE_AA)
        done = sum(is_checked(self.json_path(st)) for _, st, _ in self.frames)
        bar = (f"frame {self.k + 1}/{len(self.frames)} (#{self.idx}{', checked' if self.checked else ''})  "
               f"{done} checked   {len(self.strokes)} tufts   right-click: delete   drag ROOT->TIP: add   "
               f"f: flip   Enter: accept   h: hints   z: undo   n/p: next/prev   q: quit")
        cv2.rectangle(view, (0, 0), (self.vw, 26), (0, 0, 0), -1)
        cv2.putText(view, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        if self.message:
            cv2.rectangle(view, (0, 26), (self.vw, 50), (0, 0, 90), -1)
            cv2.putText(view, self.message, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, view)

    def on_mouse(self, event, x, y, flags, _=None):
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = [self.to_image(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing is not None:
            p = self.to_image(x, y)
            if np.hypot(p[0] - self.drawing[-1][0], p[1] - self.drawing[-1][1]) * self.scale >= 2:
                self.drawing.append(p)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing is not None:
            st = np.array(self.drawing + [self.to_image(x, y)], float)
            self.drawing = None
            self.message = ""
            if np.linalg.norm(st[-1] - st[0]) * self.scale >= 6:      # a drag: drawn by hand
                self.strokes.append(st)
            elif self.sam_fn is not None:                              # a click: SAM 2 outlines it
                self.message = "SAM 2 ..."
                got = self.sam_fn(self.img, tuple(st[0]))
                if got is None or len(got) < 2:
                    self.message = "SAM 2 found no string there: click closer to the string, or drag to draw it"
                    self.render()
                    return
                self.message = ""
                self.strokes.append(np.asarray(got, float))
            else:
                self.render()
                return
            self.undo.append(("added", None))
            self.mark_checked()
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = stroke_near(self.strokes, self.to_image(x, y), 8 / self.scale)
            if i is not None:
                self.undo.append(("removed", self.strokes.pop(i)))
                self.mark_checked()
        elif event == cv2.EVENT_MOUSEWHEEL:
            ix, iy = self.to_image(x, y)
            self.zoom = float(np.clip(self.zoom * (1.25 if flags > 0 else 0.8), 1.0, 12.0))
            self.cx, self.cy = ix - (x - self.vw / 2) / self.scale, iy - (y - self.vh / 2) / self.scale
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.pan = (x, y, self.cx, self.cy)
        elif event == cv2.EVENT_MOUSEMOVE and self.pan is not None:
            x0, y0, cx, cy = self.pan
            self.cx, self.cy = cx - (x - x0) / self.scale, cy - (y - y0) / self.scale
        elif event == cv2.EVENT_MBUTTONUP:
            self.pan = None
        else:
            return
        self.render()

    def flip_under_mouse(self):
        i = stroke_near(self.strokes, self.to_image(*self.mouse), 10 / self.scale)
        if i is not None:
            self.strokes[i] = self.strokes[i][::-1].copy()
            self.undo.append(("flipped", i))
            self.mark_checked()

    def do_undo(self):
        if not self.undo:
            return
        kind, x = self.undo.pop()
        if kind == "added" and self.strokes:
            self.strokes.pop()
        elif kind == "removed":
            self.strokes.append(x)
        elif kind == "flipped":
            self.strokes[x] = self.strokes[x][::-1].copy()
        self.save()

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.render()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), 27):
                break
            if key == ord("n") and self.k + 1 < len(self.frames):
                self.save()
                self.load(self.k + 1)
            elif key == ord("p") and self.k > 0:
                self.save()
                self.load(self.k - 1)
            elif key in (13, 10):
                self.mark_checked()
                if self.k + 1 < len(self.frames):
                    self.load(self.k + 1)
            elif key == ord("f"):
                self.flip_under_mouse()
            elif key == ord("z"):
                self.do_undo()
            elif key == ord("h"):
                self.show_hint = not self.show_hint
            else:
                continue
            self.render()
        self.save()
        cv2.destroyAllWindows()
        done = [st for _, st, _ in self.frames if is_checked(self.json_path(st))]
        n = sum(len(load_strokes(self.json_path(st))) for st in done)
        print(f"{len(done)}/{len(self.frames)} frames checked, {n} tufts -> {self.out / 'json'}")


def make_prefill(mode, video, weights):
    """-> fn(frame_idx, bgr) -> [stroke (N, 2) root -> tip], or None."""
    if mode == "none":
        return None
    from core.tuft_map import TuftMap, default_map_path
    map_path = default_map_path(video)
    if mode == "map" and not map_path.exists():
        print(f"no tuft map at {map_path} (make one with tools.process_video): using the raw skeleton detector")
        mode = "skeleton"
    if mode == "skeleton":
        from detector import TuftDetector
        dets = [TuftDetector(min_area=20, max_area=1e6, min_elongation=2.5, min_length_px=15, max_length_px=2000,
                             block_size=51, c=10, polarity=p, vector_mode="tip") for p in ("dark", "light")]

        def fn(_i, img):
            tufts = max((d.detect(img) for d in dets), key=len)    # the polarity that finds more
            return [np.asarray(t.path) for t in tufts if t.path is not None]
        return fn
    from core.map_pipeline import MapPipeline
    from core.sources import VideoFileSource
    from src.unet import UNetBackend
    from tools.process_video import pick_skeleton
    tmap = TuftMap.load(map_path)
    src = VideoFileSource(video)
    from core.tuft_map import load_ref
    ref = load_ref(map_path, src, tmap.ref_index)
    src.close()
    pipe = MapPipeline(tmap, UNetBackend(weights), ref)
    pipe.extra = pick_skeleton(pipe, ref, "auto")

    def fn(i, img):
        res = pipe.process_frame(img, index=i)
        return [np.asarray(t.path) for t in res.tufts if t.detected and t.path is not None]
    return fn


def make_sam(checkpoint, max_len=None):
    """-> fn(bgr, (x, y)) -> centre line root -> tip, or None. SAM 2 loads on the first click."""
    if not checkpoint or not Path(checkpoint).exists():
        if checkpoint:
            print(f"SAM 2 checkpoint not found ({checkpoint}): click-to-outline off, drag to draw")
        return None
    cache = {}

    def fn(img, xy):
        import torch
        from src.sam_click import tuft_from_click
        if "p" not in cache:
            from src.sam2_autolabel import DEFAULT_CONFIG, load_image_predictor
            print("loading SAM 2 ...")
            cfg = DEFAULT_CONFIG.replace("hiera_l", "hiera_s") if "small" in str(checkpoint) else DEFAULT_CONFIG
            cache["p"] = load_image_predictor(str(checkpoint), cfg, "cuda" if torch.cuda.is_available() else "cpu")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            return tuft_from_click(img, xy, cache["p"], max_len=max_len)
    return fn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--at", type=int, nargs="*", default=None, help="exact frame numbers instead")
    ap.add_argument("--pick", choices=["activity", "even"], default=None,
                    help="activity (default when a process_video CSV exists) or evenly spread")
    ap.add_argument("--between", type=float, nargs="*", default=None,
                    help="only these time ranges, in seconds: start end [start end ...]")
    ap.add_argument("--out", default=None, help="default data/drawn/<video name>")
    ap.add_argument("--weights", default="models/tuft_unet.pt", help="model for hints and the map pre-fill")
    ap.add_argument("--sam", default="models/sam2/sam2.1_hiera_large.pt",
                    help="SAM 2 checkpoint for click-to-outline ('' to turn it off)")
    ap.add_argument("--prefill", choices=["map", "skeleton", "none"], default="map",
                    help="map: detections attached to the tuft map (needs <video>.tuftmap.json); "
                         "skeleton: raw skeleton detector (lots of clutter); none: empty frames")
    a = ap.parse_args(argv)

    from core.sources import VideoFileSource
    src = VideoFileSource(a.video)
    name = Path(a.video).stem
    out = Path(a.out or f"data/drawn/{name}")
    csv_path = Path("results") / name / f"{name}_tufts.csv"
    pick = a.pick or ("activity" if csv_path.exists() else "even")
    done = [json.loads(p.read_text())["frame_idx"] for p in (out / "json").glob("*.json")
            if json.loads(p.read_text()).get("checked")] if (out / "json").exists() else []
    if a.at:
        idx = [int(i) for i in a.at]
    elif pick == "activity" and csv_path.exists():
        idx = activity_frames(csv_path, a.frames, src.fps, done, a.between)
        print(f"picked by flow activity from {csv_path} (avoiding {len(done)} checked frames)")
    else:
        idx = pick_frames(src, a.frames)
        if a.between:
            idx = [i for i in idx if any(lo <= i / src.fps <= hi for lo, hi in zip(a.between[::2], a.between[1::2]))]
    frames = [(i, f"{name}__f{i:06d}", src.read_at(i)) for i in idx]
    src.close()
    hint_fn = None
    if a.weights and Path(a.weights).exists():
        def hint_fn(img, _cache={}):
            if "b" not in _cache:
                from src.unet import UNetBackend
                _cache["b"] = UNetBackend(a.weights)
            b = _cache["b"]
            return b.predict_proba(img)[0] >= b.threshold * 0.6
    prefill_fn = make_prefill(a.prefill, a.video, a.weights) if a.weights and Path(a.weights).exists() \
        else make_prefill("skeleton" if a.prefill != "none" else "none", a.video, a.weights)
    print(f"{len(frames)} frames: {idx}")
    from core.tuft_map import TuftMap, default_map_path
    mp = default_map_path(a.video)
    max_len = 2.5 * TuftMap.load(mp).median_length() if mp.exists() else 0.4 * min(frames[0][2].shape[:2])
    DrawApp(frames, out, hint_fn, prefill_fn, a.prefill, make_sam(a.sam, max_len)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
