"""Fast label review: SAM 2 proposals now, U-Net predictions in later rounds.

    python -m tools.review_labels --data data/real/proposed --out data/real/labeled

Window shows the frame with the mask (green), each tuft's root (red dot) and
tip-tangent arrow (cyan), SAM-rejected candidates (thin red outline) and
ignore regions (blue tint: no training loss there).

  a  accept frame -> copied to --out (training data), next
  r  reject frame -> never used, next
  left click     delete the tuft under the cursor / restore a red candidate
  b / e          brush mode: add / erase pixels (left-drag); v = back to click mode
  [ ]            brush radius        wheel   zoom at cursor     right-drag  pan
  z  undo        n / p  next / prev without deciding
  m  toggle overlay    h  help    q  quit

Accepted frames keep a pixel-exact mask; every decision and edit is appended
to <out>/../review_log.jsonl with time spent.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anchor import tape_distance  # noqa: E402
from src.label_io import load_frame, load_ignore, save_frame  # noqa: E402
from src.labeling import list_samples  # noqa: E402
from src.review_core import ReviewState  # noqa: E402

WIN = "tuft review"
GREEN, RED, CYAN, YELLOW, BLUE = (60, 220, 60), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 120, 0)


def tape_root_clips(config="configs/autolabel.yaml"):
    try:
        from src.config import load_config
        cfg = load_config(config)
    except Exception:
        return set(), False
    default = bool(cfg["detector"]["default"].get("tape_root"))
    return {c for c, s in (cfg.get("clips") or {}).items() if s.get("tape_root")}, default


class ReviewApp:
    def __init__(self, data, out, stems, log_path, max_w=1600, max_h=900):
        self.data, self.out, self.stems, self.log_path = Path(data), Path(out), stems, Path(log_path)
        self.max_w, self.max_h = max_w, max_h
        self.k, self.mode, self.radius, self.show = 0, "click", 3, True
        self.zoom, self.cx, self.cy = 1.0, None, None
        self.drag, self.pan = None, None
        self.tape_clips, self.tape_default = tape_root_clips()
        self.decided = {}
        self.load(0)

    # ---------------------------------------------------------- frames
    def load(self, k):
        self.k = k
        img, mask, info, meta = load_frame(self.data, self.stems[k])
        clip = meta.get("clip_id")
        anchor = tape_distance(img) if (clip in self.tape_clips or self.tape_default) else None
        self.state = ReviewState(img, mask, info, meta, anchor)
        self.ignore = load_ignore(self.data, self.stems[k])
        h, w = img.shape[:2]
        self.base = min(self.max_w / w, self.max_h / h, 1.0)
        self.zoom, self.cx, self.cy = 1.0, w / 2, h / 2

    def decide(self, decision):
        s, stem = self.state, self.stems[self.k]
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "stem": stem, "data": str(self.data),
               **s.summary(decision)}
        meta_all = json.loads((self.data / "meta.json").read_text())
        meta = dict(meta_all.get(stem, {}))
        if decision == "accept":
            out_meta = dict(meta, reviewed=True, status="accepted", edited=rec["edited"],
                            review_seconds=rec["seconds"], proposed_by=meta.get("source"))
            save_frame(self.out, stem, s.image, s.mask, s.tufts, out_meta,
                       extra=[{"source": "reviewed"} for _ in s.tufts], ignore=self.ignore)
        meta_all[stem] = dict(meta, status="accepted" if decision == "accept" else "rejected")
        (self.data / "meta.json").write_text(json.dumps(meta_all, indent=1))
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.decided[stem] = decision
        print(f"{stem}: {decision}  tufts {rec['n_tufts_before']}->{rec['n_tufts_after']}  "
              f"+{rec['pixels_added']}/-{rec['pixels_removed']} px  {rec['seconds']}s")
        return self.next()

    def next(self):
        if self.k < len(self.stems) - 1:
            self.load(self.k + 1)
            return True
        return False

    # ---------------------------------------------------------- view
    def scale(self):
        return self.base * self.zoom

    def to_image(self, x, y):
        s = self.scale()
        w, h = self.view_size()
        return self.cx + (x - w / 2) / s, self.cy + (y - h / 2) / s

    def view_size(self):
        h, w = self.state.image.shape[:2]
        return int(w * self.base), int(h * self.base)

    def render(self):
        s = self.state
        img = s.image.copy()
        if self.show and self.ignore is not None:  # ignore regions: blue tint, no loss there
            ig = self.ignore > 0
            img[ig] = (0.6 * img[ig] + 0.4 * np.array(BLUE)).astype(np.uint8)
        if self.show:
            m = s.mask > 0
            img[m] = (0.45 * img[m] + 0.55 * np.array(GREEN)).astype(np.uint8)
            for i, c in enumerate(s.rejected):
                if not s.restored[i]:
                    cv2.drawContours(img, [c], -1, RED, 1)
            lw = max(1, int(round(1.5 / self.scale())))
            for t in s.tufts:
                r = tuple(int(round(v)) for v in t.root)
                tip = np.array(t.tip)
                end = tip + np.array(t.vector) * max(15.0, 0.25 * t.length_px)
                cv2.arrowedLine(img, tuple(int(v) for v in tip), tuple(int(v) for v in end), CYAN, lw, tipLength=0.3)
                cv2.circle(img, r, max(2, 3 * lw), RED, -1)
        vw, vh = self.view_size()
        s_ = self.scale()
        M = np.array([[s_, 0, vw / 2 - self.cx * s_], [0, s_, vh / 2 - self.cy * s_]], np.float32)
        view = cv2.warpAffine(img, M, (vw, vh), flags=cv2.INTER_AREA if s_ < 1 else cv2.INTER_NEAREST)
        stem = self.stems[self.k]
        done = self.decided.get(stem, "")
        bar = (f"{self.k + 1}/{len(self.stems)} {stem}  tufts {len(s.tufts)}  mode {self.mode}"
               f"{f' r={self.radius}' if self.mode != 'click' else ''}  zoom {self.zoom:.1f}x  {done}"
               f"   [a]ccept [r]eject click=delete b/e=brush z=undo h=help")
        cv2.rectangle(view, (0, 0), (vw, 26), (0, 0, 0), -1)
        cv2.putText(view, bar, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
        return view

    # ---------------------------------------------------------- input
    def on_mouse(self, event, x, y, flags, _=None):
        ix, iy = self.to_image(x, y)
        if event == cv2.EVENT_MOUSEWHEEL:
            old = self.zoom
            self.zoom = float(np.clip(self.zoom * (1.25 if flags > 0 else 0.8), 1.0, 12.0))
            # keep the point under the cursor fixed
            self.cx = ix - (ix - self.cx) * old / self.zoom
            self.cy = iy - (iy - self.cy) * old / self.zoom
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pan = (x, y, self.cx, self.cy)
        elif event == cv2.EVENT_MOUSEMOVE and self.pan is not None:
            x0, y0, cx0, cy0 = self.pan
            self.cx, self.cy = cx0 - (x - x0) / self.scale(), cy0 - (y - y0) / self.scale()
        elif event == cv2.EVENT_RBUTTONUP:
            self.pan = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "click":
                self.state.click(ix, iy)
            else:
                self.state.begin_stroke()
                self.drag = (ix, iy)
                self.state.paint(self.drag, self.drag, self.radius, self.mode == "add")
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.state.paint(self.drag, (ix, iy), self.radius, self.mode == "add")
            self.drag = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            self.state.end_stroke(self.mode == "add", self.radius)
            self.drag = None

    def on_key(self, key):
        """Returns False to quit."""
        c = chr(key) if 0 <= key < 256 else ""
        if c == "q":
            return False
        if c == "a":
            return self.decide("accept") or self._finished()
        if c == "r":
            return self.decide("reject") or self._finished()
        if c == "n":
            self.next()
        elif c == "p" and self.k > 0:
            self.load(self.k - 1)
        elif c == "z":
            self.state.undo()
        elif c == "b":
            self.mode = "add"
        elif c == "e":
            self.mode = "erase"
        elif c in ("v", "\x1b"):
            self.mode = "click"
        elif c == "[":
            self.radius = max(1, self.radius - 1)
        elif c == "]":
            self.radius = min(40, self.radius + 1)
        elif c == "m":
            self.show = not self.show
        elif c == "h":
            print(__doc__)
        return True

    def _finished(self):
        print("last frame reviewed")
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/real/proposed")
    ap.add_argument("--out", default="data/real/labeled")
    ap.add_argument("--log", default=None, help="default: <out>/../review_log.jsonl")
    ap.add_argument("--all", action="store_true", help="also show frames already decided")
    ap.add_argument("--clips", nargs="*", default=None)
    a = ap.parse_args(argv)

    meta = json.loads((Path(a.data) / "meta.json").read_text()) if (Path(a.data) / "meta.json").exists() else {}
    stems = [s for s in list_samples(a.data)
             if (a.all or meta.get(s, {}).get("status", "proposed") == "proposed")
             and (not a.clips or meta.get(s, {}).get("clip_id") in set(a.clips))]
    if not stems:
        print("nothing to review")
        return 0
    log = a.log or str(Path(a.out).parent / "review_log.jsonl")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    app = ReviewApp(a.data, a.out, stems, log)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, app.on_mouse)
    while True:
        cv2.imshow(WIN, app.render())
        key = cv2.waitKey(20)
        if key != -1 and not app.on_key(key & 0xFF):
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    acc = sum(v == "accept" for v in app.decided.values())
    print(f"accepted {acc}, rejected {len(app.decided) - acc}; log: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
