"""MP4 in, annotated MP4 + CSV out: AI tuft detection on a fixed tuft map.

    python -m tools.process_video flight.mp4
    python -m tools.process_video flight.mp4 --weights runs/<run>/checkpoints/best.pt --out results/flight
    python -m tools.process_video flight.mp4 --edit-map        # reopen the map screen
    python -m tools.process_video flight.mp4 --edit-map --map-only   # only fix the map, no processing
    python -m tools.process_video flight.mp4 --limit 300       # quick look at the first 300 frames
    python -m tools.process_video flight2.mp4 --map maps/aircraft.tuftmap.json   # same aircraft: same tuft numbers

First run on a video: the model scans frames across the whole video and
proposes where each tuft is anchored (core/tuft_map.py). One frame opens with
the proposed roots; fix them once, then every frame is measured against that
map (core/map_pipeline.py): one ID per tuft for the whole video, arrows from
the root, clutter away from the tufts ignored. The classical skeleton detector
(finds more tufts than the model, plus lots of clutter) fills roots the model
missed; its settings are picked on the map frame (--skeleton off to disable). The map is saved as
<video>.tuftmap.json and reused next time.

Map screen:
  left click     on a root: remove it      elsewhere: add a root there
  wheel / right-drag   zoom / pan          z  undo
  f              flip the flow direction (the big arrow) if it points the wrong way;
                 roots move to the other end of every tuft
  Enter / space  accept and process        q / Esc  quit without processing
Green = found in most frames, yellow = found less often (check those), cyan = added by hand.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.sources import VideoFileSource  # noqa: E402
from core.tuft_map import TuftMap, build_map, default_map_path, load_ref, rederive  # noqa: E402

DEFAULT_WEIGHTS = "models/tuft_unet.pt"
WIN = "tuft map: click to add / remove roots, Enter to accept"


class MapEditor:
    def __init__(self, image, tmap, scan=None, max_w=1600, max_h=900):
        self.img, self.map, self.scan = image, tmap, scan
        h, w = image.shape[:2]
        self.base = min(max_w / w, max_h / h, 1.0)
        self.zoom, self.cx, self.cy = 1.0, w / 2, h / 2
        self.vw, self.vh = int(w * self.base), int(h * self.base)
        self.undo, self.pan = [], None

    @property
    def scale(self):
        return self.base * self.zoom

    def to_image(self, x, y):
        return self.cx + (x - self.vw / 2) / self.scale, self.cy + (y - self.vh / 2) / self.scale

    def render(self):
        s = self.scale
        M = np.float32([[s, 0, self.vw / 2 - self.cx * s], [0, s, self.vh / 2 - self.cy * s]])
        view = cv2.warpAffine(self.img, M, (self.vw, self.vh), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        L = self.map.median_length()
        for a in self.map.anchors:
            p = np.array(a.root) * s + M[:, 2]
            col = (255, 255, 0) if a.manual else (60, 220, 60) if a.hits >= 0.5 else (0, 220, 255)
            if a.length_px > 0:
                d = np.array([np.cos(np.radians(a.angle_deg)), -np.sin(np.radians(a.angle_deg))])
                q = p + d * 0.6 * (a.length_px or L) * s
                cv2.line(view, tuple(p.astype(int)), tuple(q.astype(int)), col, 1, cv2.LINE_AA)
            cv2.circle(view, tuple(p.astype(int)), max(3, int(4 * min(s, 2))), col, 2, cv2.LINE_AA)
        f = np.array(self.map.flow, float)
        if f.any():                                     # the flow direction, big, top left
            c = np.array([70, 90])
            cv2.arrowedLine(view, tuple((c - f * 45).astype(int)), tuple((c + f * 45).astype(int)), (0, 0, 0), 9,
                            cv2.LINE_AA, tipLength=0.35)
            cv2.arrowedLine(view, tuple((c - f * 45).astype(int)), tuple((c + f * 45).astype(int)), (0, 255, 255), 4,
                            cv2.LINE_AA, tipLength=0.35)
            cv2.putText(view, "flow", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        bar = (f"{len(self.map.anchors)} tufts   click: add/remove   wheel: zoom   z: undo   "
               f"f: flip flow   Enter: accept   q: quit")
        cv2.rectangle(view, (0, 0), (self.vw, 26), (0, 0, 0), -1)
        cv2.putText(view, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imshow(WIN, view)

    def on_mouse(self, event, x, y, flags, _=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self.to_image(x, y)
            gone = self.map.remove_near((ix, iy), 10 / self.scale)
            if gone is not None:
                self.undo.append(("removed", gone))
            else:
                nid = self.map.add((ix, iy), self.map.median_length(), self._typical_angle())
                self.undo.append(("added", nid))
        elif event == cv2.EVENT_MOUSEWHEEL:
            ix, iy = self.to_image(x, y)
            self.zoom = float(np.clip(self.zoom * (1.25 if flags > 0 else 0.8), 1.0, 12.0))
            s = self.scale
            self.cx, self.cy = ix - (x - self.vw / 2) / s, iy - (y - self.vh / 2) / s
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pan = (x, y, self.cx, self.cy)
        elif event == cv2.EVENT_MOUSEMOVE and self.pan is not None:
            x0, y0, cx, cy = self.pan
            self.cx, self.cy = cx - (x - x0) / self.scale, cy - (y - y0) / self.scale
        elif event == cv2.EVENT_RBUTTONUP:
            self.pan = None
        else:
            return
        self.render()

    def _typical_angle(self):
        from core.tracker import circ_mean_deg
        angs = [a.angle_deg for a in self.map.anchors if a.length_px > 0]
        return circ_mean_deg(angs) if angs else 0.0

    def do_undo(self):
        if not self.undo:
            return
        kind, x = self.undo.pop()
        if kind == "added":
            self.map.anchors = [a for a in self.map.anchors if a.id != x]
        else:
            self.map.anchors.append(x)

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.render()
        while True:
            k = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1 or k in (ord("q"), 27):
                cv2.destroyAllWindows()
                return False
            if k in (13, 10, 32):
                cv2.destroyAllWindows()
                return True
            if k == ord("z"):
                self.do_undo()
                self.render()
            if k == ord("f"):
                if self.scan is None:
                    print("flip needs the scan: rerun with --rebuild-map")
                else:
                    rederive(self.map, self.scan, -np.array(self.map.flow, float))
                    self.undo.clear()
                    self.render()


def pick_skeleton(pipe, ref, mode):
    """Classical skeleton mask backend whose settings attach to the most map roots on the map frame, or None."""
    if mode == "off" or not pipe.tracks:
        return None
    from src.backends import ClassicalBackend
    pols = ("dark", "light") if mode == "auto" else (mode,)
    best, best_n, desc = None, 0, ""
    for pol in pols:
        for bs, c in ((51, 10), (101, 15), (25, 5)):
            b = ClassicalBackend(bs, c, polarity=pol)
            n = pipe.roots_matched_by(ref, b)
            if n > best_n:
                best, best_n, desc = b, n, f"{pol}, block {bs}, c {c}"
    if best is None or best_n < 0.2 * len(pipe.tracks):
        print("skeleton detector: off (attaches to too few roots)")
        return None
    print(f"skeleton detector: {desc} ({best_n}/{len(pipe.tracks)} roots on the map frame)")
    return best


def renumber(tmap):
    L = tmap.median_length()
    tmap.anchors.sort(key=lambda a: (round(a.root[1] / max(L, 1)), a.root[0]))
    for i, a in enumerate(tmap.anchors, 1):
        a.id = i


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", default=None, help="output folder (default: results/<video name>)")
    ap.add_argument("--map", default=None,
                    help="tuft map file (default: <video>.tuftmap.json); reuse one map for every video of the "
                         "same aircraft + camera mount so tuft numbers stay the same")
    ap.add_argument("--rebuild-map", action="store_true", help="build the map again from the video")
    ap.add_argument("--edit-map", action="store_true", help="open the map screen even if a map exists")
    ap.add_argument("--no-confirm", action="store_true", help="accept the proposed map without the screen")
    ap.add_argument("--map-only", action="store_true", help="build / edit and save the map, then stop")
    ap.add_argument("--samples", type=int, default=60, help="frames scanned to build the map")
    ap.add_argument("--min-hits", type=float, default=0.3, help="fraction of scanned frames a root must appear in")
    ap.add_argument("--threshold", type=float, default=None, help="model threshold (default: the checkpoint's)")
    ap.add_argument("--map-threshold-scale", type=float, default=0.6,
                    help="threshold x this while building the map (finds faint tufts)")
    ap.add_argument("--skeleton", choices=["auto", "dark", "light", "off"], default="auto",
                    help="classical skeleton detector for roots the model missed (auto: pick polarity on the map frame)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N frames")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    if not Path(a.weights).exists():
        print(f"model not found: {a.weights} (pass --weights runs/<run>/checkpoints/best.pt)")
        return 2
    from detector import TuftDetector
    from src.unet import UNetBackend
    backend = UNetBackend(a.weights, device=a.device, threshold=a.threshold)
    print(f"model {a.weights}  threshold {backend.threshold}  device {backend.device}")

    src = VideoFileSource(a.video)
    map_path = Path(a.map) if a.map else default_map_path(a.video)
    scan = None
    if map_path.exists() and not a.rebuild_map:
        tmap = TuftMap.load(map_path)
        ref = load_ref(map_path, src, tmap.ref_index)
        print(f"tuft map {map_path}: {len(tmap.anchors)} tufts")
        confirm = a.edit_map
    else:
        print(f"building the tuft map from {a.samples} frames ...")
        det = TuftDetector(min_area=20, max_area=1e6, min_elongation=2.5, min_length_px=15, max_length_px=5000,
                           backend=backend, vector_mode="tip")
        thr = backend.threshold
        backend.threshold = thr * a.map_threshold_scale    # faint far tufts; persistence filters noise
        tmap, ref, scan = build_map(src, det, a.samples, a.min_hits,
                              progress=lambda k, n: print(f"\r  {k}/{n}", end="", flush=True))
        backend.threshold = thr
        print(f"\n  proposed {len(tmap.anchors)} tufts (reference frame {tmap.ref_index})")
        confirm = not a.no_confirm
    if confirm:
        if not MapEditor(ref, tmap, scan).run():
            print("quit: nothing processed")
            return 1
        renumber(tmap)
    tmap.save(map_path, ref)
    print(f"tuft map saved: {map_path} ({len(tmap.anchors)} tufts)")
    if a.map_only:
        return 0

    from app.exporter import export_all
    import app.overlays.builtin  # noqa: F401  (registers the overlays)
    from app.overlays import overlays
    from core.map_pipeline import MapPipeline
    pipe = MapPipeline(tmap, backend, ref, fps=src.fps)
    first = src.read_at(0)
    if pipe.aligner is not None and first is not None:
        pipe.aligner.ref_to_frame(first)
        if pipe.aligner.inliers < 40:
            print(f"WARNING: this video lines up poorly with the map's reference frame "
                  f"({pipe.aligner.inliers} matching points): different camera position? "
                  f"Check with --edit-map --map-only, or build a new map with --rebuild-map")
    pipe.extra = pick_skeleton(pipe, ref, a.skeleton)
    src.close()
    src = VideoFileSource(a.video)                     # fresh reader from frame 0
    results, t0 = [], time.time()
    total = min(src.frame_count, a.limit or src.frame_count)
    while len(results) < total:
        item = src.read()
        if item is None:
            break
        i, frame = item
        results.append(pipe.process_frame(frame, index=i, time_s=i / src.fps))
        if len(results) % 50 == 0 or len(results) == total:
            fps = len(results) / (time.time() - t0)
            found = np.mean([sum(t.detected for t in r.tufts) for r in results[-50:]])
            print(f"\r  {len(results)}/{total} frames  {fps:.1f} fps  tufts found {found:.0f}/{len(tmap.anchors)}",
                  end="", flush=True)
    src.close()
    print()
    out = Path(a.out) if a.out else Path("results") / Path(a.video).stem
    paths = export_all(a.video, results, out, enabled=[ov.name for ov in overlays() if ov.name != "mask"])
    for k, p in paths.items():
        print(f"{k}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
