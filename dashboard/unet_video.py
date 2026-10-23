"""The trained-model (U-Net) video path. Same steps as tools.process_video, without its OpenCV windows:

  1. build a tuft map: the model scans ~60 frames and keeps the tuft ends that recur in place (core.tuft_map);
  2. measure every frame against that map: one ID per tuft for the whole video, arrows always from the root
     (core.map_pipeline.MapPipeline), with the classical skeleton detector filling roots the model missed.

The map is saved with the analysis and can be reviewed / corrected in the dashboard (Setup > Tuft map), the
same job tools.process_video's map screen does. A reviewed map can be saved under a name and reused for other
videos of the same aircraft + camera mount, so tuft numbers stay the same from video to video.
"""
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np

MAP_FILE = "tuft.tuftmap.json"
SCAN_FILE = "tuft.scan.pkl"
LIBRARY = Path("dashboard_data") / "maps"          # replaced by the server with <data dir>/maps
SAMPLES, MIN_HITS, MAP_THRESHOLD_SCALE = 60, 0.3, 0.6   # tools.process_video defaults


def map_path(job_dir):
    return Path(job_dir) / MAP_FILE


def _safe(name):
    return "".join(ch for ch in str(name) if ch.isalnum() or ch in "-_ .").strip()


def library_names():
    LIBRARY.mkdir(parents=True, exist_ok=True)
    return sorted(p.name[: -len(".tuftmap.json")] for p in LIBRARY.glob("*.tuftmap.json"))


def save_to_library(job_dir, name):
    name = _safe(name)
    if not name:
        raise ValueError("Give the map a name.")
    src = map_path(job_dir)
    if not src.exists():
        raise ValueError("This analysis has no tuft map to save.")
    LIBRARY.mkdir(parents=True, exist_ok=True)
    dst = LIBRARY / f"{name}.tuftmap.json"
    shutil.copy(src, dst)
    ref = Path(str(src) + ".ref.png")
    if ref.exists():
        shutil.copy(ref, Path(str(dst) + ".ref.png"))
    return name


def _copy_from_library(name, job_dir):
    src = LIBRARY / f"{_safe(name)}.tuftmap.json"
    if not src.exists():
        raise ValueError(f"The saved tuft map “{name}” was not found.")
    dst = map_path(job_dir)
    shutil.copy(src, dst)
    ref = Path(str(src) + ".ref.png")
    if ref.exists():
        shutil.copy(ref, Path(str(dst) + ".ref.png"))
    # a library map has no scan, so "flip" is unavailable for it until the map is rebuilt
    Path(job_dir, SCAN_FILE).unlink(missing_ok=True)


class MapRunner:
    """Builds (or loads) the tuft map, then measures frames against it."""

    def __init__(self, video, job_dir, opts, weights, eff_fps, progress, cancelled):
        from core.map_pipeline import MapPipeline
        from core.sources import VideoFileSource
        from core.tuft_map import TuftMap, build_map, load_ref
        from detector import TuftDetector
        from src.unet import UNetBackend
        from tools.process_video import pick_skeleton, renumber

        self.warnings = []
        self.weights = str(weights)
        self.backend = UNetBackend(weights, device=None, threshold=None)
        self.device = str(self.backend.device)
        job_dir = Path(job_dir)
        mp = map_path(job_dir)
        src = VideoFileSource(video)
        try:
            if opts.map_mode == "keep" and mp.exists():
                tmap = TuftMap.load(mp)
                ref = load_ref(mp, src, tmap.ref_index)
                self.map_origin = "reviewed"
            elif opts.map_name:
                _copy_from_library(opts.map_name, job_dir)
                tmap = TuftMap.load(mp)
                ref = load_ref(mp, src, tmap.ref_index)
                self.map_origin = f"saved map “{opts.map_name}”"
            else:
                det = TuftDetector(min_area=20, max_area=1e6, min_elongation=2.5, min_length_px=15,
                                   max_length_px=5000, backend=self.backend, vector_mode="tip")
                thr = self.backend.threshold
                self.backend.threshold = thr * MAP_THRESHOLD_SCALE       # faint far tufts; persistence filters noise

                def prog(k, n):
                    if cancelled():
                        from .analysis import Cancelled
                        raise Cancelled()
                    progress(done=k, total=n, tufts=0, eta=None, fps=0, frac=0.18 * k / n,
                             message=f"Finding where each tuft is anchored ({k} of {n})")
                try:
                    tmap, ref, scan = build_map(src, det, SAMPLES, MIN_HITS, progress=prog)
                finally:
                    self.backend.threshold = thr
                renumber(tmap)
                tmap.save(mp, ref)
                job_dir.joinpath(SCAN_FILE).write_bytes(pickle.dumps(scan))
                self.map_origin = "built from this video"
            if not tmap.anchors:
                raise ValueError("The model found no tufts to build a map from. Check that the video shows the "
                                 "tufted surface, or try the classical detector.")
            self.map = tmap
            self.ref = ref
            self.pipe = MapPipeline(tmap, self.backend, ref, fps=eff_fps)
            first = src.read_at(0)
            if self.pipe.aligner is not None and first is not None:
                self.pipe.aligner.ref_to_frame(first)
                if self.pipe.aligner.inliers < 40:
                    self.warnings.append("This video lines up poorly with the tuft map's reference picture. "
                                         "If the camera is not fixed to the surface, choose “Camera moves” in Setup.")
            self.pipe.extra = pick_skeleton(self.pipe, ref, "auto")
        finally:
            src.close()

    def process(self, frame, index, time_s):
        return self.pipe.process_frame(frame, index=index, time_s=time_s).visible()

    @property
    def note(self):
        return f"U-Net ({Path(self.weights).name}), tuft map {self.map_origin}"


# ---------------------------------------------------------------- map review (Setup > Tuft map)
def load_map(job_dir):
    from core.tuft_map import TuftMap
    mp = map_path(job_dir)
    if not mp.exists():
        return None
    return TuftMap.load(mp)


def describe(job_dir):
    tmap = load_map(job_dir)
    if tmap is None:
        return None
    return {"size": list(tmap.size), "flow": list(tmap.flow), "can_flip": Path(job_dir, SCAN_FILE).exists(),
            "anchors": [{"id": a.id, "x": round(a.root[0], 1), "y": round(a.root[1], 1), "hits": round(a.hits, 2),
                         "manual": bool(a.manual)} for a in tmap.anchors]}


def edit(job_dir, add=(), remove=(), flip=False):
    """Apply the review edits and save. -> description of the new map."""
    from core.tuft_map import rederive
    tmap = load_map(job_dir)
    if tmap is None:
        raise ValueError("This analysis has no tuft map.")
    ref = Path(str(map_path(job_dir)) + ".ref.png")
    if flip:
        sp = Path(job_dir, SCAN_FILE)
        if not sp.exists():
            raise ValueError("The flow direction can only be flipped on a map built from this video.")
        scan = pickle.loads(sp.read_bytes())
        rederive(tmap, scan, -np.array(tmap.flow, float))
    gone = {int(i) for i in remove}
    tmap.anchors = [a for a in tmap.anchors if a.id not in gone]
    for x, y in add:
        tmap.add((float(x), float(y)), tmap.median_length(), tmap.flow_angle_deg())
    if flip or add or gone:
        from tools.process_video import renumber
        renumber(tmap)
    tmap.save(map_path(job_dir), cv2.imread(str(ref)) if ref.exists() else None)
    return describe(job_dir)
