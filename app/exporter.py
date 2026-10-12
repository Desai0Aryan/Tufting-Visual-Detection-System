"""Exports (no Qt): annotated MP4 + per-frame CSV of all tufts + CSV of the
selected tufts. Usable headless and from the app's export thread."""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from core.export import write_csv
from core.sources import VideoFileSource

from .overlays import OverlayContext, line_scale_for, render


class _VideoWriter:
    """H.264 through ffmpeg when it is installed (plays everywhere), else OpenCV mp4v."""

    def __init__(self, path, fps, size):
        self.path, self.proc, self.cv = Path(path), None, None
        ff = shutil.which("ffmpeg")
        if ff:
            w, h = size
            cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(self.path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed while writing the annotated video")
        else:
            self.cv.release()


def export_all(video_path, results, out_dir, enabled, selected=(), stem=None, progress=None):
    """results: FrameResults 0..N-1 in order. Returns dict of written paths.
    progress(done, total) is called per frame; return False from it to cancel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or Path(video_path).stem
    paths = {"video": out_dir / f"{stem}_annotated.mp4", "csv_all": out_dir / f"{stem}_tufts.csv"}
    src = VideoFileSource(video_path)
    try:
        writer = _VideoWriter(paths["video"], src.fps, src.size)
        ctx = OverlayContext(enabled=set(enabled), selected=set(selected), line_scale=line_scale_for(src.size[::-1]))
        n = len(results)
        for k, r in enumerate(results):
            frame = src.read_at(r.index)
            if frame is None:
                break
            writer.write(render(frame, r, ctx))
            if progress is not None and progress(k + 1, n) is False:
                break
        writer.close()
    finally:
        src.close()
    write_csv(paths["csv_all"], results)
    if selected:
        paths["csv_selected"] = out_dir / f"{stem}_selected_tufts.csv"
        write_csv(paths["csv_selected"], results, track_ids=selected)
    return paths
