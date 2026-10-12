"""Video <-> frames. The detector only ever sees still images; this module is the part that
turns an MP4 into a stream of frames and turns annotated frames back into an MP4."""
import shutil
import subprocess
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}


class VideoError(Exception):
    """Raised when a video cannot be opened, decoded or written."""


def is_video_path(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


class VideoReader:
    """Iterates a video's frames as BGR arrays. Use as a context manager."""

    def __init__(self, path):
        p = Path(path)
        if not p.exists():
            raise VideoError(f"File not found: {path}")
        self.path = str(p)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise VideoError(f"Could not open video (unsupported codec or corrupt file): {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 and fps == fps else 30.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = n if n > 0 else None  # some containers can't tell us

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.cap.release()

    def frames(self):
        """Yield (frame_index, timestamp_s, frame) for every frame."""
        i = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield i, i / self.fps, frame
            i += 1

    def first_frame(self):
        ok, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not ok:
            raise VideoError(f"Could not read any frame from: {self.path}")
        return frame


class VideoWriter:
    """Writes frames to an MP4. OpenCV's own encoder ('mp4v') is written first; if ffmpeg is on
    PATH the file is then re-encoded to H.264 so it plays in browsers, QuickTime and PowerPoint."""

    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size  # (width, height)
        self._raw = self.path.with_name(self.path.stem + "_raw.mp4")
        self.writer = cv2.VideoWriter(str(self._raw), cv2.VideoWriter_fourcc(*"mp4v"),
                                      float(fps), size)
        if not self.writer.isOpened():
            raise VideoError(f"Could not open a video writer for: {path}")
        self.fps = float(fps)
        self.h264 = False

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(frame)

    def close(self):
        self.writer.release()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(self._raw),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                   "-movflags", "+faststart", str(self.path)]
            if subprocess.run(cmd).returncode == 0:
                self._raw.unlink(missing_ok=True)
                self.h264 = True
                return
        self._raw.replace(self.path)  # no ffmpeg (or it failed): keep the OpenCV encode

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
