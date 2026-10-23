"""Video writing (browser-playable H.264) and small drawing helpers. No detector code in here."""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# BGR colours by flow state (0 unknown, 1 attached, 2 unsteady, 3 separated)
STATE_BGR = {0: (200, 170, 110), 1: (96, 200, 70), 2: (40, 190, 255), 3: (70, 70, 240)}
STATE_NAME = {0: "learning", 1: "steady", 2: "fluttering", 3: "stalled"}


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class H264Writer:
    """Pipe BGR frames into ffmpeg (libx264, yuv420p, faststart) so every browser can play the result.
    Falls back to OpenCV's mp4v writer when ffmpeg is not available (then `browser_ok` is False)."""

    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.size = (int(size[0]), int(size[1]))
        self.fps = float(fps)
        self.proc = None
        self.cv = None
        self.browser_ok = False
        exe = ffmpeg_exe()
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{self.size[0]}x{self.size[1]}", "-r", f"{self.fps:.5f}", "-i", "-",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self.browser_ok = True
            except OSError:
                self.proc = None
        if self.proc is None:
            self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self.cv.isOpened():
                raise IOError(f"cannot write video {self.path}")

    def write(self, frame):
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        if self.proc is not None:
            try:
                self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError):
                raise IOError("ffmpeg stopped while writing the video")
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            proc, self.proc = self.proc, None
            err = proc.stderr.read().decode("utf8", "ignore")
            rc = proc.wait()
            proc.stderr.close()
            if rc != 0:
                raise IOError(f"ffmpeg failed: {err.strip()[-300:]}")
        if self.cv is not None:
            self.cv.release()
            self.cv = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _outlined_text(img, text, org, scale, color, thick):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_axes(img, cs, k):
    """Draw the user's coordinate axes at the origin (only for a defined coordinate system)."""
    if cs is None or not cs.defined:
        return
    try:
        o = cs.to_pixel([(0.0, 0.0)])[0]
        span = max(img.shape[:2]) * 0.12
        ux = cs.units_per_pixel(o) or 1.0
        px = cs.to_pixel([(span * ux, 0.0)])[0]
        py = cs.to_pixel([(0.0, span * ux)])[0]
    except Exception:
        return
    o_i = tuple(int(round(v)) for v in o)
    for tip, label, col in ((px, "X", (60, 60, 240)), (py, "Y", (70, 200, 70))):
        t = tuple(int(round(v)) for v in tip)
        cv2.arrowedLine(img, o_i, t, (20, 20, 20), int(4 * k) + 2, cv2.LINE_AA, tipLength=0.12)
        cv2.arrowedLine(img, o_i, t, col, int(2.4 * k) + 1, cv2.LINE_AA, tipLength=0.12)
        _outlined_text(img, label, (t[0] + 6, t[1] - 6), 0.6 * k + 0.2, col, max(1, int(1.6 * k)))
    cv2.circle(img, o_i, int(5 * k) + 1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, o_i, int(5 * k) + 1, (20, 20, 20), 1, cv2.LINE_AA)


def draw_frame(img, tufts, banner=None, cs=None, show_ids=True):
    """Annotated copy of `img`. tufts: iterable of dicts/rows with id, root, tip, vector, length, state."""
    out = img.copy()
    h, w = out.shape[:2]
    k = max(w, h) / 1000.0                          # thickness scales with resolution
    thick = max(1, int(round(2 * k)))
    draw_axes(out, cs, k)
    for t in tufts:
        col = STATE_BGR.get(int(t["state"]), STATE_BGR[0])
        rx, ry = t["root"]
        vx, vy = t["vector"]
        ln = max(22.0 * k, 0.8 * t["length"])
        end = (rx + vx * ln, ry + vy * ln)
        a = (int(round(rx)), int(round(ry)))
        b = (int(round(end[0])), int(round(end[1])))
        cv2.arrowedLine(out, a, b, (25, 25, 25), thick + 2, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(out, a, b, col, thick, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(out, a, max(2, int(3 * k)), (255, 255, 255), -1, cv2.LINE_AA)
        if show_ids:
            _outlined_text(out, str(t["id"]), (a[0] + int(5 * k), a[1] - int(6 * k)), 0.42 * k + 0.12,
                           (255, 255, 255), 1)
    if banner:
        bh = int(26 * k) + 10
        ov = out.copy()
        cv2.rectangle(ov, (0, 0), (w, bh), (18, 24, 34), -1)
        out = cv2.addWeighted(ov, 0.72, out, 0.28, 0)
        cv2.putText(out, banner, (int(10 * k) + 4, int(bh * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * k + 0.15, (235, 240, 245), 1, cv2.LINE_AA)
    return out
