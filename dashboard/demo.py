"""A small made-up wind-tunnel clip (white tufts on a grey panel) so the dashboard can be tried without
any footage. Part of the left side of the panel stays attached, a patch on the right separates and flaps."""
import math
from pathlib import Path

import cv2
import numpy as np

from .media import H264Writer


def make_demo_video(path, seconds=7.0, fps=24, size=(960, 540), seed=4):
    rng = np.random.default_rng(seed)
    w, h = size
    bg = np.full((h, w, 3), (92, 88, 84), np.float32)
    bg += np.linspace(-10, 14, w, dtype=np.float32)[None, :, None]
    cols, rows = 9, 5
    tufts = []
    for r in range(rows):
        for c in range(cols):
            rx = 90 + c * 90 + rng.integers(-6, 7)
            ry = 90 + r * 84 + rng.integers(-6, 7)
            tufts.append({"rx": float(rx), "ry": float(ry), "L": float(rng.uniform(46, 58)),
                          "ph": rng.uniform(0, 6.28), "sep": c >= 6 and 1 <= r <= 3})
    n = int(seconds * fps)
    wr = H264Writer(path, fps, (w, h))        # H.264 so every browser can play the sample
    base = np.clip(bg, 0, 255).astype(np.uint8)
    noises = [rng.normal(0, 2.5, base.shape).round().astype(np.int16) for _ in range(6)]
    for f in range(n):
        t = f / fps
        img = base.copy()
        img = np.clip(img.astype(np.int16) + noises[f % len(noises)], 0, 255).astype(np.uint8)
        for tf in tufts:
            rx, ry, L = tf["rx"], tf["ry"], tf["L"]
            if tf["sep"]:
                ang = 18 + 85 * math.sin(2 * math.pi * 1.7 * t + tf["ph"]) + 25 * math.sin(7.3 * t + tf["ph"] * 2)
                Lc = L * (0.62 + 0.22 * math.sin(2 * math.pi * 2.3 * t + tf["ph"]))
            else:
                ang = 12 + 3.0 * math.sin(2 * math.pi * 0.6 * t + tf["ph"])
                Lc = L
            pts, p = [], np.array([rx, ry])
            for i in range(21):
                s = i / 20
                phi = math.radians(ang + (18 * s * s) * math.sin(3 * t + tf["ph"]))
                pts.append(p.copy())
                p = p + (Lc / 20) * np.array([math.cos(phi), -math.sin(phi)])
            cv2.rectangle(img, (int(rx) - 5, int(ry) - 5), (int(rx) + 5, int(ry) + 5), (60, 170, 210), -1)
            for i in range(20):
                a = tuple(int(round(v)) for v in pts[i])
                b = tuple(int(round(v)) for v in pts[i + 1])
                cv2.line(img, a, b, (238, 238, 238), 3 if i < 10 else 2, cv2.LINE_AA)
        wr.write(img)
    wr.close()
    return Path(path)
