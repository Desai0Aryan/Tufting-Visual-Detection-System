from dataclasses import dataclass, field

import cv2
import numpy as np

BACKGROUNDS = {
    "white": (235, 235, 235),
    "gray": (150, 150, 150),
    "red": (40, 40, 200),      # BGR
    "dark": (30, 30, 30),
    "blue": (180, 90, 30),
}


@dataclass
class SynthScene:
    image: np.ndarray
    tuft_mask: np.ndarray
    root_mask: np.ndarray
    truth: list = field(default_factory=list)  # dicts: root(x,y), tip(x,y), flow_deg


def _tuft_color(bg):
    """Pick a string colour with solid contrast against the background."""
    lum = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
    return (25, 25, 25) if lum > 110 else (225, 225, 225)


def make_scene(width=640, height=480, flow_deg=15.0, background="white", seed=0,
               spacing=(78, 62), length_range=(34, 58), hook_prob=0.5,
               heading_jitter_deg=6.0, noise=4.0):
    rng = np.random.default_rng(seed)
    bg = np.array(BACKGROUNDS.get(background, background), dtype=np.float32)
    img = np.ones((height, width, 3), np.float32) * bg
    # gentle lighting gradient + sensor noise, like an unevenly lit surface
    gx = np.linspace(-14, 14, width, dtype=np.float32)[None, :, None]
    gy = np.linspace(-8, 8, height, dtype=np.float32)[:, None, None]
    img += gx + gy
    img += rng.normal(0, noise, img.shape).astype(np.float32)

    color = _tuft_color(bg)
    tape_color = tuple(int(c) for c in np.clip(bg * 0.8 + 40, 0, 255))
    body = np.zeros((height, width), np.uint8)
    roots = np.zeros((height, width), np.uint8)
    canvas = np.clip(img, 0, 255).astype(np.uint8)
    truth = []

    for gy0 in range(40, height - 70, spacing[1]):
        for gx0 in range(30, width - 90, spacing[0]):
            rx = gx0 + int(rng.integers(-6, 7))
            ry = gy0 + int(rng.integers(-6, 7))
            L = float(rng.uniform(*length_range))
            theta = np.radians(flow_deg + rng.normal(0, heading_jitter_deg))
            hook = (rng.choice([-1, 1]) * rng.uniform(40, 110)) if rng.random() < hook_prob else 0.0

            n = 40
            pts, p = [], np.array([rx, ry], float)
            for i in range(n + 1):
                s = i / n
                phi = theta + np.radians(hook) * max(0.0, s - 0.55) ** 2 * 4.0  # hook near the free end
                pts.append(p.copy())
                p = p + (L / n) * np.array([np.cos(phi), -np.sin(phi)])
            pts = np.array(pts)

            # tape square at the root
            cv2.rectangle(canvas, (rx - 5, ry - 5), (rx + 5, ry + 5), tape_color, -1)
            cv2.rectangle(roots, (rx - 5, ry - 5), (rx + 5, ry + 5), 255, -1)
            for i in range(n):
                thick = 3.4 - 2.0 * (i / n)  # taper root -> tip
                a = tuple(int(round(v)) for v in pts[i])
                b = tuple(int(round(v)) for v in pts[i + 1])
                cv2.line(canvas, a, b, color, max(1, int(round(thick))), cv2.LINE_AA)
                cv2.line(body, a, b, 255, max(1, int(round(thick))))
            truth.append({
                "root": (float(rx), float(ry)),
                "tip": (float(pts[-1][0]), float(pts[-1][1])),
                "flow_deg": float(flow_deg),
            })
    return SynthScene(canvas, body, roots, truth)


def angular_error_deg(a_deg, b_deg):
    """Smallest absolute difference between two angles, in degrees (0..180)."""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return abs(d)
