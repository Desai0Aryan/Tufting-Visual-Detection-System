"""Realistic synthetic tufts: a tufted panel seen in perspective.

Modelled on real tuft tests (Images/sample_tufts1.jpg, the car door): a dense
grid of short thin yarn tufts, each anchored under a small tape tab, on a
painted panel (panel lines, rivets, a coloured edge stripe) that recedes in
perspective over a sky / sea / ground background. Tufts follow a smooth flow
field, hook at the tip, and turn wavy inside separated-flow patches.

The panel is drawn flat at 2x, then warped into the frame with a random
homography, so far tufts are smaller and closer together, exactly as on
camera. Labels go through the same warp: the tuft mask is every yarn pixel
(including yarn seen through translucent tape, never the tape), the root
mask a disk at each anchor.

    scene = make_scene(rng)                                 # procedural background
    scene = make_scene(rng, background=bgr_image)           # real background behind the panel
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

SURFACES = {   # BGR panel paint, weight
    "white": ((236, 236, 232), 0.45), "light_grey": ((200, 200, 196), 0.12), "silver": ((175, 178, 180), 0.08),
    "yellow": ((60, 200, 230), 0.05), "red": ((45, 45, 190), 0.06), "dark_grey": ((70, 72, 70), 0.08),
    "black": ((32, 32, 32), 0.05), "blue": ((150, 90, 40), 0.05), "olive": ((70, 110, 105), 0.06),
}
YARN_DARK = [(30, 30, 30), (45, 40, 35), (35, 35, 110), (110, 60, 30), (30, 90, 40)]
YARN_LIGHT = [(238, 238, 238), (225, 235, 240), (60, 215, 240), (40, 130, 245), (190, 140, 245)]
TAPE = {       # BGR, alpha range, weight: mostly small white / clear tabs like real tests
    "clear": ((225, 225, 225), (0.15, 0.35), 0.3), "white": ((240, 240, 240), (0.75, 0.95), 0.3),
    "silver": ((190, 190, 190), (0.8, 0.95), 0.12), "kapton": ((20, 170, 210), (0.45, 0.7), 0.1),
    "black": ((28, 28, 28), (0.85, 0.97), 0.06), "blue": ((200, 130, 40), (0.8, 0.95), 0.06),
    "red": ((40, 40, 200), (0.8, 0.95), 0.06),
}
STRIPES = [(45, 45, 190), (40, 40, 40), (150, 90, 40), (60, 200, 230)]


@dataclass
class SynthScene:
    image: np.ndarray          # BGR uint8
    tuft_mask: np.ndarray      # uint8 0/255
    root_mask: np.ndarray      # uint8 0/255
    truth: list = field(default_factory=list)   # per tuft: root, tip (frame px), flow_deg, separated


def _pick(rng, table):
    names = list(table)
    w = np.array([table[n][1] for n in names], float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def _smooth_noise(rng, h, w, cells):
    g = rng.uniform(-1, 1, (cells, cells)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _lum(bgr):
    b, g, r = bgr
    return 0.114 * b + 0.587 * g + 0.299 * r


# ---------------------------------------------------------------- backgrounds
def sky_background(rng, h, w):
    """Sky over sea / ground: vertical gradient, soft clouds, horizon."""
    top = np.array(rng.choice([[230, 210, 190], [200, 190, 180], [170, 150, 130], [240, 235, 230]]), np.float32)
    low = np.array(rng.choice([[90, 75, 55], [70, 80, 60], [110, 110, 100], [60, 55, 50]]), np.float32)
    horizon = rng.uniform(0.2, 0.7)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    t = 1 / (1 + np.exp(-(yy - horizon) * rng.uniform(15, 60)))
    img = top * (1 - t) + low * t
    img = img * np.ones((1, w, 1), np.float32)
    clouds = np.clip(_smooth_noise(rng, h, w, 6) + 0.5 * _smooth_noise(rng, h, w, 18), 0, None)
    img += (clouds * rng.uniform(10, 50) * (1 - t[..., 0]))[..., None]
    img += (_smooth_noise(rng, h, w, 40) * rng.uniform(3, 12) * t[..., 0])[..., None]   # ground / sea texture
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- the panel
def _paint_panel(rng, H, W, paint):
    img = np.ones((H, W, 3), np.float32) * np.array(paint, np.float32)
    img *= (1 + 0.08 * _smooth_noise(rng, H, W, 5))[..., None]
    line_col = (np.array(paint, np.float32) * (0.75 if _lum(paint) > 90 else 1.4)).tolist()
    for _ in range(int(rng.integers(0, 4))):                       # panel seams (straight, thin)
        if rng.random() < 0.5:
            x = rng.uniform(0.1, 0.9) * W
            cv2.line(img, (int(x), 0), (int(x + rng.normal(0, 30)), H), line_col, int(rng.integers(1, 4)), cv2.LINE_AA)
        else:
            y = rng.uniform(0.1, 0.9) * H
            cv2.line(img, (0, int(y)), (W, int(y + rng.normal(0, 30))), line_col, int(rng.integers(1, 4)), cv2.LINE_AA)
    if rng.random() < 0.4:                                         # rivet rows
        y0, gap = rng.uniform(0.05, 0.95) * H, rng.uniform(25, 60)
        for x in np.arange(gap / 2, W, gap):
            cv2.circle(img, (int(x), int(y0)), 3, line_col, -1, cv2.LINE_AA)
    stripe = None
    if rng.random() < 0.45:                                        # coloured edge stripe (leading edge)
        col = np.array(STRIPES[int(rng.integers(len(STRIPES)))], np.float32)
        x1 = int(W * rng.uniform(0.8, 0.92))
        img[:, x1:] = col * (1 + 0.08 * _smooth_noise(rng, H, W - x1, 4))[..., None]
        if rng.random() < 0.7:                                     # thin trim line along the stripe
            cv2.line(img, (x1 - 6, 0), (x1 - 6, H), (40, 170, 200), int(rng.integers(1, 6)), cv2.LINE_AA)
        stripe = (x1, col)
    return img, stripe


def _flow_field(rng, H, W):
    """Smooth flow angle (radians) over the panel, roughly along +x on average."""
    base = rng.normal(0, 0.15)
    return base + np.radians(rng.uniform(5, 25)) * _smooth_noise(rng, H, W, 4)


def _tuft_path(rng, root, theta, length, separated):
    """Centre-line: straight-ish shaft, hook near the tip; wavy when separated."""
    step = 1.5
    n = max(6, int(length / step))
    hook = rng.choice([-1, 1]) * rng.uniform(0.02, 0.09) if rng.random() < 0.65 else 0.0
    amp = rng.uniform(0.08, 0.25) if separated else rng.uniform(0, 0.02)
    freq, ph = rng.uniform(0.08, 0.2), rng.uniform(0, 6.3)
    p, a, pts = np.array(root, float), theta, []
    for i in range(n + 1):
        s = i / n
        pts.append(p.copy())
        a += (amp * np.sin(freq * i * step + ph) + hook * (s > 0.55) * (s - 0.55) * 6) * step * 0.5
        p = p + step * np.array([np.cos(a), np.sin(a)])
    return np.array(pts)


def make_panel(rng, H=1400, W=2000):
    """Flat tufted panel at 2x scale.
    -> (image f32, yarn alpha f32, tape alpha f32, opaque-tape mask, roots [(x, y)], truth)"""
    paint_name = _pick(rng, SURFACES)
    paint = SURFACES[paint_name][0]
    img, stripe = _paint_panel(rng, H, W, paint)

    dark_on_light = _lum(paint) > 120
    if rng.random() < 0.85:                                        # yarn contrasts with the paint
        palette = YARN_DARK if dark_on_light else YARN_LIGHT
    else:
        palette = YARN_LIGHT if dark_on_light else YARN_DARK
    yarn = np.array(palette[int(rng.integers(len(palette)))], np.float32)
    names = list(TAPE)
    tw = np.array([TAPE[n][2] for n in names])
    tape_name = names[int(rng.choice(len(names), p=tw / tw.sum()))]
    tape_col, (alo, ahi), _ = TAPE[tape_name]
    tape_alpha = rng.uniform(alo, ahi)

    length = rng.uniform(70, 150)                                  # at 2x: 35-75 px flat
    width = rng.uniform(2.5, 5.5)                                 # at 2x: 1.3-2.8 px flat
    tape_w, tape_h = length * rng.uniform(0.18, 0.3), length * rng.uniform(0.14, 0.25)
    pitch_x, pitch_y = length * rng.uniform(1.4, 2.2), length * rng.uniform(0.7, 1.3)
    stagger = rng.random() < 0.5
    flow = _flow_field(rng, H, W)
    # separated-flow patches: tufts wave and point anywhere
    sep = np.zeros((H, W), np.uint8)
    for _ in range(int(rng.choice([0, 0, 1, 2]))):
        c = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
        cv2.ellipse(sep, c, (int(rng.uniform(150, 600)), int(rng.uniform(100, 400))), rng.uniform(0, 180), 0, 360, 1, -1)

    yarn_a = np.zeros((H, W), np.uint8)
    shadow = np.zeros((H, W), np.uint8)
    tape_a = np.zeros((H, W), np.uint8)
    roots, truth = [], []
    margin = length * 0.6
    for r, y in enumerate(np.arange(margin, H - margin, pitch_y)):
        x0 = margin + (pitch_x / 2 if stagger and r % 2 else 0)
        for x in np.arange(x0, W - length * 1.2, pitch_x):
            if rng.random() < 0.03:                                # a missing tuft
                continue
            root = np.array([x, y]) + rng.normal(0, pitch_x * 0.04, 2)
            iy, ix = int(np.clip(root[1], 0, H - 1)), int(np.clip(root[0], 0, W - 1))
            separated = bool(sep[iy, ix])
            theta = flow[iy, ix] + (rng.uniform(-np.pi, np.pi) * 0.8 if separated else rng.normal(0, 0.08))
            pts = _tuft_path(rng, root, theta, length * rng.uniform(0.85, 1.15), separated)
            poly = np.round(pts * 4).astype(np.int32)
            w = max(1, int(round(width * rng.uniform(0.85, 1.15))))
            cv2.polylines(yarn_a, [poly], False, 255, w, cv2.LINE_AA, shift=2)
            off = np.round(np.array([2, 3]) * 4).astype(np.int32)
            cv2.polylines(shadow, [poly + off], False, 255, w + 1, cv2.LINE_AA, shift=2)
            # tape tab over the root, the yarn leaves it downstream
            d = np.array([np.cos(theta), np.sin(theta)])
            ang = rng.normal(0, 0.12)
            c, s = np.cos(ang), np.sin(ang)
            half = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * [tape_w / 2, tape_h / 2]
            centre = root + d * tape_w * 0.15
            quad = half @ np.array([[c, s], [-s, c]]) + centre
            cv2.fillPoly(tape_a, [np.round(quad * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
            roots.append(root)
            truth.append({"root_flat": root.tolist(), "tip_flat": pts[-1].tolist(),
                          "flow_deg": float(np.degrees(theta)), "separated": separated})

    ya = yarn_a.astype(np.float32) / 255
    ta = tape_a.astype(np.float32) / 255
    sh = cv2.GaussianBlur(shadow.astype(np.float32) / 255, (9, 9), 0)
    img *= (1 - 0.25 * sh)[..., None]
    tex = 1 + 0.07 * rng.normal(0, 1, (H, W, 1)).astype(np.float32)          # fibre texture
    core = cv2.GaussianBlur(ya, (3, 3), 0)
    ycol = yarn * tex * (0.85 + 0.15 * core[..., None])
    img = img * (1 - ya[..., None]) + ycol * ya[..., None]
    tcol = np.array(tape_col, np.float32) * (1 + 0.06 * _smooth_noise(rng, H, W, 6))[..., None]
    a = tape_alpha * ta
    img = img * (1 - a[..., None]) + tcol * a[..., None]
    opaque = ta > 0.5 if tape_alpha > 0.8 else np.zeros((H, W), bool)
    meta = {"paint": paint_name, "tape": tape_name, "stripe": stripe is not None}
    return img, ya, opaque, roots, truth, meta


# ---------------------------------------------------------------- scene
def _perspective_quad(rng, W, H):
    """Where the panel's corners land in the frame: near edge wide, far edge narrow."""
    near_w = rng.uniform(0.9, 1.6) * W
    far_w = near_w * rng.uniform(0.35, 0.85)
    near_y = rng.uniform(0.85, 1.15) * H
    far_y = rng.uniform(-0.1, 0.25) * H
    cx_near, cx_far = W * rng.uniform(0.35, 0.65), W * rng.uniform(0.35, 0.65)
    quad = np.array([[cx_far - far_w / 2, far_y], [cx_far + far_w / 2, far_y],
                     [cx_near + near_w / 2, near_y], [cx_near - near_w / 2, near_y]], np.float32)
    ang = rng.uniform(-0.6, 0.6)                                    # camera roll
    c, s = np.cos(ang), np.sin(ang)
    centre = np.array([W / 2, H / 2], np.float32)
    return ((quad - centre) @ np.array([[c, s], [-s, c]], np.float32) + centre).astype(np.float32)


def make_scene(rng, width=1280, height=720, background=None):
    img_p, ya, opaque, roots, truth, meta = make_panel(rng)
    Hp, Wp = ya.shape
    # the flat panel's x axis (flow) maps to "into the scene" or across it at random
    src = np.array([[0, 0], [Wp, 0], [Wp, Hp], [0, Hp]], np.float32)
    k = int(rng.integers(4))
    src = np.roll(src, k, axis=0)
    M = cv2.getPerspectiveTransform(src, _perspective_quad(rng, width, height))

    if background is None:
        bg = sky_background(rng, height, width).astype(np.float32)
    else:
        bg = cv2.resize(background, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    warp = lambda a, interp=cv2.INTER_AREA: cv2.warpPerspective(a, M, (width, height), flags=interp,
                                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    panel_a = warp(np.ones((Hp, Wp), np.float32), cv2.INTER_LINEAR)
    panel = warp(img_p, cv2.INTER_LINEAR)
    img = bg * (1 - panel_a[..., None]) + panel * panel_a[..., None]
    yarn = warp(np.where(opaque, 0, ya).astype(np.float32), cv2.INTER_LINEAR)
    tuft = (yarn > 0.35).astype(np.uint8) * 255

    root_mask = np.zeros((height, width), np.uint8)
    out_truth = []
    if roots:
        rp = cv2.perspectiveTransform(np.array(roots, np.float32)[None], M)[0]
        tips = cv2.perspectiveTransform(np.array([t["tip_flat"] for t in truth], np.float32)[None], M)[0]
        for (x, y), tip, t in zip(rp, tips, truth):
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(root_mask, (int(round(x)), int(round(y))), 3, 255, -1)
                out_truth.append({"root": [float(x), float(y)], "tip": tip.tolist(), "flow_deg": t["flow_deg"],
                                  "separated": t["separated"]})
    # camera: slight defocus, optional motion blur, noise, vignette
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.0))
    if rng.random() < 0.3:
        k = int(rng.integers(3, 8))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2] = 1 / k
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1), (k, k))
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    vig = 1 - rng.uniform(0, 0.25) * (((xx - width / 2) / width) ** 2 + ((yy - height / 2) / height) ** 2) * 2
    img = img * vig[..., None] + rng.normal(0, rng.uniform(1, 4), img.shape)
    meta["tufts"] = len(out_truth)
    return SynthScene(np.clip(img, 0, 255).astype(np.uint8), tuft, root_mask, out_truth), meta


def background_crops(image, exclude, size, rng, n=1, tries=200):
    """Random size=(w, h) crops of `image` not touching any (x, y, w, h) box in `exclude`."""
    H, W = image.shape[:2]
    w, h = size
    out = []
    for _ in range(tries):
        if len(out) >= n or w > W or h > H:
            break
        x, y = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
        if all(x + w <= ex or ex + ew <= x or y + h <= ey or ey + eh <= y for ex, ey, ew, eh in exclude):
            out.append(image[y:y + h, x:x + w])
    return out
