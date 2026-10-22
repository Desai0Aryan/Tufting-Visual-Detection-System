"""Albumentations training pipeline (image and mask always transformed together).

The label is an HxWx3 uint8 mask (tuft, root, root-valid), passed as one
`mask` so every geometric op moves all three channels with the image.
Masks use nearest-neighbour interpolation, so they stay binary.

Geometric ops run on a window larger than the output crop, followed by a
centre crop, so rotation and scale never expose padded borders.
Nothing here changes the output resolution: there is no downscaling.
"""
import cv2
import numpy as np
import albumentations as A

DEFAULT_AUG = {
    "flip_p": 0.5,
    "affine_p": 0.7, "rotate_deg": 15, "scale": [0.8, 1.25],
    "elastic_p": 0.15, "elastic_alpha": 20.0, "elastic_sigma": 8.0,
    "brightness_contrast_p": 0.7, "brightness": 0.25, "contrast": 0.25,
    "gamma_p": 0.4, "gamma": [70, 150],
    "hue_p": 0.2,
    "shadow_gradient_p": 0.4, "shadow_polygon_p": 0.15,
    "glare_p": 0.3,
    "blur_p": 0.3, "motion_blur_max": 9,
    "noise_p": 0.4, "noise_std": [0.01, 0.05],
    "compression_p": 0.5, "compression_quality": [30, 90],
}

# How much larger the pre-crop window must be so a crop rotated by `rotate_deg`
# and zoomed out by min(scale) still lies inside real pixels.
def window_factor(aug):
    # |cos| + |sin| peaks at 45 deg (sqrt 2), so any range reaching 45 deg needs the full factor
    a = np.radians(min(abs(aug.get("rotate_deg", 0)), 45.0))
    return float((abs(np.cos(a)) + abs(np.sin(a))) / min(1.0, aug.get("scale", [1, 1])[0])) + 0.02


class ShadowGradient(A.ImageOnlyTransform):
    """A soft-edged cast shadow: one side of a random line (or the inside of a
    random ellipse) is darkened, with a blurred transition. Unlike a global
    illumination ramp, the shadow edge is sharp enough to look like a tuft-ish
    edge to a threshold, which is exactly the failure we train against."""

    def __init__(self, strength=(0.35, 0.8), softness=(2, 40), p=0.5):
        super().__init__(p=p)
        self.strength, self.softness = strength, softness

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]
        r = self.random_generator
        m = np.zeros((h, w), np.float32)
        if r.random() < 0.6:
            ang = r.uniform(0, np.pi)
            c = (r.uniform(0, w), r.uniform(0, h))
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            m = ((xx - c[0]) * np.cos(ang) + (yy - c[1]) * np.sin(ang) > 0).astype(np.float32)
        else:
            cv2.ellipse(m, (int(r.uniform(0, w)), int(r.uniform(0, h))),
                        (int(r.uniform(0.1, 0.6) * w), int(r.uniform(0.1, 0.6) * h)),
                        r.uniform(0, 180), 0, 360, 1.0, -1)
        s = int(r.uniform(*self.softness)) | 1
        m = cv2.GaussianBlur(m, (s, s), 0) if s > 1 else m
        return {"gain": 1.0 - r.uniform(*self.strength) * m}

    def apply(self, img, gain=None, **params):
        return np.clip(img.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


class GlareBlobs(A.ImageOnlyTransform):
    """Additive specular glare: 1-3 bright Gaussian blobs, some stretched into
    streaks, that can saturate to white (the same colour as the tufts)."""

    def __init__(self, count=(1, 3), intensity=(80, 255), p=0.3):
        super().__init__(p=p)
        self.count, self.intensity = count, intensity

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]
        r = self.random_generator
        add = np.zeros((h, w), np.float32)
        for _ in range(int(r.integers(self.count[0], self.count[1] + 1))):
            blob = np.zeros((h, w), np.float32)
            axes = (max(2, int(r.uniform(0.01, 0.15) * w)), max(2, int(r.uniform(0.005, 0.06) * h)))
            cv2.ellipse(blob, (int(r.uniform(0, w)), int(r.uniform(0, h))), axes,
                        r.uniform(0, 180), 0, 360, 1.0, -1)
            k = int(max(axes) * r.uniform(0.3, 1.0)) * 2 + 1
            add += cv2.GaussianBlur(blob, (k, k), 0) * r.uniform(*self.intensity)
        return {"add": add}

    def apply(self, img, add=None, **params):
        return np.clip(img.astype(np.float32) + add[..., None], 0, 255).astype(np.uint8)


def shuffle_tufts(img, lab, rng, drop_frac=0.25, paste_max=6):
    """Breaks the grid: erases some tufts (surface inpainted, label removed) and pastes
    copies of others at random places and angles (label pasted too), so tuft rows and
    spacing stop being a cue. img: RGB uint8 window; lab: HxWx4 uint8 (tuft, root,
    root-valid, ignore). Returns new (img, lab)."""
    tuft = (lab[..., 0] > 127).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(tuft, connectivity=8)
    comps = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= 20]
    if not comps:
        return img, lab
    img, lab = img.copy(), lab.copy()
    ker = np.ones((7, 7), np.uint8)
    # 1. erase: paint the surface over a few tufts; that area is plain background now
    drop = [k for k in comps if rng.random() < drop_frac]
    if drop:
        hole = cv2.dilate(np.isin(cc, drop).astype(np.uint8), ker)
        hole &= (lab[..., 3] < 128).astype(np.uint8)          # never touch ignore areas
        img = cv2.inpaint(img, hole, 5, cv2.INPAINT_TELEA)
        lab[hole > 0, 0] = 0
        lab[hole > 0, 1] = 0
    keep = [k for k in comps if k not in drop]
    # 2. paste: copies of kept tufts at random places / angles, blended on the yarn only
    H, W = tuft.shape
    for _ in range(int(rng.integers(0, paste_max + 1)) if keep else 0):
        k = int(rng.choice(keep))
        x, y, w, h = (int(v) for v in stats[k, :4])
        pad = 6
        x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
        patch, plab = img[y0:y1, x0:x1], lab[y0:y1, x0:x1]
        m = cv2.dilate((cc[y0:y1, x0:x1] == k).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        ph, pw = m.shape
        side = int(np.ceil(np.hypot(ph, pw))) + 2
        R = cv2.getRotationMatrix2D((pw / 2, ph / 2), rng.uniform(0, 360), 1.0)
        R[:, 2] += (side - pw) / 2, (side - ph) / 2
        warp = lambda a, interp: cv2.warpAffine(a, R, (side, side), flags=interp, borderValue=0)
        rp, rm = warp(patch, cv2.INTER_LINEAR), warp(m, cv2.INTER_LINEAR)
        rl0, rl1 = warp(plab[..., 0], cv2.INTER_NEAREST), warp(plab[..., 1], cv2.INTER_NEAREST)
        if side >= min(H, W):
            continue
        body = cv2.dilate((rm > 0.2).astype(np.uint8), ker) > 0     # the pasted yarn plus a margin
        for _try in range(10):                                  # dense grids: look for a free spot
            tx, ty = int(rng.integers(0, W - side)), int(rng.integers(0, H - side))
            dst = (slice(ty, ty + side), slice(tx, tx + side))
            if not ((lab[dst][..., 0] > 127) & body).any() and not ((lab[dst][..., 3] > 127) & body).any():
                break                                           # not on top of another tuft / ignore area
        else:
            continue
        a = cv2.GaussianBlur(rm, (3, 3), 0)[..., None]
        img[dst] = (img[dst] * (1 - a) + rp * a).astype(np.uint8)
        lab[dst][..., 0] = np.maximum(lab[dst][..., 0], np.where(rm > 0.5, rl0, 0))
        lab[dst][..., 1] = np.maximum(lab[dst][..., 1], np.where(rm > 0.5, rl1, 0))
    return img, lab


YARN_COLORS = np.array([[240, 240, 240], [25, 25, 25], [210, 40, 40], [245, 130, 30], [240, 220, 40],
                        [40, 110, 200], [60, 170, 60], [245, 120, 180], [120, 120, 120]], np.float32)   # RGB


def recolor_tuft_background(img, tuft, rng, tuft_p=0.5, background_p=0.5, min_contrast=40.0,
                            white_p=0.0, gray_bg_p=0.0):
    """Colour variety that keeps the label right: the yarn gets a new colour (its own
    shading kept) and / or the surface gets a new hue, brightness and saturation,
    independently, so the model learns tuft shape, not one tuft / paint colour pair.

    img: RGB uint8 crop; tuft: bool label mask, which may be drawn thicker than the
    yarn: only label pixels that stand out from the local surface (the yarn itself)
    are recoloured, with soft edges, so no fat painted band appears.
    white_p / gray_bg_p bias toward a target look (white yarn on grey paint): of the
    recoloured surfaces, gray_bg_p become a grey (often darker) paint; of the recoloured
    yarns, white_p become white whenever that keeps the contrast."""
    out = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    surface = cv2.medianBlur(img, 15).astype(np.float32)
    s_gray = cv2.cvtColor(surface.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    zone = cv2.dilate(tuft.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    alpha = np.where(zone, np.clip((np.abs(gray - s_gray) - 6.0) / 30.0, 0.0, 1.0), 0.0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[..., None]
    if rng.random() < background_p:                       # new surface colour, yarn untouched
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        if rng.random() < gray_bg_p:                      # grey paint, light to dark
            hsv[..., 1] *= rng.uniform(0.0, 0.15)
            hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.3, 0.9), 0, 255)
        else:
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(0, 180)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.3, 2.0) + rng.uniform(0, 40), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.6, 1.3), 0, 255)
        bg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        out = bg * (1 - alpha) + out * alpha
        surface = cv2.medianBlur(out.astype(np.uint8), 15).astype(np.float32)
    if rng.random() < tuft_p and alpha.max() > 0:         # new yarn colour, contrast kept
        s_lum = float(cv2.cvtColor(surface.astype(np.uint8), cv2.COLOR_RGB2GRAY)[zone].mean()) if zone.any() else 128.0
        lum = YARN_COLORS @ np.array([0.299, 0.587, 0.114], np.float32)
        ok = np.flatnonzero(np.abs(lum - s_lum) >= min_contrast)
        if ok.size:
            pick = 0 if (0 in ok and rng.random() < white_p) else int(rng.choice(ok))   # YARN_COLORS[0] = white
            col = YARN_COLORS[pick] + rng.normal(0, 10, 3)
            m = alpha[..., 0] > 0.3
            shade = gray / max(float(gray[m].mean()) if m.any() else 128.0, 1.0)   # the yarn's own shading
            yarn = np.clip(col[None, None, :] * np.clip(shade, 0.6, 1.4)[..., None], 0, 255)
            out = out * (1 - alpha) + yarn * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def build_domain_transform(spec):
    """Extra colour augmentation for one data domain (e.g. DLR's red/dark tufts,
    while ours are white): hue shift, grayscale and colour inversion, so the
    model has to learn tuft shape rather than tuft colour."""
    return A.Compose([
        A.HueSaturationValue(hue_shift_limit=180, sat_shift_limit=40, val_shift_limit=0,
                             p=spec.get("hue_p", 0.5)),
        A.ToGray(p=spec.get("gray_p", 0.3)),
        A.InvertImg(p=spec.get("invert_p", 0.3)),
    ], seed=0)


def build_train_transform(size, aug=None):
    """Returns (transform, window): crop a `window`-px square, then call
    transform(image=rgb, mask=label) -> size x size outputs."""
    a = dict(DEFAULT_AUG, **(aug or {}))
    window = int(np.ceil(size * window_factor(a)))
    ops = [
        A.HorizontalFlip(p=a["flip_p"]),
        A.VerticalFlip(p=a["flip_p"]),
        A.Affine(rotate=(-a["rotate_deg"], a["rotate_deg"]), scale=tuple(a["scale"]),
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                 border_mode=cv2.BORDER_REFLECT_101, p=a["affine_p"]),
        A.ElasticTransform(alpha=a["elastic_alpha"], sigma=a["elastic_sigma"],
                           mask_interpolation=cv2.INTER_NEAREST,
                           border_mode=cv2.BORDER_REFLECT_101, p=a["elastic_p"]),
        A.CenterCrop(size, size),
        A.RandomBrightnessContrast(brightness_limit=a["brightness"], contrast_limit=a["contrast"],
                                   p=a["brightness_contrast_p"]),
        A.RandomGamma(gamma_limit=tuple(a["gamma"]), p=a["gamma_p"]),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=0, p=a["hue_p"]),
        ShadowGradient(p=a["shadow_gradient_p"]),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=a["shadow_polygon_p"]),
        GlareBlobs(p=a["glare_p"]),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)),
                 A.MotionBlur(blur_limit=(3, a["motion_blur_max"]))], p=a["blur_p"]),
        A.GaussNoise(std_range=tuple(a["noise_std"]), p=a["noise_p"]),
        A.OneOf([A.ImageCompression(compression_type="jpeg", quality_range=tuple(a["compression_quality"])),
                 A.ImageCompression(compression_type="webp", quality_range=tuple(a["compression_quality"]))],
                p=a["compression_p"]),
    ]
    return A.Compose(ops, seed=0), window
