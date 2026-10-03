import cv2
import numpy as np


# ------------------------------------------------------------------ geometric
def random_geometric(img, label, rng, rotate=True, scale=(0.75, 1.3), flip=True):
    h, w = img.shape[:2]
    if flip:
        if rng.random() < 0.5:
            img, label = img[:, ::-1], label[:, ::-1]
        if rng.random() < 0.5:
            img, label = img[::-1], label[::-1]
    img, label = np.ascontiguousarray(img), np.ascontiguousarray(label)
    angle = rng.uniform(-180, 180) if rotate else 0.0
    s = float(np.exp(rng.uniform(np.log(scale[0]), np.log(scale[1]))))
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, s)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    label = cv2.warpAffine(label, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    if label.ndim == 2:  # warpAffine drops a singleton channel axis
        label = label[..., None]
    return img, label


# --------------------------------------------------------------- photometric
def _hue_shift(img, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(rng.integers(0, 180))) % 180
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * rng.uniform(0.6, 1.5), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _brightness_contrast(img, rng):
    alpha = rng.uniform(0.6, 1.5)    # contrast
    beta = rng.uniform(-35, 35)      # brightness
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _gamma(img, rng):
    g = float(np.exp(rng.uniform(np.log(0.6), np.log(1.7))))
    lut = (255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8)
    return cv2.LUT(img, lut)


def _illumination_gradient(img, rng):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5))
    gain = 1.0 + rng.uniform(0.1, 0.45) * ramp * rng.choice([-1, 1])
    return np.clip(img.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


def _blur(img, rng):
    k = int(rng.choice([3, 5]))
    return cv2.GaussianBlur(img, (k, k), 0)


def _motion_blur(img, rng):
    k = int(rng.choice([3, 5, 7]))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def _noise(img, rng):
    sigma = rng.uniform(2, 10)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _jpeg(img, rng):
    q = int(rng.integers(25, 85))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def _channel_permute(img, rng):
    return np.ascontiguousarray(img[..., rng.permutation(3)])


def _grayscale(img, rng):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _invert(img, rng):
    return 255 - img


# (function, probability). Order matters a little: colour first, sensor effects last.
PHOTOMETRIC = [
    (_hue_shift, 0.5),
    (_channel_permute, 0.2),
    (_grayscale, 0.15),
    (_invert, 0.15),
    (_brightness_contrast, 0.7),
    (_gamma, 0.4),
    (_illumination_gradient, 0.4),
    (_blur, 0.25),
    (_motion_blur, 0.15),
    (_noise, 0.4),
    (_jpeg, 0.4),
]


def random_photometric(img, rng, ops=PHOTOMETRIC):
    for fn, p in ops:
        if rng.random() < p:
            img = fn(img, rng)
    return img


def augment(img, label, rng=None, geometric=True, photometric=True):
    """Apply a random augmentation to an (image, label) pair. Returns new arrays."""
    rng = rng if rng is not None else np.random.default_rng()
    if geometric:
        img, label = random_geometric(img, label, rng)
    if photometric:
        img = random_photometric(img, rng)
    return img, label
