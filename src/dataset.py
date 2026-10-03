import numpy as np

from .augment import augment
from .labeling import list_samples, load_sample, split_samples


def tile_coords(h, w, size):
    """Non-overlapping tile origins (y, x) covering the image; edge tiles are
    shifted back so every tile is full-size (images smaller than `size` are
    padded by `crop_or_pad`)."""
    ys = list(range(0, max(h - size, 0) + 1, size))
    xs = list(range(0, max(w - size, 0) + 1, size))
    if h > size and ys[-1] + size < h:
        ys.append(h - size)
    if w > size and xs[-1] + size < w:
        xs.append(w - size)
    return [(y, x) for y in ys for x in xs]


def crop_or_pad(img, label, y, x, size):
    """Crop (size x size) at (y, x); pad with reflection if the image is smaller.
    Label padding is 0 for every channel, i.e. also root-valid = 0 (ignored)."""
    h, w = img.shape[:2]
    c_img = img[y:y + size, x:x + size]
    c_lab = label[y:y + size, x:x + size]
    ph, pw = size - c_img.shape[0], size - c_img.shape[1]
    if ph > 0 or pw > 0:
        c_img = np.pad(c_img, ((0, ph), (0, pw), (0, 0)), mode="reflect" if min(h, w) > 1 else "edge")
        c_lab = np.pad(c_lab, ((0, ph), (0, pw), (0, 0)), mode="constant")
    return c_img, c_lab


def sample_crop_origin(label, size, rng, p_positive=0.8, tries=12):
    """Pick a crop origin; with prob p_positive centre it on a random tuft pixel."""
    h, w = label.shape[:2]
    if h <= size and w <= size:
        return 0, 0
    if rng.random() < p_positive:
        ys, xs = np.nonzero(label[..., 0] > 127)
        if len(ys):
            k = int(rng.integers(len(ys)))
            # jitter so the tuft is not always dead-centre
            cy = int(ys[k] + rng.integers(-size // 4, size // 4 + 1))
            cx = int(xs[k] + rng.integers(-size // 4, size // 4 + 1))
            return (int(np.clip(cy - size // 2, 0, max(h - size, 0))),
                    int(np.clip(cx - size // 2, 0, max(w - size, 0))))
    return int(rng.integers(0, max(h - size, 0) + 1)), int(rng.integers(0, max(w - size, 0) + 1))


def make_training_crop(img, label, size, rng, p_positive=0.8, aug=True):
    # draw a larger window first so rotation/scale don't expose reflected borders
    big = int(size * 1.5)
    y, x = sample_crop_origin(label, big, rng, p_positive)
    c_img, c_lab = crop_or_pad(img, label, y, x, big)
    if aug:
        c_img, c_lab = augment(c_img, c_lab, rng)
    o = (big - size) // 2
    return c_img[o:o + size, o:o + size].copy(), c_lab[o:o + size, o:o + size].copy()


def to_arrays(img, label):
    """uint8 BGR image + uint8 label -> float32 CHW tensors-as-numpy (RGB, 0..1;
    targets (2,H,W) in {0,1}; root weight (1,H,W) in {0,1})."""
    rgb = img[..., ::-1].astype(np.float32).transpose(2, 0, 1) / 255.0
    tgt = (label[..., :2] > 127).astype(np.float32).transpose(2, 0, 1)
    wgt = (label[..., 2:3] > 127).astype(np.float32).transpose(2, 0, 1)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(tgt), np.ascontiguousarray(wgt)


class CropDataset:
    """torch.utils.data.Dataset-compatible (duck-typed; torch only needed by the loader).

    train: `epoch_size` random augmented crops per epoch, each from a randomly
           picked photo (optionally weighted); deterministic per (seed, index).
    val:   fixed tiling of each photo, no augmentation.
    """

    def __init__(self, roots, split="train", size=256, epoch_size=400, val_percent=20,
                 seed=0, p_positive=0.8, weights=None):
        if isinstance(roots, (str, bytes)) or not hasattr(roots, "__iter__"):
            roots = [roots]
        self.roots = [str(r) for r in roots]
        self.split, self.size, self.epoch_size = split, size, epoch_size
        self.seed, self.p_positive = seed, p_positive
        self.epoch = 0
        self.items = []  # (root, stem)
        for r in self.roots:
            stems = split_samples(list_samples(r), val_percent)[split]
            self.items += [(r, s) for s in stems]
        if not self.items:
            raise ValueError(f"no {split} samples found in {self.roots}")
        self._cache = {}
        w = np.ones(len(self.items)) if weights is None else np.asarray(
            [weights.get(r, 1.0) for r, _ in self.items], float)
        self.p = w / w.sum()
        if split == "val":
            self.tiles = []
            for k, (r, s) in enumerate(self.items):
                img, _ = self._get(k)
                self.tiles += [(k, y, x) for y, x in tile_coords(*img.shape[:2], size)]

    def set_epoch(self, epoch):
        """Different crops each epoch (still reproducible for a given seed)."""
        self.epoch = int(epoch)

    def set_root_shares(self, shares):
        """Sampling probability mass per dataset root, e.g. {real: 0.8, synthetic: 0.2},
        independent of how many photos each root holds. Roots not listed get 0."""
        counts = {}
        for r, _ in self.items:
            counts[r] = counts.get(r, 0) + 1
        w = np.array([shares.get(r, 0.0) / counts[r] for r, _ in self.items], float)
        if w.sum() <= 0:
            raise ValueError("root shares select no samples")
        self.p = w / w.sum()

    def _get(self, k):
        if k not in self._cache:
            self._cache[k] = load_sample(*self.items[k])
        return self._cache[k]

    def __len__(self):
        return self.epoch_size if self.split == "train" else len(self.tiles)

    def __getitem__(self, i):
        if self.split == "train":
            rng = np.random.default_rng([self.seed, self.epoch, i, 0xC0FFEE])
            k = int(rng.choice(len(self.items), p=self.p))
            img, lab = self._get(k)
            c_img, c_lab = make_training_crop(img, lab, self.size, rng, self.p_positive, aug=True)
        else:
            k, y, x = self.tiles[i]
            img, lab = self._get(k)
            c_img, c_lab = crop_or_pad(img, lab, y, x, self.size)
        return to_arrays(c_img, c_lab)
