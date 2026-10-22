"""Training / validation datasets for the smp U-Net.

Train: `epoch_size` random crops per epoch at NATIVE resolution (default
512 x 512), mostly centred near a tuft (p_positive), drawn from real and
synthetic samples at a configurable ratio, augmented with src.transforms.
Every crop is reproducible from (seed, global index); EpochSampler hands each
epoch a fresh index range, which also works with persistent loader workers
(they never see a set_epoch() call on their copy of the dataset).

Val: fixed non-overlapping full-resolution tiles of real frames, no augmentation.

Samples use the existing on-disk format (images/<stem>.png, labels/<stem>.png
with channels tuft/root/root-valid, meta.json) plus an optional
ignore/<stem>.png: pixels there get zero tuft loss and are left out of
validation metrics (e.g. tufts a dataset never labelled).

Each sample yields (image, target, valid) where valid[0] is root-valid and
valid[1] is the tuft loss weight (0 inside ignore regions).
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .dataset import sample_crop_origin, tile_coords
from .labeling import load_sample


@dataclass(frozen=True)
class Item:
    root: str
    stem: str
    group: str            # "real" or "synthetic"
    domain: str = "own"   # where the frames come from: "own", "dlr", ... (meta.json "domain")
    weight: float = 1.0   # tuft-loss weight for this sample's label source (data.source_weights)
    source: str = ""      # label source from meta.json, e.g. dlr_polygon, sam2_box, sam2, unet


def _to_tensors(rgb, label, classes):
    import torch
    img = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
    lab = label > 127
    tgt = torch.from_numpy(np.ascontiguousarray(lab[..., :classes].transpose(2, 0, 1))).float()
    weight = ~lab[..., 3] if lab.shape[-1] > 3 else np.ones(lab.shape[:2], bool)
    valid = torch.from_numpy(np.ascontiguousarray(np.stack([lab[..., 2], weight]))).float()
    return img, tgt, valid


def crop_pad(img, label, y, x, size):
    """size x size crop at (y, x). Frames smaller than `size` are padded by
    reflection on BOTH image and label (reflecting only the image would teach
    "tuft = background"), and with a 4-channel label the padded pixels are also
    marked ignore, so the model never learns from mirrored content. The ignore
    channel is transformed with the image, so it stays exact after rotation."""
    c_img, c_lab = img[y:y + size, x:x + size], label[y:y + size, x:x + size]
    h, w = c_img.shape[:2]
    ph, pw = size - h, size - w
    if ph > 0 or pw > 0:
        mode = cv2.BORDER_REFLECT_101 if min(c_img.shape[:2]) > 1 else cv2.BORDER_REPLICATE
        c_img = cv2.copyMakeBorder(c_img, 0, ph, 0, pw, mode)
        c_lab = cv2.copyMakeBorder(c_lab, 0, ph, 0, pw, mode)
        if c_lab.ndim == 3 and c_lab.shape[2] >= 4:  # mirrored pixels: no loss, not scored
            c_lab[h:, :, 3] = 255
            c_lab[:, w:, 3] = 255
    return c_img, c_lab


def _load_rgb(item):
    """RGB image and a 4-channel label: tuft, root, root-valid, ignore."""
    from .label_io import load_ignore
    img, lab = load_sample(item.root, item.stem)
    ign = load_ignore(item.root, item.stem)
    ign = np.zeros(lab.shape[:2], np.uint8) if ign is None else ign
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.dstack([lab, ign])


class TrainCrops:
    """torch Dataset of augmented random crops (len = epoch_size)."""

    def __init__(self, items, size=512, epoch_size=2000, seed=0, p_positive=0.7,
                 synthetic_share=0.3, aug=None, classes=1, augment=True, source_shares=None):
        from .transforms import build_domain_transform, build_train_transform
        aug = dict(aug or {})
        self.domain_tf = {d: build_domain_transform(spec)
                          for d, spec in (aug.pop("domain", None) or {}).items()} if augment else {}
        self.recolor = (aug.pop("recolor", None) or {}) if augment else {}   # yarn / surface colour, label-aware
        self.shuffle = (aug.pop("shuffle", None) or {}) if augment else {}   # erase / paste tufts: no grid cue
        if not items:
            raise ValueError("no training samples")
        self.items, self.size, self.epoch_size = list(items), size, epoch_size
        self.seed, self.p_positive, self.classes = seed, p_positive, classes
        self.tf, self.window = build_train_transform(size, aug)
        if not augment:
            import albumentations as A
            self.tf, self.window = A.Compose([A.CenterCrop(size, size)]), size
        self.p = (self._source_probabilities(source_shares) if source_shares
                  else self._probabilities(synthetic_share))

    def _source_probabilities(self, shares):
        """Fraction of crops per label source, independent of how many frames each
        source has. Sources not listed share whatever fraction is left."""
        srcs = np.array([it.source for it in self.items])
        listed = {s: float(v) for s, v in shares.items() if (srcs == s).any()}
        rest = max(0.0, 1.0 - sum(listed.values()))
        n_rest = int((~np.isin(srcs, list(listed))).sum())
        p = np.array([listed[s] / (srcs == s).sum() if s in listed else (rest / n_rest if n_rest else 0.0)
                      for s in srcs])
        if p.sum() <= 0:
            raise ValueError("source_shares select no training samples")
        return p / p.sum()

    def _probabilities(self, share):
        groups = np.array([it.group for it in self.items])
        n_real, n_syn = (groups == "real").sum(), (groups == "synthetic").sum()
        if n_real == 0 or n_syn == 0:
            return np.full(len(self.items), 1.0 / len(self.items))
        share = float(np.clip(share, 0.0, 1.0))
        return np.where(groups == "real", (1 - share) / n_real, share / n_syn)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, i):
        """i is a global index: epoch e uses e*epoch_size ... (e+1)*epoch_size-1."""
        rng = np.random.default_rng([self.seed, int(i), 0x7AF7])
        item = self.items[int(rng.choice(len(self.items), p=self.p))]
        rgb, lab = _load_rgb(item)
        y, x = sample_crop_origin(lab, self.window, rng, self.p_positive)
        c_img, c_lab = crop_pad(rgb, lab, y, x, self.window)
        if self.shuffle and rng.random() < self.shuffle.get("p", 0.5):
            from .transforms import shuffle_tufts
            c_img, c_lab = shuffle_tufts(c_img, c_lab, rng, self.shuffle.get("drop_frac", 0.25),
                                         self.shuffle.get("paste_max", 6))
        self.tf.set_random_seed(int(rng.integers(2 ** 31)))
        out = self.tf(image=c_img, mask=c_lab)
        if self.recolor:
            from .transforms import recolor_tuft_background
            out["image"] = recolor_tuft_background(out["image"], out["mask"][..., 0] > 127, rng,
                                                   self.recolor.get("tuft_p", 0.5),
                                                   self.recolor.get("background_p", 0.5),
                                                   white_p=self.recolor.get("white_p", 0.0),
                                                   gray_bg_p=self.recolor.get("gray_bg_p", 0.0))
        dtf = self.domain_tf.get(item.domain)
        if dtf is not None:
            dtf.set_random_seed(int(rng.integers(2 ** 31)))
            out["image"] = dtf(image=out["image"])["image"]
        img, tgt, valid = _to_tensors(out["image"], out["mask"], self.classes)
        if item.weight != 1.0:
            valid[1] *= item.weight
        return img, tgt, valid


class EpochSampler:
    """Yields epoch * n ... (epoch + 1) * n - 1 so every epoch sees new crops."""

    def __init__(self, n):
        self.n, self.epoch = n, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(range(self.epoch * self.n, (self.epoch + 1) * self.n))

    def __len__(self):
        return self.n


class ValTiles:
    """Deterministic full-resolution tiles of each validation frame."""

    def __init__(self, items, size=512, classes=1):
        self.items, self.size, self.classes = list(items), size, classes
        self.tiles = []
        for k, it in enumerate(self.items):
            h, w = cv2.imread(f"{it.root}/images/{it.stem}.png", cv2.IMREAD_UNCHANGED).shape[:2]
            self.tiles += [(k, y, x) for y, x in tile_coords(h, w, size)]
        self._cache_k, self._cache = None, None

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        k, y, x = self.tiles[i]
        if self._cache_k != k:  # tiles of one frame are consecutive
            self._cache_k, self._cache = k, _load_rgb(self.items[k])
        rgb, lab = self._cache
        c_img, c_lab = crop_pad(rgb, lab, y, x, self.size)
        return _to_tensors(c_img, c_lab, self.classes)
