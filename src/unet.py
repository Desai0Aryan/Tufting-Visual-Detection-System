import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .backends import MaskResult

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Legacy from-scratch U-Net (kept so older checkpoints still load)."""

    def __init__(self, in_ch=3, out_ch=2, base=16, depth=4):
        super().__init__()
        self.depth = depth
        self.pad_multiple = 2 ** depth
        chans = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chans[:-1]:
            self.enc.append(_Block(c, ch))
            c = ch
        self.mid = _Block(chans[-2], chans[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2))
            self.dec.append(_Block(chans[i] * 2, chans[i]))
        self.head = nn.Conv2d(chans[0], out_ch, 1)

    def forward(self, x):  # H, W must be divisible by 2**depth
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([skip, up(x)], dim=1))
        return self.head(x)  # logits


class TuftSegNet(nn.Module):
    """segmentation_models_pytorch U-Net with a pretrained encoder.

    Input: RGB float in [0, 1], (N, 3, H, W) with H, W multiples of 32.
    Output: raw logits (N, classes, H, W); channel 0 = tuft, channel 1 = root
    (only when root_head is on). The encoder's input normalisation lives
    inside forward(), so training and inference cannot disagree about it.
    """

    pad_multiple = 32

    def __init__(self, encoder="resnet34", encoder_weights="imagenet", classes=1):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                            in_channels=3, classes=classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build_model(config, pretrained=True):
    """config: dict from the YAML `model` section or a checkpoint.
    arch "smp_unet" (default) or "legacy_unet" (no 'arch' key = legacy)."""
    arch = config.get("arch", "legacy_unet")
    if arch == "smp_unet":
        return TuftSegNet(encoder=config.get("encoder", "resnet34"),
                          encoder_weights=config.get("encoder_weights", "imagenet") if pretrained else None,
                          classes=config.get("classes", 1))
    if arch == "legacy_unet":
        kw = {k: v for k, v in config.items() if k in ("in_ch", "out_ch", "base", "depth")}
        return UNet(**kw)
    raise ValueError(f"unknown model arch {arch!r}")


def save_checkpoint(model, path, config, extra=None):
    ckpt = {"state_dict": model.state_dict(), "config": config}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_model(weights, device="cpu"):
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    model = build_model(ckpt.get("config", {}), pretrained=False)  # weights come from the file
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def suggested_threshold(weights, default=0.5):
    """The threshold the checkpoint's own validation picked (written by the trainer
    to <weights>.json). Falling back to 0.5 silently was a common source of
    over/under-detection because the best value is dataset dependent."""
    meta = Path(str(weights) + ".json")
    try:
        data = json.loads(meta.read_text())
        chosen = data.get("selected_on")
        return float(data["best_metrics"][chosen]["best_threshold"])
    except Exception:
        return default


def _ramp(n, overlap):
    """1-D blending weight: ramps up over `overlap` px at both ends."""
    w = np.ones(n, np.float32)
    if overlap > 0:
        r = np.linspace(1.0 / (overlap + 1), 1.0, overlap, dtype=np.float32)
        k = min(overlap, n // 2)
        w[:k], w[n - k:] = r[:k], r[::-1][:k]
    return w


class UNetBackend:
    """Full-frame (padded to the model's multiple) or overlapping-tile inference.

    tile=None runs the whole frame in one pass (fits 1080p on a 5090);
    tiles are blended with linear ramps over the overlap so seams vanish.
    Never resizes the image.
    """

    name = "unet"

    def __init__(self, weights, device=None, threshold=None, tile=None, overlap=64,
                 batch_tiles=8, amp=True):
        if device is None:
            mps = getattr(torch.backends, "mps", None)
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if mps is not None and mps.is_available()
                      else "cpu")
        self.device = device
        self.model = load_model(weights, self.device)
        self.threshold = suggested_threshold(weights) if threshold is None else threshold
        self.tile = tile
        self.overlap = overlap
        self.batch_tiles = batch_tiles
        self.amp = amp and str(device).startswith("cuda")
        m = getattr(self.model, "pad_multiple", 32)
        if tile is not None and tile % m:
            self.tile = tile + (-tile) % m  # tiles must also be valid model inputs

    @torch.no_grad()
    def _forward(self, x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            return torch.sigmoid(self.model(x).float())

    def _run(self, x):
        """x: (1, 3, H, W) float tensor on device -> (C, H, W) probabilities."""
        h, w = x.shape[-2:]
        m = getattr(self.model, "pad_multiple", 32)
        ph, pw = (-h) % m, (-w) % m
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return self._forward(x)[0, :, :h, :w]

    def _tiled(self, x):
        _, _, h, w = x.shape
        t, ov = self.tile, self.overlap
        stride = max(t - ov, 1)
        ys = list(range(0, max(h - t, 0) + 1, stride))
        xs = list(range(0, max(w - t, 0) + 1, stride))
        if ys[-1] + t < h:
            ys.append(h - t)
        if xs[-1] + t < w:
            xs.append(w - t)
        th, tw = min(t, h), min(t, w)
        wy, wx = _ramp(th, ov), _ramp(tw, ov)
        wt = torch.from_numpy(np.outer(wy, wx)).to(x.device)
        acc = cnt = None
        coords = [(y0, x0) for y0 in ys for x0 in xs]
        for i in range(0, len(coords), self.batch_tiles):
            chunk = coords[i:i + self.batch_tiles]
            batch = torch.cat([x[..., y0:y0 + th, x0:x0 + tw] for y0, x0 in chunk])
            m = getattr(self.model, "pad_multiple", 32)
            batch = F.pad(batch, (0, (-tw) % m, 0, (-th) % m), mode="replicate")
            probs = self._forward(batch)[..., :th, :tw]
            if acc is None:
                acc = torch.zeros((probs.shape[1], h, w), device=x.device)
                cnt = torch.zeros((h, w), device=x.device)
            for (y0, x0), p in zip(chunk, probs):
                acc[:, y0:y0 + th, x0:x0 + tw] += p * wt
                cnt[y0:y0 + th, x0:x0 + tw] += wt
        return acc / cnt.clamp(min=1e-6)

    def predict_proba(self, image):
        """BGR/gray uint8 image -> (C, H, W) float32 probabilities (numpy)."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        h, w = rgb.shape[:2]
        if self.tile is None or max(h, w) <= self.tile:
            probs = self._run(x)
        else:
            probs = self._tiled(x)
        return probs.cpu().numpy()

    def predict(self, image):
        probs = self.predict_proba(image)
        root = probs[1] if probs.shape[0] > 1 else None
        return MaskResult(tuft=(probs[0] >= self.threshold).astype(np.uint8) * 255, root=root)
