import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .backends import MaskResult


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
    def __init__(self, in_ch=3, out_ch=2, base=16, depth=4):
        super().__init__()
        self.depth = depth
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


def save_checkpoint(model, path, config):
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_model(weights, device="cpu"):
    ckpt = torch.load(weights, map_location="cpu")
    model = UNet(**ckpt.get("config", {}))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


class UNetBackend:
    name = "unet"

    def __init__(self, weights, device=None, threshold=0.5, tile=768, overlap=64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(weights, self.device)
        self.threshold = threshold
        self.tile = tile
        self.overlap = overlap

    @torch.no_grad()
    def _run(self, rgb):
        """rgb: HxWx3 float32 in [0,1] -> (C, H, W) probabilities."""
        h, w = rgb.shape[:2]
        m = 2 ** self.model.depth
        ph, pw = (-h) % m, (-w) % m
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        probs = torch.sigmoid(self.model(x))[0, :, :h, :w]
        return probs.cpu().numpy()

    def _probs(self, rgb):
        h, w = rgb.shape[:2]
        if max(h, w) <= self.tile:
            return self._run(rgb)
        stride = self.tile - self.overlap
        acc, cnt = None, np.zeros((h, w), np.float32)
        for y0 in range(0, max(h - self.overlap, 1), stride):
            for x0 in range(0, max(w - self.overlap, 1), stride):
                y1, x1 = min(y0 + self.tile, h), min(x0 + self.tile, w)
                p = self._run(rgb[y0:y1, x0:x1])
                if acc is None:
                    acc = np.zeros((p.shape[0], h, w), np.float32)
                acc[:, y0:y1, x0:x1] += p
                cnt[y0:y1, x0:x1] += 1
        return acc / np.maximum(cnt, 1)

    def predict(self, image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        probs = self._probs(rgb)
        root = probs[1] if probs.shape[0] > 1 else None
        return MaskResult(tuft=(probs[0] >= self.threshold).astype(np.uint8) * 255,
                          root=root)
