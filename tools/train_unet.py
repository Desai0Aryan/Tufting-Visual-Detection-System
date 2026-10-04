import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.dataset import CropDataset  # noqa: E402
from src.labeling import list_samples, split_samples  # noqa: E402
from src.metrics import SegMeter  # noqa: E402
from src.unet import UNet, save_checkpoint  # noqa: E402

cv2.setNumThreads(0)  # augmentation runs inside loader workers; avoid oversubscription
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


# ----------------------------------------------------------------- losses
def bce(logits, target, weight=None, pos_weight=4.0):
    pw = torch.tensor(pos_weight, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    if weight is None:
        return loss.mean()
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def dice_loss(logits, target, weight=None, eps=1.0):
    p = torch.sigmoid(logits)
    if weight is not None:
        p, target = p * weight, target * weight
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def total_loss(logits, target, root_weight, root_scale=0.5, pos_weight=2.0):
    """Tuft mask: BCE + Dice everywhere. Root mask: same, but only where the
    label says the root decision is trustworthy (root_weight == 1).

    pos_weight was 4.0: strings are ~1-3% of pixels, so a heavy positive weight buys
    recall by making the net fire on anything string-like (stripe edges, texture,
    sky streaks). 2.0 keeps it sensitive without that bias."""
    tuft = (bce(logits[:, 0:1], target[:, 0:1], pos_weight=pos_weight)
            + dice_loss(logits[:, 0:1], target[:, 0:1]))
    if root_weight.sum() > 0:
        root = (bce(logits[:, 1:2], target[:, 1:2], root_weight)
                + dice_loss(logits[:, 1:2], target[:, 1:2], root_weight))
    else:
        root = logits.sum() * 0.0
    return tuft + root_scale * root, tuft.detach(), root.detach()


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device, beta=0.5):
    model.eval()
    tuft = {t: SegMeter() for t in THRESHOLDS}
    root = SegMeter()
    for img, tgt, wgt in loader:
        img = img.to(device)
        prob = torch.sigmoid(model(img)).cpu().numpy()
        tgt, wgt = tgt.numpy(), wgt.numpy()
        for t in THRESHOLDS:
            tuft[t].update(prob[:, 0] >= t, tgt[:, 0] > 0.5)
        root.update(prob[:, 1] >= 0.5, tgt[:, 1] > 0.5, weight=wgt[:, 0] > 0.5)
    return {
        "tuft_dice": tuft[0.5].dice, "tuft_iou": tuft[0.5].iou,
        "tuft_precision": tuft[0.5].precision, "tuft_recall": tuft[0.5].recall,
        "root_dice": root.dice,
        # threshold chosen on F-beta (beta<1 favours precision) instead of plain Dice, so the
        # suggested --threshold trades a little recall for far fewer false detections
        "best_threshold": max(THRESHOLDS, key=lambda t: tuft[t].fbeta(beta)),
        "dice_by_threshold": {str(t): tuft[t].dice for t in THRESHOLDS},
        "precision_by_threshold": {str(t): tuft[t].precision for t in THRESHOLDS},
        "recall_by_threshold": {str(t): tuft[t].recall for t in THRESHOLDS},
    }


def make_loader(ds, batch, shuffle, workers):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      drop_last=shuffle, pin_memory=torch.cuda.is_available())


def _warn_unnormalised(roots):
    """Datasets labelled before scale normalisation hold tufts at native photo scale, which
    will not match what main.py now feeds the network."""
    import json as _json
    for r in roots:
        mp = Path(r) / "meta.json"
        if not mp.exists():
            continue
        try:
            meta = _json.loads(mp.read_text())
        except Exception:
            continue
        missing = [k for k, v in meta.items() if "scale" not in v]
        if missing:
            print(f"[warn] {r}: {len(missing)}/{len(meta)} samples have no recorded scale "
                  f"(labelled before --norm-size existed). Re-run tools.autolabel on those photos "
                  f"or tufts in big photos will be 2-3x larger than at inference.")


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[], help="real labelled dataset folder(s)")
    ap.add_argument("--synthetic", nargs="*", default=[], help="synthetic dataset folder(s)")
    ap.add_argument("--synthetic-share", type=float, default=0.2,
                    help="fraction of training crops drawn from synthetic data (if real data exists)")
    ap.add_argument("--out", default="models/tuft_unet.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epoch-size", type=int, default=400, help="random crops per epoch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256, help="training crop size (px)")
    ap.add_argument("--base", type=int, default=16, help="U-Net base channels")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-weight", type=float, default=2.0,
                    help="BCE weight on tuft pixels (was 4.0; high values cause false detections)")
    ap.add_argument("--p-positive", type=float, default=0.6,
                    help="fraction of training crops centred on a tuft (was 0.8). Lower = more "
                         "tuft-free background crops, i.e. more hard negatives (sky, texture, edges)")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="F-beta used to pick the suggested threshold (<1 favours precision)")
    ap.add_argument("--val-percent", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    if not a.data and not a.synthetic:
        ap.error("give at least one of --data / --synthetic")
    if a.size % 16:
        ap.error("--size must be divisible by 16 (U-Net depth 4)")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    mps = getattr(torch.backends, "mps", None)
    device = a.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if mps is not None and mps.is_available() else "cpu")
    real, syn = list(a.data), list(a.synthetic)

    roots = real + syn
    _warn_unnormalised(real)
    train_ds = CropDataset(roots, "train", a.size, a.epoch_size, a.val_percent, a.seed,
                           p_positive=a.p_positive)
    if real and syn:
        share = float(np.clip(a.synthetic_share, 0.0, 0.9))
        shares = {str(r): (1 - share) / len(real) for r in real}
        shares.update({str(r): share / len(syn) for r in syn})
        train_ds.set_root_shares(shares)

    val_sets = {}
    for name, group in (("real", real), ("synthetic", syn)):
        if group:
            try:
                val_sets[name] = CropDataset(group, "val", a.size, 0, a.val_percent, a.seed)
            except ValueError:
                print(f"[warn] no validation photos for the {name} data (too few photos?)")
    select_on = "real" if "real" in val_sets else ("synthetic" if "synthetic" in val_sets else None)
    if select_on is None:
        print("[warn] no validation set: the 'best' checkpoint is chosen by training loss")

    model = UNet(in_ch=3, out_ch=2, base=a.base, depth=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.02)
    print(f"device={device}  params={n_params / 1e6:.2f}M  train photos={len(train_ds.items)}  "
          + "  ".join(f"val[{k}] tiles={len(v)}" for k, v in val_sets.items()))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {"in_ch": 3, "out_ch": 2, "base": a.base, "depth": 4}
    log_path = Path(str(out) + ".log.csv")
    best_score, best_epoch, best_metrics = -1.0, -1, None
    best_loss = float("inf")
    val_loaders = {k: make_loader(v, a.batch, False, a.workers) for k, v in val_sets.items()}

    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "tuft_loss", "root_loss", "lr"]
                        + [f"{k}_{m}" for k in val_sets for m in ("tuft_dice", "tuft_iou", "root_dice")])
        for epoch in range(1, a.epochs + 1):
            train_ds.set_epoch(epoch)
            loader = make_loader(train_ds, a.batch, True, a.workers)
            model.train()
            t0, sums, n = time.time(), np.zeros(3), 0
            for img, tgt, wgt in loader:
                img, tgt, wgt = img.to(device), tgt.to(device), wgt.to(device)
                loss, lt, lr_ = total_loss(model(img), tgt, wgt, pos_weight=a.pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sums += [loss.item(), lt.item(), lr_.item()]
                n += 1
            sched.step()
            avg = sums / max(n, 1)

            metrics = {k: evaluate(model, ld, device, a.beta) for k, ld in val_loaders.items()}
            row = [epoch, f"{avg[0]:.4f}", f"{avg[1]:.4f}", f"{avg[2]:.4f}", f"{sched.get_last_lr()[0]:.2e}"]
            for k in val_sets:
                row += [f"{metrics[k]['tuft_dice']:.4f}", f"{metrics[k]['tuft_iou']:.4f}",
                        f"{metrics[k]['root_dice']:.4f}"]
            writer.writerow(row)
            fh.flush()

            if select_on:
                score, improved = metrics[select_on]["tuft_dice"], metrics[select_on]["tuft_dice"] > best_score
            else:
                score, improved = -avg[0], avg[0] < best_loss
                best_loss = min(best_loss, avg[0])
            if improved:
                best_score, best_epoch, best_metrics = score, epoch, metrics
                save_checkpoint(model, str(out), config)
            val_txt = "  ".join(f"{k}: dice {m['tuft_dice']:.3f} P {m['tuft_precision']:.3f} "
                                f"R {m['tuft_recall']:.3f} root {m['root_dice']:.3f}"
                                for k, m in metrics.items())
            print(f"epoch {epoch:3d}/{a.epochs}  loss {avg[0]:.3f}  {val_txt}  "
                  f"({time.time() - t0:.0f}s){'  *best*' if improved else ''}")

    save_checkpoint(model, str(out) + ".last.pt", config)
    Path(str(out) + ".json").write_text(json.dumps({
        "args": vars(a), "config": config, "best_epoch": best_epoch,
        "selected_on": select_on, "best_metrics": best_metrics,
        "split": {str(r): split_samples(list_samples(r), a.val_percent) for r in roots},
    }, indent=1))
    print(f"\nbest checkpoint: {out}  (epoch {best_epoch})")
    if best_metrics and select_on:
        print(f"suggested --threshold for inference: {best_metrics[select_on]['best_threshold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
