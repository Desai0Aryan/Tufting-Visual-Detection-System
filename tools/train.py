"""Train the smp U-Net tuft segmenter from a YAML config.

    python -m tools.train --config configs/unet_resnet34.yaml [--set train.epochs=80] [--resume runs/.../checkpoints/last.pt]

Writes runs/<name>_<timestamp>/:
    config.yaml       resolved config
    split.json        which clips / samples went to train and val (test: counts only)
    log.csv, steps.csv, tb/   per-epoch metrics, per-step loss, TensorBoard
    samples/epoch_NNN.png     image | ground truth | prediction overlay
    checkpoints/best.pt (+ .json with the chosen threshold), last.pt (resumable)

The frozen test split is never loaded here.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import dump_config, load_config  # noqa: E402
from src.labeling import list_samples  # noqa: E402
from src.losses import SegLoss  # noqa: E402
from src.metrics import ClDiceMeter, SegMeter  # noqa: E402
from src.seg_dataset import EpochSampler, Item, TrainCrops, ValTiles  # noqa: E402
from src.splits import load_split_map, split_real  # noqa: E402
from src.unet import build_model, save_checkpoint  # noqa: E402

cv2.setNumThreads(0)  # loader workers already run in parallel


# ----------------------------------------------------------------- data
def assemble_splits(cfg):
    """Returns (train_items, val_items, manifest). Test items are only counted."""
    d = cfg["data"]
    train, val = [], []
    manifest = {"real": {}, "synthetic": {}, "test_samples": 0, "clips": {}}
    if d.get("real"):
        split_map = load_split_map(d["sources_csv"])
        for root in d["real"]:
            if not Path(root, "images").is_dir():
                raise FileNotFoundError(f"real dataset {root} not found (images/ missing)")
            parts = split_real(root, list_samples(root), split_map)
            train += [Item(str(root), s, "real") for s in parts["train"]]
            val += [Item(str(root), s, "real") for s in parts["val"]]
            manifest["test_samples"] += len(parts["test"])
            manifest["real"][str(root)] = {"train": parts["train"], "val": parts["val"],
                                           "test_count": len(parts["test"])}
        for split in ("train", "val", "test"):
            manifest["clips"][split] = sorted(c for c, s in split_map.items() if s == split)
    for root in d.get("synthetic") or []:
        if not Path(root, "images").is_dir():
            raise FileNotFoundError(f"synthetic dataset {root} not found (images/ missing)")
        stems = list_samples(root)
        held = []
        if d.get("allow_synthetic_val") and not val:
            k = max(1, int(round(len(stems) * d.get("synthetic_val_fraction", 0.2))))
            held = stems[-k:]  # deterministic; synthetic scenes are independent of each other
        train += [Item(str(root), s, "synthetic") for s in stems if s not in held]
        val += [Item(str(root), s, "synthetic") for s in held]
        manifest["synthetic"][str(root)] = {"train": len(stems) - len(held), "val_heldout": held}
    tr_keys = {(i.root, i.stem) for i in train}
    assert not tr_keys & {(i.root, i.stem) for i in val}, "train/val overlap"
    if val and all(i.group == "synthetic" for i in val):
        print("[warn] validating on SYNTHETIC frames (allow_synthetic_val): smoke tests only")
    return train, val, manifest


# ----------------------------------------------------------------- helpers
def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def lr_lambda(total_steps, warmup_steps, min_frac):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


def param_groups(model, lr, enc_mult, wd):
    enc = [p for n, p in model.named_parameters() if ".encoder." in f".{n}"]
    ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": enc, "lr": lr * enc_mult, "weight_decay": wd},
            {"params": rest, "lr": lr, "weight_decay": wd}]


def autocast_ctx(device, amp):
    if amp == "off" or not str(device).startswith("cuda"):
        return torch.autocast("cpu", enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def overlay_sheet(images, gts, preds):
    """Rows of image | ground truth | prediction overlay (green = predicted,
    red = missed ground truth). Inputs: lists of HxWx3 RGB uint8 / HxW bool."""
    rows = []
    for img, gt, pr in zip(images, gts, preds):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt_vis = np.dstack([gt.astype(np.uint8) * 255] * 3)
        ov = bgr.copy()
        ov[pr] = (0.35 * ov[pr] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
        miss = gt & ~pr
        ov[miss] = (0.35 * ov[miss] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        sep = np.full((bgr.shape[0], 4, 3), 255, np.uint8)
        rows.append(np.hstack([bgr, sep, gt_vis, sep, ov]))
    return np.vstack([np.vstack([r, np.full((4, r.shape[1], 3), 255, np.uint8)]) for r in rows])


# ----------------------------------------------------------------- validation
@torch.no_grad()
def validate(model, loader, device, cfg, loss_fn):
    model.eval()
    ths = cfg["train"]["thresholds"]
    counts = {t: torch.zeros(3, dtype=torch.float64, device=device) for t in ths}  # tp fp fn
    cl = ClDiceMeter()
    loss_sum, n = 0.0, 0
    for img, tgt, valid in loader:
        img, tgt, valid = img.to(device), tgt.to(device), valid.to(device)
        with autocast_ctx(device, cfg["train"]["amp"]):
            logits = model(img)
        logits = logits.float()
        loss_sum += float(loss_fn(logits, tgt, valid)[0]) * img.shape[0]
        n += img.shape[0]
        prob, gt = torch.sigmoid(logits[:, 0]), tgt[:, 0] > 0.5
        for t in ths:
            pr = prob >= t
            counts[t] += torch.stack([(pr & gt).sum(), (pr & ~gt).sum(), (~pr & gt).sum()]).double()
        cl.update((prob >= 0.5).cpu().numpy(), gt.cpu().numpy())
    meters = {}
    for t, c in counts.items():
        m = SegMeter()
        m.tp, m.fp, m.fn = (int(v) for v in c.tolist())
        meters[t] = m
    beta = cfg["train"]["threshold_beta"]
    best_t = max(ths, key=lambda t: meters[t].fbeta(beta))
    b = meters[best_t]
    return {"val_loss": loss_sum / max(n, 1), "val_dice": b.dice, "val_iou": b.iou,
            "val_precision": b.precision, "val_recall": b.recall,
            "val_dice@0.5": meters[0.5].dice if 0.5 in meters else float("nan"),
            "val_cldice@0.5": cl.cldice, "best_threshold": best_t}


@torch.no_grad()
def write_samples(model, ds, idx, device, cfg, path, threshold):
    model.eval()
    imgs, gts, preds = [], [], []
    for i in idx:
        img, tgt, _ = ds[i]
        with autocast_ctx(device, cfg["train"]["amp"]):
            logit = model(img[None].to(device))
        prob = torch.sigmoid(logit.float())[0, 0].cpu().numpy()
        imgs.append((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        gts.append(tgt[0].numpy() > 0.5)
        preds.append(prob >= threshold)
    cv2.imwrite(str(path), overlay_sheet(imgs, gts, preds))


def pick_sample_tiles(ds, k):
    """The k validation tiles with the most tuft pixels (fixed for the run)."""
    scores = []
    for i in range(len(ds)):
        _, tgt, _ = ds[i]
        scores.append(float(tgt[0].sum()))
    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[:k]]


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", default=None, help="checkpoints/last.pt to continue from")
    ap.add_argument("--device", default=None, help="default: cuda (required unless set)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    device = a.device or "cuda"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run `python -m tools.check_env` "
                         "(or pass --device cpu for a tiny test run)")
    seed_everything(cfg["run"]["seed"], cfg["run"]["deterministic"])
    classes = 2 if cfg["model"]["root_head"] else 1

    train_items, val_items, manifest = assemble_splits(cfg)
    d, tcfg = cfg["data"], cfg["train"]
    train_ds = TrainCrops(train_items, d["crop"], d["epoch_size"], cfg["run"]["seed"], d["p_positive"],
                          d["synthetic_share"], cfg.get("aug"), classes)
    sampler = EpochSampler(len(train_ds))
    workers = int(d["workers"])
    train_loader = DataLoader(train_ds, batch_size=d["batch_size"], sampler=sampler,
                              num_workers=workers, pin_memory=device.startswith("cuda"),
                              drop_last=True, persistent_workers=workers > 0)
    val_ds = ValTiles(val_items, d["val_tile"], classes) if val_items else None
    val_loader = (DataLoader(val_ds, batch_size=d["batch_size"], shuffle=False, num_workers=0)
                  if val_ds else None)

    run_dir = Path(cfg["run"]["out_dir"]) / f"{cfg['run']['name']}_{datetime.now():%Y%m%d_%H%M%S}"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "samples").mkdir()
    dump_config(cfg, run_dir / "config.yaml")
    (run_dir / "split.json").write_text(json.dumps(manifest, indent=1))
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(str(run_dir / "tb"))

    model_cfg = {"arch": cfg["model"]["arch"], "encoder": cfg["model"]["encoder"],
                 "encoder_weights": cfg["model"]["encoder_weights"], "classes": classes}
    model = build_model(model_cfg).to(device)
    if tcfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    o = cfg["optim"]
    opt = torch.optim.AdamW(param_groups(model, o["lr"], o["encoder_lr_mult"], o["weight_decay"]))
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(tcfg["epochs"] * steps_per_epoch, o["warmup_epochs"] * steps_per_epoch,
                       o["min_lr_frac"]))
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg["amp"] == "fp16" and device.startswith("cuda"))
    loss_fn = SegLoss(**{k: cfg["loss"][k] for k in
                         ("pos_weight", "w_bce", "w_dice", "w_cldice", "cldice_iters", "root_scale")})

    start_epoch, best, best_epoch, stale = 1, -1.0, 0, 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, best_epoch, stale = ck["epoch"] + 1, ck["best"], ck["best_epoch"], ck["stale"]
        print(f"resumed from {a.resume} at epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"run dir {run_dir}\ndevice {device} ({torch.cuda.get_device_name(0) if device.startswith('cuda') else 'cpu'})"
          f"  encoder {model_cfg['encoder']}  params {n_params / 1e6:.1f}M  classes {classes}")
    print(f"train samples {len(train_items)} (real {sum(i.group == 'real' for i in train_items)})  "
          f"val samples {len(val_items)} -> {len(val_ds) if val_ds else 0} tiles  "
          f"test samples held out (never loaded): {manifest['test_samples']}")
    sample_idx = pick_sample_tiles(val_ds, tcfg["sample_count"]) if val_ds else []

    log_f = open(run_dir / "log.csv", "a", newline="")
    step_f = open(run_dir / "steps.csv", "a", newline="")
    log_w, step_w = csv.writer(log_f), csv.writer(step_f)
    cols = ["epoch", "lr", "train_loss", "bce", "dice", "cldice", "root", "val_loss", "val_dice",
            "val_iou", "val_precision", "val_recall", "val_dice@0.5", "val_cldice@0.5",
            "best_threshold", "seconds"]
    if start_epoch == 1:
        log_w.writerow(cols)
        step_w.writerow(["epoch", "step", "loss"])
    global_step = (start_epoch - 1) * steps_per_epoch

    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        t0 = time.time()
        sampler.set_epoch(epoch - 1)
        model.train()
        sums, n = {}, 0
        for img, tgt, valid in train_loader:
            img = img.to(device, non_blocking=True)
            if tcfg["channels_last"]:
                img = img.contiguous(memory_format=torch.channels_last)
            tgt, valid = tgt.to(device, non_blocking=True), valid.to(device, non_blocking=True)
            with autocast_ctx(device, tcfg["amp"]):
                logits = model(img)
            loss, parts = loss_fn(logits.float(), tgt, valid)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            parts["loss"] = float(loss.detach())
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            step_w.writerow([epoch, global_step, f"{parts['loss']:.5f}"])
            tb.add_scalar("train/loss_step", parts["loss"], global_step)
        step_f.flush()
        tr = {k: v / max(n, 1) for k, v in sums.items()}

        vm = validate(model, val_loader, device, cfg, loss_fn) if val_loader else {}
        score = vm.get(tcfg["select_metric"], -tr["loss"])
        improved = score > best
        if improved:
            best, best_epoch, stale = score, epoch, 0
            save_checkpoint(model, run_dir / "checkpoints" / "best.pt", model_cfg)
            (run_dir / "checkpoints" / "best.pt.json").write_text(json.dumps({
                "epoch": epoch, "selected_on": "val", "select_metric": tcfg["select_metric"],
                "best_metrics": {"val": vm}, "train": tr, "config": cfg}, indent=1))
        else:
            stale += 1
        save_checkpoint(model, run_dir / "checkpoints" / "last.pt", model_cfg, extra={
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best": best, "best_epoch": best_epoch, "stale": stale})

        lr = opt.param_groups[1]["lr"]
        row = {"epoch": epoch, "lr": lr, "train_loss": tr["loss"], "seconds": round(time.time() - t0, 1),
               **{k: tr.get(k, "") for k in ("bce", "dice", "cldice", "root")}, **vm}
        log_w.writerow([row.get(c, "") for c in cols])
        log_f.flush()
        for k, v in row.items():
            if isinstance(v, (int, float)) and k != "epoch":
                tb.add_scalar(f"epoch/{k}", v, epoch)
        if val_ds and (epoch % tcfg["sample_every"] == 0 or epoch == 1 or epoch == tcfg["epochs"]):
            write_samples(model, val_ds, sample_idx, device, cfg,
                          run_dir / "samples" / f"epoch_{epoch:03d}.png", vm.get("best_threshold", 0.5))
        vtxt = (f"val dice {vm['val_dice']:.3f} P {vm['val_precision']:.3f} R {vm['val_recall']:.3f} "
                f"clDice {vm['val_cldice@0.5']:.3f} thr {vm['best_threshold']}") if vm else "no val"
        print(f"epoch {epoch:3d}/{tcfg['epochs']}  loss {tr['loss']:.4f}  {vtxt}  "
              f"({row['seconds']:.0f}s){'  *best*' if improved else ''}", flush=True)
        if stale >= tcfg["early_stop_patience"]:
            print(f"early stop: no improvement for {stale} epochs (best epoch {best_epoch})")
            break

    log_f.close(); step_f.close(); tb.close()
    print(f"\nbest epoch {best_epoch}  {tcfg['select_metric']} {best:.4f}\n"
          f"checkpoint {run_dir / 'checkpoints' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
