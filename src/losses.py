"""Segmentation losses on raw logits: BCE + Dice + optional soft-clDice.

soft-clDice: Shit et al., "clDice - a Novel Topology-Preserving Loss Function
for Tubular Structure Segmentation", CVPR 2021. It scores the predicted mask's
soft skeleton against the label and vice versa, so a gap that splits one tuft
in two costs much more than the same number of pixels shaved off its edge.
That matters here because the downstream skeleton/BFS path breaks on gaps.
"""
import torch
import torch.nn.functional as F


def bce_loss(logits, target, pos_weight=1.0, weight=None):
    pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    if weight is None:
        return loss.mean()
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def dice_loss(logits, target, weight=None, eps=1.0, per_sample=False):
    """Soft Dice on sigmoid(logits). Batch-global by default, which is stable
    when many crops are empty hard negatives."""
    p = torch.sigmoid(logits)
    if weight is not None:
        p, target = p * weight, target * weight
    dims = (1, 2, 3) if per_sample else (0, 1, 2, 3)
    inter = (p * target).sum(dim=dims)
    denom = p.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


# ---------------------------------------------------------------- soft skeleton
def soft_erode(x):
    a = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    b = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.min(a, b)


def soft_dilate(x):
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skeleton(x, iters=10):
    """Differentiable skeleton of a [0,1] map. `iters` must exceed the largest
    half-width (px) of the structures; tufts are a few px wide."""
    skel = F.relu(x - soft_open(x))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(logits, target, iters=10, smooth=1.0):
    p = torch.sigmoid(logits)
    sp, st = soft_skeleton(p, iters), soft_skeleton(target, iters)
    tprec = ((sp * target).sum() + smooth) / (sp.sum() + smooth)
    tsens = ((st * p).sum() + smooth) / (st.sum() + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


class SegLoss:
    """w_bce * BCE + w_dice * Dice + w_cldice * soft-clDice on channel 0 (tuft),
    plus root_scale * (BCE + Dice) on channel 1 (root) where root-valid == 1,
    when the model has a root head."""

    def __init__(self, pos_weight=2.0, w_bce=1.0, w_dice=1.0, w_cldice=0.0,
                 cldice_iters=10, root_scale=0.5):
        self.pos_weight, self.w_bce, self.w_dice = pos_weight, w_bce, w_dice
        self.w_cldice, self.cldice_iters, self.root_scale = w_cldice, cldice_iters, root_scale

    def __call__(self, logits, target, root_valid=None):
        """Returns (total, parts dict of detached floats)."""
        t_logit, t_tgt = logits[:, 0:1], target[:, 0:1]
        parts = {"bce": bce_loss(t_logit, t_tgt, self.pos_weight),
                 "dice": dice_loss(t_logit, t_tgt)}
        total = self.w_bce * parts["bce"] + self.w_dice * parts["dice"]
        if self.w_cldice > 0:
            parts["cldice"] = soft_cldice_loss(t_logit.float(), t_tgt, self.cldice_iters)
            total = total + self.w_cldice * parts["cldice"]
        if logits.shape[1] > 1 and root_valid is not None and root_valid.sum() > 0:
            r_logit, r_tgt = logits[:, 1:2], target[:, 1:2]
            parts["root"] = (bce_loss(r_logit, r_tgt, 1.0, root_valid)
                             + dice_loss(r_logit, r_tgt, root_valid))
            total = total + self.root_scale * parts["root"]
        return total, {k: float(v.detach()) for k, v in parts.items()}
