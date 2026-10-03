import numpy as np


class SegMeter:
    """Accumulate TP/FP/FN so Dice/IoU are computed over the whole set, not as
    an average of per-tile scores (tiles with no tufts would otherwise score 1
    or nan and distort the mean)."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def update(self, pred, target, weight=None):
        p, t = np.asarray(pred).astype(bool), np.asarray(target).astype(bool)
        if weight is not None:
            w = np.asarray(weight).astype(bool)
            p, t = p & w, t & w
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum())

    @property
    def dice(self):
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else 1.0

    @property
    def iou(self):
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else 1.0

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0
