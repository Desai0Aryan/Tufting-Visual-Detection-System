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

    def fbeta(self, beta=1.0):
        """beta < 1 weights precision more (fewer false detections); beta > 1 favours recall."""
        p, r = self.precision, self.recall
        b2 = beta * beta
        d = b2 * p + r
        return (1 + b2) * p * r / d if d else 0.0


class ClDiceMeter:
    """Hard clDice (Shit et al. 2021) accumulated over a set:
    Tprec = |skel(P) & G| / |skel(P)|, Tsens = |skel(G) & P| / |skel(G)|,
    clDice = 2 Tprec Tsens / (Tprec + Tsens). Counts are summed across images,
    like SegMeter, so empty images do not distort the score."""

    def __init__(self):
        self.sp_in_g = self.sp = self.sg_in_p = self.sg = 0

    def update(self, pred, target):
        from skimage.morphology import skeletonize
        p, t = np.asarray(pred).astype(bool), np.asarray(target).astype(bool)
        for pi, ti in zip(p.reshape(-1, *p.shape[-2:]), t.reshape(-1, *t.shape[-2:])):
            sp, sg = skeletonize(pi), skeletonize(ti)
            self.sp_in_g += int((sp & ti).sum())
            self.sp += int(sp.sum())
            self.sg_in_p += int((sg & pi).sum())
            self.sg += int(sg.sum())

    @property
    def tprec(self):
        return self.sp_in_g / self.sp if self.sp else 1.0

    @property
    def tsens(self):
        return self.sg_in_p / self.sg if self.sg else 1.0

    @property
    def cldice(self):
        p, s = self.tprec, self.tsens
        return 2 * p * s / (p + s) if (p + s) else 0.0
