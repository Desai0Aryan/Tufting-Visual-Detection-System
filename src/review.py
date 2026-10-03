import numpy as np

from .labeling import (LABEL_EXTRACT_KW, decode_label, flip_root, render_label)
from .mask_to_tufts import extract_tufts


class ReviewSession:
    def __init__(self, image, label):
        self.image = image
        tuft, root, _ = decode_label(label)
        # the stored root disk decides orientation where it is decisive; thickness otherwise
        self.tufts = extract_tufts(tuft.astype(np.uint8) * 255, root.astype(np.uint8) * 255,
                                   **LABEL_EXTRACT_KW)
        self.keep = [True] * len(self.tufts)
        self._undo = []

    # ---- queries
    def nearest(self, x, y, max_dist=25.0):
        """Index of the tuft whose centerline is closest to (x, y), or None."""
        best, bd = None, max_dist
        for i, t in enumerate(self.tufts):
            pts = t.path if t.path is not None else np.array([t.root, t.tip])
            d = float(np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))
            if d < bd:
                best, bd = i, d
        return best

    @property
    def n_kept(self):
        return sum(self.keep)

    # ---- edits (each pushes an undo snapshot)
    def _snap(self):
        self._undo.append((list(self.tufts), list(self.keep)))

    def toggle(self, i):
        self._snap()
        self.keep[i] = not self.keep[i]

    def flip(self, i):
        self._snap()
        self.tufts[i] = flip_root(self.tufts[i])

    def reject_box(self, x0, y0, x1, y1):
        """Reject every kept tuft whose root lies in the box. Returns the count."""
        lo_x, hi_x = sorted((x0, x1))
        lo_y, hi_y = sorted((y0, y1))
        hit = [i for i, t in enumerate(self.tufts)
               if self.keep[i] and lo_x <= t.root[0] <= hi_x and lo_y <= t.root[1] <= hi_y]
        if hit:
            self._snap()
            for i in hit:
                self.keep[i] = False
        return len(hit)

    def undo(self):
        if self._undo:
            self.tufts, self.keep = self._undo.pop()

    # ---- output
    def render(self):
        """Final label. Reviewed tufts are all trusted for the root head."""
        return render_label(self.image.shape, self.tufts, keep=self.keep,
                            trusted=[True] * len(self.tufts))
