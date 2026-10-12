"""Editing state for the label-review tool (no GUI code, so it is testable).

The mask is edited pixel-exactly: deleting a tuft removes its connected
component, the brush adds or erases pixels, restoring a SAM-rejected
candidate fills its outline. Geometry (root, tip, tip-tangent arrow) is
recomputed from the current mask for display and for the saved JSON.
Every edit is logged.
"""
import time

import cv2
import numpy as np

from .label_io import tufts_from_mask

MAX_UNDO = 40


class ReviewState:
    def __init__(self, image, mask, info=None, meta=None, anchor_dist=None):
        self.image = image
        self.mask = (mask > 0).astype(np.uint8) * 255
        self.info = info or {"tufts": [], "rejected_candidates": []}
        self.meta = meta or {}
        self.anchor_dist = anchor_dist
        self.rejected = [np.asarray(r["contour"], np.int32).reshape(-1, 1, 2)
                         for r in self.info.get("rejected_candidates", [])]
        self.restored = [False] * len(self.rejected)
        self.start_mask = self.mask.copy()
        self.n_tufts_start = len(self.info.get("tufts", []))
        self.undo_stack, self.edits = [], []
        self.t0 = time.time()
        self._tufts = None

    # ------------------------------------------------------------ queries
    @property
    def tufts(self):
        if self._tufts is None:
            self._tufts = (tufts_from_mask(self.mask, anchor_dist=self.anchor_dist)
                           if self.mask.any() else [])
        return self._tufts

    def component_at(self, x, y, radius=8):
        """Label map and id of the mask component nearest (x, y) within radius, or (None, 0)."""
        h, w = self.mask.shape
        x0, y0 = max(int(x) - radius, 0), max(int(y) - radius, 0)
        x1, y1 = min(int(x) + radius + 1, w), min(int(y) + radius + 1, h)
        win = self.mask[y0:y1, x0:x1] > 0
        if not win.any():
            return None, 0
        n, cc = cv2.connectedComponents((self.mask > 0).astype(np.uint8), connectivity=8)
        ys, xs = np.nonzero(win)
        k = int(np.argmin((xs + x0 - x) ** 2 + (ys + y0 - y) ** 2))
        return cc, int(cc[ys[k] + y0, xs[k] + x0])

    def rejected_at(self, x, y):
        for i, c in enumerate(self.rejected):
            if not self.restored[i] and cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0:
                return i
        return None

    # ------------------------------------------------------------ edits
    def _push(self):
        self.undo_stack.append((self.mask.copy(), list(self.restored)))
        if len(self.undo_stack) > MAX_UNDO:
            self.undo_stack.pop(0)
        self._tufts = None

    def click(self, x, y):
        """Delete the tuft under the cursor, or restore a rejected candidate there."""
        cc, label = self.component_at(x, y)
        if label:
            self._push()
            removed = int((cc == label).sum())
            self.mask[cc == label] = 0
            self.edits.append({"action": "delete_tuft", "x": int(x), "y": int(y), "pixels": removed})
            return "deleted"
        i = self.rejected_at(x, y)
        if i is not None:
            self._push()
            before = int((self.mask > 0).sum())
            cv2.drawContours(self.mask, [self.rejected[i]], -1, 255, cv2.FILLED)
            self.restored[i] = True
            self.edits.append({"action": "restore_candidate", "x": int(x), "y": int(y),
                               "pixels": int((self.mask > 0).sum()) - before})
            return "restored"
        return None

    def begin_stroke(self):
        self._push()
        self._stroke = {"added": 0, "removed": 0}

    def paint(self, p0, p1, radius, add=True):
        before = int((self.mask > 0).sum())
        cv2.line(self.mask, tuple(int(v) for v in p0), tuple(int(v) for v in p1),
                 255 if add else 0, max(1, int(round(2 * radius))))
        cv2.circle(self.mask, tuple(int(v) for v in p1), int(radius), 255 if add else 0, -1)
        delta = int((self.mask > 0).sum()) - before
        self._stroke["added" if add else "removed"] += abs(delta)
        self._tufts = None

    def end_stroke(self, add, radius):
        self.edits.append({"action": "brush_add" if add else "brush_erase", "radius": radius,
                           **getattr(self, "_stroke", {})})

    def undo(self):
        if self.undo_stack:
            self.mask, self.restored = self.undo_stack.pop()
            self._tufts = None
            self.edits.append({"action": "undo"})

    # ------------------------------------------------------------ output
    def summary(self, decision):
        m0, m1 = self.start_mask > 0, self.mask > 0
        return {"decision": decision, "seconds": round(time.time() - self.t0, 1),
                "n_tufts_before": self.n_tufts_start, "n_tufts_after": len(self.tufts),
                "pixels_added": int((m1 & ~m0).sum()), "pixels_removed": int((m0 & ~m1).sum()),
                "edited": bool((m0 != m1).any()), "edits": self.edits}
