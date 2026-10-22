from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment


class TuftTracker:
    def __init__(self, max_dist=30.0, max_missing=5):
        """max_dist: furthest a root may move between frames (pixels) and still be the same
        tuft. max_missing: frames a tuft may go undetected before its ID is retired."""
        self.max_dist = float(max_dist)
        self.max_missing = int(max_missing)
        self._tracks = {}   # id -> {"pos": (x, y), "missing": int}
        self._next_id = 0

    def update(self, tufts):
        """Return copies of `tufts` carrying persistent IDs."""
        ids = list(self._tracks)
        assigned = {}
        if ids and tufts:
            prev = np.array([self._tracks[i]["pos"] for i in ids], float)
            cur = np.array([t.root for t in tufts], float)
            cost = np.linalg.norm(prev[:, None, :] - cur[None, :, :], axis=2)
            big = self.max_dist * 1e3
            rows, cols = linear_sum_assignment(np.where(cost <= self.max_dist, cost, big))
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.max_dist:
                    assigned[c] = ids[r]

        out = []
        seen = set()
        for j, t in enumerate(tufts):
            tid = assigned.get(j)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            seen.add(tid)
            self._tracks[tid] = {"pos": tuple(t.root), "missing": 0}
            out.append(replace(t, id=tid))

        for tid in list(self._tracks):
            if tid not in seen:
                self._tracks[tid]["missing"] += 1
                if self._tracks[tid]["missing"] > self.max_missing:
                    del self._tracks[tid]
        return out