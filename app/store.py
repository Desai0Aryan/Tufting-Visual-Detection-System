"""Thread-safe cache of FrameResults, one per frame (no Qt)."""
import threading


class ResultStore:
    def __init__(self, total=0):
        self._lock = threading.Lock()
        self._results = {}
        self.total = total
        self.frontier = -1     # highest index such that 0..frontier are all processed

    def clear(self, total=None):
        with self._lock:
            self._results.clear()
            self.frontier = -1
            if total is not None:
                self.total = total

    def put(self, result):
        with self._lock:
            self._results[result.index] = result
            while self.frontier + 1 in self._results:
                self.frontier += 1

    def get(self, index):
        with self._lock:
            return self._results.get(index)

    def ordered(self, upto=None):
        """Processed results 0..min(upto, frontier) in order."""
        with self._lock:
            end = self.frontier if upto is None else min(upto, self.frontier)
            return [self._results[i] for i in range(end + 1)]

    def series(self, track_id, upto=None, field="angle_deg"):
        """[(frame, value)] for one track over processed frames (detected frames only)."""
        out = []
        for r in self.ordered(upto):
            t = r.by_id(track_id)
            if t is not None and t.detected:
                out.append((r.index, getattr(t, field)))
        return out

    def __len__(self):
        with self._lock:
            return len(self._results)
