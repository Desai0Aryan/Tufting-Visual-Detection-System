"""Tuft map: where every tuft is anchored, found once per video (no Qt).

On a camera fixed to the tested surface, every tuft's taped root stays put.
So instead of re-finding every tuft in every frame, the map is built once:

  1. sample frames across the video, align each to a reference frame
     (ORB features + RANSAC similarity: camera shake / wing flex), detect
     tufts with the AI model, and collect both ends of every detection;
  2. cluster the ends; an end that recurs in many frames is a candidate root;
  3. a detection links its two ends. Which end is the root cannot be told
     from one frame (the model's guess is a coin flip), but over a video the
     root stays put while the tip wanders: that sets the flow direction, and
     of each linked pair the upstream end is the root. Roots seen with and
     without their tape tab, and steady tips, merge into one tuft;
  4. a human confirms the map once (tools.process_video: click to add or
     remove roots, f flips the flow if the estimate is wrong), then it is
     saved as <video>.tuftmap.json plus its reference frame (<map>.ref.png).

A map belongs to the aircraft + camera mount, not to one video: pass it to any
later video of the same setup (tools.process_video new.mp4 --map aircraft.json)
and every frame is aligned to the saved reference frame, so tuft 52 is the same
tape on the wing in every video.

Measurement per frame against the map lives in core/map_pipeline.py.
"""
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Anchor:
    id: int
    root: tuple                 # (x, y) in the reference frame
    length_px: float = 0.0      # typical tuft length (median over samples), 0 = unknown
    angle_deg: float = 0.0      # typical direction root -> tip (math convention)
    hits: float = 0.0           # fraction of sampled frames it was seen in (1.0 = always)
    manual: bool = False        # added by hand


@dataclass
class TuftMap:
    video: str
    ref_index: int
    size: tuple                 # (width, height)
    anchors: list = field(default_factory=list)
    flow: tuple = (0.0, 0.0)    # unit flow direction in the reference frame (image coords, y down)

    def save(self, path, ref_frame=None):
        """Writes the map; with ref_frame also <path>.ref.png (needed to reuse it on other videos)."""
        if ref_frame is not None:
            cv2.imwrite(str(ref_path(path)), ref_frame)
        d = {"video": self.video, "ref_index": self.ref_index, "size": list(self.size),
             "flow": list(self.flow), "anchors": [asdict(a) for a in self.anchors]}
        Path(path).write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        anchors = [Anchor(**{**a, "root": tuple(a["root"])}) for a in d["anchors"]]
        return cls(d["video"], int(d["ref_index"]), tuple(d["size"]), anchors,
                   tuple(d.get("flow", (0.0, 0.0))))

    def add(self, xy, length_px=0.0, angle_deg=0.0):
        nid = max((a.id for a in self.anchors), default=0) + 1
        self.anchors.append(Anchor(nid, (float(xy[0]), float(xy[1])), length_px, angle_deg, 0.0, True))
        return nid

    def remove_near(self, xy, radius):
        """Remove the anchor nearest to xy if within radius; returns it or None."""
        if not self.anchors:
            return None
        d = [np.hypot(a.root[0] - xy[0], a.root[1] - xy[1]) for a in self.anchors]
        i = int(np.argmin(d))
        return self.anchors.pop(i) if d[i] <= radius else None

    def median_length(self):
        ls = [a.length_px for a in self.anchors if a.length_px > 0]
        return float(np.median(ls)) if ls else 40.0

    def flow_angle_deg(self):
        return float(np.degrees(np.arctan2(-self.flow[1], self.flow[0])))


def ref_path(map_path):
    return Path(str(map_path) + ".ref.png")


def load_ref(map_path, video_source=None, ref_index=0):
    """The map's reference frame: the saved image, else that frame of the given video."""
    p = ref_path(map_path)
    if p.exists():
        return cv2.imread(str(p))
    return video_source.read_at(ref_index) if video_source is not None else None


def default_map_path(video):
    return Path(str(video) + ".tuftmap.json")


# ---------------------------------------------------------------- alignment
class Aligner:
    """Similarity transform from the reference frame to any frame (ORB + RANSAC).
    Static parts (wing, cockpit) dominate the inliers; sky and sea are outliers."""

    def __init__(self, ref_bgr, n_features=4000, scale=0.5):
        self.scale = scale
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_kp, self.ref_des = self._features(ref_bgr)
        self.last = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        self.inliers = 0                     # of the last successful fit (0 = never aligned)

    def _features(self, bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        kp, des = self.orb.detectAndCompute(g, None)
        return kp, des

    def ref_to_frame(self, bgr, min_inliers=25):
        """2x3 matrix mapping reference pixels to this frame's pixels (last good on failure)."""
        kp, des = self._features(bgr)
        if des is None or self.ref_des is None or len(kp) < min_inliers:
            return self.last
        pairs = self.matcher.knnMatch(self.ref_des, des, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
        if len(good) < min_inliers:
            return self.last
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good]) / self.scale
        dst = np.float32([kp[m.trainIdx].pt for m in good]) / self.scale
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or inl is None or int(inl.sum()) < min_inliers:
            return self.last
        self.last, self.inliers = M, int(inl.sum())
        return M


def apply(M, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def invert(M):
    return cv2.invertAffineTransform(M)


# ---------------------------------------------------------------- clustering
def _cluster(points, radius):
    """Greedy densest-first clustering. -> labels (N,) int, centres [(x, y)]."""
    from scipy.spatial import cKDTree
    pts = np.asarray(points, float)
    labels = -np.ones(len(pts), int)
    if not len(pts):
        return labels, []
    tree = cKDTree(pts)
    nbrs = tree.query_ball_point(pts, radius)
    order = np.argsort([-len(n) for n in nbrs], kind="stable")
    centres = []
    for i in order:
        if labels[i] >= 0:
            continue
        members = [j for j in nbrs[i] if labels[j] < 0]
        c = pts[members].mean(0)
        members = [j for j in tree.query_ball_point(c, radius) if labels[j] < 0]   # re-centre once
        if not members:
            continue
        labels[members] = len(centres)
        centres.append(tuple(pts[members].mean(0)))
    return labels, centres


def det_axis(dets):
    """Direction-free flow axis (unit, image coords) from the doubled-angle mean of all detections."""
    v = np.array([np.subtract(d[2], d[1]) for d in dets], float)
    th = np.arctan2(v[:, 1], v[:, 0])
    a = 0.5 * np.arctan2(np.sin(2 * th).mean(), np.cos(2 * th).mean())
    return np.array([np.cos(a), np.sin(a)])


def anchors_from_detections(dets, n_frames, min_hits=0.3, radius=None, flow=None):
    """dets: [(frame_no, end_xy, other_end_xy, length_px)] in reference coordinates
    (which end the detector called root does not matter: that guess is a coin flip).
    flow: unit flow vector (image coords), or None to estimate it: over a video a
    taped root stays put while its tip wanders, so the partner positions seen
    from a root end scatter more than those seen from a tip end.
    -> (list[Anchor] sorted top-to-bottom then left-to-right, flow unit vector)"""
    if not dets:
        return [], (np.zeros(2) if flow is None else np.asarray(flow, float))
    n = len(dets)
    L = float(np.median([d[3] for d in dets]))
    radius = radius or max(5.0, 0.18 * L)
    ends = np.array([d[1] for d in dets] + [d[2] for d in dets], float)
    frame_of = np.r_[[d[0] for d in dets], [d[0] for d in dets]]
    partner = np.r_[np.arange(n, 2 * n), np.arange(n)]          # index of the other end
    labels, centres = _cluster(ends, radius)
    centres = np.array(centres, float)

    frames_in, members = defaultdict(set), defaultdict(list)
    for k, (lab, f) in enumerate(zip(labels, frame_of)):
        frames_in[lab].add(int(f))
        members[lab].append(k)

    def partner_scatter(c):
        pts = ends[partner[members[c]]]
        return float(np.median(np.linalg.norm(pts - pts.mean(0), axis=1))) if len(pts) > 2 else 0.0

    links = Counter()
    for i in range(n):
        a, b = labels[i], labels[n + i]
        if a != b:
            links[(min(a, b), max(a, b))] += 1
    pairs = [(a, b, cnt) for (a, b), cnt in links.most_common() if cnt >= 2]

    axis = det_axis(dets)
    if flow is None:
        vote = 0.0
        for a, b, cnt in pairs:
            sa, sb = partner_scatter(a), partner_scatter(b)
            if sa == sb:
                continue
            root, tip = (a, b) if sa > sb else (b, a)
            vote += np.sign((centres[tip] - centres[root]) @ axis) * abs(sa - sb) * cnt
        flow = axis if vote >= 0 else -axis
    flow = np.asarray(flow, float) / (np.linalg.norm(flow) + 1e-12)

    tip_clusters = set()
    for a, b, cnt in pairs:
        if a in tip_clusters or b in tip_clusters:
            continue
        tip_clusters.add(b if (centres[b] - centres[a]) @ flow > 0 else a)    # the downstream end

    anchors = []
    for c, centre in enumerate(centres):
        hits = len(frames_in[c]) / max(n_frames, 1)
        if c in tip_clusters or hits < min_hits:
            continue
        vecs = [ends[partner[k]] - ends[k] for k in members[c]]
        v = np.mean([u / (np.linalg.norm(u) + 1e-9) for u in vecs], 0) if vecs else flow
        if v @ flow < 0:          # its tufts point upstream from here: it is a tip end
            continue
        lens = [dets[k % n][3] for k in members[c]]
        anchors.append(Anchor(0, (float(centre[0]), float(centre[1])), float(np.median(lens)) if lens else L,
                              float(np.degrees(np.arctan2(-v[1], v[0]))), round(hits, 3)))
    anchors = drop_tip_duplicates(anchors, L, flow)
    anchors.sort(key=lambda a: (round(a.root[1] / max(L, 1)), a.root[0]))
    for i, a in enumerate(anchors, 1):
        a.id = i
    return anchors, flow


def drop_tip_duplicates(anchors, L, flow, reach=0.85, cone_deg=30.0):
    """Two anchors closer than about one tuft length and lined up with the flow
    are one tuft: the root seen with and without its tape tab, or a steady tip.
    The upstream one (the tape end) is kept and set to point downstream."""
    cos_min = np.cos(np.radians(cone_deg))
    order = sorted(range(len(anchors)), key=lambda k: float(np.dot(anchors[k].root, flow)))   # upstream first
    drop = set()
    for ii, i in enumerate(order):
        if i in drop:
            continue
        a = anchors[i]
        for j in order[ii + 1:]:
            if j in drop:
                continue
            b = anchors[j]
            v = np.subtract(b.root, a.root)
            dist = float(np.linalg.norm(v))
            if dist < 1e-6 or dist > reach * max(a.length_px or L, b.length_px or L):
                continue
            if float(v @ flow) / dist < cos_min:       # b must lie downstream of a, along the flow
                continue
            drop.add(j)
            a.length_px = max(a.length_px, b.length_px)
            a.hits = max(a.hits, b.hits)
    return [a for k, a in enumerate(anchors) if k not in drop]


# ---------------------------------------------------------------- building a map
def sample_indices(frame_count, n=60, skip_s=0.0, fps=30.0):
    start = int(skip_s * fps)
    if frame_count <= start:
        return [0]
    return sorted(set(np.linspace(start, frame_count - 1, min(n, frame_count - start)).astype(int).tolist()))


def build_map(source, detector, n_samples=60, min_hits=0.3, progress=None):
    """source: core.sources.VideoFileSource (seekable). detector: TuftDetector-like
    (.detect(bgr) -> [Tuft]). Reference frame = the sampled frame with most detections.
    -> (TuftMap, reference frame, detections) ; pass detections to rederive() to
    rebuild the anchors with a flipped flow without scanning the video again."""
    idx = sample_indices(source.frame_count, n_samples, fps=source.fps)
    frames, per_frame = {}, {}
    for k, i in enumerate(idx):
        f = source.read_at(i)
        if f is None:
            continue
        frames[i] = f
        per_frame[i] = detector.detect(f)
        if progress:
            progress(k + 1, len(idx))
    if not frames:
        raise ValueError("could not read any frames from the video")
    ref = max(per_frame, key=lambda i: len(per_frame[i]))
    aligner = Aligner(frames[ref])
    dets = []
    for fno, (i, tufts) in enumerate(sorted(per_frame.items())):
        to_ref = invert(aligner.ref_to_frame(frames[i])) if i != ref else np.array([[1, 0, 0], [0, 1, 0]], float)
        for t in tufts:
            r, p = apply(to_ref, [t.root, t.tip])
            dets.append((fno, tuple(r), tuple(p), float(t.length_px)))
    h, w = frames[ref].shape[:2]
    anchors, flow = anchors_from_detections(dets, len(per_frame), min_hits)
    tmap = TuftMap(str(getattr(source, "path", "")), int(ref), (w, h), anchors, tuple(float(v) for v in flow))
    return tmap, frames[ref], {"dets": dets, "n_frames": len(per_frame), "min_hits": min_hits}


def rederive(tmap, scan, flow):
    """Rebuild the automatic anchors for a given flow (e.g. flipped); hand-added ones are kept."""
    anchors, flow = anchors_from_detections(scan["dets"], scan["n_frames"], scan["min_hits"], flow=flow)
    manual = [a for a in tmap.anchors if a.manual]
    tmap.anchors = anchors
    for a in manual:
        tmap.add(a.root, a.length_px, a.angle_deg)
    tmap.flow = tuple(float(v) for v in flow)
    return tmap
