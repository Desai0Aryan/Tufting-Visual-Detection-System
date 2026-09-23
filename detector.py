"""Tuft detection and flow-vector estimation from a single image.

A tuft is a short length of dark string taped to the surface at one end
(the wider "root" blob) with the free end (the thin, tapering "tip")
lying along the local airflow direction. Real tufts are often curled or
hook-shaped rather than straight, so this module finds each tuft's
contour via thresholding/shape filtering, then extracts a thin skeleton
of the contour and walks it to find the two free ends. The root/tip are
told apart by local thickness (root = wider, taped end), and the stored
flow vector is the *local* tangent direction at the tip end (not the
straight root->tip chord), since for a curled tuft the chord can point
in a very different direction than the string is actually indicating.
"""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Tuft:
    id: int
    root: tuple      # (x, y) anchor/tape end, in image pixel coords
    tip: tuple       # (x, y) free/tail end, in image pixel coords
    vector: tuple    # unit vector: local tangent direction at the tip end
    angle_deg: float # standard math angle (0 = +x, CCW positive, y flipped)
    length_px: float # geodesic (along-string) length, not straight-line chord
    contour: np.ndarray


class TuftDetector:
    def __init__(
        self,
        min_area=15,
        max_area=600,
        min_elongation=2.0,
        min_length_px=12,
        max_length_px=110,
        block_size=25,
        c=5,
        clahe_clip=3.0,
        clahe_tile=8,
        roi=None,  # (x, y, w, h) restricting the search area, or None for full image
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_elongation = min_elongation
        self.min_length_px = min_length_px
        self.max_length_px = max_length_px
        if not isinstance(block_size, int) or block_size <= 1 or block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer greater than 1")
        if min_area < 0 or max_area <= min_area:
            raise ValueError("max_area must be greater than min_area >= 0")
        if min_length_px < 0 or max_length_px <= min_length_px:
            raise ValueError("max_length_px must be greater than min_length_px >= 0")
        if min_elongation <= 0:
            raise ValueError("min_elongation must be greater than 0")

        self.block_size = block_size
        self.c = c
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.roi = roi

    def _threshold(self, gray):
        # Local contrast (not just brightness) varies a lot across a curved,
        # unevenly-lit wing surface; CLAHE equalizes contrast within small
        # tiles so a shadowed region gets thresholded as reliably as a bright
        # one, instead of only the well-lit region producing clean edges.
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile)
        )
        equalized = clahe.apply(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        block_size = self.block_size
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, self.c,
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return thresh

    @staticmethod
    def _elongation(contour, area):
        """perimeter^2 / (4*pi*area): ~1 for a compact/round blob, growing
        large for a thin elongated shape. Unlike a minAreaRect aspect ratio,
        this stays large even when the shape curls back on itself, since
        perimeter tracks the string's actual (curled) length on both edges.
        """
        if area < 1e-6:
            return 0.0
        perimeter = cv2.arcLength(contour, True)
        return (perimeter * perimeter) / (4.0 * np.pi * area)

    @staticmethod
    def _skeletonize(mask):
        """Zhang-Suen thinning: reduces a filled blob to a true 1px-wide
        topological skeleton. This matters for distinguishing real tuft
        strings from jagged noise blobs (e.g. cloud texture) later, since a
        genuinely thin/simple shape thins down to a clean path with few
        branches, while a rough/blobby shape thins into a bushy tangle."""
        img = (mask > 0).astype(np.uint8)
        while True:
            changed = False
            for step in (0, 1):
                p = np.pad(img, 1)
                P2, P3, P4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
                P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
                P8, P9 = p[1:-1, :-2], p[:-2, :-2]
                ring = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
                B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
                A = np.zeros_like(img)
                for a, b in zip(ring[:-1], ring[1:]):
                    A += ((a == 0) & (b == 1)).astype(np.uint8)
                if step == 0:
                    cond = (
                        (img == 1) & (B >= 2) & (B <= 6) & (A == 1)
                        & (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
                    )
                else:
                    cond = (
                        (img == 1) & (B >= 2) & (B <= 6) & (A == 1)
                        & (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
                    )
                if np.any(cond):
                    img[cond] = 0
                    changed = True
            if not changed:
                break
        return (img * 255).astype(np.uint8)

    @staticmethod
    def _skeleton_points_and_endpoints(skel):
        """Return (set of (row, col) skeleton pixels, list of endpoints).
        An endpoint is a skeleton pixel with exactly one 8-connected
        neighbor, i.e. a free end of an (assumed) simple open curve."""
        ys, xs = np.nonzero(skel)
        pts = set(zip(ys.tolist(), xs.tolist()))
        endpoints = []
        for (y, x) in pts:
            count = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    if (y + dy, x + dx) in pts:
                        count += 1
            if count == 1:
                endpoints.append((y, x))
        return pts, endpoints

    @staticmethod
    def _skeleton_path(pts, start, end):
        """BFS shortest path (8-connected) between two skeleton pixels."""
        if start == end:
            return [start]
        visited = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == end:
                break
            y, x = cur
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nxt = (y + dy, x + dx)
                    if nxt in pts and nxt not in visited:
                        visited[nxt] = cur
                        queue.append(nxt)
        if end not in visited:
            return None
        path = [end]
        while path[-1] != start:
            path.append(visited[path[-1]])
        path.reverse()
        return path

    @staticmethod
    def _path_length(path):
        return float(sum(
            np.hypot(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])
            for i in range(len(path) - 1)
        ))

    @classmethod
    def _best_endpoint_path(cls, pts, endpoints):
        """Among all endpoint pairs, return the (path, length) for the pair
        with the longest geodesic path. A clean tuft skeleton has exactly
        two endpoints; extra ones (small thinning spurs) are ignored by
        preferring the longest path rather than an arbitrary pair."""
        if len(endpoints) < 2:
            return None
        best = None
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                path = cls._skeleton_path(pts, endpoints[i], endpoints[j])
                if path is None:
                    continue
                length = cls._path_length(path)
                if best is None or length > best[1]:
                    best = (path, length)
        return best

    @staticmethod
    def _root_tip_from_contour(contour):
        """Fallback for contours whose skeleton has no clear pair of free
        ends (e.g. near-blob shapes): PCA on the contour points, root = the
        locally wider projection extreme."""
        pts = contour.reshape(-1, 2).astype(np.float32)
        mean = pts.mean(axis=0)
        centered = pts - mean

        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        proj = centered @ axis
        end_a = pts[np.argmin(proj)]
        end_b = pts[np.argmax(proj)]

        perp = np.array([-axis[1], axis[0]])
        span = proj.max() - proj.min()
        tol = max(span * 0.15, 2.0)

        def local_width(target_proj):
            mask = np.abs(proj - target_proj) <= tol
            if not np.any(mask):
                return 0.0
            perp_vals = centered[mask] @ perp
            return float(perp_vals.max() - perp_vals.min())

        width_a = local_width(proj.min())
        width_b = local_width(proj.max())

        if width_a >= width_b:
            root, tip = end_a, end_b
        else:
            root, tip = end_b, end_a
        return tuple(root.tolist()), tuple(tip.tolist())

    @classmethod
    def _analyze_contour(cls, contour, global_skeleton):
        """Return (root_xy, tip_xy, unit_vector, length_px, angle_deg) in
        the coordinate frame of `global_skeleton` (i.e. the working image),
        or None if the contour can't be resolved into a usable tuft."""
        x, y, w, h = cv2.boundingRect(contour)
        pad = 2
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1 = min(x + w + pad, global_skeleton.shape[1])
        y1 = min(y + h + pad, global_skeleton.shape[0])

        local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.drawContours(
            local_mask, [contour - np.array([x0, y0])], -1, 255, thickness=cv2.FILLED
        )
        local_skel = cv2.bitwise_and(global_skeleton[y0:y1, x0:x1], local_mask)

        pts, endpoints = cls._skeleton_points_and_endpoints(local_skel)
        best = cls._best_endpoint_path(pts, endpoints)

        if best is None:
            root, tip = cls._root_tip_from_contour(contour)
            vec = np.array(tip) - np.array(root)
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                return None
            unit = vec / length
            angle = float(np.degrees(np.arctan2(-unit[1], unit[0])))
            return root, tip, tuple(unit.tolist()), length, angle

        path, length = best
        if length < 1e-6:
            return None

        dist_transform = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
        k = max(2, len(path) // 8)
        width_start = dist_transform[path[min(k, len(path) - 1)]]
        width_end = dist_transform[path[max(0, len(path) - 1 - k)]]

        if width_start >= width_end:
            root_px, tip_px, tip_inward_px = path[0], path[-1], path[max(0, len(path) - 1 - k)]
        else:
            root_px, tip_px, tip_inward_px = path[-1], path[0], path[min(k, len(path) - 1)]

        def to_xy(p):
            return (float(p[1] + x0), float(p[0] + y0))

        root_xy = to_xy(root_px)
        tip_xy = to_xy(tip_px)
        tip_inward_xy = to_xy(tip_inward_px)

        vec = np.array(tip_xy) - np.array(tip_inward_xy)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            vec = np.array(tip_xy) - np.array(root_xy)
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                return None
        unit = vec / norm
        angle = float(np.degrees(np.arctan2(-unit[1], unit[0])))

        return root_xy, tip_xy, tuple(unit.tolist()), length, angle

    def detect(self, image):
        """Detect tufts in a BGR (or grayscale) image and return a list of Tuft."""
        if image is None or not hasattr(image, "shape") or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")

        if self.roi is not None:
            x, y, w, h = self.roi
            height, width = image.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise ValueError(
                    f"ROI {self.roi} is outside the image bounds ({width}x{height})"
                )
            work = image[y:y + h, x:x + w]
            offset = np.array([x, y], dtype=np.float32)
        else:
            work = image
            offset = np.array([0, 0], dtype=np.float32)

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
        thresh = self._threshold(gray)
        skeleton = self._skeletonize(thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        tufts = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            if self._elongation(contour, area) < self.min_elongation:
                continue

            result = self._analyze_contour(contour, skeleton)
            if result is None:
                continue
            root, tip, unit, length, angle = result
            if length < self.min_length_px or length > self.max_length_px:
                continue

            root = np.array(root) + offset
            tip = np.array(tip) + offset

            tufts.append(Tuft(
                id=len(tufts),
                root=tuple(root.tolist()),
                tip=tuple(tip.tolist()),
                vector=unit,
                angle_deg=angle,
                length_px=length,
                contour=contour + offset.astype(np.int32),
            ))

        return tufts


def draw_tufts(image, tufts, color=(0, 255, 0), thickness=2):
    """Return a copy of image with each tuft's root->tip vector drawn as an arrow."""
    out = image.copy()
    for t in tufts:
        root = tuple(int(round(v)) for v in t.root)
        tip = tuple(int(round(v)) for v in t.tip)
        cv2.arrowedLine(out, root, tip, color, thickness, tipLength=0.35)
        cv2.circle(out, root, 3, (0, 0, 255), -1)
        cv2.putText(out, str(t.id), (root[0] + 5, root[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return out
