"""Tuft detection and flow-vector estimation from a single image.

A tuft is a short length of dark string taped to the surface at one end
(the wider "root" blob) with the free end (the thin, tapering "tip")
lying along the local airflow direction. This module finds each tuft's
contour via thresholding/shape filtering, then estimates a root->tip
vector from the contour geometry (the root is the locally wider end).
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Tuft:
    id: int
    root: tuple      # (x, y) anchor/tape end, in image pixel coords
    tip: tuple       # (x, y) free/tail end, in image pixel coords
    vector: tuple    # unit vector (ux, uy) pointing root -> tip
    angle_deg: float # standard math angle (0 = +x, CCW positive, y flipped)
    length_px: float
    contour: np.ndarray


class TuftDetector:
    def __init__(
        self,
        min_area=15,
        max_area=600,
        min_aspect_ratio=5.5,
        min_length_px=12,
        max_length_px=110,
        block_size=25,
        c=5,
        roi=None,  # (x, y, w, h) restricting the search area, or None for full image
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.min_length_px = min_length_px
        self.max_length_px = max_length_px
        self.block_size = block_size
        self.c = c
        self.roi = roi

    def _threshold(self, gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        block_size = self.block_size | 1  # adaptiveThreshold requires odd block size
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, self.c,
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return thresh

    @staticmethod
    def _root_tip_from_contour(contour):
        """Root = locally wider end of the contour (tape patch); tip = narrower end."""
        pts = contour.reshape(-1, 2).astype(np.float32)
        mean = pts.mean(axis=0)
        centered = pts - mean

        # Principal axis via SVD (equivalent to PCA on the contour points).
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

    def detect(self, image):
        """Detect tufts in a BGR (or grayscale) image and return a list of Tuft."""
        if self.roi is not None:
            x, y, w, h = self.roi
            work = image[y:y + h, x:x + w]
            offset = np.array([x, y], dtype=np.float32)
        else:
            work = image
            offset = np.array([0, 0], dtype=np.float32)

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
        thresh = self._threshold(gray)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        tufts = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
            if min(rw, rh) < 1e-3:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect < self.min_aspect_ratio:
                continue
            longest_side = max(rw, rh)
            if longest_side < self.min_length_px or longest_side > self.max_length_px:
                continue

            root, tip = self._root_tip_from_contour(contour)
            root = np.array(root) + offset
            tip = np.array(tip) + offset
            vec = tip - root
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            unit = vec / length
            angle = float(np.degrees(np.arctan2(-unit[1], unit[0])))

            tufts.append(Tuft(
                id=len(tufts),
                root=tuple(root.tolist()),
                tip=tuple(tip.tolist()),
                vector=tuple(unit.tolist()),
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
