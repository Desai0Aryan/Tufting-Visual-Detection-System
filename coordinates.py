"""User-defined coordinate system and per-tuft (X, Y, Z) locations (requirements 2 and 4).

Everything here is plain numpy, no Qt, no detector imports.

A CoordinateSystem maps image pixels (origin top-left, y down) to the user's own
world coordinates (any unit, any origin, any axis direction):

    cs = CoordinateSystem.from_origin_axis(origin_px=(120, 640), axis_px=(980, 628),
                                           distance=500, units="mm")
    cs.to_world([(300, 400)])               # -> [[x_mm, y_mm]]
    cs.to_world_vectors(px, unit_vectors)   # flow directions in the same axes

Ways to define it:
  * from_origin_axis: click the origin, click a second point along the X axis, type the
    real distance between them. Y is perpendicular (up by default). Scale + rotation.
  * from_points: any number of reference points with known world X, Y.
        2 points  -> scale + rotation (+ shift)
        3 points  -> affine (also corrects skew / unequal scale)
        4+ points -> perspective (homography), for a camera that views the surface at an angle
    The surface is assumed flat; its height is `z_plane` (the Z written for tufts that
    have no user-entered Z).

TuftLocations holds the optional (X, Y, Z) a user typed for individual tufts; any
component they leave empty falls back to the automatic value.
"""
import csv
import io
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

UNITS = ("px", "mm", "cm", "m", "in", "ft")


class CoordinateError(ValueError):
    """A coordinate system cannot be built from the given points (message is user-readable)."""


def _as_pts(p) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.size == 0:
        return a.reshape(0, 2)
    return a.reshape(-1, 2)


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = _as_pts(pts)
    if len(pts) == 0:
        return pts
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = ph[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return ph[:, :2] / w


def _fit_similarity(src: np.ndarray, dst: np.ndarray, allow_reflection: bool) -> np.ndarray:
    """Least-squares scale+rotation(+reflection)+shift, 3x3 matrix (Umeyama)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var = (s0 ** 2).sum() / len(src)
    if var < 1e-12:
        raise CoordinateError("The reference points are all at the same place in the image.")
    cov = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if not allow_reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D) / var)
    t = mu_d - scale * R @ mu_s
    H = np.eye(3)
    H[:2, :2] = scale * R
    H[:2, 2] = t
    return H


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    A = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(A, dst, rcond=None)      # (3, 2)
    H = np.eye(3)
    H[:2, :] = sol.T
    return H


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalised DLT."""
    def norm(p):
        c = p.mean(0)
        d = np.sqrt(((p - c) ** 2).sum(1)).mean() or 1.0
        s = math.sqrt(2) / d
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
        return T, _apply_h(T, p)
    Ts, s = norm(src)
    Td, d = norm(dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    return H / H[2, 2]


def _non_collinear(pts: np.ndarray, min_abs: float = 0.0) -> bool:
    c = pts - pts.mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    return len(sv) > 1 and sv[1] > 1e-3 * max(sv[0], 1e-12) and sv[1] > min_abs


@dataclass
class CoordinateSystem:
    """pixel -> world mapping. `H` is a 3x3 matrix; identity means "no system defined" (pixels)."""
    H: np.ndarray = field(default_factory=lambda: np.eye(3))
    units: str = "px"
    z_plane: float = 0.0
    defined: bool = False
    method: str = "none"                       # none / origin_axis / points
    points: List[Tuple[float, float, float, float]] = field(default_factory=list)  # (px, py, X, Y)
    origin_label: str = ""
    y_up: bool = True

    # ---------------------------------------------------------------- builders
    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def from_origin_axis(cls, origin_px, axis_px, distance, units="mm", y_up=True, z_plane=0.0):
        """Origin at `origin_px`; +X points toward `axis_px`, which is `distance` units from it."""
        o, a = np.asarray(origin_px, float), np.asarray(axis_px, float)
        d_px = float(np.hypot(*(a - o)))
        if d_px < 3:
            raise CoordinateError("The two points are too close together. Click two points that are far apart.")
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            raise CoordinateError("Enter the real distance between the two points as a number.")
        if not distance > 0:
            raise CoordinateError("The distance between the two points must be greater than zero.")
        scale = distance / d_px
        ex = (a - o) / d_px                         # image-space unit vector of +X
        ey = np.array([ex[1], -ex[0]]) if y_up else np.array([-ex[1], ex[0]])   # +Y (up = CCW on screen)
        R = np.array([ex, ey]) * scale              # rows: world x, world y
        H = np.eye(3)
        H[:2, :2] = R
        H[:2, 2] = -R @ o
        pts = [(o[0], o[1], 0.0, 0.0), (a[0], a[1], distance, 0.0)]
        return cls(H=H, units=_unit(units), z_plane=float(z_plane), defined=True, method="origin_axis", points=pts, y_up=bool(y_up))

    @classmethod
    def from_points(cls, points, units="mm", z_plane=0.0):
        """points: iterable of (px, py, X, Y). 2 -> similarity, 3 -> affine, 4+ -> perspective."""
        pts = [tuple(float(v) for v in p[:4]) for p in points]
        if len(pts) < 2:
            raise CoordinateError("Add at least 2 reference points (3 or more for a camera that is not square-on).")
        arr = np.asarray(pts)
        src, dst = arr[:, :2], arr[:, 2:4]
        if len(pts) == 2:
            H = _fit_similarity(src, dst, allow_reflection=False)
        else:
            if not _non_collinear(src, 2.0):
                raise CoordinateError("The reference points lie on one line. Add a point off that line.")
            if not _non_collinear(dst):
                raise CoordinateError("The real-world positions you entered lie on one line.")
            H = _fit_affine(src, dst) if len(pts) == 3 else _fit_homography(src, dst)
        cs = cls(H=H, units=_unit(units), z_plane=float(z_plane), defined=True, method="points", points=pts)
        if not cs.is_valid():
            raise CoordinateError("Those reference points do not make a usable coordinate system. Check the numbers.")
        return cs

    # ---------------------------------------------------------------- mapping
    def is_valid(self) -> bool:
        if not np.all(np.isfinite(self.H)):
            return False
        return abs(np.linalg.det(self.H[:2, :2])) > 1e-12 or abs(np.linalg.det(self.H)) > 1e-12

    def to_world(self, px) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) world X, Y."""
        return _apply_h(self.H, px)

    def to_pixel(self, world) -> np.ndarray:
        return _apply_h(np.linalg.inv(self.H), world)

    def to_world_vectors(self, px, vectors, eps: float = 1.0) -> np.ndarray:
        """Unit flow vectors in image space at points `px` -> unit vectors in world axes.

        Works for a perspective map too: the direction is measured by moving `eps` pixels
        along the vector and seeing where that lands in the world."""
        px, v = _as_pts(px), _as_pts(vectors)
        if len(px) == 0:
            return v
        a = self.to_world(px)
        b = self.to_world(px + v * eps)
        d = b - a
        n = np.linalg.norm(d, axis=1, keepdims=True)
        out = np.where(n > 1e-12, d / np.where(n > 1e-12, n, 1), 0.0)
        return out

    def units_per_pixel(self, at_px=(0.0, 0.0)) -> float:
        p = np.asarray(at_px, float)
        a = self.to_world([p])[0]
        b = self.to_world([p + [1, 0]])[0]
        c = self.to_world([p + [0, 1]])[0]
        return float(math.sqrt(abs((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0])))

    def angle_offset_deg(self) -> float:
        """Rotation of the world +X axis relative to the image +X axis (degrees, CCW on screen)."""
        a = self.to_world([(0.0, 0.0)])[0]
        b = self.to_world([(1.0, 0.0)])[0]
        d = b - a
        return float(math.degrees(math.atan2(d[1], d[0])))

    def residuals(self) -> List[float]:
        """Distance (in world units) between each reference point and where the fit puts it."""
        if not self.points:
            return []
        arr = np.asarray(self.points)
        pred = self.to_world(arr[:, :2])
        return [float(v) for v in np.hypot(*(pred - arr[:, 2:4]).T)]

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"defined": self.defined, "method": self.method, "units": self.units,
                "z_plane": self.z_plane, "points": [list(p) for p in self.points],
                "H": np.asarray(self.H).tolist(), "origin_label": self.origin_label, "y_up": self.y_up}

    @classmethod
    def from_dict(cls, d: Optional[dict]):
        if not d or not d.get("defined"):
            return cls()
        try:
            if d.get("points"):
                cs = cls.from_points(d["points"], d.get("units", "mm"), d.get("z_plane", 0.0))
                if d.get("method") == "origin_axis" and len(d["points"]) == 2:
                    p0, p1 = d["points"]
                    cs = cls.from_origin_axis(p0[:2], p1[:2], p1[2], d.get("units", "mm"),
                                              y_up=bool(d.get("y_up", True)), z_plane=d.get("z_plane", 0.0))
                cs.method = d.get("method", cs.method)
                return cs
        except CoordinateError:
            pass
        H = np.asarray(d.get("H", np.eye(3)), float).reshape(3, 3)
        return cls(H=H, units=_unit(d.get("units", "px")), z_plane=float(d.get("z_plane", 0.0)),
                   defined=True, method=d.get("method", "points"))


def _unit(u) -> str:
    u = str(u or "px").strip()
    return u if u in UNITS else (u[:8] or "px")


# ===================================================================== per-tuft locations
_ALIASES = {
    "id": "tuft_id", "tuft": "tuft_id", "tuftid": "tuft_id", "tuft_id": "tuft_id", "tuft id": "tuft_id",
    "x": "x", "y": "y", "z": "z",
}


class LocationError(ValueError):
    pass


@dataclass
class TuftLocations:
    """User-entered (X, Y, Z) per tuft ID. Any component may be None (= use the automatic value)."""
    items: Dict[int, Tuple[Optional[float], Optional[float], Optional[float]]] = field(default_factory=dict)

    def set(self, tuft_id, x=None, y=None, z=None):
        tid = int(tuft_id)
        vals = tuple(_num(v, n) for v, n in ((x, "X"), (y, "Y"), (z, "Z")))
        if all(v is None for v in vals):
            self.items.pop(tid, None)
        else:
            self.items[tid] = vals

    def remove(self, tuft_id):
        self.items.pop(int(tuft_id), None)

    def get(self, tuft_id):
        return self.items.get(int(tuft_id), (None, None, None))

    def __len__(self):
        return len(self.items)

    def resolve(self, tuft_id, auto_xyz):
        """Final (x, y, z) and where each came from ('user' / 'auto')."""
        u = self.get(tuft_id)
        xyz, src = [], []
        for uv, av in zip(u, auto_xyz):
            xyz.append(uv if uv is not None else float(av))
            src.append("user" if uv is not None else "auto")
        return tuple(xyz), tuple(src)

    # persistence -------------------------------------------------------
    def to_dict(self):
        return {str(k): list(v) for k, v in sorted(self.items.items())}

    @classmethod
    def from_dict(cls, d):
        out = cls()
        for k, v in (d or {}).items():
            out.set(int(k), *list(v)[:3])
        return out

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["tuft_id", "x", "y", "z"])
        for k, v in sorted(self.items.items()):
            w.writerow([k] + ["" if c is None else c for c in v])
        return buf.getvalue()

    @classmethod
    def from_csv(cls, text: str):
        """Header row with tuft_id and any of x, y, z (case-insensitive). Blank cells = automatic."""
        text = text.lstrip("﻿")
        rows = list(csv.reader(io.StringIO(text)))
        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows:
            raise LocationError("The file is empty.")
        head = [_ALIASES.get(c.strip().lower().replace("-", "_"), c.strip().lower()) for c in rows[0]]
        if "tuft_id" not in head or not any(c in head for c in ("x", "y", "z")):
            raise LocationError("The first row must name the columns: tuft_id, x, y and/or z.")
        out = cls()
        for n, r in enumerate(rows[1:], start=2):
            rec = dict(zip(head, [c.strip() for c in r]))
            try:
                tid = int(float(rec.get("tuft_id", "")))
            except ValueError:
                raise LocationError(f"Row {n}: tuft_id must be a whole number.")
            try:
                out.set(tid, *(rec.get(k, "") or None for k in ("x", "y", "z")))
            except LocationError as exc:
                raise LocationError(f"Row {n}: {exc}")
        return out


def _num(v, name="value"):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise LocationError(f"{name} must be a number (got {v!r}).")
    if not math.isfinite(f):
        raise LocationError(f"{name} must be a finite number.")
    return f


def auto_locations(roots_by_tuft: Dict[int, Iterable[Tuple[float, float]]], cs: CoordinateSystem):
    """Automatic location of each tuft: the median root position over the video, mapped to world.
    Z is the surface height (cs.z_plane). -> {tuft_id: (x, y, z)}"""
    out = {}
    for tid, pts in roots_by_tuft.items():
        p = _as_pts(list(pts))
        if len(p) == 0:
            continue
        m = np.median(p, axis=0)
        w = cs.to_world([m])[0]
        out[int(tid)] = (float(w[0]), float(w[1]), float(cs.z_plane))
    return out
