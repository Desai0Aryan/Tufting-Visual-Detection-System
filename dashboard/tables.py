"""Turn the raw pixel results + the user's setup into the tables the customer sees and downloads.

Raw results never change after analysis; the coordinate system, tuft positions and the optional
lift estimate are applied here, so editing them is instant (no re-analysis).
"""
import io
import math
import zipfile

import numpy as np
import pandas as pd

from coordinates import CoordinateSystem, TuftLocations, auto_locations

STATE_LABEL = {0: "learning", 1: "steady", 2: "fluttering", 3: "stalled"}

HISTORY_COLUMNS = ["time_s", "frame", "tuft_id", "x", "y", "z", "vector_x", "vector_y", "vector_z",
                   "angle_deg", "flow_state", "length", "confidence",
                   "root_px_x", "root_px_y", "tip_px_x", "tip_px_y"]
SUMMARY_COLUMNS = ["tuft_id", "x", "y", "z", "position_source", "seen_pct", "first_seen_s", "last_seen_s",
                   "mean_angle_deg", "angle_spread_deg", "mean_length", "steady_pct", "fluttering_pct",
                   "stalled_pct", "main_state"]


def default_setup():
    return {"coords": CoordinateSystem().to_dict(), "locations": {}, "estimate_lift": False}


def effective_cs(setup, height):
    """The user's coordinate system, or (when none is defined) pixels with the origin bottom-left, Y up."""
    cs = CoordinateSystem.from_dict((setup or {}).get("coords"))
    if cs.defined:
        return cs
    d = CoordinateSystem()
    d.H = np.array([[1.0, 0, 0], [0, -1.0, float(height)], [0, 0, 1.0]])
    d.units = "px"
    return d


def _flat(results):
    fr = results["frames"]
    counts = np.fromiter((len(f) for f in fr), int, count=len(fr))
    rows = [r for f in fr for r in f]
    arr = np.asarray(rows, dtype=float).reshape(-1, len(results["fields"]))
    fidx = np.repeat(np.arange(len(fr)), counts)
    t = np.asarray(results["times"], float)[fidx] if len(fidx) else np.zeros(0)
    return fidx, t, arr


def history_frame(results, setup) -> pd.DataFrame:
    """Vector time history: one row per tuft per processed frame."""
    setup = setup or default_setup()
    cs = effective_cs(setup, results["height"])
    locs = TuftLocations.from_dict(setup.get("locations"))
    fidx, t, a = _flat(results)
    if len(a) == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    ids = a[:, 0].astype(int)
    root, tip, vec, ratio = a[:, 1:3], a[:, 3:5], a[:, 5:7], a[:, 9]
    wv = cs.to_world_vectors(root, vec)
    vz = np.zeros(len(a))
    if setup.get("estimate_lift"):
        r = np.clip(np.nan_to_num(ratio, nan=1.0) / 0.9, 0.0, 1.0)     # 0.9 = normal foreshortening margin
        vz = np.sqrt(np.clip(1.0 - r ** 2, 0, 1))
        wv = wv * r[:, None]
    angle = np.degrees(np.arctan2(wv[:, 1], wv[:, 0]))
    upp = cs.units_per_pixel(root.mean(0)) or 1.0
    # tuft location: user value per component, else automatic (median root over the video)
    by = {}
    for tid, p in zip(ids, root):
        by.setdefault(int(tid), []).append(p)
    auto = auto_locations(by, cs)
    loc = {tid: locs.resolve(tid, auto[tid])[0] for tid in auto}
    xyz = np.array([loc[i] for i in ids])
    df = pd.DataFrame({
        "time_s": t.round(4), "frame": (np.round(t * results["fps"])).astype(int), "tuft_id": ids,
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "vector_x": wv[:, 0], "vector_y": wv[:, 1], "vector_z": vz,
        "angle_deg": angle, "flow_state": ["unmeasured"] * len(a) if results.get("kind") == "image" else [STATE_LABEL[int(s)] for s in a[:, 10]],
        "length": a[:, 8] * upp, "confidence": a[:, 11],
        "root_px_x": root[:, 0], "root_px_y": root[:, 1], "tip_px_x": tip[:, 0], "tip_px_y": tip[:, 1],
    })
    for c in ("x", "y", "z", "vector_x", "vector_y", "vector_z", "length"):
        df[c] = df[c].round(5 if c.startswith("vector") else 3)
    df["angle_deg"] = df["angle_deg"].round(2)
    return df.sort_values(["time_s", "tuft_id"], kind="stable").reset_index(drop=True)


def _circ_stats(deg):
    r = np.radians(deg)
    c, s = np.cos(r).mean(), np.sin(r).mean()
    R = min(1.0, math.hypot(c, s))
    mean = math.degrees(math.atan2(s, c))
    spread = math.degrees(math.sqrt(-2.0 * math.log(max(R, 1e-9)))) if R > 1e-9 else 180.0
    return mean, min(spread, 180.0)


def summary_frame(results, setup, history=None) -> pd.DataFrame:
    """One row per tuft: where it is, which way the air pushes it, and how steady it is."""
    setup = setup or default_setup()
    h = history if history is not None else history_frame(results, setup)
    if h.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    locs = TuftLocations.from_dict(setup.get("locations"))
    n_frames = max(len(results["frames"]), 1)
    out = []
    for tid, g in h.groupby("tuft_id", sort=True):
        mean, spread = _circ_stats(g["angle_deg"].to_numpy())
        st = g["flow_state"].value_counts()
        is_img = results.get("kind") == "image"
        known = 0 if is_img else max(len(g) - int(st.get("learning", 0)), 0)

        def pct(name):
            return round(100.0 * int(st.get(name, 0)) / known, 1) if known else float("nan")
        shares = {k: int(st.get(k, 0)) for k in ("steady", "fluttering", "stalled")}
        main = max(shares, key=shares.get) if known else ("unmeasured" if results.get("kind") == "image" else "learning")
        u = locs.get(int(tid))
        src = "entered by you" if any(v is not None for v in u) else "automatic"
        if any(v is not None for v in u) and any(v is None for v in u):
            src = "partly entered"
        out.append({"tuft_id": int(tid), "x": g["x"].iloc[0], "y": g["y"].iloc[0], "z": g["z"].iloc[0],
                    "position_source": src, "seen_pct": round(100.0 * len(g) / n_frames, 1),
                    "first_seen_s": g["time_s"].min(), "last_seen_s": g["time_s"].max(),
                    "mean_angle_deg": round(mean, 1), "angle_spread_deg": round(spread, 1),
                    "mean_length": round(float(g["length"].mean()), 2),
                    "steady_pct": pct("steady"), "fluttering_pct": pct("fluttering"),
                    "stalled_pct": pct("stalled"), "main_state": main})
    return pd.DataFrame(out, columns=SUMMARY_COLUMNS)


def overview(results, setup):
    """Numbers for the headline cards and the 'flow health over time' chart."""
    fr, times = results["frames"], results["times"]
    series = []
    for t, f in zip(times, fr):
        c = [0, 0, 0, 0]
        for r in f:
            c[int(r[10])] += 1
        series.append([t, len(f)] + c)                     # time, total, learning, steady, fluttering, stalled
    ids = {r[0] for f in fr for r in f}
    allang = [r[7] for f in fr for r in f]
    mean = _circ_stats(np.asarray(allang))[0] if allang else 0.0
    cs = effective_cs(setup, results["height"])
    # bulk flow direction in the user's axes (average of every tuft vector)
    h = history_frame(results, setup)
    bulk = _circ_stats(h["angle_deg"].to_numpy())[0] if not h.empty else 0.0
    last = [0, 0, 0, 0]
    for s in series:
        for i in range(4):
            last[i] += s[2 + i]
    known = last[1] + last[2] + last[3]
    return {"series": series, "tufts": len(ids), "frames": len(fr), "duration": times[-1] if times else 0,
            "mean_angle_deg": round(bulk, 1), "units": cs.units,
            "steady_pct": round(100 * last[1] / known, 1) if known else None,
            "fluttering_pct": round(100 * last[2] / known, 1) if known else None,
            "stalled_pct": round(100 * last[3] / known, 1) if known else None}


def locations_frame(results, setup) -> pd.DataFrame:
    s = summary_frame(results, setup)
    return s[["tuft_id", "x", "y", "z", "position_source"]].copy() if not s.empty else s


def excel_bytes(results, setup, name="analysis") -> bytes:
    h = history_frame(results, setup)
    s = summary_frame(results, setup, h)
    cs = effective_cs(setup, results["height"])
    info = pd.DataFrame({"item": ["File", "Units", "Coordinate system", "Frames analysed", "Tufts found",
                                  "Detection"],
                         "value": [name, cs.units, "defined by you" if cs.defined else "pixels, origin bottom-left",
                                   len(results["frames"]), int(s.shape[0]), results.get("engine", "")]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        s.to_excel(xw, sheet_name="Tuft summary", index=False)
        h.drop(columns=["root_px_x", "root_px_y", "tip_px_x", "tip_px_y"]).to_excel(xw, sheet_name="Vector time history", index=False)
        info.to_excel(xw, sheet_name="About", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 34)
    return buf.getvalue()
