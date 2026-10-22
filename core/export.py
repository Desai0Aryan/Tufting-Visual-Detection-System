"""CSV export of pipeline results (no Qt). Every tracked tuft is written, with a
`state` column (tentative / confirmed / missing) to filter on. The annotated video is written by the
app, because it reuses the app's overlay modules."""
import csv

COLUMNS = ["frame", "time_s", "track_id", "state", "stall", "root_x", "root_y", "tip_x", "tip_y",
           "vector_x", "vector_y", "angle_deg", "length_px", "length_ratio", "circ_std_deg",
           "reversed", "confidence", "detected"]


def rows(results, track_ids=None, confirmed_only=False):
    keep = set(track_ids) if track_ids is not None else None
    for r in results:
        for t in r.tufts:
            if confirmed_only and t.state not in ("confirmed", "missing"):
                continue
            if keep is not None and t.track_id not in keep:
                continue
            yield {"frame": r.index, "time_s": round(r.time_s, 4), "track_id": t.track_id, "state": t.state,
                   "stall": t.stall, "root_x": round(t.root[0], 2), "root_y": round(t.root[1], 2),
                   "tip_x": round(t.tip[0], 2), "tip_y": round(t.tip[1], 2),
                   "vector_x": round(t.vector[0], 5), "vector_y": round(t.vector[1], 5),
                   "angle_deg": round(t.angle_deg, 3), "length_px": round(t.length_px, 2),
                   "length_ratio": round(t.length_ratio, 4) if t.length_ratio == t.length_ratio else "",
                   "circ_std_deg": round(t.circ_std_deg, 3) if t.circ_std_deg == t.circ_std_deg else "",
                   "reversed": int(t.reversed), "confidence": round(t.confidence, 3), "detected": int(t.detected)}


def write_csv(path, results, track_ids=None, confirmed_only=False):
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows(results, track_ids, confirmed_only):
            w.writerow(row)
            n += 1
    return n
