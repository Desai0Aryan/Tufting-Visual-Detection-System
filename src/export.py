import pandas as pd

COLUMNS = [
    "timestamp_s", "frame", "tuft_id",
    "root_x", "root_y", "tip_x", "tip_y",
    "unit_vector_x", "unit_vector_y", "unit_vector_z",
    "angle_deg", "length_px", "curl", "confidence", "source",
]


def tufts_to_rows(tufts, frame=0, timestamp_s=0.0):
    return [
        {
            "timestamp_s": float(timestamp_s), "frame": int(frame), "tuft_id": t.id,
            "root_x": t.root[0], "root_y": t.root[1],
            "tip_x": t.tip[0], "tip_y": t.tip[1],
            "unit_vector_x": t.vector[0], "unit_vector_y": t.vector[1],
            "unit_vector_z": t.vector_z,
            "angle_deg": t.angle_deg, "length_px": t.length_px,
            "curl": t.curl, "confidence": t.confidence, "source": t.source,
        }
        for t in tufts
    ]


def tufts_dataframe(tufts, frame=0, timestamp_s=0.0):
    return pd.DataFrame(tufts_to_rows(tufts, frame, timestamp_s), columns=COLUMNS)
