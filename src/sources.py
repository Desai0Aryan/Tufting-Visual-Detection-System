"""sources.csv: one row per clip, the single place that records where footage
came from, its license, what it is used for and which split it belongs to.

Rules enforced here:
- every clip is assigned whole to train, val or test (never split);
- external clips must be Creative Commons, or carry an explicit permission note;
- clips below 720p are refused; heavily compressed clips are refused unless forced;
- `use` is one of: tufts (auto-label + review), background (synthetic
  backgrounds), negative (tuft-free surface -> empty-mask hard negatives);
- part of the test set must be our own footage (checked by `check`).
"""
import csv
import re
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from .splits import SOURCES_COLUMNS, SPLITS

USES = ("tufts", "background", "negative")
MIN_HEIGHT = 720
MIN_BPP = 0.02          # bits per pixel per frame; below this, compression is heavy
MAX_BLOCKINESS = 1.35   # 8x8 block-edge gradient ratio; above this, visible blocking


def read_sources(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_sources(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SOURCES_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SOURCES_COLUMNS})


def is_creative_commons(license_text):
    t = (license_text or "").lower()
    return "creative commons" in t or re.search(r"\bcc[- ]?by\b", t) is not None or t.startswith("cc0")


def video_stats(path, sample_frames=12):
    """Resolution, fps, duration, bits per pixel and a JPEG/H.264 blockiness score."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps if fps else 0.0
    size_bits = Path(path).stat().st_size * 8
    bpp = size_bits / (w * h * n) if w and h and n else 0.0
    scores = []
    for k in np.linspace(0, max(n - 1, 0), min(sample_frames, max(n, 1))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, frame = cap.read()
        if ok:
            scores.append(blockiness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    cap.release()
    return {"width": w, "height": h, "fps": round(fps, 3), "frames": n, "duration_s": round(dur, 2),
            "bpp": round(bpp, 4), "blockiness": round(float(np.median(scores)) if scores else 0.0, 3)}


def blockiness(gray):
    """Mean |horizontal gradient| on 8-px block boundaries / elsewhere.
    ~1.0 for clean video; grows with visible compression blocks."""
    g = np.abs(np.diff(gray.astype(np.float32), axis=1))
    cols = np.arange(g.shape[1])
    on = g[:, (cols % 8) == 7].mean()
    off = g[:, (cols % 8) != 7].mean()
    return float(on / max(off, 1e-6))


def quality_problems(stats, force=False):
    problems = []
    if stats["height"] < MIN_HEIGHT:
        problems.append(f"{stats['width']}x{stats['height']} is below {MIN_HEIGHT}p")
    if not force:
        if stats["bpp"] and stats["bpp"] < MIN_BPP:
            problems.append(f"bitrate {stats['bpp']} bits/pixel/frame < {MIN_BPP} (heavy compression)")
        if stats["blockiness"] > MAX_BLOCKINESS:
            problems.append(f"blockiness {stats['blockiness']} > {MAX_BLOCKINESS} (visible artifacts)")
    return problems


def make_row(clip_id, split, use, origin, url="", title="", uploader="", license_text="",
             resolution="", notes=""):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if use not in USES:
        raise ValueError(f"use must be one of {USES}")
    return {"clip_id": clip_id, "url": url, "title": title, "uploader": uploader,
            "license": license_text, "date_downloaded": date.today().isoformat(), "split": split,
            "use": use, "origin": origin, "resolution": resolution, "notes": notes}


def add_row(path, row):
    rows = read_sources(path)
    if any(r["clip_id"] == row["clip_id"] for r in rows):
        raise ValueError(f"clip {row['clip_id']!r} is already in {path}; one clip, one split")
    rows.append(row)
    write_sources(path, rows)


def check(path, clips_dir):
    """Returns (errors, summary)."""
    rows = read_sources(path)
    errors, summary = [], {s: {u: 0 for u in USES} for s in SPLITS}
    seen = set()
    for r in rows:
        cid = r["clip_id"]
        if cid in seen:
            errors.append(f"{cid}: listed twice")
        seen.add(cid)
        if r["split"] not in SPLITS:
            errors.append(f"{cid}: bad split {r['split']!r}")
            continue
        if r["use"] not in USES:
            errors.append(f"{cid}: bad use {r['use']!r}")
            continue
        summary[r["split"]][r["use"]] += 1
        if r["origin"] != "own" and not (is_creative_commons(r["license"]) or "permission" in r["license"].lower()):
            errors.append(f"{cid}: external clip without a Creative Commons license or permission note")
        if not list(Path(clips_dir).glob(f"{cid}.*")):
            errors.append(f"{cid}: no video file in {clips_dir}")
    if rows and not any(r["origin"] == "own" and r["split"] == "test" for r in rows):
        errors.append("test split has no clip of our own footage (origin=own)")
    return errors, summary
