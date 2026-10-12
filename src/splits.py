"""Train / val / test assignment BY CLIP (video or recording session).

Every real sample must carry a `clip_id` in its dataset's meta.json, and every
clip must appear exactly once in sources.csv with split in {train, val, test}.
A whole clip goes to one split, so frames of the same video can never leak
between training and evaluation. Missing information is an error, never a
silent fallback to a random or hash-based split.
"""
import csv
import json
from pathlib import Path

SPLITS = ("train", "val", "test")
SOURCES_COLUMNS = ["clip_id", "url", "title", "uploader", "license", "date_downloaded",
                   "split", "use", "origin", "resolution", "notes"]


def load_split_map(sources_csv):
    """sources.csv -> {clip_id: split}. Rejects duplicates and unknown splits."""
    out = {}
    with open(sources_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid, split = row["clip_id"].strip(), row["split"].strip().lower()
            if not cid:
                continue
            if split not in SPLITS:
                raise ValueError(f"{sources_csv}: clip {cid!r} has split {split!r}, expected one of {SPLITS}")
            if cid in out:
                raise ValueError(f"{sources_csv}: clip {cid!r} is listed twice")
            out[cid] = split
    return out


def read_meta(root):
    mp = Path(root) / "meta.json"
    return json.loads(mp.read_text()) if mp.exists() else {}


def split_real(root, stems, split_map):
    """Assign the samples of one real dataset folder. Returns {split: [stem]}."""
    meta = read_meta(root)
    out = {s: [] for s in SPLITS}
    missing, unknown = [], set()
    for stem in stems:
        cid = meta.get(stem, {}).get("clip_id")
        if not cid:
            missing.append(stem)
            continue
        if cid not in split_map:
            unknown.add(cid)
            continue
        out[split_map[cid]].append(stem)
    if missing:
        raise ValueError(f"{root}: {len(missing)} sample(s) have no clip_id in meta.json "
                         f"(e.g. {missing[:3]}); every real frame must name its clip")
    if unknown:
        raise ValueError(f"{root}: clip(s) {sorted(unknown)[:5]} are not in sources.csv")
    return out


def clips_by_split(roots, split_map):
    """{split: set(clip_id)} actually present in the given dataset folders."""
    out = {s: set() for s in SPLITS}
    for r in roots:
        for stem, m in read_meta(r).items():
            cid = m.get("clip_id")
            if cid in split_map:
                out[split_map[cid]].add(cid)
    return out
