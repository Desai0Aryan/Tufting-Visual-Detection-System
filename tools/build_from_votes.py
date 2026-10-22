"""Votes from tools.vote_tufts -> training frames.

    python -m tools.build_from_votes
    python -m tools.train --config configs/unet_resnet34.yaml --set "data.real=[data/real/voted]"

Every proposed frame with at least one y/n vote is written to --out (rebuilt
from scratch each time): y = tuft, n = background, unsure / unvoted
candidates = ignore (dilated), see src/tuft_cards.py. A rejected candidate
voted y is labelled with the classical outline you saw on the card.
A contact sheet (green tuft, red voted-no, blue ignore) goes next to --out.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.label_io import load_frame, load_ignore, save_frame, tufts_from_mask  # noqa: E402
from src.tuft_cards import build_frame, load_votes, stems_with_votes  # noqa: E402


def overlay(image, tuft, ignore, neg=None):
    out = image.copy()
    out[ignore] = (0.5 * out[ignore] + [127, 60, 0]).astype(np.uint8)
    if neg is not None:
        out[neg] = (0, 0, 255)
    out[tuft] = (60, 220, 60)
    return out


def contact_sheet(tiles, path, cols=3, width=640):
    if not tiles:
        return
    rows = []
    small = [cv2.resize(t, (width, int(t.shape[0] * width / t.shape[1])), interpolation=cv2.INTER_AREA)
             for t in tiles]
    h = max(t.shape[0] for t in small)
    small = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in small]
    small += [np.zeros_like(small[0])] * (-len(small) % cols)
    for i in range(0, len(small), cols):
        rows.append(np.hstack(small[i:i + cols]))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/real/proposed")
    ap.add_argument("--votes", default="data/real/votes.jsonl")
    ap.add_argument("--out", default="data/real/voted")
    ap.add_argument("--dilate", type=int, default=5, help="px of ignore around unsure / unvoted candidates")
    ap.add_argument("--sheet-frames", type=int, default=12)
    ap.add_argument("--exclude-clips", nargs="*", default=[],
                    help="leave these clips out (e.g. tufts the candidates could not separate)")
    a = ap.parse_args(argv)

    votes = load_votes(a.votes)
    stems = [s for s in stems_with_votes(votes) if s.rsplit("__", 1)[0] not in set(a.exclude_clips)]
    if not stems:
        print(f"no y/n votes in {a.votes}")
        return 1
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    tiles, total = [], {"y": 0, "n": 0, "u": 0, "unvoted": 0}
    for i, stem in enumerate(stems):
        img, mask, info, meta = load_frame(a.data, stem)
        tuft, ignore, counts = build_frame(stem, mask, info, votes, dilate=a.dilate)
        prop_ignore = load_ignore(a.data, stem)   # e.g. tuft-coloured pixels no candidate covered
        if prop_ignore is not None:
            neg = build_frame(stem, mask, info, {c: "y" for c, v in votes.items()
                                                 if v == "n" and c.startswith(stem + "#")}, dilate=0)[0]
            ignore |= (prop_ignore > 0) & ~tuft & ~neg
        m = tuft.astype(np.uint8) * 255
        tufts = tufts_from_mask(m) if m.any() else []
        meta = {**meta, "source": "votes", "status": "voted", "domain": meta.get("domain", "own"),
                "votes": counts}
        save_frame(out, stem, img, m, tufts, meta, ignore=ignore)
        for k in total:
            total[k] += counts[k]
        if len(tiles) < a.sheet_frames:
            # voted-no pixels, for display only: the same cards with their n turned into y
            as_y = {c: "y" for c, v in votes.items() if v == "n" and c.startswith(stem + "#")}
            neg = build_frame(stem, mask, info, as_y, dilate=0)[0] if as_y else None
            tiles.append(overlay(img, tuft, ignore, neg))
        print(f"\r{i + 1}/{len(stems)} {stem}", end="", flush=True)
    print()
    sheet = out.parent / f"{out.name}_sheet.jpg"
    contact_sheet(tiles, sheet)
    print(f"{len(stems)} frames -> {out}  (votes used: {total['y']} tuft, {total['n']} not tuft, "
          f"{total['u']} unsure; {total['unvoted']} unvoted candidates ignored)")
    print(f"sheet: {sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
