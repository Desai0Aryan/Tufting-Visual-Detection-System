"""Per-tuft yes/no votes on proposals (no GUI code, so it is testable).

A proposed frame (tools.auto_label output) holds hundreds of candidates, too
many to review frame by frame. Each candidate becomes a card instead:
    <stem>#s<k>   k-th connected component of the SAM 2 mask (accepted tufts)
    <stem>#r<k>   k-th SAM-rejected classical candidate (json rejected_candidates)
and gets one vote: y (tuft), n (not a tuft), u (unsure / bad mask).

Votes become a training label (build_frame):
    y                         tuft
    n                         background (hard negative)
    u, not voted, mask pixels
    belonging to no card      ignore, dilated (no tuft loss)
    everything else           background
Candidates are never guessed: a tuft nobody voted on cannot teach "background".

Votes are an append-only JSONL log; the last record per card wins, and a
record with vote null clears it (undo).
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VOTES = ("y", "n", "u")
MIN_AREA = 15  # px; smaller SAM fragments get no card (they end up ignore)


@dataclass
class Card:
    id: str
    stem: str
    kind: str            # "sam" or "rejected"
    bbox: tuple          # x, y, w, h in frame pixels
    contours: list       # [int32 (N, 1, 2)], frame pixels


def _sam_components(mask):
    n, cc, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return n, cc, stats


def frame_cards(stem, mask, info, min_area=MIN_AREA):
    """Cards of one proposed frame, in a stable order."""
    cards = []
    n, cc, stats = _sam_components(mask)
    k = 0
    for lab in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[lab])
        if area < min_area:
            continue
        comp = (cc[y:y + h, x:x + w] == lab).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cards.append(Card(f"{stem}#s{k}", stem, "sam", (x, y, w, h), [c + [x, y] for c in cs]))
        k += 1
    for i, r in enumerate(info.get("rejected_candidates", [])):
        c = np.asarray(r["contour"], np.int32).reshape(-1, 1, 2)
        if len(c) < 3:
            continue
        cards.append(Card(f"{stem}#r{i}", stem, "rejected", tuple(int(v) for v in cv2.boundingRect(c)), [c]))
    return cards


def card_mask(card, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(m, card.contours, -1, 255, cv2.FILLED)
    return m > 0


def sam_component_masks(mask, min_area=MIN_AREA):
    """{card suffix 's<k>': bool mask}, pixel-exact (contour fill would lose holes)."""
    n, cc, stats = _sam_components(mask)
    out, k = {}, 0
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < min_area:
            continue
        out[f"s{k}"] = cc == lab
        k += 1
    return out


def build_frame(stem, mask, info, votes, dilate=5, min_area=MIN_AREA):
    """-> (tuft bool, ignore bool, counts {y, n, u, unvoted})."""
    shape = mask.shape[:2]
    tuft, neg, unsure = (np.zeros(shape, bool) for _ in range(3))
    sam = sam_component_masks(mask, min_area)
    counts = {"y": 0, "n": 0, "u": 0, "unvoted": 0}
    for c in frame_cards(stem, mask, info, min_area):
        m = sam[c.id.split("#")[1]] if c.kind == "sam" else card_mask(c, shape)
        v = votes.get(c.id)
        counts[v if v in VOTES else "unvoted"] += 1
        {"y": tuft, "n": neg}.get(v, unsure)[m] = True
    unsure |= (mask > 0) & ~tuft & ~neg   # fragments without a card
    if dilate > 0 and unsure.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        unsure = cv2.dilate(unsure.astype(np.uint8), k) > 0
    return tuft, unsure & ~tuft & ~neg, counts


# ---------------------------------------------------------------- vote log
def load_votes(path):
    """{card id: vote} from the log; the last record per card wins."""
    votes, p = {}, Path(path)
    if not p.exists():
        return votes
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("vote") in VOTES:
            votes[r["card"]] = r["vote"]
        else:
            votes.pop(r["card"], None)
    return votes


def append_vote(path, card_id, vote, secs=None, **extra):
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card_id, "vote": vote}
    if secs is not None:
        rec["secs"] = round(float(secs), 2)
    rec.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def stems_with_votes(votes):
    return sorted({cid.split("#")[0] for cid, v in votes.items() if v in ("y", "n")})
