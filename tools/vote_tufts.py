"""One tuft at a time: is this a tuft? (votes -> tools.build_from_votes)

    python -m tools.vote_tufts                                  # train + val clips
    python -m tools.vote_tufts --splits train --clips yt_FcL8hJFZftU

Each card is one candidate from data/real/proposed (SAM 2 tuft or SAM-rejected
classical candidate, shown alike so the source does not bias the vote):
left the plain crop, middle the same crop with its outline, right the whole
frame with a box where it is.

  y  tuft            n  not a tuft        u  unsure / tuft but bad outline
  s  skip (stays unvoted)   z  undo last vote   q  quit

Cards come in a shuffled order across clips and frames, so a short session
already covers every clip. Votes are appended to data/real/votes.jsonl (with
seconds per vote); quitting and restarting continues where you left off.
"""
import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labeling import list_samples  # noqa: E402
from src.splits import load_split_map  # noqa: E402
from src.tuft_cards import append_vote, frame_cards, load_votes  # noqa: E402

WIN = "is this a tuft?"
PANEL, THUMB_W = 480, 360
OUTLINE, BOX = (60, 220, 60), (0, 200, 255)
KEYS = {ord("y"): "y", ord("n"): "n", ord("u"): "u"}


def load_mask(data, stem):
    lab = cv2.imread(str(Path(data) / "labels" / f"{stem}.png"), cv2.IMREAD_COLOR)
    return (lab[..., 0] > 127).astype(np.uint8) * 255


def collect_cards(data, stems, seed=0):
    cards = []
    for stem in stems:
        p = Path(data) / "json" / f"{stem}.json"
        info = json.loads(p.read_text()) if p.exists() else {}
        cards += frame_cards(stem, load_mask(data, stem), info)
    random.Random(seed).shuffle(cards)
    return cards


def crop_box(bbox, shape, factor=3.0, min_side=96):
    x, y, w, h = bbox
    H, W = shape[:2]
    side = int(min(max(min_side, factor * max(w, h)), max(H, W)))
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(round(cx - side / 2), 0, max(W - side, 0)))
    y0 = int(np.clip(round(cy - side / 2), 0, max(H - side, 0)))
    return x0, y0, min(x0 + side, W), min(y0 + side, H)


def render_card(image, card, status=""):
    x0, y0, x1, y1 = crop_box(card.bbox, image.shape)
    crop = image[y0:y1, x0:x1]
    s = min(PANEL / crop.shape[1], PANEL / crop.shape[0], 8.0)
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    plain = cv2.resize(crop, None, fx=s, fy=s, interpolation=interp)
    outlined = plain.copy()
    # outline drawn after upscaling so it stays 1 px and does not hide the yarn
    cs = [np.round((c - [x0, y0]) * s + s / 2).astype(np.int32) for c in card.contours]
    cv2.drawContours(outlined, cs, -1, OUTLINE, 1, cv2.LINE_AA)
    H, W = image.shape[:2]
    ts = THUMB_W / W
    thumb = cv2.resize(image, (THUMB_W, int(round(H * ts))), interpolation=cv2.INTER_AREA)
    bx, by, bw, bh = card.bbox
    cv2.rectangle(thumb, (int(bx * ts) - 4, int(by * ts) - 4),
                  (int((bx + bw) * ts) + 4, int((by + bh) * ts) + 4), BOX, 2)
    canvas = np.zeros((PANEL + 34, 2 * PANEL + THUMB_W + 20, 3), np.uint8)
    canvas[:plain.shape[0], :plain.shape[1]] = plain
    canvas[:outlined.shape[0], PANEL + 10:PANEL + 10 + outlined.shape[1]] = outlined
    canvas[:thumb.shape[0], 2 * PANEL + 20:] = thumb
    cv2.putText(canvas, status, (8, PANEL + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


class VoteApp:
    def __init__(self, data, cards, log, cache=6):
        self.data, self.cards, self.log = Path(data), cards, Path(log)
        self.votes = load_votes(log)
        self.queue = [c for c in cards if c.id not in self.votes]
        self.k, self.history, self.cache, self.cache_n = 0, [], OrderedDict(), cache
        self.session = 0

    def image(self, stem):
        if stem not in self.cache:
            self.cache[stem] = cv2.imread(str(self.data / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
            if len(self.cache) > self.cache_n:
                self.cache.popitem(last=False)
        self.cache.move_to_end(stem)
        return self.cache[stem]

    def status(self, card):
        done = len(self.votes)
        return (f"{done} voted ({self.session} this session) | {len(self.queue) - self.k} left | "
                f"{card.stem}   y tuft  n no  u unsure  s skip  z undo  q quit")

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        while self.k < len(self.queue):
            card = self.queue[self.k]
            cv2.imshow(WIN, render_card(self.image(card.stem), card, self.status(card)))
            cv2.waitKey(1)
            if self.k + 1 < len(self.queue):           # decode the next frame while you look
                self.image(self.queue[self.k + 1].stem)
            t0 = time.time()
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in KEYS:
                v = KEYS[key]
                append_vote(self.log, card.id, v, time.time() - t0, kind=card.kind)
                self.votes[card.id] = v
                self.history.append(self.k)
                self.session += 1
                self.k += 1
            elif key == ord("s"):
                self.k += 1
            elif key == ord("z") and self.history:
                self.k = self.history.pop()
                cid = self.queue[self.k].id
                append_vote(self.log, cid, None)
                self.votes.pop(cid, None)
                self.session -= 1
        cv2.destroyAllWindows()
        print(f"{self.session} votes this session, {len(self.votes)} in {self.log}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/real/proposed")
    ap.add_argument("--log", default="data/real/votes.jsonl")
    ap.add_argument("--sources-csv", default="sources.csv")
    ap.add_argument("--splits", nargs="*", default=["train", "val"],
                    help="clips of these splits (test frames are kept for evaluation)")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    meta = json.loads((Path(a.data) / "meta.json").read_text())
    split_map = load_split_map(a.sources_csv)
    stems = [s for s in list_samples(a.data)
             if split_map.get(meta.get(s, {}).get("clip_id")) in set(a.splits)
             and (not a.clips or meta[s]["clip_id"] in set(a.clips))]
    if not stems:
        print("no proposed frames match")
        return 1
    print(f"building cards from {len(stems)} frames ...")
    app = VoteApp(a.data, collect_cards(a.data, stems, a.seed), a.log)
    print(f"{len(app.cards)} cards, {len(app.cards) - len(app.queue)} already voted")
    if not app.queue:
        print("nothing left to vote on")
        return 0
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
