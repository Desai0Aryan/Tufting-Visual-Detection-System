"""Vote look-alike tuft cards a whole grid at a time (votes -> tools.build_from_votes).

    python -m tools.vote_groups                        # train + val clips
    python -m tools.vote_groups --splits train --per-group 30

Unvoted cards from data/real/proposed are grouped by appearance (src/card_groups.py)
and shown up to 48 per page, the most typical first, the odd ones in the last rows.

  click a cell   its own vote: none -> n (red) -> y (green) -> u (grey) -> none
  y / n / u      every cell without its own vote gets this vote; next group
  s  skip group (stays unvoted)    z  undo last page    q  quit

Merged tufts ("too close"): outline only on yarn (two touching strings) -> y,
pixels are right; outline also covers background between them -> u.

Votes go to data/real/votes.jsonl like tools.vote_tufts (same format, plus
mode/group), so both tools can be mixed. Features are cached in
data/real/card_features.npz; only new cards are embedded.
"""
import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.card_groups import GridPage, combine, crop_for_embedding, crop_square, embed, load_encoder, \
    make_groups, shape_features  # noqa: E402
from src.labeling import list_samples  # noqa: E402
from src.splits import load_split_map  # noqa: E402
from src.tuft_cards import append_vote, load_votes  # noqa: E402
from tools.vote_tufts import collect_cards  # noqa: E402

WIN = "tuft groups"
COLS, ROWS, CELL = 8, 6, 130
BAR = 30
OUTLINE = (60, 220, 60)
BORDER = {"n": (0, 0, 255), "y": (60, 220, 60), "u": (160, 160, 160)}


class FrameCache:
    def __init__(self, data, size=16):
        self.data, self.size, self.d = Path(data), size, OrderedDict()

    def __call__(self, stem):
        if stem not in self.d:
            self.d[stem] = cv2.imread(str(self.data / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
            if len(self.d) > self.size:
                self.d.popitem(last=False)
        self.d.move_to_end(stem)
        return self.d[stem]


def features(cards, data, cache_path, device="cuda"):
    """(cnn (N, C), shape (N, 4)) for cards, embedding only those not in the cache."""
    cache_path = Path(cache_path)
    have = {}
    if cache_path.exists():
        z = np.load(cache_path)
        have = {cid: (c, s) for cid, c, s in zip(z["ids"], z["cnn"], z["shape"])}
    todo = [c for c in cards if c.id not in have]
    if todo:
        print(f"embedding {len(todo)} cards ...")
        frames, enc = FrameCache(data, 2), load_encoder(device)
        todo.sort(key=lambda c: c.stem)          # each frame decoded once
        t0 = time.time()
        for i in range(0, len(todo), 512):
            chunk = todo[i:i + 512]
            crops, shapes = [], []
            for c in chunk:
                img = frames(c.stem)
                crops.append(crop_for_embedding(img, c))
                shapes.append(shape_features(img, c))
            for c, f, s in zip(chunk, embed(np.stack(crops), enc, device), shapes):
                have[c.id] = (f, s)
            print(f"\r{min(i + 512, len(todo))}/{len(todo)}  {time.time() - t0:.0f} s", end="", flush=True)
        print()
        ids = np.array(list(have))
        np.savez(cache_path, ids=ids, cnn=np.stack([have[i][0] for i in ids]),
                 shape=np.stack([have[i][1] for i in ids]))
    return (np.stack([have[c.id][0] for c in cards]), np.stack([have[c.id][1] for c in cards]))


def render_page(cards, page, frames, status=""):
    canvas = np.zeros((ROWS * CELL + BAR, COLS * CELL, 3), np.uint8)
    for i, c in enumerate(cards):
        img = frames(c.stem)
        x0, y0, x1, y1 = crop_square(c.bbox, img.shape, factor=2.5, min_side=80)
        s = CELL / max(x1 - x0, y1 - y0)
        cell = cv2.resize(img[y0:y1, x0:x1], (CELL, CELL),
                          interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        sx, sy = CELL / (x1 - x0), CELL / (y1 - y0)
        cs = [np.round((k - [x0, y0]) * [sx, sy]).astype(np.int32) for k in c.contours]
        cv2.drawContours(cell, cs, -1, OUTLINE, 1, cv2.LINE_AA)
        own = page.own[i]
        if own:
            cv2.rectangle(cell, (1, 1), (CELL - 2, CELL - 2), BORDER[own], 4)
            cv2.putText(cell, own, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BORDER[own], 2, cv2.LINE_AA)
        r, k = divmod(i, COLS)
        canvas[r * CELL:(r + 1) * CELL, k * CELL:(k + 1) * CELL] = cell
        cv2.rectangle(canvas, (k * CELL, r * CELL), ((k + 1) * CELL - 1, (r + 1) * CELL - 1), (40, 40, 40), 1)
    cv2.putText(canvas, status, (8, ROWS * CELL + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1,
                cv2.LINE_AA)
    return canvas


class GroupApp:
    def __init__(self, data, cards, groups, log):
        self.cards, self.groups, self.log = cards, groups, Path(log)
        self.frames = FrameCache(data)
        self.g, self.history, self.n_session = 0, [], 0
        self.left = sum(len(g) for g in groups)
        self.page = None

    def status(self):
        g = self.groups[self.g]
        return (f"group {self.g + 1}/{len(self.groups)} ({len(g)} cards) | {self.left} cards left | "
                f"{self.n_session} voted   click: own vote   y/n/u: rest + next   s skip  z undo  q quit")

    def show(self):
        idx = self.groups[self.g]
        if self.page is None:
            self.page = GridPage([self.cards[i].id for i in idx])
        cv2.imshow(WIN, render_page([self.cards[i] for i in idx], self.page, self.frames, self.status()))

    def on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and self.page and y < ROWS * CELL:
            self.page.click((y // CELL) * COLS + x // CELL)
            self.show()

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        t0 = time.time()
        while self.g < len(self.groups):
            self.show()
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("y"), ord("n"), ord("u")):
                votes = self.page.commit(chr(key))
                secs = (time.time() - t0) / len(votes)
                for cid, v in votes.items():
                    append_vote(self.log, cid, v, secs, mode="group", group=self.g)
                self.history.append(self.g)
                self.n_session += len(votes)
                self.left -= len(votes)
                self.g, self.page, t0 = self.g + 1, None, time.time()
            elif key == ord("s"):
                self.left -= len(self.groups[self.g])
                self.g, self.page, t0 = self.g + 1, None, time.time()
            elif key == ord("z") and self.history:
                back = self.history.pop()
                for i in self.groups[back]:
                    append_vote(self.log, self.cards[i].id, None)
                self.n_session -= len(self.groups[back])
                self.left += sum(len(self.groups[j]) for j in range(back, self.g))
                self.g, self.page, t0 = back, None, time.time()
        cv2.destroyAllWindows()
        print(f"{self.n_session} votes this session, {len(load_votes(self.log))} in {self.log}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/real/proposed")
    ap.add_argument("--log", default="data/real/votes.jsonl")
    ap.add_argument("--cache", default="data/real/card_features.npz")
    ap.add_argument("--sources-csv", default="sources.csv")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--per-group", type=int, default=40, help="average group size before splitting")
    ap.add_argument("--shape-weight", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    meta = json.loads((Path(a.data) / "meta.json").read_text())
    split_map = load_split_map(a.sources_csv)
    stems = [s for s in list_samples(a.data)
             if split_map.get(meta.get(s, {}).get("clip_id")) in set(a.splits)
             and (not a.clips or meta[s]["clip_id"] in set(a.clips))]
    voted = load_votes(a.log)
    cards = [c for c in collect_cards(a.data, stems) if c.id not in voted]
    print(f"{len(stems)} frames, {len(cards)} unvoted cards ({len(voted)} already voted)")
    if not cards:
        return 0
    cnn, shape = features(cards, a.data, a.cache, a.device)
    groups = make_groups(combine(cnn, shape, a.shape_weight), per_group=a.per_group,
                         max_size=COLS * ROWS, seed=a.seed)
    print(f"{len(groups)} groups")
    GroupApp(a.data, cards, groups, a.log).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
