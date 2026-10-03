import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labeling import list_samples, load_sample, save_sample  # noqa: E402
from src.review import ReviewSession  # noqa: E402

WIN = "review labels"


def draw(session, scale=1.0):
    out = session.image.copy()
    for i, t in enumerate(session.tufts):
        if not session.keep[i]:
            c = (0, 0, 255)
            cv2.drawContours(out, [t.contour.astype(np.int32)], -1, c, 1)
            continue
        if t.path is not None:
            cv2.polylines(out, [np.round(t.path).astype(np.int32).reshape(-1, 1, 2)],
                          False, (255, 160, 0), 1, cv2.LINE_AA)
        r = tuple(int(round(v)) for v in t.root)
        e = tuple(int(round(v)) for v in
                  np.array(t.root) + np.array(t.vector) * max(14.0, 0.7 * t.length_px))
        cv2.arrowedLine(out, r, e, (0, 220, 0), 1, tipLength=0.3)
        cv2.circle(out, r, 3, (0, 0, 255), -1)
    cv2.putText(out, f"kept {session.n_kept}/{len(session.tufts)}  [s]ave [n]ext [p]rev [u]ndo [q]uit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--all", action="store_true", help="also revisit already-reviewed samples")
    ap.add_argument("--scale", type=float, default=1.0, help="display scale (e.g. 0.7 for big tiles)")
    a = ap.parse_args(argv)

    meta_path = Path(a.data) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    stems = [s for s in list_samples(a.data) if a.all or not meta.get(s, {}).get("reviewed")]
    if not stems:
        print("nothing to review")
        return 0

    k, state = 0, {"drag": None, "session": None}

    def load(i):
        img, lab = load_sample(a.data, stems[i])
        state["session"] = ReviewSession(img, lab)

    def on_mouse(event, x, y, flags, _):
        s = state["session"]
        x, y = x / a.scale, y / a.scale
        if event == cv2.EVENT_LBUTTONDOWN:
            i = s.nearest(x, y)
            if i is not None:
                s.flip(i) if flags & cv2.EVENT_FLAG_SHIFTKEY else s.toggle(i)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["drag"] = (x, y)
        elif event == cv2.EVENT_RBUTTONUP and state["drag"] is not None:
            s.reject_box(*state["drag"], x, y)
            state["drag"] = None

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    load(k)
    while True:
        cv2.setWindowTitle(WIN, f"{stems[k]}  ({k + 1}/{len(stems)})")
        cv2.imshow(WIN, draw(state["session"], a.scale))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u"):
            state["session"].undo()
        elif key == ord("s"):
            s = state["session"]
            m = dict(meta.get(stems[k], {}), reviewed=True, kept=s.n_kept, rejected=len(s.tufts) - s.n_kept)
            save_sample(a.data, stems[k], s.image, s.render(), meta=m)
            meta[stems[k]] = m
            print(f"saved {stems[k]}: kept {s.n_kept}, rejected {len(s.tufts) - s.n_kept}")
            key = ord("n")
        if key == ord("n") and k < len(stems) - 1:
            k += 1
            load(k)
        elif key == ord("p") and k > 0:
            k -= 1
            load(k)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
