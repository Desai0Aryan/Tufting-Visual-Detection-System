"""Drawing helpers for annotated stills (and, in a later part, video frames)."""
import cv2
import numpy as np


def _conf_color(conf):
    """Green = trusted orientation, orange = ambiguous (BGR)."""
    return (0, 220, 0) if conf >= 0.5 else (0, 165, 255)


def draw_tufts(image, tufts, thickness=2, show_ids=True, show_path=True):
    """Return a copy of `image` with each tuft's centerline and flow arrow.

    The arrow starts at the root and points along the stored flow vector (so it
    is correct for every vector mode); its length follows the tuft's length.
    """
    out = image.copy()
    for t in tufts:
        color = _conf_color(t.confidence)
        root = np.array(t.root, dtype=float)
        if show_path and t.path is not None:
            cv2.polylines(out, [np.round(t.path).astype(np.int32).reshape(-1, 1, 2)],
                          False, (255, 160, 0), 1, cv2.LINE_AA)
        end = root + np.array(t.vector) * max(20.0, 0.8 * t.length_px)
        cv2.arrowedLine(out, tuple(int(round(v)) for v in root),
                        tuple(int(round(v)) for v in end), color, thickness, tipLength=0.3)
        cv2.circle(out, tuple(int(round(v)) for v in root), 3, (0, 0, 255), -1)
        if show_ids:
            cv2.putText(out, str(t.id), (int(root[0]) + 5, int(root[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return out
