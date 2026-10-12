"""Built-in overlays: mask, flow arrows, tuft IDs, stall colour, selection."""
import cv2
import numpy as np

from . import SELECT_BGR, Overlay, register_overlay


@register_overlay
class MaskOverlay(Overlay):
    name, label, default_on, order = "mask", "Tuft mask", False, 0

    def draw(self, img, result, ctx):
        m = result.mask()
        if m is None:
            return
        sel = m > 0
        img[sel] = (0.55 * img[sel] + 0.45 * np.array((255, 160, 40))).astype(np.uint8)


@register_overlay
class StallColorOverlay(Overlay):
    """Colours arrows and IDs by stall state (attached green, unsteady yellow,
    separated red, unknown grey) and rings each root."""
    name, label, default_on, order = "stall", "Stall colour", True, 5

    def draw(self, img, result, ctx):
        for t in result.visible():
            r = tuple(int(round(v)) for v in t.root)
            cv2.circle(img, r, ctx.px(6), ctx.color_for(t), ctx.px(2), cv2.LINE_AA)


@register_overlay
class ArrowOverlay(Overlay):
    """Centre-line plus a flow arrow along the tip tangent, starting at the tip."""
    name, label, default_on, order = "arrows", "Flow arrows", True, 10

    def draw(self, img, result, ctx):
        for t in result.visible():
            c = ctx.color_for(t)
            if t.path is not None and len(t.path) > 1:
                cv2.polylines(img, [np.round(t.path).astype(np.int32).reshape(-1, 1, 2)], False, c,
                              ctx.px(1.5), cv2.LINE_AA)
            tip = np.asarray(t.tip, float)
            end = tip + np.asarray(t.vector, float) * max(18.0 * ctx.line_scale, 0.5 * t.length_px)
            cv2.arrowedLine(img, tuple(int(v) for v in tip), tuple(int(v) for v in end), c, ctx.px(2),
                            cv2.LINE_AA, tipLength=0.35)


@register_overlay
class IdOverlay(Overlay):
    name, label, default_on, order = "ids", "Tuft IDs", True, 20

    def draw(self, img, result, ctx):
        fs = 0.45 * ctx.line_scale
        for t in result.visible():
            org = (int(t.root[0]) + ctx.px(7), int(t.root[1]) - ctx.px(7))
            cv2.putText(img, str(t.track_id), org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ctx.px(3), cv2.LINE_AA)
            cv2.putText(img, str(t.track_id), org, cv2.FONT_HERSHEY_SIMPLEX, fs, ctx.color_for(t), ctx.px(1), cv2.LINE_AA)


@register_overlay
class SelectionOverlay(Overlay):
    """Always on: highlights the tufts selected in the side panel."""
    name, label, toggleable, order = "selection", "Selection", False, 30

    def draw(self, img, result, ctx):
        for t in result.visible():
            if t.track_id not in ctx.selected:
                continue
            if t.path is not None and len(t.path) > 1:
                cv2.polylines(img, [np.round(t.path).astype(np.int32).reshape(-1, 1, 2)], False, SELECT_BGR,
                              ctx.px(4), cv2.LINE_AA)
            cv2.circle(img, tuple(int(round(v)) for v in t.root), ctx.px(10), SELECT_BGR, ctx.px(2), cv2.LINE_AA)
