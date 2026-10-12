"""Overlay registry. An overlay draws onto a BGR frame (OpenCV), so the same code
renders the live view and the exported video.

Add one without touching the main window:

    from app.overlays import Overlay, register_overlay

    @register_overlay
    class StallHeatmap(Overlay):
        name, label, default_on, order = "heatmap", "Stall heatmap", False, 5
        def draw(self, img, result, ctx): ...

and import its module in app/overlays/builtin.py (or anywhere before the window opens).
"""
from dataclasses import dataclass, field

_REGISTRY = {}

STALL_BGR = {"attached": (60, 200, 60), "unsteady": (0, 215, 255), "separated": (40, 40, 230),
             "unknown": (170, 170, 170)}
NEUTRAL_BGR = (255, 220, 120)
SELECT_BGR = (255, 0, 255)


@dataclass
class OverlayContext:
    enabled: set                      # names of overlays switched on
    selected: set = field(default_factory=set)
    line_scale: float = 1.0           # ~1 at 720p; grows with resolution

    def color_for(self, tuft):
        if "stall" in self.enabled:
            return STALL_BGR.get(tuft.stall, STALL_BGR["unknown"])
        return NEUTRAL_BGR

    def px(self, v):
        return max(1, int(round(v * self.line_scale)))


class Overlay:
    name = ""
    label = ""
    default_on = True
    toggleable = True
    order = 10          # lower draws first

    def draw(self, img, result, ctx):
        raise NotImplementedError


def register_overlay(cls):
    if not cls.name:
        raise ValueError("overlay needs a name")
    _REGISTRY[cls.name] = cls()
    return cls


def overlays():
    """Registered overlays in drawing order."""
    from . import builtin  # noqa: F401  (registers the built-in overlays)
    return sorted(_REGISTRY.values(), key=lambda o: o.order)


def render(frame, result, ctx):
    """Copy of `frame` with every enabled overlay drawn (non-toggleable ones always)."""
    img = frame.copy()
    if result is None:
        return img
    for ov in overlays():
        if not ov.toggleable or ov.name in ctx.enabled:
            ov.draw(img, result, ctx)
    return img


def line_scale_for(shape):
    return max(1.0, min(shape[0], shape[1]) / 720.0)
