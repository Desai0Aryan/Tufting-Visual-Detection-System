"""Side-panel registry. A panel is a QWidget that receives frame and selection
updates; add one without touching the main window:

    from app.panels import Panel, register_panel

    @register_panel
    class StreamlinePanel(Panel):
        title, order = "Streamlines", 30
        def on_frame(self, result, index): ...
        def on_selection(self, track_ids): ...

and import its module in app/panels/builtin.py.
"""
from PySide6.QtWidgets import QWidget

_REGISTRY = []


class Panel(QWidget):
    title = "Panel"
    order = 10

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state   # AppState: store, selection, fps, ...

    def on_frame(self, result, index):
        pass

    def on_selection(self, track_ids):
        pass

    def on_reset(self):
        pass


def register_panel(cls):
    _REGISTRY.append(cls)
    return cls


def panels():
    from . import builtin  # noqa: F401  (registers the built-in panels)
    return sorted(_REGISTRY, key=lambda c: c.order)
