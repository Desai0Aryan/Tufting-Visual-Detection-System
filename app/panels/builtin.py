"""Built-in panels: selected-tuft details with an angle-over-time plot, and a
frame summary of stall states."""
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from . import Panel, register_panel

# categorical palette (light mode), fixed order: selection colours follow the entity, not rank
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STALL_HEX = {"attached": "#3cc83c", "unsteady": "#ffd700", "separated": "#e62828", "unknown": "#aaaaaa"}


class AnglePlot(QWidget):
    """Angle (deg) over time for the selected tufts; breaks lines at the +-180 wrap."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.series, self.current, self.fps, self.window_s = {}, 0, 30.0, 10.0

    def set_data(self, series, current, fps):
        self.series, self.current, self.fps = series, current, fps or 30.0
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#fcfcfb"))
        l, r, t, b = 40, 8, 8, 22
        w, h = self.width() - l - r, self.height() - t - b
        span = int(self.window_s * self.fps)
        x0 = max(0, self.current - span)
        X = lambda f: l + (f - x0) / max(span, 1) * w
        Y = lambda a: t + (180 - a) / 360.0 * h
        p.setFont(QFont(self.font().family(), 8))
        for a in (-180, -90, 0, 90, 180):
            p.setPen(QPen(QColor("#e4e3df"), 1))
            p.drawLine(QPointF(l, Y(a)), QPointF(l + w, Y(a)))
            p.setPen(QColor("#52514e"))
            p.drawText(QPointF(4, Y(a) + 4), f"{a:+d}" if a else "0")
        p.drawText(QPointF(l, self.height() - 6), f"angle (deg), last {self.window_s:.0f} s")
        for k, (tid, pts) in enumerate(self.series.items()):
            p.setPen(QPen(QColor(SERIES[k % len(SERIES)]), 2))
            prev = None
            for f, a in pts:
                if f < x0:
                    continue
                cur = QPointF(X(f), Y(a))
                if prev is not None and abs(a - prev[1]) < 180 and f - prev[0] <= 3:
                    p.drawLine(prev[2], cur)
                prev = (f, a, cur)
        p.setPen(QPen(QColor("#52514e"), 1, Qt.DashLine))
        p.drawLine(QPointF(X(self.current), t), QPointF(X(self.current), t + h))
        p.end()


@register_panel
class SelectedTuftsPanel(Panel):
    title, order = "Selected tufts", 10
    cols = ["ID", "angle (deg)", "length (px)", "length ratio", "stall", "angle std (deg)"]

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        lay = QVBoxLayout(self)
        self.hint = QLabel("Click a tuft to select it; Ctrl+click to select several.")
        self.hint.setWordWrap(True)
        self.table = QTableWidget(0, len(self.cols))
        self.table.setHorizontalHeaderLabels(self.cols)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.plot = AnglePlot()
        lay.addWidget(self.hint)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.plot, 1)
        self.ids, self.index = [], 0

    def on_selection(self, track_ids):
        self.ids = sorted(track_ids)
        self._refresh()

    def on_frame(self, result, index):
        self.index = index
        self._refresh(result)

    def on_reset(self):
        self.ids = []
        self._refresh()

    def _refresh(self, result=None):
        result = result or self.state.store.get(self.index)
        self.table.setRowCount(len(self.ids))
        for row, tid in enumerate(self.ids):
            t = result.by_id(tid) if result else None
            vals = [str(tid)] + (["-"] * 5 if t is None else
                                 [f"{t.angle_deg:+.1f}", f"{t.length_px:.0f}",
                                  "-" if t.length_ratio != t.length_ratio else f"{t.length_ratio:.2f}",
                                  t.stall if t.detected else f"{t.stall} (missing)",
                                  "-" if t.circ_std_deg != t.circ_std_deg else f"{t.circ_std_deg:.1f}"])
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 0:
                    it.setForeground(QColor(SERIES[row % len(SERIES)]))
                if c == 4 and t is not None:
                    it.setBackground(QColor(STALL_HEX.get(t.stall, "#aaaaaa")))
                self.table.setItem(row, c, it)
        self.table.resizeColumnsToContents()
        series = {tid: self.state.store.series(tid, upto=self.index) for tid in self.ids}
        self.plot.set_data(series, self.index, self.state.fps)


@register_panel
class FrameSummaryPanel(Panel):
    title, order = "Frame summary", 20

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        lay = QVBoxLayout(self)
        self.label = QLabel("-")
        self.label.setTextFormat(Qt.RichText)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(self.label)

    def on_frame(self, result, index):
        if result is None:
            self.label.setText(f"Frame {index}: not processed yet")
            return
        vis = result.visible()
        counts = {k: sum(t.stall == k for t in vis) for k in STALL_HEX}
        rows = "".join(f"<tr><td><span style='color:{STALL_HEX[k]}'>&#9632;</span> {k}</td>"
                       f"<td align='right'>{counts[k]}</td></tr>" for k in STALL_HEX)
        self.label.setText(f"<b>Frame {index}</b> ({result.time_s:.2f} s)<br>"
                           f"tracked tufts: {len(vis)} &nbsp; detections: {result.detections}<br>"
                           f"processing: {result.process_ms:.0f} ms<br><table>{rows}</table>")
