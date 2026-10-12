"""TuftView main window. All processing lives in core/ and runs in a worker
thread; this module only displays results, handles input and starts jobs."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressDialog, QPushButton, QSlider, QStyle,
                               QTabWidget, QToolBar, QVBoxLayout, QWidget)

from core.sources import VIDEO_EXTS, VideoFileSource

from . import settings as settings_mod
from .overlays import OverlayContext, line_scale_for, overlays, render
from .panels import panels
from .selection import nearest_tuft, update_selection
from .store import ResultStore
from .worker import ExportWorker, ProcessingWorker, start_in_thread


@dataclass
class AppState:
    store: ResultStore = field(default_factory=ResultStore)
    selected: set = field(default_factory=set)
    fps: float = 30.0
    video: str = ""


# ---------------------------------------------------------------- video view
class VideoView(QWidget):
    clicked = Signal(float, float, bool)   # image x, image y, ctrl held

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.image, self.target = None, QRect()
        self.message = "Open a video (File > Open) or drop one here"

    def set_frame(self, rgb):
        h, w = rgb.shape[:2]
        self.image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self.image is None:
            p.setPen(Qt.lightGray)
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        s = min(self.width() / self.image.width(), self.height() / self.image.height())
        w, h = int(self.image.width() * s), int(self.image.height() * s)
        self.target = QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(self.target, self.image)

    def mousePressEvent(self, e):
        if self.image is None or not self.target.contains(e.position().toPoint()):
            return
        s = self.image.width() / self.target.width()
        x = (e.position().x() - self.target.x()) * s
        y = (e.position().y() - self.target.y()) * s
        self.clicked.emit(x, y, bool(e.modifiers() & Qt.ControlModifier))


# ---------------------------------------------------------------- settings dialog
class SettingsDialog(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TuftView settings")
        self.cfg = cfg
        form = QFormLayout()
        self.backend = QComboBox()
        self.backend.addItems(["unet", "classical"])
        self.backend.setCurrentText(cfg.backend)
        self.ckpt = QLineEdit(cfg.checkpoint)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.ckpt, 1)
        row.addWidget(browse)
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0.0, 1.0)
        self.thr.setSingleStep(0.05)
        self.thr.setSpecialValueText("checkpoint default")
        self.thr.setValue(cfg.threshold)
        self.polarity = QComboBox()
        self.polarity.addItems(["light", "dark"])
        self.polarity.setCurrentText(cfg.polarity)
        self.minlen = QDoubleSpinBox()
        self.minlen.setRange(1, 2000)
        self.minlen.setValue(cfg.min_length_px)
        form.addRow("Detector backend", self.backend)
        form.addRow("U-Net checkpoint", row)
        form.addRow("Confidence threshold", self.thr)
        form.addRow("Classical: tufts are", self.polarity)
        form.addRow("Min tuft length (px)", self.minlen)
        self.ov_boxes = {}
        for ov in overlays():
            if ov.toggleable:
                cb = QCheckBox(ov.label)
                cb.setChecked(ov.name in cfg.overlays)
                self.ov_boxes[ov.name] = cb
                form.addRow("Overlay" if len(self.ov_boxes) == 1 else "", cb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(bb)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "U-Net checkpoint", self.ckpt.text(), "PyTorch checkpoint (*.pt)")
        if p:
            self.ckpt.setText(p)

    def result_settings(self):
        c = self.cfg
        c.backend, c.checkpoint = self.backend.currentText(), self.ckpt.text().strip()
        c.threshold, c.polarity, c.min_length_px = self.thr.value(), self.polarity.currentText(), self.minlen.value()
        c.overlays = [n for n, cb in self.ov_boxes.items() if cb.isChecked()]
        return c


# ---------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]

    def __init__(self, qsettings=None):
        super().__init__()
        self.qsettings = qsettings
        self.cfg = settings_mod.load(qsettings)
        self.state = AppState()
        self.src = None
        self.worker = self.thread = None
        self.export_job = None
        self.index, self.playing, self.device = 0, False, "-"
        self.setWindowTitle("TuftView")
        self.setAcceptDrops(True)
        self.resize(1500, 900)

        self.view = VideoView()
        self.view.clicked.connect(self.on_click)
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.clicked.connect(self.toggle_play)
        back = QPushButton()
        back.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        back.clicked.connect(lambda: self.step(-1))
        fwd = QPushButton()
        fwd.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        fwd.clicked.connect(lambda: self.step(1))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_scrub)
        self.speed = QComboBox()
        self.speed.addItems([f"{s:g}x" for s in self.SPEEDS])
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self._retime)
        self.frame_lbl = QLabel("0 / 0")
        controls = QHBoxLayout()
        for w in (back, self.play_btn, fwd):
            controls.addWidget(w)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.frame_lbl)
        controls.addWidget(QLabel("speed"))
        controls.addWidget(self.speed)
        central = QWidget()
        v = QVBoxLayout(central)
        v.addWidget(self.view, 1)
        v.addLayout(controls)
        self.setCentralWidget(central)

        # panels (registered modules)
        self.tabs = QTabWidget()
        self.panels = [cls(self.state) for cls in panels()]
        for pnl in self.panels:
            self.tabs.addTab(pnl, pnl.title)
        dock = QDockWidget("Tufts", self)
        dock.setWidget(self.tabs)
        dock.setMinimumWidth(420)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self._build_menus()
        self.st_fps, self.st_frame, self.st_tufts, self.st_dev = (QLabel("-") for _ in range(4))
        for w in (self.st_frame, self.st_tufts, self.st_fps, self.st_dev):
            self.statusBar().addPermanentWidget(w)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._update_status()

    # ------------------------------------------------------------ menus
    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        for text, key, fn in (("&Open video...", QKeySequence.Open, self.open_dialog),
                              ("&Export...", "Ctrl+E", self.export_dialog),
                              ("&Settings...", "Ctrl+,", self.settings_dialog),
                              ("&Quit", QKeySequence.Quit, self.close)):
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(fn)
            m.addAction(a)
        view = self.menuBar().addMenu("&View")
        tb = QToolBar("Overlays")
        self.addToolBar(tb)
        self.ov_actions = {}
        for ov in overlays():
            if not ov.toggleable:
                continue
            a = QAction(ov.label, self, checkable=True)
            a.setChecked(ov.name in self.cfg.overlays)
            a.toggled.connect(self._overlays_changed)
            view.addAction(a)
            tb.addAction(a)
            self.ov_actions[ov.name] = a
        a = QAction("Play / pause", self)
        a.setShortcut(Qt.Key_Space)
        a.triggered.connect(self.toggle_play)
        self.addAction(a)
        for key, d in ((Qt.Key_Right, 1), (Qt.Key_Left, -1)):
            a = QAction(self)
            a.setShortcut(key)
            a.triggered.connect(lambda _=False, d=d: self.step(d))
            self.addAction(a)

    def enabled_overlays(self):
        return {n for n, a in self.ov_actions.items() if a.isChecked()}

    def _overlays_changed(self):
        self.cfg.overlays = sorted(self.enabled_overlays())
        settings_mod.save(self.cfg, self.qsettings)
        self.show_frame(self.index)

    # ------------------------------------------------------------ opening and processing
    def dragEnterEvent(self, e):
        if any(Path(u.toLocalFile()).suffix.lower() in VIDEO_EXTS for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if Path(u.toLocalFile()).suffix.lower() in VIDEO_EXTS:
                self.open_video(u.toLocalFile())
                return

    def open_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open video", self.cfg.last_dir,
                                           "Video (" + " ".join(f"*{e}" for e in VIDEO_EXTS) + ")")
        if p:
            self.open_video(p)

    def open_video(self, path):
        self.stop_processing()
        if self.src:
            self.src.close()
        try:
            self.src = VideoFileSource(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "TuftView", str(e))
            return
        self.cfg.last_dir = str(Path(path).parent)
        settings_mod.save(self.cfg, self.qsettings)
        self.state.video, self.state.fps = str(path), self.src.fps
        self.state.store.clear(self.src.frame_count)
        self.state.selected = set()
        for p in self.panels:
            p.on_reset()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self.src.frame_count - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.setWindowTitle(f"TuftView - {Path(path).name}")
        self.index = 0
        self.show_frame(0)
        self.start_processing()

    def start_processing(self):
        if not self.state.video:
            return
        if self.cfg.backend == "unet" and not Path(self.cfg.checkpoint).is_file():
            QMessageBox.warning(self, "TuftView", "The U-Net backend needs a checkpoint. "
                                "Choose one in File > Settings, or switch to the classical backend.")
            return
        self.worker = ProcessingWorker(self.state.video, self.cfg.pipeline_dict(), self.state.store)
        self.worker.progress.connect(self.on_progress)
        self.worker.started_ok.connect(self.on_started)
        self.worker.failed.connect(lambda m: QMessageBox.critical(self, "Processing failed", m))
        self.thread = start_in_thread(self.worker, self)

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait(10000)
        self.worker = self.thread = None

    def on_started(self, device):
        self.device = device
        self._update_status()

    def on_progress(self, index, fps):
        self.proc_fps = fps
        self.st_fps.setText(f"processing {fps:.1f} fps ({index + 1}/{self.state.store.total})")
        if index >= self.index and self.state.store.get(self.index) is not None:
            self.show_frame(self.index)  # the frame on screen just got its result

    # ------------------------------------------------------------ playback
    def toggle_play(self):
        self.playing = not self.playing and self.src is not None
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause if self.playing else QStyle.SP_MediaPlay))
        self._retime()

    def _retime(self):
        if self.playing:
            self.timer.start(int(1000 / (self.state.fps * self.SPEEDS[self.speed.currentIndex()])))
        else:
            self.timer.stop()

    def _tick(self):
        nxt = self.index + 1
        if nxt >= self.state.store.total:
            self.toggle_play()
            return
        if nxt <= self.state.store.frontier:
            self.show_frame(nxt)
        # else: wait for processing to catch up

    def step(self, d):
        if self.src:
            self.show_frame(int(np.clip(self.index + d, 0, self.state.store.total - 1)))

    def on_scrub(self, value):
        if self.src and value != self.index:
            self.show_frame(value)

    def show_frame(self, i):
        if self.src is None:
            return
        frame = self.src.read_at(i)
        if frame is None:
            return
        self.index = i
        result = self.state.store.get(i)
        ctx = OverlayContext(enabled=self.enabled_overlays(), selected=self.state.selected,
                             line_scale=line_scale_for(frame.shape))
        img = render(frame, result, ctx)
        self.view.set_frame(np.ascontiguousarray(img[..., ::-1]))
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        for p in self.panels:
            p.on_frame(result, i)
        self._update_status(result)

    def _update_status(self, result=None):
        total = self.state.store.total
        self.frame_lbl.setText(f"{self.index + 1} / {total}")
        self.st_frame.setText(f"frame {self.index}" + ("" if result else " (not processed yet)"))
        self.st_tufts.setText(f"tufts {len(result.visible()) if result else '-'}")
        self.st_dev.setText(f"{self.cfg.backend} on {self.device}")

    # ------------------------------------------------------------ selection (view only)
    def on_click(self, x, y, ctrl):
        result = self.state.store.get(self.index)
        if result is None:
            return
        scale = line_scale_for(result.shape)
        hit = nearest_tuft(result.visible(), x, y, max_dist=25 * scale)
        self.state.selected = update_selection(self.state.selected, hit, ctrl)
        for p in self.panels:
            p.on_selection(self.state.selected)
        self.show_frame(self.index)

    # ------------------------------------------------------------ settings and export
    def settings_dialog(self):
        before = self.cfg.pipeline_dict()
        dlg = SettingsDialog(settings_mod.AppSettings(**vars(self.cfg)), self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.cfg = dlg.result_settings()
        settings_mod.save(self.cfg, self.qsettings)
        for n, a in self.ov_actions.items():
            a.blockSignals(True)
            a.setChecked(n in self.cfg.overlays)
            a.blockSignals(False)
        if self.cfg.pipeline_dict() != before and self.state.video:
            self.stop_processing()
            self.state.store.clear()
            self.start_processing()
        self.show_frame(self.index)

    def export_dialog(self):
        store = self.state.store
        if store.frontier < 0:
            QMessageBox.information(self, "Export", "Nothing processed yet.")
            return
        if store.frontier < store.total - 1:
            ok = QMessageBox.question(self, "Export", f"Only frames 0-{store.frontier} of {store.total} are "
                                      "processed. Export those now?")
            if ok != QMessageBox.Yes:
                return
        out = QFileDialog.getExistingDirectory(self, "Export to folder", self.cfg.last_dir)
        if not out:
            return
        results = store.ordered()
        job = ExportWorker(self.state.video, results, out, sorted(self.enabled_overlays()),
                           sorted(self.state.selected))
        dlg = QProgressDialog("Writing annotated video and CSV files...", "Cancel", 0, len(results), self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.canceled.connect(job.cancel)
        job.progress.connect(lambda k, n: dlg.setValue(k))
        job.done.connect(lambda paths: (dlg.close(), QMessageBox.information(
            self, "Export finished", "\n".join(f"{k}: {v}" for k, v in paths.items()))))
        job.failed.connect(lambda m: (dlg.close(), QMessageBox.critical(self, "Export failed", m)))
        self.export_job = (job, start_in_thread(job, self))

    def closeEvent(self, e):
        self.timer.stop()
        self.stop_processing()
        if self.export_job:
            self.export_job[0].cancel()
            self.export_job[1].quit()
            self.export_job[1].wait(10000)
        if self.src:
            self.src.close()
        super().closeEvent(e)
