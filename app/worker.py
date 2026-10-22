"""Background workers (QThread). The UI thread never runs detection or export."""
import time

from PySide6.QtCore import QObject, QThread, Signal

from core import Pipeline, PipelineConfig, VideoFileSource


class ProcessingWorker(QObject):
    """Runs the pipeline over every frame in order and fills the ResultStore."""
    progress = Signal(int, float)        # last processed index, processing fps
    started_ok = Signal(str)             # device description
    finished = Signal()
    failed = Signal(str)

    def __init__(self, video_path, settings_dict, store):
        super().__init__()
        self.video_path, self.cfg, self.store = video_path, settings_dict, store
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            src = VideoFileSource(self.video_path)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
            return
        try:
            pipe = Pipeline(PipelineConfig.from_dict(self.cfg), fps=src.fps)
            self.started_ok.emit(pipe.device)
            t_last, n_last = time.perf_counter(), 0
            for i, frame in src:
                if self._stop:
                    break
                self.store.put(pipe.process_frame(frame, i))
                n_last += 1
                now = time.perf_counter()
                if now - t_last >= 0.5 or i == src.frame_count - 1:
                    self.progress.emit(i, n_last / (now - t_last))
                    t_last, n_last = now, 0
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            src.close()
            self.finished.emit()


class ExportWorker(QObject):
    progress = Signal(int, int)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, video_path, results, out_dir, enabled, selected):
        super().__init__()
        self.args = (video_path, results, out_dir, enabled, selected)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from .exporter import export_all
        try:
            v, res, out, en, sel = self.args
            paths = export_all(v, res, out, en, sel, progress=lambda k, n: (self.progress.emit(k, n),
                                                                            not self._cancel)[1])
            self.done.emit({k: str(p) for k, p in paths.items()})
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


def start_in_thread(worker, parent=None):
    """Move `worker` to a new QThread, run worker.run, clean up when it finishes."""
    th = QThread(parent)
    worker.moveToThread(th)
    th.started.connect(worker.run)
    for sig in ("finished", "done", "failed"):
        if hasattr(worker, sig):
            getattr(worker, sig).connect(th.quit)
    th.start()
    return th
