"""python -m app [video]"""
import sys
from pathlib import Path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:  # core/, src/ and detector.py live next to app/
        sys.path.insert(0, str(root))
    from PySide6.QtWidgets import QApplication
    from app.main_window import MainWindow
    qapp = QApplication.instance() or QApplication(sys.argv[:1])
    qapp.setApplicationName("TuftView")
    win = MainWindow()
    win.show()
    if argv:
        win.open_video(argv[0])
    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main())
