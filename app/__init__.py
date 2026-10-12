"""TuftView desktop app (PySide6). UI only: processing lives in core/."""
import sys
from pathlib import Path


def resource_path(relative):
    """Path to a bundled resource; works from source and from a PyInstaller bundle."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative
