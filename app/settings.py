"""Persistent settings (QSettings: the Windows registry under PSU-ESM/TuftView)."""
from dataclasses import asdict, dataclass, field

from PySide6.QtCore import QSettings

ORG, APP = "PSU-ESM", "TuftView"


@dataclass
class AppSettings:
    backend: str = "unet"           # unet / classical
    checkpoint: str = ""
    threshold: float = 0.0          # 0 = use the checkpoint's own threshold
    polarity: str = "light"         # classical backend
    min_length_px: float = 15.0
    max_length_px: float = 400.0
    overlays: list = field(default_factory=lambda: ["arrows", "ids", "stall"])
    last_dir: str = ""

    def pipeline_dict(self):
        return {"backend": self.backend, "checkpoint": self.checkpoint, "threshold": self.threshold,
                "polarity": self.polarity, "min_length_px": self.min_length_px,
                "max_length_px": self.max_length_px}


def load(settings=None):
    s = settings or QSettings(ORG, APP)
    out = AppSettings()
    for k, v in asdict(out).items():
        if s.contains(k):
            raw = s.value(k)
            if isinstance(v, list):
                out.__dict__[k] = [x for x in (raw if isinstance(raw, list) else str(raw).split(",")) if x]
            elif isinstance(v, float):
                out.__dict__[k] = float(raw)
            else:
                out.__dict__[k] = str(raw)
    return out


def save(cfg, settings=None):
    s = settings or QSettings(ORG, APP)
    for k, v in asdict(cfg).items():
        s.setValue(k, ",".join(v) if isinstance(v, list) else v)
    s.sync()
