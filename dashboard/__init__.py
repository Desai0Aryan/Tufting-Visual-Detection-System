"""TuftView dashboard: a local web app around the existing detection / tracking code.

    python -m dashboard            # opens http://127.0.0.1:8765 in the browser

Nothing in src/, core/, tools/ or app/ is changed by this package; it only calls them.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # core/, src/, detector.py live at the repo root
    sys.path.insert(0, str(ROOT))

__version__ = "1.0.0"
