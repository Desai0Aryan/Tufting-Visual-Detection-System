"""One-click start for TuftView. Used by Start-TuftView.command (Mac) and Start-TuftView.bat (Windows).

On the first run (and whenever a requirements file changes) it:
  1. makes a private Python environment in .venv/ (nothing is installed system-wide),
  2. installs everything the dashboard needs,
  3. installs everything the trained U-Net needs to run (PyTorch etc.; the NVIDIA build when a GPU is found),
then starts the dashboard. Later starts skip straight to step 4. Needs internet the first time only.
Standard library only, so any Python 3.10+ can run it.
"""
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
STAMP = VENV / ".tuftview-installed"
BASE = ["Requirements.txt", "Requirements-dashboard.txt"]
MODEL = "Requirements-model.txt"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def say(msg):
    print(f"\n  {msg}", flush=True)


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def has_nvidia():
    if platform.system() == "Darwin":
        return False
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def fingerprint(gpu):
    h = hashlib.sha256()
    for name in BASE + [MODEL]:
        p = ROOT / name
        h.update(p.read_bytes() if p.exists() else b"-")
    h.update(b"cuda" if gpu else b"default")
    h.update(sys.version.split()[0].encode())
    return h.hexdigest()


def pip(py, *args):
    return subprocess.run([str(py), "-m", "pip", "install", "--disable-pip-version-check", *args], cwd=ROOT).returncode == 0


def model_ready(py):
    code = "import torch, segmentation_models_pytorch, scipy"
    return subprocess.run([str(py), "-c", code], capture_output=True).returncode == 0


def ensure_env():
    gpu = has_nvidia()
    fp = fingerprint(gpu)
    py = venv_python()
    if py.exists() and STAMP.exists() and STAMP.read_text().strip() == fp:
        return py
    if not py.exists():
        say("Setting up TuftView for the first time. This takes a few minutes and needs internet.")
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    elif STAMP.exists():
        say("Updating TuftView's components...")
    say("Installing the dashboard...")
    if not pip(py, "--upgrade", "pip") and not py.exists():
        sys.exit("Could not set up Python. Please install Python 3.10 or newer from python.org and try again.")
    if not pip(py, *sum((["-r", f] for f in BASE), [])):
        sys.exit("The dashboard could not be installed. Check the internet connection and start it again.")
    if (ROOT / MODEL).exists():
        say("Installing the trained model's components (PyTorch). This is the big download...")
        args = ["-r", MODEL] + (["--extra-index-url", CUDA_INDEX] if gpu else [])
        ok = pip(py, *args) and model_ready(py)
        if not ok and gpu:
            say("The NVIDIA build did not install; trying the standard one...")
            ok = pip(py, "-r", MODEL) and model_ready(py)
        if not ok:
            say("The trained model's components could not be installed. TuftView will still start and use the "
                "classic detector. Start it again with internet to retry.")
            return py                           # no stamp: retry next time
    STAMP.write_text(fp)
    say("Everything is installed.")
    return py


def main():
    if sys.version_info < (3, 10):
        sys.exit("TuftView needs Python 3.10 or newer (python.org).")
    os.chdir(ROOT)
    py = ensure_env()
    say("Starting TuftView...")
    sys.exit(subprocess.call([str(py), "-m", "dashboard", *sys.argv[1:]], cwd=ROOT))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
