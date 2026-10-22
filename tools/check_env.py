"""Verify the training environment before anything else.

    python -m tools.check_env

Exits non-zero if CUDA is unavailable, the torch build lacks the GPU's
architecture (RTX 5090 = sm_120 needs CUDA 12.8+), or OpenCV has no GUI
(the label-review tool needs cv2.imshow).
"""
import sys


def main():
    ok = True
    import torch
    print(f"torch {torch.__version__}  CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False (CPU-only build or driver problem)")
        return 1
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    print(f"GPU {name}  capability {major}.{minor}  build archs {torch.cuda.get_arch_list()}")
    if arch not in torch.cuda.get_arch_list():
        print(f"FAIL: this torch build has no {arch} kernels; install the cu128 build")
        ok = False
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    print(f"matmul on GPU ok: {bool(torch.isfinite(x @ x).all())}")

    import cv2
    gui = any(k in cv2.getBuildInformation() for k in ("Win32 UI", "GTK", "Cocoa", "QT"))
    print(f"opencv {cv2.__version__}  GUI {'yes' if gui else 'NO'}")
    if not gui:
        print("FAIL: headless OpenCV; run: pip uninstall -y opencv-python opencv-python-headless "
              "&& pip install --no-deps opencv-python==5.0.0.93")
        ok = False

    for mod in ("segmentation_models_pytorch", "albumentations", "yaml"):
        m = __import__(mod)
        print(f"{mod} {getattr(m, '__version__', '?')}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
