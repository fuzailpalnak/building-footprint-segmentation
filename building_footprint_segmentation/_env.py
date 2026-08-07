import os
import sys


def configure_windows_openmp() -> None:
    """Avoid OpenMP duplicate-runtime crashes on Windows (PyTorch + MKL/OpenCV)."""
    if sys.platform == "win32":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
