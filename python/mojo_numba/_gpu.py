"""ctypes bridge to the optional GPU matrix kernel."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

from ._lib import ROOT, addr

LIB = os.path.join(ROOT, "dist", "libmojo-numba-gpu.so")
I = ctypes.c_int64
MIN_MATMUL_N = 256
MAX_MATMUL_N = 8192

_handle: ctypes.CDLL | None = None
_available: bool | None = None


def lib() -> ctypes.CDLL | None:
    global _handle
    if _handle is None:
        if not os.path.exists(LIB):
            return None
        try:
            _handle = ctypes.CDLL(LIB)
        except OSError:
            return None
        _handle.mn_gpu_available.argtypes = []
        _handle.mn_gpu_available.restype = I
        _handle.mn_gpu_matmul_square_f64.argtypes = [I] * 4
        _handle.mn_gpu_matmul_square_f64.restype = I
    return _handle


def available() -> bool:
    global _available
    if _available is None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            free_mib = max(
                int(line.strip()) for line in output.splitlines() if line.strip()
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            free_mib = 0
        handle = lib() if free_mib >= 4000 else None
        _available = bool(handle is not None and handle.mn_gpu_available())
    return _available


def matmul(a: np.ndarray, b: np.ndarray, result: np.ndarray) -> bool:
    if not MIN_MATMUL_N <= a.shape[0] <= MAX_MATMUL_N or not available():
        return False
    handle = lib()
    if handle is None:
        return False
    return handle.mn_gpu_matmul_square_f64(
        addr(a), addr(b), addr(result), a.shape[0]
    ) == 0
