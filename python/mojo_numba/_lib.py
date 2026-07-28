"""ctypes access to the prebuilt Mojo execution engine."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-numba.so")

I = ctypes.c_int64
D = ctypes.c_double

_library: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build() -> str:
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            build()
        _library = ctypes.CDLL(LIB)
        _library.mn_execute.argtypes = [I] * 12
        _library.mn_execute.restype = I
        _library.mn_sum_squares_f64.argtypes = [I, I]
        _library.mn_sum_squares_f64.restype = D
        _library.mn_axpy_f64.argtypes = [I, I, I, I, D]
        _library.mn_axpy_f64.restype = None
        _library.mn_threshold_count_f64.argtypes = [I, I, D]
        _library.mn_threshold_count_f64.restype = I
        _library.mn_nonlinear_f64.argtypes = [I, I, I]
        _library.mn_nonlinear_f64.restype = None
        _library.mn_matmul_square_f64.argtypes = [I, I, I, I]
        _library.mn_matmul_square_f64.restype = None
    return _library


def addr(value: np.ndarray) -> int:
    return value.ctypes.data
