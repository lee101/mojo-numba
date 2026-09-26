"""ctypes access to the prebuilt Mojo execution engine."""

from __future__ import annotations

import ctypes
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

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
        _library.mn_nonlinear_f64.argtypes = [I, I, I, I]
        _library.mn_nonlinear_f64.restype = None
        _library.mn_matmul_square_f64.argtypes = [I, I, I, I, I, I]
        _library.mn_matmul_square_f64.restype = None
    return _library


def addr(value: np.ndarray) -> int:
    return value.ctypes.data


# Threading only pays above roughly 2 flops/byte, so the fan-out threshold is
# a total-flop count: memory-bound kernels never clear it. Element chunks are
# multiples of 8 doubles, which keeps every vector access exactly as aligned
# as the caller's base pointer.
PARALLEL_WORKERS = 4
PARALLEL_MIN_FLOPS = 1 << 24
_pool: ThreadPoolExecutor | None = None


def run_chunks(call, units: int, flops: int, granularity: int = 8) -> None:
    """Fan ``call(start, stop)`` out over a process-wide thread pool."""
    global _pool
    workers = min(PARALLEL_WORKERS, os.cpu_count() or 1)
    step = (units // workers // granularity) * granularity
    if workers <= 1 or step == 0 or flops < PARALLEL_MIN_FLOPS:
        call(0, units)
        return
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=workers)
    bounds = [(part * step, (part + 1) * step) for part in range(workers - 1)]
    bounds.append((bounds[-1][1], units))
    for future in [_pool.submit(call, lo, hi) for lo, hi in bounds]:
        future.result()
