"""Warmed execution benchmarks against upstream Numba and plain CPython."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
from numba import prange

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_numba as mnb  # noqa: E402
import numba  # noqa: E402


def sum_squares(x):
    total = 0.0
    for i in range(x.size):
        total += x[i] * x[i]
    return total


def axpy(x, y, alpha):
    result = np.empty_like(x)
    for i in range(x.size):
        result[i] = alpha * x[i] + y[i]
    return result


def threshold_count(x, limit):
    count = 0
    for i in prange(x.size):
        if x[i] > limit:
            count += 1
    return count


def nonlinear(x):
    result = np.empty_like(x)
    for i in range(x.size):
        value = x[i]
        result[i] = math.sin(value) + math.exp(-abs(value))
    return result


def matmul_square(a, b):
    result = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[1]):
                result[i, j] += a[i, k] * b[k, j]
    return result


def best_time(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def milliseconds(seconds):
    return f"{seconds * 1000:.2f} ms"


def machine():
    model = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return model or platform.machine()


def gpu_free_mib():
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
        return max(int(line.strip()) for line in output.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def main():
    rng = np.random.default_rng(42)
    x = np.ascontiguousarray(rng.normal(size=1_000_000))
    y = np.ascontiguousarray(rng.normal(size=1_000_000))
    a = np.ascontiguousarray(rng.normal(size=(80, 80)))
    b = np.ascontiguousarray(rng.normal(size=(80, 80)))

    cases = [
        ("sum of squares, 1M", sum_squares, (x,)),
        ("AXPY allocation, 1M", axpy, (x, y, 0.75)),
        ("threshold count, 1M", threshold_count, (x, 0.2)),
        ("sin + exp transform, 1M", nonlinear, (x,)),
        ("naive matmul, 80x80", matmul_square, (a, b)),
    ]

    rows = []
    for name, function, arguments in cases:
        mojo_function = mnb.njit(function)
        numba_function = numba.njit(function)
        mojo_function(*arguments)
        numba_function(*arguments)
        mojo_time = best_time(lambda: mojo_function(*arguments))
        numba_time = best_time(lambda: numba_function(*arguments))
        python_time = best_time(lambda: function(*arguments), repeat=1)
        rows.append((name, mojo_time, numba_time, python_time))

    print(f"Machine: {machine()}; Python {platform.python_version()}; Numba {numba.__version__}")
    print()
    print("| kernel | mojo-numba | upstream Numba | CPython | vs Numba | vs CPython |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, mojo_time, numba_time, python_time in rows:
        print(
            f"| {name} | {milliseconds(mojo_time)} | {milliseconds(numba_time)} | "
            f"{milliseconds(python_time)} | {numba_time / mojo_time:.2f}x | "
            f"{python_time / mojo_time:.2f}x |"
        )

    free_mib = gpu_free_mib()
    if free_mib < 4000:
        print(f"\nGPU benchmark skipped: {free_mib} MiB free (requires 4000 MiB).")
        return
    from mojo_numba import _gpu

    if not _gpu.available():
        print("\nGPU benchmark skipped: no usable GPU runtime.")
        return
    size = 384
    ga = np.ascontiguousarray(rng.normal(size=(size, size)))
    gb = np.ascontiguousarray(rng.normal(size=(size, size)))
    mojo_cpu = mnb.njit(matmul_square)
    mojo_gpu = mnb.njit(matmul_square, device="gpu")
    upstream = numba.njit(matmul_square)
    cpu_result = mojo_cpu(ga, gb)
    gpu_result = mojo_gpu(ga, gb)
    upstream(ga, gb)
    if not np.allclose(gpu_result, cpu_result, rtol=1e-12, atol=1e-12):
        raise RuntimeError("GPU matmul parity failed")
    cpu_time = best_time(lambda: mojo_cpu(ga, gb))
    gpu_time = best_time(lambda: mojo_gpu(ga, gb))
    upstream_time = best_time(lambda: upstream(ga, gb))
    print("\n| GPU kernel | mojo-numba GPU | mojo-numba CPU | upstream Numba | vs CPU |")
    print("| --- | ---: | ---: | ---: | ---: |")
    print(
        f"| naive matmul, {size}x{size} | {milliseconds(gpu_time)} | "
        f"{milliseconds(cpu_time)} | {milliseconds(upstream_time)} | "
        f"{cpu_time / gpu_time:.2f}x |"
    )


if __name__ == "__main__":
    main()
