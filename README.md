# mojo-numba

`mojo-numba` is a small typed-loop compiler with a Mojo execution backend. It
implements a useful, deliberately bounded subset of Numba's Python API: write an
ordinary array loop, decorate it with `jit` or `njit`, and subsequent calls run
the loop inside the compiled Mojo shared library.

This is not a fork of Numba and does not use LLVM. It is a standalone experiment
in how much of Numba's most common CPU-loop workflow can be expressed through a
stable Python-to-Mojo FFI boundary.

## Covered subset

The parity suite proves these behaviors against the installed upstream Numba:

- `jit` and `njit`, lazy specialization, explicit string/type-object signatures,
  `inspect_types`, and `typeof`
- `range` and `prange`, nested loops, negative steps, loop `else`, `while`, `if`/`else`,
  `break`, and `continue`
- 1-D and 2-D C-contiguous `float64`, `float32`, `int64`, `int32`, `uint8`, and boolean
  arrays
- indexed loads and stores, in-place updates, scalar/array returns, and
  `np.empty_like`/`np.zeros_like` output allocation
- arithmetic, comparisons, short-circuit booleans, `len`, `.size`, `.shape`, `abs`,
  `min`, `max`, `sin`, `exp`, and `sqrt`
- `vectorize` for tested unary/binary scalar expressions, equal-shaped arrays, scalar
  broadcasting, and declared `float32` or boolean output

Unsupported syntax raises `mojo_numba.errors.UnsupportedError`; unsupported argument
types and layouts raise `TypingError`. Execution never silently falls back to Python.

This is a tested subset, not API or implementation coverage of upstream Numba. Not
covered are arbitrary Python, object mode, non-contiguous or rank-3+ arrays,
allocation shapes other than `*_like`, tuple returns, user-function calls from compiled
code, ufunc broadcasting between different array shapes, `guvectorize`, CUDA, Numba
extension APIs, or disk caching. Generic `prange` plans currently have serial semantics,
and options such as `parallel` and `fastmath` are accepted for source compatibility.
Recognized large independent loops use thresholded native parallel kernels. Float32
expressions execute in the Float64 VM stack and are cast on store, so their final
rounding can differ from upstream by an ulp. Integer expressions also use that stack and
are therefore exact only through `2**53`; this project is not suitable for general
full-range `int64` arithmetic.

## Install

The repository pins the Mojo nightly that its ABI was tested with.

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-numba.so`. Pixi activates `python/` on
`PYTHONPATH`, so no editable install is required.

## Usage

Save this as a Python file; compilation needs inspectable source.

```python
import numpy as np
import mojo_numba as numba

@numba.njit
def energy(x):
    total = 0.0
    for i in range(x.size):
        total += x[i] * x[i]
    return total

values = np.arange(1000, dtype=np.float64)
print(energy(values))
```

Run the checked-in version with:

```bash
pixi run python examples/quickstart.py
```

Explicit signatures and elementwise compilation use Numba's familiar spelling:

```python
@numba.njit("float64(float64[:])")
def sum_values(x):
    total = 0.0
    for i in range(x.size):
        total += x[i]
    return total

@numba.vectorize(["float64(float64, float64)"])
def fused(x, y):
    return x * y + x
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Python
3.13.14, and upstream Numba 0.66.0. Times are the best of five warmed calls; the
CPython column is one call. Compilation is excluded for both compiled systems.

| kernel | mojo-numba | upstream Numba | CPython | against upstream | over CPython |
| --- | ---: | ---: | ---: | ---: | ---: |
| sum of squares, 1M | 0.52 ms | 1.10 ms | 327.50 ms | 2.12x | 630.13x |
| AXPY allocation, 1M | 2.42 ms | 2.45 ms | 534.56 ms | 1.01x | 221.23x |
| threshold count, 1M | 0.29 ms | 0.26 ms | 223.19 ms | 0.90x | 763.10x |
| sin + exp transform, 1M | 23.31 ms | 26.56 ms | 606.58 ms | 1.14x | 26.02x |
| naive matmul, 80x80 | 0.28 ms | 0.46 ms | 401.88 ms | 1.63x | 1433.00x |

Ratios are upstream or CPython time divided by mojo-numba time, so values above one
favor mojo-numba. These are single-machine microbenchmarks, not general performance
claims. Canonical reductions, elementwise transforms, counts, and
square matrix multiplication are recognized once during lowering and dispatched to
prebuilt native SIMD kernels. Other supported programs continue to use the compact
typed VM, preserving the bounded language subset without per-function Mojo builds.

## How it works

At decoration time, mojo-numba retains the Python function and its inspectable AST.
The first call for a new argument signature assigns concrete array and scalar types,
checks the supported subset, recognizes supported native fast paths, and lowers control
flow and expressions to fixed-width typed instructions. Plans are cached in memory by
dtype, rank, and scalar type.

Python owns every allocation. NumPy arrays remain C-contiguous row-major buffers;
2-D indexing is lowered to `row * columns + column`. ctypes passes integer addresses,
explicit buffer lengths, and dtype/shape metadata to `dist/libmojo-numba.so`. Python
keeps every NumPy and metadata buffer referenced for the complete native call. The
Mojo entry point validates bytecode, stack/local indices, ranks, and array bounds
before dereferencing data pointers, then writes into caller-owned buffers. No Mojo
allocation crosses the ABI, so there is no cross-runtime ownership or deallocation
protocol.
Native fast paths pass the existing NumPy addresses directly and avoid the VM's
temporary metadata, local, stack, and return arrays. Matrix multiplication also writes
the complete output into `empty_like` storage, avoiding a redundant zero fill.

The test suite compares every supported behavior to the real upstream `numba`
package, not to a hand-written approximation.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

`src/kernels.mojo` is intentionally one compilation unit because shared-library build
cost is effectively fixed. The benchmark task takes a machine-wide file lock; run it
through Pixi so concurrent jobs do not distort results.

MIT licensed.
