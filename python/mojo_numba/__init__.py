"""A Mojo execution backend for a useful, typed subset of Numba's loop API."""

from __future__ import annotations

import numpy as np

from . import errors, types
from .compiler import Dispatcher, VectorizedDispatcher
from .types import boolean, bool_, float32, float64, int32, int64, uint8, void

__version__ = "0.1.0"


def jit(signature_or_function=None, **options):
    if callable(signature_or_function):
        return Dispatcher(signature_or_function, options=options)

    def decorate(function):
        return Dispatcher(function, signature=signature_or_function, options=options)

    return decorate


def njit(signature_or_function=None, **options):
    options = {"nopython": True, **options}
    return jit(signature_or_function, **options)


def vectorize(signatures=None, **options):
    def decorate(function):
        return VectorizedDispatcher(function, signatures=signatures, options=options)

    return decorate


def prange(*args):
    return range(*args)


def typeof(value):
    if isinstance(value, np.ndarray):
        mapping = {
            np.dtype(np.float64): float64,
            np.dtype(np.float32): float32,
            np.dtype(np.int64): int64,
            np.dtype(np.int32): int32,
            np.dtype(np.uint8): uint8,
            np.dtype(np.bool_): boolean,
        }
        if value.dtype not in mapping:
            raise ValueError(f"unsupported dtype {value.dtype}")
        base = mapping[value.dtype]
        return types.Type(base.name, base.dtype, value.ndim)
    if isinstance(value, (bool, np.bool_)):
        return boolean
    if isinstance(value, (int, np.integer)):
        return int64
    if isinstance(value, (float, np.floating)):
        return float64
    raise ValueError(f"unsupported value type {type(value).__name__}")


__all__ = [
    "Dispatcher",
    "boolean",
    "bool_",
    "errors",
    "float32",
    "float64",
    "int32",
    "int64",
    "jit",
    "njit",
    "prange",
    "typeof",
    "types",
    "uint8",
    "vectorize",
    "void",
]
