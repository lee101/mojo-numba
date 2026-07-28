"""A small Numba-compatible type vocabulary for supported signatures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Signature:
    return_type: "Type"
    args: tuple["Type", ...]

    def __str__(self) -> str:
        return f"{self.return_type}({', '.join(map(str, self.args))})"


@dataclass(frozen=True)
class Type:
    name: str
    dtype: np.dtype | None
    ndim: int = 0

    def __getitem__(self, item) -> "Type":
        dims = item if isinstance(item, tuple) else (item,)
        if not all(isinstance(dim, slice) and dim == slice(None) for dim in dims):
            raise TypeError("array types use ':' dimensions, for example float64[:]")
        return Type(self.name, self.dtype, len(dims))

    def __call__(self, *args: "Type") -> Signature:
        return Signature(self, tuple(args))

    def __str__(self) -> str:
        return self.name + "[:]" * self.ndim


float64 = Type("float64", np.dtype(np.float64))
float32 = Type("float32", np.dtype(np.float32))
int64 = Type("int64", np.dtype(np.int64))
int32 = Type("int32", np.dtype(np.int32))
uint8 = Type("uint8", np.dtype(np.uint8))
boolean = Type("boolean", np.dtype(np.bool_))
bool_ = boolean
void = Type("void", None)


def parse_signature(value) -> Signature | None:
    if value is None:
        return None
    if isinstance(value, Signature):
        return value
    if isinstance(value, (list, tuple)):
        return parse_signature(value[0]) if value else None
    if not isinstance(value, str):
        raise TypeError(f"unsupported signature {value!r}")
    namespace = {
        "float64": float64,
        "float32": float32,
        "int64": int64,
        "int32": int32,
        "uint8": uint8,
        "boolean": boolean,
        "bool_": boolean,
        "void": void,
    }
    try:
        parsed = eval(value, {"__builtins__": {}}, namespace)
    except Exception as exc:
        raise TypeError(f"invalid signature {value!r}") from exc
    if not isinstance(parsed, Signature):
        raise TypeError(f"invalid signature {value!r}")
    return parsed
