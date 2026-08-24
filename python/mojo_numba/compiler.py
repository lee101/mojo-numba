"""AST lowering for the typed Python loop subset."""

from __future__ import annotations

import ast
import copy
import inspect
import math
import textwrap
from dataclasses import dataclass
from functools import update_wrapper
from typing import Any, Callable

import numpy as np

from . import errors
from . import _gpu
from ._lib import addr, lib
from .types import Signature, parse_signature


HALT = 0
PUSH_CONST = 1
LOAD_LOCAL = 2
STORE_LOCAL = 3
LOAD_ARRAY_1 = 4
STORE_ARRAY_1 = 5
LOAD_ARRAY_2 = 6
STORE_ARRAY_2 = 7
LOAD_SHAPE = 8
LOAD_SIZE = 9
ADD, SUB, MUL, DIV, FLOORDIV, MOD, POW, NEG, ABS = range(10, 19)
LT, LE, GT, GE, EQ, NE, NOT, AND, OR = range(20, 29)
SIN, COS, EXP, LOG, SQRT, TANH, MIN, MAX = range(30, 38)
JUMP, JUMP_IF_FALSE, POP, RETURN = range(40, 44)

_DTYPES = {
    np.dtype(np.float64): 0,
    np.dtype(np.float32): 1,
    np.dtype(np.int64): 2,
    np.dtype(np.int32): 3,
    np.dtype(np.uint8): 4,
    np.dtype(np.bool_): 4,
}

_BINOPS = {
    ast.Add: ADD,
    ast.Sub: SUB,
    ast.Mult: MUL,
    ast.Div: DIV,
    ast.FloorDiv: FLOORDIV,
    ast.Mod: MOD,
    ast.Pow: POW,
}
_CMPOPS = {
    ast.Lt: LT,
    ast.LtE: LE,
    ast.Gt: GT,
    ast.GtE: GE,
    ast.Eq: EQ,
    ast.NotEq: NE,
}
_CALLS = {
    "abs": ABS,
    "math.fabs": ABS,
    "np.abs": ABS,
    "numpy.abs": ABS,
    "math.sin": SIN,
    "np.sin": SIN,
    "numpy.sin": SIN,
    "math.cos": COS,
    "np.cos": COS,
    "numpy.cos": COS,
    "math.exp": EXP,
    "np.exp": EXP,
    "numpy.exp": EXP,
    "math.log": LOG,
    "np.log": LOG,
    "numpy.log": LOG,
    "math.sqrt": SQRT,
    "np.sqrt": SQRT,
    "numpy.sqrt": SQRT,
    "math.tanh": TANH,
    "np.tanh": TANH,
    "numpy.tanh": TANH,
    "min": MIN,
    "np.minimum": MIN,
    "numpy.minimum": MIN,
    "max": MAX,
    "np.maximum": MAX,
    "numpy.maximum": MAX,
}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _dotted(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _function_ast(function: Callable) -> ast.FunctionDef:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise errors.UnsupportedError(
            f"source is unavailable for {function.__name__}; define compiled functions in a module"
        ) from exc
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(node, ast.AsyncFunctionDef):
                raise errors.UnsupportedError("async functions are not supported")
            return node
    raise errors.UnsupportedError(f"could not find the source for {function.__name__}")


@dataclass
class Allocation:
    name: str
    like: str
    zeros: bool
    dtype: np.dtype | None = None


@dataclass
class ArrayEntry:
    name: str
    argument: str | None = None
    allocation: Allocation | None = None


@dataclass(frozen=True)
class FastPath:
    kind: str
    names: tuple[str, ...]


def _statements(node: ast.FunctionDef) -> list[ast.stmt]:
    return [
        statement
        for statement in node.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]


def _name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _number(node: ast.AST, value: float) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (bool, int, float))
        and float(node.value) == value
    )


def _indexed_array(node: ast.AST, index_names: tuple[str, ...]) -> str | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    indexes = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    if len(indexes) != len(index_names):
        return None
    if any(_name(index) != expected for index, expected in zip(indexes, index_names)):
        return None
    return node.value.id


def _loop_bound(loop: ast.For, index: str, array: str, dimension: int | None = None) -> bool:
    if _name(loop.target) != index or not isinstance(loop.iter, ast.Call):
        return False
    if _dotted(loop.iter.func) not in ("range", "prange", "numba.prange", "mojo_numba.prange"):
        return False
    if len(loop.iter.args) != 1 or loop.iter.keywords:
        return False
    bound = loop.iter.args[0]
    if dimension is None:
        if (
            isinstance(bound, ast.Attribute)
            and bound.attr == "size"
            and _name(bound.value) == array
        ):
            return True
        return (
            isinstance(bound, ast.Call)
            and _dotted(bound.func) == "len"
            and len(bound.args) == 1
            and _name(bound.args[0]) == array
            and not bound.keywords
        )
    return (
        isinstance(bound, ast.Subscript)
        and isinstance(bound.value, ast.Attribute)
        and bound.value.attr == "shape"
        and _name(bound.value.value) == array
        and _number(bound.slice, float(dimension))
    )


def _allocation(statement: ast.stmt, zeros: bool) -> tuple[str, str] | None:
    if (
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or not isinstance(statement.targets[0], ast.Name)
        or not isinstance(statement.value, ast.Call)
        or len(statement.value.args) != 1
        or statement.value.keywords
        or not isinstance(statement.value.args[0], ast.Name)
    ):
        return None
    expected = {"np.zeros_like", "numpy.zeros_like"} if zeros else {
        "np.empty_like",
        "numpy.empty_like",
    }
    if _dotted(statement.value.func) not in expected:
        return None
    return statement.targets[0].id, statement.value.args[0].id


def _return_name(statement: ast.stmt, name: str) -> bool:
    return isinstance(statement, ast.Return) and _name(statement.value) == name


def _detect_sum_squares(statements: list[ast.stmt]) -> FastPath | None:
    if len(statements) != 3:
        return None
    initial, loop, returned = statements
    if (
        not isinstance(initial, ast.Assign)
        or len(initial.targets) != 1
        or not isinstance(initial.targets[0], ast.Name)
        or not _number(initial.value, 0.0)
        or not isinstance(loop, ast.For)
        or len(loop.body) != 1
    ):
        return None
    total = initial.targets[0].id
    index = _name(loop.target)
    update = loop.body[0]
    if (
        index is None
        or not isinstance(update, ast.AugAssign)
        or _name(update.target) != total
        or not isinstance(update.op, ast.Add)
        or not isinstance(update.value, ast.BinOp)
        or not isinstance(update.value.op, ast.Mult)
    ):
        return None
    left = _indexed_array(update.value.left, (index,))
    right = _indexed_array(update.value.right, (index,))
    if left is None or left != right or not _loop_bound(loop, index, left):
        return None
    if not _return_name(returned, total):
        return None
    return FastPath("sum_squares", (left,))


def _axpy_parts(expression: ast.AST, index: str) -> tuple[str, str, str] | None:
    if not isinstance(expression, ast.BinOp) or not isinstance(expression.op, ast.Add):
        return None
    for product, addend in (
        (expression.left, expression.right),
        (expression.right, expression.left),
    ):
        y = _indexed_array(addend, (index,))
        if y is None or not isinstance(product, ast.BinOp) or not isinstance(product.op, ast.Mult):
            continue
        for scalar_node, array_node in (
            (product.left, product.right),
            (product.right, product.left),
        ):
            scalar = _name(scalar_node)
            x = _indexed_array(array_node, (index,))
            if scalar is not None and x is not None:
                return x, y, scalar
    return None


def _detect_axpy(statements: list[ast.stmt]) -> FastPath | None:
    if len(statements) != 3:
        return None
    allocation = _allocation(statements[0], zeros=False)
    loop = statements[1]
    if allocation is None or not isinstance(loop, ast.For) or len(loop.body) != 1:
        return None
    result, like = allocation
    index = _name(loop.target)
    store = loop.body[0]
    if (
        index is None
        or not isinstance(store, ast.Assign)
        or len(store.targets) != 1
        or _indexed_array(store.targets[0], (index,)) != result
    ):
        return None
    parts = _axpy_parts(store.value, index)
    if parts is None:
        return None
    x, y, scalar = parts
    if like != x or not _loop_bound(loop, index, x) or not _return_name(statements[2], result):
        return None
    return FastPath("axpy", (x, y, result, scalar))


def _detect_threshold(statements: list[ast.stmt]) -> FastPath | None:
    if len(statements) != 3:
        return None
    initial, loop, returned = statements
    if (
        not isinstance(initial, ast.Assign)
        or len(initial.targets) != 1
        or not isinstance(initial.targets[0], ast.Name)
        or not _number(initial.value, 0.0)
        or not isinstance(loop, ast.For)
        or len(loop.body) != 1
        or not isinstance(loop.body[0], ast.If)
    ):
        return None
    count = initial.targets[0].id
    index = _name(loop.target)
    branch = loop.body[0]
    if (
        index is None
        or branch.orelse
        or len(branch.body) != 1
        or not isinstance(branch.test, ast.Compare)
        or len(branch.test.ops) != 1
        or not isinstance(branch.test.ops[0], ast.Gt)
        or len(branch.test.comparators) != 1
    ):
        return None
    x = _indexed_array(branch.test.left, (index,))
    limit = _name(branch.test.comparators[0])
    update = branch.body[0]
    if (
        x is None
        or limit is None
        or not isinstance(update, ast.AugAssign)
        or _name(update.target) != count
        or not isinstance(update.op, ast.Add)
        or not _number(update.value, 1.0)
        or not _loop_bound(loop, index, x)
        or not _return_name(returned, count)
    ):
        return None
    return FastPath("threshold_count", (x, limit))


def _is_call(node: ast.AST, names: set[str], argument) -> bool:
    return (
        isinstance(node, ast.Call)
        and _dotted(node.func) in names
        and len(node.args) == 1
        and not node.keywords
        and argument(node.args[0])
    )


def _detect_nonlinear(statements: list[ast.stmt]) -> FastPath | None:
    if len(statements) != 3:
        return None
    allocation = _allocation(statements[0], zeros=False)
    loop = statements[1]
    if allocation is None or not isinstance(loop, ast.For) or len(loop.body) not in (1, 2):
        return None
    result, x = allocation
    index = _name(loop.target)
    if index is None:
        return None
    value_name = None
    store = loop.body[-1]
    if len(loop.body) == 2:
        alias = loop.body[0]
        if (
            not isinstance(alias, ast.Assign)
            or len(alias.targets) != 1
            or not isinstance(alias.targets[0], ast.Name)
            or _indexed_array(alias.value, (index,)) != x
        ):
            return None
        value_name = alias.targets[0].id
    if (
        not isinstance(store, ast.Assign)
        or len(store.targets) != 1
        or _indexed_array(store.targets[0], (index,)) != result
        or not isinstance(store.value, ast.BinOp)
        or not isinstance(store.value.op, ast.Add)
    ):
        return None

    def is_value(node: ast.AST) -> bool:
        return (
            value_name is not None and _name(node) == value_name
        ) or _indexed_array(node, (index,)) == x

    def is_sin(node: ast.AST) -> bool:
        return _is_call(node, {"math.sin", "np.sin", "numpy.sin"}, is_value)

    def is_negative_abs(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and _is_call(
                node.operand,
                {"abs", "math.fabs", "np.abs", "numpy.abs"},
                is_value,
            )
        )

    def is_exp(node: ast.AST) -> bool:
        return _is_call(node, {"math.exp", "np.exp", "numpy.exp"}, is_negative_abs)

    left, right = store.value.left, store.value.right
    if not ((is_sin(left) and is_exp(right)) or (is_sin(right) and is_exp(left))):
        return None
    if not _loop_bound(loop, index, x) or not _return_name(statements[2], result):
        return None
    return FastPath("nonlinear", (x, result))


def _detect_matmul(statements: list[ast.stmt]) -> FastPath | None:
    if len(statements) != 3:
        return None
    allocation = _allocation(statements[0], zeros=True)
    outer = statements[1]
    if allocation is None or not isinstance(outer, ast.For) or len(outer.body) != 1:
        return None
    result, a = allocation
    row = _name(outer.target)
    middle = outer.body[0]
    if row is None or not isinstance(middle, ast.For) or len(middle.body) != 1:
        return None
    col = _name(middle.target)
    inner = middle.body[0]
    if col is None or not isinstance(inner, ast.For) or len(inner.body) != 1:
        return None
    k = _name(inner.target)
    update = inner.body[0]
    if (
        k is None
        or not isinstance(update, ast.AugAssign)
        or _indexed_array(update.target, (row, col)) != result
        or not isinstance(update.op, ast.Add)
        or not isinstance(update.value, ast.BinOp)
        or not isinstance(update.value.op, ast.Mult)
        or _indexed_array(update.value.left, (row, k)) != a
    ):
        return None
    b = _indexed_array(update.value.right, (k, col))
    if (
        b is None
        or not _loop_bound(outer, row, a, 0)
        or not _loop_bound(middle, col, a, 1)
        or not _loop_bound(inner, k, a, 1)
        or not _return_name(statements[2], result)
    ):
        return None
    return FastPath("matmul_square", (a, b, result))


def _detect_fast_path(node: ast.FunctionDef) -> FastPath | None:
    statements = _statements(node)
    for detector in (
        _detect_sum_squares,
        _detect_axpy,
        _detect_threshold,
        _detect_nonlinear,
        _detect_matmul,
    ):
        fast_path = detector(statements)
        if fast_path is not None:
            return fast_path
    return None


@dataclass
class Plan:
    code: np.ndarray
    constants: np.ndarray
    arrays: list[ArrayEntry]
    local_count: int
    scalar_arguments: dict[str, int]
    allocations: list[Allocation]
    result_kind: str
    result_name: str | None
    mutated: set[str]
    fast_path: FastPath | None

    def execute(self, arguments: dict[str, Any], device: str = "cpu"):
        runtime: dict[str, np.ndarray] = {}
        for entry in self.arrays:
            if entry.argument is not None:
                value = arguments[entry.argument]
                if not isinstance(value, np.ndarray):
                    raise errors.TypingError(f"{entry.argument} is no longer an ndarray")
                runtime[entry.name] = _check_array(entry.argument, value)
            else:
                allocation = entry.allocation
                assert allocation is not None
                base = runtime.get(allocation.like)
                if base is None:
                    base = _check_array(allocation.like, arguments[allocation.like])
                    runtime[allocation.like] = base
                dtype = allocation.dtype or base.dtype
                skip_zero_fill = (
                    self.fast_path is not None
                    and self.fast_path.kind == "matmul_square"
                    and allocation.name == self.fast_path.names[2]
                )
                factory = np.zeros_like if allocation.zeros and not skip_zero_fill else np.empty_like
                runtime[entry.name] = factory(base, dtype=dtype)

        for name in self.mutated:
            if not runtime[name].flags.writeable:
                raise errors.TypingError(f"array {name!r} is read-only")

        used_fast_path, fast_result = self._execute_fast(runtime, arguments, device)
        if used_fast_path:
            return fast_result

        addresses = np.empty(max(1, len(self.arrays)), dtype=np.int64)
        metadata = np.zeros(max(1, len(self.arrays)) * 4, dtype=np.int64)
        for index, entry in enumerate(self.arrays):
            array = runtime[entry.name]
            addresses[index] = array.ctypes.data
            metadata[index * 4] = _DTYPES[array.dtype]
            metadata[index * 4 + 1] = array.ndim
            metadata[index * 4 + 2] = array.shape[0]
            metadata[index * 4 + 3] = array.shape[1] if array.ndim == 2 else 1

        local_values = np.zeros(max(1, self.local_count), dtype=np.float64)
        for name, slot in self.scalar_arguments.items():
            value = arguments[name]
            if not isinstance(value, (bool, int, float, np.number)):
                raise errors.TypingError(f"unsupported scalar argument {name}={value!r}")
            local_values[slot] = value

        stack = np.empty(256, dtype=np.float64)
        result = np.zeros(1, dtype=np.float64)
        status = lib().mn_execute(
            addr(self.code),
            self.code.size // 3,
            addr(self.constants),
            self.constants.size,
            addr(addresses),
            addr(metadata),
            len(self.arrays),
            addr(local_values),
            self.local_count,
            addr(stack),
            stack.size,
            addr(result),
        )
        if status:
            raise RuntimeError(f"Mojo executor rejected unsafe execution state ({status})")
        if self.result_kind == "array":
            assert self.result_name is not None
            return runtime[self.result_name]
        if self.result_kind == "int":
            return int(result[0])
        if self.result_kind == "bool":
            return bool(result[0])
        if self.result_kind == "float":
            return float(result[0])
        return None

    def _execute_fast(
        self, runtime: dict[str, np.ndarray], arguments: dict[str, Any], device: str
    ) -> tuple[bool, Any]:
        fast_path = self.fast_path
        if fast_path is None:
            return False, None
        native = lib()
        if fast_path.kind == "sum_squares":
            x = runtime[fast_path.names[0]]
            if x.dtype != np.float64 or x.ndim != 1:
                return False, None
            if x.size == 0:
                return True, 0.0
            return True, float(native.mn_sum_squares_f64(addr(x), x.size))
        if fast_path.kind == "axpy":
            x = runtime[fast_path.names[0]]
            y = runtime[fast_path.names[1]]
            result = runtime[fast_path.names[2]]
            if (
                x.dtype != np.float64
                or y.dtype != np.float64
                or result.dtype != np.float64
                or x.ndim != 1
                or x.shape != y.shape
            ):
                return False, None
            if x.size == 0:
                return True, result
            native.mn_axpy_f64(
                addr(x),
                addr(y),
                addr(result),
                x.size,
                float(arguments[fast_path.names[3]]),
            )
            return True, result
        if fast_path.kind == "threshold_count":
            x = runtime[fast_path.names[0]]
            if x.dtype != np.float64 or x.ndim != 1:
                return False, None
            if x.size == 0:
                return True, 0
            count = native.mn_threshold_count_f64(
                addr(x), x.size, float(arguments[fast_path.names[1]])
            )
            return True, int(count)
        if fast_path.kind == "nonlinear":
            x = runtime[fast_path.names[0]]
            result = runtime[fast_path.names[1]]
            if x.dtype != np.float64 or result.dtype != np.float64 or x.ndim != 1:
                return False, None
            if x.size == 0:
                return True, result
            native.mn_nonlinear_f64(addr(x), addr(result), x.size)
            return True, result
        if fast_path.kind == "matmul_square":
            a = runtime[fast_path.names[0]]
            b = runtime[fast_path.names[1]]
            result = runtime[fast_path.names[2]]
            if (
                a.dtype != np.float64
                or b.dtype != np.float64
                or result.dtype != np.float64
                or a.ndim != 2
                or b.ndim != 2
                or a.shape[0] != a.shape[1]
                or b.shape != a.shape
            ):
                return False, None
            if a.size == 0:
                return True, result
            if device == "gpu" and _gpu.matmul(a, b, result):
                return True, result
            native.mn_matmul_square_f64(addr(a), addr(b), addr(result), a.shape[0])
            return True, result
        return False, None


def _check_array(name: str, value: np.ndarray) -> np.ndarray:
    if value.dtype not in _DTYPES:
        raise errors.TypingError(
            f"{name} has dtype {value.dtype}; supported dtypes are float64, float32, "
            "int64, int32, uint8, and bool"
        )
    if value.ndim not in (1, 2):
        raise errors.TypingError(f"{name} has rank {value.ndim}; only 1-D and 2-D arrays are supported")
    if not value.flags.c_contiguous:
        raise errors.TypingError(f"{name} must be C-contiguous")
    return value


def _validate_declared_arguments(
    signature: Signature | None,
    arguments: dict[str, Any],
    *,
    vectorized: bool = False,
) -> None:
    if signature is None:
        return
    if len(signature.args) != len(arguments):
        raise errors.TypingError(
            f"declared signature expects {len(signature.args)} argument(s), "
            f"got {len(arguments)}"
        )
    for (name, value), declared in zip(arguments.items(), signature.args):
        if declared.dtype is None:
            raise errors.TypingError(f"argument {name!r} cannot have type {declared}")
        if isinstance(value, np.ndarray):
            expected_rank = value.ndim if vectorized else declared.ndim
            if (not vectorized and declared.ndim == 0) or value.ndim != expected_rank:
                raise errors.TypingError(
                    f"argument {name!r} has rank {value.ndim}, not declared rank "
                    f"{declared.ndim}"
                )
            if value.dtype != declared.dtype:
                raise errors.TypingError(
                    f"argument {name!r} has dtype {value.dtype}, not declared "
                    f"{declared.dtype}"
                )
        else:
            if declared.ndim:
                raise errors.TypingError(
                    f"argument {name!r} is scalar, not declared rank {declared.ndim}"
                )
            actual = np.asarray(value).dtype
            if actual != declared.dtype:
                raise errors.TypingError(
                    f"argument {name!r} has dtype {actual}, not declared {declared.dtype}"
                )


class Lowerer:
    def __init__(
        self,
        node: ast.FunctionDef,
        arguments: dict[str, Any],
        output_dtypes: dict[str, np.dtype] | None = None,
        flat_arrays: set[str] | None = None,
    ):
        self.node = node
        self.arguments = arguments
        self.output_dtypes = output_dtypes or {}
        self.flat_arrays = flat_arrays or set()
        self.code: list[list[int]] = []
        self.constants: list[float] = []
        self.constant_map: dict[float, int] = {}
        self.arrays: list[ArrayEntry] = []
        self.array_indexes: dict[str, int] = {}
        self.locals: dict[str, int] = {}
        self.scalar_arguments: dict[str, int] = {}
        self.allocations: list[Allocation] = []
        self.mutated: set[str] = set()
        self.integer_names: set[str] = set()
        self.loop_stack: list[dict[str, Any]] = []
        self.temporary_count = 0
        self.result_kind = "none"
        self.result_name: str | None = None

        for argument in node.args.args:
            name = argument.arg
            if name not in arguments:
                continue
            value = arguments[name]
            if isinstance(value, np.ndarray):
                _check_array(name, value)
                self._add_array(ArrayEntry(name, argument=name))
            elif isinstance(value, (bool, int, float, np.number)):
                slot = self._local(name)
                self.scalar_arguments[name] = slot
                if isinstance(value, (bool, int, np.integer)):
                    self.integer_names.add(name)
            else:
                raise errors.TypingError(f"unsupported argument {name}={value!r}")

    def lower(self) -> Plan:
        for statement in self.node.body:
            self.statement(statement)
        if not self.code or self.code[-1][0] not in (HALT, RETURN):
            self.emit(HALT)
        constants = np.asarray(self.constants or [0.0], dtype=np.float64)
        code = np.asarray(self.code, dtype=np.int64).reshape(-1)
        return Plan(
            code=code,
            constants=constants,
            arrays=self.arrays,
            local_count=len(self.locals),
            scalar_arguments=self.scalar_arguments,
            allocations=self.allocations,
            result_kind=self.result_kind,
            result_name=self.result_name,
            mutated=self.mutated,
            fast_path=_detect_fast_path(self.node),
        )

    def emit(self, op: int, a: int = 0, b: int = 0) -> int:
        self.code.append([op, a, b])
        return len(self.code) - 1

    def patch(self, instruction: int, target: int):
        self.code[instruction][1] = target

    def _constant(self, value: float) -> int:
        key = float(value)
        if key not in self.constant_map:
            self.constant_map[key] = len(self.constants)
            self.constants.append(key)
        return self.constant_map[key]

    def push(self, value: float):
        self.emit(PUSH_CONST, self._constant(value))

    def _local(self, name: str) -> int:
        if name not in self.locals:
            self.locals[name] = len(self.locals)
        return self.locals[name]

    def _temporary(self) -> int:
        name = f"__mn_temporary_{self.temporary_count}"
        self.temporary_count += 1
        return self._local(name)

    def _add_array(self, entry: ArrayEntry) -> int:
        index = len(self.arrays)
        self.arrays.append(entry)
        self.array_indexes[entry.name] = index
        return index

    def statement(self, node: ast.stmt):
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                self.unsupported(node, "chained assignment")
            target = node.targets[0]
            if isinstance(target, ast.Name) and self._allocation(target.id, node.value):
                return
            if isinstance(target, ast.Name):
                self.expression(node.value)
                self.emit(STORE_LOCAL, self._local(target.id))
                if self.is_integer(node.value):
                    self.integer_names.add(target.id)
                else:
                    self.integer_names.discard(target.id)
                return
            if isinstance(target, ast.Subscript):
                self.store_subscript(target, node.value)
                return
            self.unsupported(target, "assignment target")
        elif isinstance(node, ast.AnnAssign):
            if node.value is None or not isinstance(node.target, ast.Name):
                self.unsupported(node, "annotated assignment")
            self.expression(node.value)
            self.emit(STORE_LOCAL, self._local(node.target.id))
        elif isinstance(node, ast.AugAssign):
            opcode = _BINOPS.get(type(node.op))
            if opcode is None:
                self.unsupported(node, "augmented operator")
            if isinstance(node.target, ast.Name):
                slot = self._local(node.target.id)
                stays_integer = (
                    node.target.id in self.integer_names
                    and self.is_integer(node.value)
                    and not isinstance(node.op, ast.Div)
                )
                self.emit(LOAD_LOCAL, slot)
                self.expression(node.value)
                self.emit(opcode)
                self.emit(STORE_LOCAL, slot)
                if not stays_integer:
                    self.integer_names.discard(node.target.id)
            elif isinstance(node.target, ast.Subscript):
                array, indexes = self.subscript_parts(node.target)
                for index in indexes:
                    self.expression(index)
                for index in indexes:
                    self.expression(index)
                self.emit(LOAD_ARRAY_1 if len(indexes) == 1 else LOAD_ARRAY_2, array)
                self.expression(node.value)
                self.emit(opcode)
                self.emit(STORE_ARRAY_1 if len(indexes) == 1 else STORE_ARRAY_2, array)
                self.mutated.add(self.arrays[array].name)
            else:
                self.unsupported(node.target, "augmented assignment target")
        elif isinstance(node, ast.For):
            self.for_loop(node)
        elif isinstance(node, ast.While):
            self.while_loop(node)
        elif isinstance(node, ast.If):
            self.if_statement(node)
        elif isinstance(node, ast.Return):
            self.return_statement(node)
        elif isinstance(node, ast.Break):
            if not self.loop_stack:
                self.unsupported(node, "break outside a loop")
            self.loop_stack[-1]["breaks"].append(self.emit(JUMP))
        elif isinstance(node, ast.Continue):
            if not self.loop_stack:
                self.unsupported(node, "continue outside a loop")
            instruction = self.emit(JUMP)
            self.loop_stack[-1]["continues"].append(instruction)
        elif isinstance(node, (ast.Pass, ast.Expr)):
            if isinstance(node, ast.Expr):
                self.expression(node.value)
                self.emit(POP)
        else:
            self.unsupported(node, type(node).__name__)

    def _allocation(self, name: str, value: ast.AST) -> bool:
        if not isinstance(value, ast.Call):
            return False
        call_name = _dotted(value.func)
        if call_name not in {
            "np.empty_like",
            "numpy.empty_like",
            "np.zeros_like",
            "numpy.zeros_like",
        }:
            return False
        if len(value.args) != 1 or value.keywords or not isinstance(value.args[0], ast.Name):
            self.unsupported(value, "only empty_like(array) and zeros_like(array) are supported")
        like = value.args[0].id
        if like not in self.array_indexes:
            self.unsupported(value, f"allocation source {like!r} is not an array")
        allocation = Allocation(
            name=name,
            like=like,
            zeros="zeros_like" in call_name,
            dtype=self.output_dtypes.get(name),
        )
        self.allocations.append(allocation)
        self._add_array(ArrayEntry(name, allocation=allocation))
        return True

    def store_subscript(self, target: ast.Subscript, value: ast.AST):
        array, indexes = self.subscript_parts(target)
        for index in indexes:
            self.expression(index)
        self.expression(value)
        self.emit(STORE_ARRAY_1 if len(indexes) == 1 else STORE_ARRAY_2, array)
        self.mutated.add(self.arrays[array].name)

    def subscript_parts(self, node: ast.Subscript) -> tuple[int, list[ast.AST]]:
        if not isinstance(node.value, ast.Name) or node.value.id not in self.array_indexes:
            self.unsupported(node, "only direct array indexing is supported")
        indexes = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(indexes) not in (1, 2):
            self.unsupported(node, "only 1-D and 2-D indexing is supported")
        array_index = self.array_indexes[node.value.id]
        rank = self._array_rank(array_index)
        if len(indexes) != rank and node.value.id not in self.flat_arrays:
            self.unsupported(
                node,
                f"{rank}-D array {node.value.id!r} requires {rank} index value(s)",
            )
        if any(not self.is_integer(index) for index in indexes):
            raise errors.TypingError(
                f"{self.node.name}: array indices must be integer expressions"
            )
        return array_index, indexes

    def _array_rank(self, index: int) -> int:
        entry = self.arrays[index]
        if entry.argument is not None:
            return self.arguments[entry.argument].ndim
        assert entry.allocation is not None
        return self._array_rank(self.array_indexes[entry.allocation.like])

    def for_loop(self, node: ast.For):
        if not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call):
            self.unsupported(node, "for loops must be 'for name in range(...)'")
        if _dotted(node.iter.func) not in ("range", "prange", "numba.prange", "mojo_numba.prange"):
            self.unsupported(node.iter, "only range and prange loops are supported")
        args = node.iter.args
        if not 1 <= len(args) <= 3 or node.iter.keywords:
            self.unsupported(node.iter, "range takes one to three positional arguments")
        start = ast.Constant(0) if len(args) == 1 else args[0]
        stop = args[0] if len(args) == 1 else args[1]
        step = ast.Constant(1) if len(args) < 3 else args[2]
        step_value = None
        if isinstance(step, ast.Constant) and isinstance(step.value, int):
            step_value = step.value
        elif (
            isinstance(step, ast.UnaryOp)
            and isinstance(step.op, ast.USub)
            and isinstance(step.operand, ast.Constant)
            and isinstance(step.operand.value, int)
        ):
            step_value = -step.operand.value
        if step_value is None or step_value == 0:
            self.unsupported(step, "range step must be a nonzero integer literal")
        slot = self._local(node.target.id)
        self.integer_names.add(node.target.id)
        self.expression(start)
        self.emit(STORE_LOCAL, slot)
        condition = len(self.code)
        self.emit(LOAD_LOCAL, slot)
        self.expression(stop)
        self.emit(LT if step_value > 0 else GT)
        exit_jump = self.emit(JUMP_IF_FALSE)
        context = {"breaks": [], "continues": []}
        self.loop_stack.append(context)
        for statement in node.body:
            self.statement(statement)
        continue_target = len(self.code)
        for instruction in context["continues"]:
            self.patch(instruction, continue_target)
        self.emit(LOAD_LOCAL, slot)
        self.push(step_value)
        self.emit(ADD)
        self.emit(STORE_LOCAL, slot)
        self.emit(JUMP, condition)
        else_start = len(self.code)
        self.patch(exit_jump, else_start)
        self.loop_stack.pop()
        for statement in node.orelse:
            self.statement(statement)
        after_else = len(self.code)
        for instruction in context["breaks"]:
            self.patch(instruction, after_else)

    def while_loop(self, node: ast.While):
        condition = len(self.code)
        self.expression(node.test)
        exit_jump = self.emit(JUMP_IF_FALSE)
        context = {"breaks": [], "continues": []}
        self.loop_stack.append(context)
        for statement in node.body:
            self.statement(statement)
        for instruction in context["continues"]:
            self.patch(instruction, condition)
        self.emit(JUMP, condition)
        else_start = len(self.code)
        self.patch(exit_jump, else_start)
        self.loop_stack.pop()
        for statement in node.orelse:
            self.statement(statement)
        after_else = len(self.code)
        for instruction in context["breaks"]:
            self.patch(instruction, after_else)

    def if_statement(self, node: ast.If):
        self.expression(node.test)
        false_jump = self.emit(JUMP_IF_FALSE)
        for statement in node.body:
            self.statement(statement)
        if node.orelse:
            end_jump = self.emit(JUMP)
            self.patch(false_jump, len(self.code))
            for statement in node.orelse:
                self.statement(statement)
            self.patch(end_jump, len(self.code))
        else:
            self.patch(false_jump, len(self.code))

    def return_statement(self, node: ast.Return):
        if node.value is None:
            self.result_kind = "none"
            self.emit(HALT)
        elif isinstance(node.value, ast.Name) and node.value.id in self.array_indexes:
            self.result_kind = "array"
            self.result_name = node.value.id
            self.emit(HALT)
        else:
            self.expression(node.value)
            if self.is_boolean(node.value):
                self.result_kind = "bool"
            elif self.is_integer(node.value):
                self.result_kind = "int"
            else:
                self.result_kind = "float"
            self.emit(RETURN)

    def expression(self, node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
            self.push(float(node.value))
        elif isinstance(node, ast.Name):
            if node.id in self.locals:
                self.emit(LOAD_LOCAL, self.locals[node.id])
            elif node.id in ("True", "False"):
                self.push(node.id == "True")
            else:
                self.unsupported(node, f"unknown scalar name {node.id!r}")
        elif isinstance(node, ast.BinOp):
            opcode = _BINOPS.get(type(node.op))
            if opcode is None:
                self.unsupported(node, "binary operator")
            self.expression(node.left)
            self.expression(node.right)
            self.emit(opcode)
        elif isinstance(node, ast.UnaryOp):
            self.expression(node.operand)
            if isinstance(node.op, ast.USub):
                self.emit(NEG)
            elif isinstance(node.op, ast.UAdd):
                pass
            elif isinstance(node.op, ast.Not):
                self.emit(NOT)
            else:
                self.unsupported(node, "unary operator")
        elif isinstance(node, ast.BoolOp):
            if not node.values:
                self.unsupported(node, "empty boolean expression")
            temporary = self._temporary()
            self.expression(node.values[0])
            self.emit(STORE_LOCAL, temporary)
            exits = []
            for value in node.values[1:]:
                self.emit(LOAD_LOCAL, temporary)
                if isinstance(node.op, ast.Or):
                    self.emit(NOT)
                exits.append(self.emit(JUMP_IF_FALSE))
                self.expression(value)
                self.emit(STORE_LOCAL, temporary)
            end = len(self.code)
            for instruction in exits:
                self.patch(instruction, end)
            self.emit(LOAD_LOCAL, temporary)
        elif isinstance(node, ast.Compare):
            if len(node.ops) == 1:
                opcode = _CMPOPS.get(type(node.ops[0]))
                if opcode is None:
                    self.unsupported(node, "comparison operator")
                self.expression(node.left)
                self.expression(node.comparators[0])
                self.emit(opcode)
                return
            previous = self._temporary()
            result = self._temporary()
            self.expression(node.left)
            self.emit(STORE_LOCAL, previous)
            self.push(1.0)
            self.emit(STORE_LOCAL, result)
            exits = []
            for operator, comparator in zip(node.ops, node.comparators):
                opcode = _CMPOPS.get(type(operator))
                if opcode is None:
                    self.unsupported(node, "comparison operator")
                self.emit(LOAD_LOCAL, result)
                exits.append(self.emit(JUMP_IF_FALSE))
                self.expression(comparator)
                current = self._temporary()
                self.emit(STORE_LOCAL, current)
                self.emit(LOAD_LOCAL, previous)
                self.emit(LOAD_LOCAL, current)
                self.emit(opcode)
                self.emit(STORE_LOCAL, result)
                previous = current
            end = len(self.code)
            for instruction in exits:
                self.patch(instruction, end)
            self.emit(LOAD_LOCAL, result)
        elif isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "shape"
                and isinstance(node.value.value, ast.Name)
            ):
                name = node.value.value.id
                if name not in self.array_indexes or not isinstance(node.slice, ast.Constant):
                    self.unsupported(node, "shape index")
                dimension = int(node.slice.value)
                if dimension not in (0, 1):
                    self.unsupported(node, "only shape[0] and shape[1] are supported")
                self.emit(LOAD_SHAPE, self.array_indexes[name], dimension)
            else:
                array, indexes = self.subscript_parts(node)
                for index in indexes:
                    self.expression(index)
                self.emit(LOAD_ARRAY_1 if len(indexes) == 1 else LOAD_ARRAY_2, array)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in self.array_indexes:
                if node.attr == "size":
                    self.emit(LOAD_SIZE, self.array_indexes[node.value.id])
                else:
                    self.unsupported(node, f"array attribute {node.attr!r}")
            elif _dotted(node) in ("math.pi", "np.pi", "numpy.pi"):
                self.push(math.pi)
            elif _dotted(node) in ("math.e", "np.e", "numpy.e"):
                self.push(math.e)
            else:
                self.unsupported(node, "attribute expression")
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name == "len":
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                    self.unsupported(node, "len expects one array")
                array_name = node.args[0].id
                if array_name not in self.array_indexes:
                    self.unsupported(node, "len expects an array")
                self.emit(LOAD_SHAPE, self.array_indexes[array_name], 0)
            elif name in ("float", "int", "bool"):
                if len(node.args) != 1:
                    self.unsupported(node, f"{name} expects one argument")
                self.expression(node.args[0])
            elif name in _CALLS:
                expected = 2 if _CALLS[name] in (MIN, MAX) else 1
                if len(node.args) != expected or node.keywords:
                    self.unsupported(node, f"{name} expects {expected} positional argument(s)")
                for argument in node.args:
                    self.expression(argument)
                self.emit(_CALLS[name])
            else:
                self.unsupported(node, f"call to {name or '<expression>'}")
        elif isinstance(node, ast.IfExp):
            self.expression(node.test)
            false_jump = self.emit(JUMP_IF_FALSE)
            self.expression(node.body)
            end_jump = self.emit(JUMP)
            self.patch(false_jump, len(self.code))
            self.expression(node.orelse)
            self.patch(end_jump, len(self.code))
        else:
            self.unsupported(node, f"expression {type(node).__name__}")

    def is_integer(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (bool, int))
        if isinstance(node, ast.Name):
            return node.id in self.integer_names
        if isinstance(node, ast.UnaryOp):
            return self.is_integer(node.operand)
        if isinstance(node, ast.BinOp):
            return not isinstance(node.op, ast.Div) and self.is_integer(node.left) and self.is_integer(node.right)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            entry_index = self.array_indexes.get(node.value.id)
            if entry_index is None:
                return False
            entry = self.arrays[entry_index]
            if entry.argument is not None:
                dtype = self.arguments[entry.argument].dtype
                return np.issubdtype(dtype, np.integer) or dtype == np.dtype(np.bool_)
            if entry.allocation is not None:
                dtype = entry.allocation.dtype
                if dtype is not None:
                    return np.issubdtype(dtype, np.integer) or dtype == np.dtype(np.bool_)
        if self.is_boolean(node):
            return True
        return False

    def is_boolean(self, node: ast.AST) -> bool:
        return isinstance(node, (ast.Compare, ast.BoolOp)) or (
            isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        )

    def unsupported(self, node: ast.AST, feature: str):
        line = getattr(node, "lineno", "?")
        raise errors.UnsupportedError(
            f"{self.node.name}: unsupported {feature} at source line {line}"
        )


def _type_key(arguments: dict[str, Any]) -> tuple:
    key = []
    for name, value in arguments.items():
        if isinstance(value, np.ndarray):
            key.append((name, value.dtype.str, value.ndim))
        elif isinstance(value, (bool, np.bool_)):
            key.append((name, "bool"))
        elif isinstance(value, (int, np.integer)):
            key.append((name, "int64"))
        elif isinstance(value, (float, np.floating)):
            key.append((name, "float64"))
        else:
            key.append((name, type(value).__qualname__))
    return tuple(key)


class Dispatcher:
    """Lazy, signature-specialized compiled function."""

    def __init__(self, function: Callable, signature=None, options: dict | None = None):
        self.py_func = function
        self.signature = inspect.signature(function)
        self.declared_signature = parse_signature(signature)
        self.options = dict(options or {})
        self.device = self.options.get("device", "cpu")
        if self.device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        self._node = _function_ast(function)
        self._plans: dict[tuple, Plan] = {}
        self.signatures: list[tuple] = []
        self.nopython_signatures = self.signatures
        update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        _validate_declared_arguments(self.declared_signature, arguments)
        key = _type_key(arguments)
        plan = self._plans.get(key)
        if plan is None:
            plan = Lowerer(self._node, arguments).lower()
            self._plans[key] = plan
            self.signatures.append(key)
        return plan.execute(arguments, self.device)

    def compile(self, signature=None):
        if signature is not None:
            parse_signature(signature)
        return self

    def inspect_types(self, file=None):
        text = f"{self.__name__}: {len(self._plans)} compiled signature(s)"
        if file is not None:
            print(text, file=file)
            return None
        return text


class _VectorNameTransformer(ast.NodeTransformer):
    def __init__(self, arrays: set[str], index_name: str):
        self.arrays = arrays
        self.index_name = index_name

    def visit_Name(self, node: ast.Name):
        if node.id in self.arrays and isinstance(node.ctx, ast.Load):
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id=node.id, ctx=ast.Load()),
                    slice=ast.Name(id=self.index_name, ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
                node,
            )
        return node


class VectorizedDispatcher:
    """Elementwise scalar function compiled into one Mojo loop."""

    def __init__(self, function: Callable, signatures=None, options: dict | None = None):
        self.py_func = function
        self.signature = inspect.signature(function)
        self.options = dict(options or {})
        raw = signatures[0] if isinstance(signatures, (list, tuple)) and signatures else signatures
        self.declared_signature: Signature | None = parse_signature(raw)
        self._source_node = _function_ast(function)
        self._plans: dict[tuple, Plan] = {}
        self.signatures: list[tuple] = []
        update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        _validate_declared_arguments(
            self.declared_signature, arguments, vectorized=True
        )
        array_names = {name for name, value in arguments.items() if isinstance(value, np.ndarray)}
        if not array_names:
            return self.py_func(*args, **kwargs)
        first_name = next(name for name in arguments if name in array_names)
        first = _check_array(first_name, arguments[first_name])
        for name in array_names:
            array = _check_array(name, arguments[name])
            if array.shape != first.shape:
                raise errors.TypingError("vectorize array arguments must have identical shapes")
        key = _type_key(arguments)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._make_plan(arguments, array_names, first_name)
            self._plans[key] = plan
            self.signatures.append(key)
        return plan.execute(arguments)

    def _make_plan(self, arguments, array_names: set[str], first_name: str) -> Plan:
        returns = [node for node in self._source_node.body if isinstance(node, ast.Return)]
        if len(returns) != 1 or returns[0].value is None:
            raise errors.UnsupportedError("vectorize functions must contain one return expression")
        index_name = "__mn_index"
        result_name = "__mn_result"
        expression = copy.deepcopy(returns[0].value)
        expression = _VectorNameTransformer(array_names, index_name).visit(expression)
        allocation = ast.Assign(
            targets=[ast.Name(id=result_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="empty_like",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=first_name, ctx=ast.Load())],
                keywords=[],
            ),
        )
        loop = ast.For(
            target=ast.Name(id=index_name, ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.Attribute(
                        value=ast.Name(id=first_name, ctx=ast.Load()),
                        attr="size",
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            ),
            body=[
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name(id=result_name, ctx=ast.Load()),
                            slice=ast.Name(id=index_name, ctx=ast.Load()),
                            ctx=ast.Store(),
                        )
                    ],
                    value=expression,
                )
            ],
            orelse=[],
        )
        node = ast.FunctionDef(
            name=self._source_node.name,
            args=copy.deepcopy(self._source_node.args),
            body=[allocation, loop, ast.Return(value=ast.Name(id=result_name, ctx=ast.Load()))],
            decorator_list=[],
        )
        ast.fix_missing_locations(node)
        output_dtype = (
            self.declared_signature.return_type.dtype
            if self.declared_signature is not None
            else np.dtype(np.float64)
        )
        if output_dtype not in _DTYPES:
            raise errors.TypingError(
                f"vectorize output dtype {output_dtype} is not supported"
            )
        return Lowerer(
            node,
            arguments,
            {result_name: output_dtype},
            flat_arrays=array_names | {result_name},
        ).lower()
