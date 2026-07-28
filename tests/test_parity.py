"""Behavioural parity with upstream Numba for the documented loop subset."""

import math

import numba
import numpy as np
import pytest
from numba import prange

import mojo_numba as mnb


def sum_squares_impl(x):
    total = 0.0
    for i in range(len(x)):
        total += x[i] * x[i]
    return total


def axpy_impl(x, y, alpha):
    result = np.empty_like(x)
    for i in range(x.size):
        result[i] = alpha * x[i] + y[i]
    return result


def threshold_impl(x, limit):
    count = 0
    for i in prange(x.size):
        if x[i] > limit:
            count += 1
    return count


def nonlinear_impl(x):
    result = np.empty_like(x)
    for i in range(x.size):
        value = x[i]
        result[i] = math.sin(value) + math.exp(-abs(value))
    return result


def matmul_square_impl(a, b):
    result = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[1]):
                result[i, j] += a[i, k] * b[k, j]
    return result


def transform_2d_impl(x):
    result = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] >= 0.0:
                result[i, j] = math.sqrt(x[i, j]) + math.sin(x[i, j])
            else:
                result[i, j] = -math.sqrt(-x[i, j])
    return result


def mutate_impl(x, scale):
    for i in range(x.size):
        x[i] *= scale


def integer_checksum_impl(x):
    total = 0
    for i in range(x.size):
        total += x[i] * (i + 1)
    return total


def while_impl(n):
    i = 0
    total = 0
    while i < n:
        i += 1
        if i % 2 == 0:
            continue
        if i > 25:
            break
        total += i
    return total


def clipped_impl(x, lo, hi):
    result = np.empty_like(x)
    for i in range(x.size):
        result[i] = min(max(x[i], lo), hi)
    return result


def scalar_bool_impl(x):
    total = 0.0
    for i in range(x.size):
        total += x[i]
    return total > 0.0


def boolean_short_circuit_impl(x):
    return x != 0.0 and 1.0 / x > 0.25


def descending_integer_impl(n):
    total = 0
    for i in range(n, -7, -3):
        total += (i // 4) * 11 + (i % 4)
    return total


def loop_else_impl(n, stop):
    total = 0
    for i in range(n):
        if i == stop:
            break
        total += i
    else:
        total += 1000
    return total


def vector_formula(x, y):
    return math.sin(x) + x * y


def vector_predicate(x):
    return x > 0.0


def first_item_impl(x):
    return x[0]


@pytest.fixture(scope="module")
def values():
    return np.ascontiguousarray(np.random.default_rng(7).normal(size=2048))


def assert_compiled_parity(function, *args, **kwargs):
    ours = mnb.njit(function)
    upstream = numba.njit(function)
    expected = upstream(*args, **kwargs)
    actual = ours(*args, **kwargs)
    if isinstance(expected, np.ndarray):
        assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert actual.dtype == expected.dtype
    else:
        assert actual == pytest.approx(expected)
        assert type(actual) is type(expected)
    return ours


def test_reduction_parity(values):
    compiled = assert_compiled_parity(sum_squares_impl, values)
    assert compiled.py_func is sum_squares_impl
    assert len(compiled.signatures) == 1


def test_jit_call_and_decorator_forms(values):
    assert mnb.jit(sum_squares_impl)(values) == pytest.approx(sum_squares_impl(values))
    assert mnb.jit()(sum_squares_impl)(values) == pytest.approx(sum_squares_impl(values))


def test_allocating_elementwise_parity(values):
    y = np.ascontiguousarray(values[::-1])
    assert_compiled_parity(axpy_impl, values, y, 0.75)


def test_branch_and_prange_parity(values):
    assert_compiled_parity(threshold_impl, values, 0.2)


def test_simd_tail_fast_paths():
    x = np.ascontiguousarray(np.linspace(-2.0, 2.0, 37))
    y = np.ascontiguousarray(np.linspace(1.0, -1.0, 37))
    cases = [
        (sum_squares_impl, (x,), "sum_squares"),
        (axpy_impl, (x, y, 0.75), "axpy"),
        (threshold_impl, (x, 0.2), "threshold_count"),
        (nonlinear_impl, (x,), "nonlinear"),
    ]
    for function, arguments, kind in cases:
        compiled = assert_compiled_parity(function, *arguments)
        plan = next(iter(compiled._plans.values()))
        assert plan.fast_path is not None
        assert plan.fast_path.kind == kind


@pytest.mark.parametrize("size", [0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65])
def test_fast_paths_at_empty_and_tail_boundaries(size):
    x = np.ascontiguousarray(np.linspace(-2.0, 2.0, size))
    y = np.ascontiguousarray(np.linspace(1.0, -1.0, size))
    assert_compiled_parity(sum_squares_impl, x)
    assert_compiled_parity(axpy_impl, x, y, 0.75)
    assert_compiled_parity(threshold_impl, x, 0.2)
    assert_compiled_parity(nonlinear_impl, x)


def test_parallel_nonlinear_threshold():
    size = 262_147
    x = np.ascontiguousarray(np.linspace(-3.0, 3.0, size))
    assert_compiled_parity(nonlinear_impl, x)


def test_nested_matrix_loop_parity():
    rng = np.random.default_rng(1)
    a = np.ascontiguousarray(rng.normal(size=(12, 12)))
    b = np.ascontiguousarray(rng.normal(size=(12, 12)))
    assert_compiled_parity(matmul_square_impl, a, b)


def test_parallel_matmul_threshold():
    rng = np.random.default_rng(11)
    a = np.ascontiguousarray(rng.normal(size=(127, 127)))
    b = np.ascontiguousarray(rng.normal(size=(127, 127)))
    compiled = assert_compiled_parity(matmul_square_impl, a, b)
    plan = next(iter(compiled._plans.values()))
    assert plan.fast_path is not None
    assert plan.fast_path.kind == "matmul_square"


def test_2d_indexing_math_and_branch_parity():
    x = np.ascontiguousarray(np.linspace(-3.0, 3.0, 99).reshape(9, 11))
    assert_compiled_parity(transform_2d_impl, x)


def test_in_place_mutation_parity(values):
    ours_x = values.copy()
    upstream_x = values.copy()
    ours = mnb.njit(mutate_impl)
    upstream = numba.njit(mutate_impl)
    assert ours(ours_x, 1.25) is None
    assert upstream(upstream_x, 1.25) is None
    assert np.array_equal(ours_x, upstream_x)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_integer_array_parity(dtype):
    x = np.arange(200, dtype=dtype)
    assert_compiled_parity(integer_checksum_impl, x)


@pytest.mark.parametrize("dtype", [np.uint8, np.bool_])
def test_uint8_and_boolean_array_parity(dtype):
    x = np.arange(64, dtype=np.uint8).astype(dtype)
    assert_compiled_parity(integer_checksum_impl, x)


def test_while_break_continue_parity():
    assert_compiled_parity(while_impl, 100)


def test_min_max_and_scalar_arguments(values):
    assert_compiled_parity(clipped_impl, values, -0.5, 0.75)


def test_boolean_scalar_return(values):
    assert_compiled_parity(scalar_bool_impl, values)


@pytest.mark.parametrize("value", [0.0, -2.0, 0.5, 8.0])
def test_boolean_short_circuit_and_chained_comparison(value):
    assert_compiled_parity(boolean_short_circuit_impl, value)


@pytest.mark.parametrize("value", [0, 5, 31])
def test_negative_range_floor_division_and_modulo(value):
    assert_compiled_parity(descending_integer_impl, value)


@pytest.mark.parametrize("stop", [3, 99])
def test_loop_else_skipped_only_by_break(stop):
    assert_compiled_parity(loop_else_impl, 10, stop)


def test_explicit_signature_and_dispatch_cache(values):
    compiled = mnb.njit("float64(float64[:])")(sum_squares_impl)
    first = compiled(values)
    second = compiled(values * 2)
    assert first == pytest.approx(sum_squares_impl(values))
    assert second == pytest.approx(sum_squares_impl(values * 2))
    assert len(compiled.signatures) == 1
    assert "1 compiled signature" in compiled.inspect_types()


def test_declared_signature_rejects_dtype_mismatch():
    compiled = mnb.njit("float64(float64[:])")(sum_squares_impl)
    with pytest.raises(mnb.errors.TypingError, match="not declared"):
        compiled(np.arange(8, dtype=np.float32))


def test_signature_objects_and_typeof(values):
    signature = mnb.float64(mnb.float64[:])
    compiled = mnb.njit(signature)(sum_squares_impl)
    assert compiled(values) == pytest.approx(sum_squares_impl(values))
    assert str(mnb.typeof(values)) == "float64[:]"
    assert mnb.typeof(3) == mnb.int64
    assert mnb.typeof(2.0) == mnb.float64


def test_vectorize_binary_parity(values):
    y = np.ascontiguousarray(np.linspace(-2.0, 2.0, values.size))
    ours = mnb.vectorize(["float64(float64, float64)"])(vector_formula)
    upstream = numba.vectorize(["float64(float64, float64)"])(vector_formula)
    assert np.allclose(ours(values, y), upstream(values, y), rtol=1e-12, atol=1e-12)
    assert ours(2.0, 3.0) == pytest.approx(vector_formula(2.0, 3.0))


def test_vectorize_scalar_broadcast_and_shape():
    x = np.ascontiguousarray(np.linspace(-1.0, 1.0, 120).reshape(10, 12))
    ours = mnb.vectorize(["float64(float64, float64)"])(vector_formula)
    upstream = numba.vectorize(["float64(float64, float64)"])(vector_formula)
    expected = upstream(x, 0.25)
    actual = ours(x, 0.25)
    assert actual.shape == x.shape
    assert np.allclose(actual, expected)


def test_vectorize_declared_boolean_output(values):
    ours = mnb.vectorize(["boolean(float64)"])(vector_predicate)
    upstream = numba.vectorize(["boolean(float64)"])(vector_predicate)
    assert np.array_equal(ours(values), upstream(values))
    assert ours(values).dtype == np.bool_


def test_float32_output_preserved():
    x = np.ascontiguousarray(np.linspace(-1.0, 1.0, 100, dtype=np.float32))
    y = x[::-1].copy()
    ours = mnb.vectorize(["float32(float32, float32)"])(vector_formula)
    upstream = numba.vectorize(["float32(float32, float32)"])(vector_formula)
    assert np.allclose(ours(x, y), upstream(x, y), rtol=1e-6, atol=1e-7)
    assert ours(x, y).dtype == np.float32


def test_unsupported_dtype_fails_explicitly():
    compiled = mnb.njit(sum_squares_impl)
    with pytest.raises(mnb.errors.TypingError, match="supported dtypes"):
        compiled(np.arange(10, dtype=np.float16))


def test_noncontiguous_input_fails_explicitly(values):
    compiled = mnb.njit(sum_squares_impl)
    with pytest.raises(mnb.errors.TypingError, match="C-contiguous"):
        compiled(values[::2])


def test_out_of_bounds_access_is_rejected_before_pointer_dereference():
    compiled = mnb.njit(first_item_impl)
    with pytest.raises(RuntimeError, match="unsafe execution state"):
        compiled(np.empty(0, dtype=np.float64))


def unsupported_list_comprehension(x):
    return sum([x[i] for i in range(x.size)])


def test_unsupported_syntax_fails_without_python_fallback(values):
    compiled = mnb.njit(unsupported_list_comprehension)
    with pytest.raises(mnb.errors.UnsupportedError, match="unsupported"):
        compiled(values)
