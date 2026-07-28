"""Typed bytecode execution kernels for mojo-numba."""

from std.algorithm import parallelize
from std.math import cos, exp, floor, log, pow, sin, sqrt, tanh
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime PARALLEL_ELEMENTS = 262_144
comptime PARALLEL_AXPY_ELEMENTS = 16_777_216
comptime PARALLEL_MATMUL_WORK = 2_000_000


@export("mn_sum_squares_f64")
def mn_sum_squares_f64(x_addr: Int, n: Int) abi("C") -> Float64:
    var x = F64Ptr(unsafe_from_address=x_addr)
    var vector_total = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var values = x.load[width=W](i)
        vector_total += values * values
        i += W
    var total = vector_total.reduce_add()
    while i < n:
        total += x[i] * x[i]
        i += 1
    return total


@export("mn_axpy_f64")
def mn_axpy_f64(
    x_addr: Int,
    y_addr: Int,
    dst_addr: Int,
    n: Int,
    alpha: Float64,
) abi("C"):
    var x = F64Ptr(unsafe_from_address=x_addr)
    var y = F64Ptr(unsafe_from_address=y_addr)
    var dst = F64Ptr(unsafe_from_address=dst_addr)
    var workers = num_physical_cores() if n >= PARALLEL_AXPY_ELEMENTS else 1

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        var i = start
        while i + 4 * W <= end:
            dst.store(
                i,
                SIMD[DType.float64, W](alpha) * x.load[width=W](i)
                + y.load[width=W](i),
            )
            dst.store(
                i + W,
                SIMD[DType.float64, W](alpha) * x.load[width=W](i + W)
                + y.load[width=W](i + W),
            )
            dst.store(
                i + 2 * W,
                SIMD[DType.float64, W](alpha) * x.load[width=W](i + 2 * W)
                + y.load[width=W](i + 2 * W),
            )
            dst.store(
                i + 3 * W,
                SIMD[DType.float64, W](alpha) * x.load[width=W](i + 3 * W)
                + y.load[width=W](i + 3 * W),
            )
            i += 4 * W
        while i + W <= end:
            dst.store(
                i,
                SIMD[DType.float64, W](alpha) * x.load[width=W](i)
                + y.load[width=W](i),
            )
            i += W
        while i < end:
            dst[i] = alpha * x[i] + y[i]
            i += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mn_threshold_count_f64")
def mn_threshold_count_f64(
    x_addr: Int, n: Int, limit: Float64
) abi("C") -> Int64:
    var x = F64Ptr(unsafe_from_address=x_addr)
    var total0 = SIMD[DType.int64, W](0)
    var total1 = SIMD[DType.int64, W](0)
    var total2 = SIMD[DType.int64, W](0)
    var total3 = SIMD[DType.int64, W](0)
    var i = 0
    while i + 4 * W <= n:
        total0 += x.load[width=W](i).gt(limit).cast[DType.int64]()
        total1 += x.load[width=W](i + W).gt(limit).cast[DType.int64]()
        total2 += x.load[width=W](i + 2 * W).gt(limit).cast[DType.int64]()
        total3 += x.load[width=W](i + 3 * W).gt(limit).cast[DType.int64]()
        i += 4 * W
    while i + W <= n:
        total0 += x.load[width=W](i).gt(limit).cast[DType.int64]()
        i += W
    var total = (total0 + total1 + total2 + total3).reduce_add()
    while i < n:
        if x[i] > limit:
            total += 1
        i += 1
    return total


@export("mn_nonlinear_f64")
def mn_nonlinear_f64(x_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var x = F64Ptr(unsafe_from_address=x_addr)
    var dst = F64Ptr(unsafe_from_address=dst_addr)
    var workers = num_physical_cores() if n >= PARALLEL_ELEMENTS else 1

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        var i = start
        while i + W <= end:
            var values = x.load[width=W](i)
            dst.store(i, sin(values) + exp(-abs(values)))
            i += W
        while i < end:
            var value = x[i]
            dst[i] = sin(value) + exp(-abs(value))
            i += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mn_matmul_square_f64")
def mn_matmul_square_f64(
    a_addr: Int, b_addr: Int, dst_addr: Int, n: Int
) abi("C"):
    var a = F64Ptr(unsafe_from_address=a_addr)
    var b = F64Ptr(unsafe_from_address=b_addr)
    var dst = F64Ptr(unsafe_from_address=dst_addr)
    var work = n * n * n
    var workers = num_physical_cores() if work >= PARALLEL_MATMUL_WORK else 1
    workers = min(workers, n)

    @parameter
    def process(worker: Int):
        var row_start = worker * n // workers
        var row_end = (worker + 1) * n // workers
        for row in range(row_start, row_end):
            var target = dst + row * n
            var col = 0
            while col + W <= n:
                var acc = SIMD[DType.float64, W](0.0)
                for k in range(n):
                    acc += (
                        SIMD[DType.float64, W](a[row * n + k])
                        * b.load[width=W](k * n + col)
                    )
                target.store(col, acc)
                col += W
            while col < n:
                var total = 0.0
                for k in range(n):
                    total += a[row * n + k] * b[k * n + col]
                target[col] = total
                col += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


def load_value(address: Int, dtype: Int, index: Int) -> Float64:
    if dtype == 0:
        return F64Ptr(unsafe_from_address=address)[index]
    if dtype == 1:
        return Float64(F32Ptr(unsafe_from_address=address)[index])
    if dtype == 2:
        return Float64(I64Ptr(unsafe_from_address=address)[index])
    if dtype == 3:
        return Float64(I32Ptr(unsafe_from_address=address)[index])
    return Float64(U8Ptr(unsafe_from_address=address)[index])


def store_value(address: Int, dtype: Int, index: Int, value: Float64):
    if dtype == 0:
        F64Ptr(unsafe_from_address=address)[index] = value
    elif dtype == 1:
        F32Ptr(unsafe_from_address=address)[index] = Float32(value)
    elif dtype == 2:
        I64Ptr(unsafe_from_address=address)[index] = Int64(value)
    elif dtype == 3:
        I32Ptr(unsafe_from_address=address)[index] = Int32(value)
    else:
        U8Ptr(unsafe_from_address=address)[index] = UInt8(value)


@export("mn_execute")
def mn_execute(
    code_addr: Int,
    code_count: Int,
    constants_addr: Int,
    constants_count: Int,
    arrays_addr: Int,
    metadata_addr: Int,
    array_count: Int,
    locals_addr: Int,
    local_count: Int,
    stack_addr: Int,
    stack_count: Int,
    return_addr: Int,
) abi("C") -> Int:
    var code = I64Ptr(unsafe_from_address=code_addr)
    var constants = F64Ptr(unsafe_from_address=constants_addr)
    var arrays = I64Ptr(unsafe_from_address=arrays_addr)
    var metadata = I64Ptr(unsafe_from_address=metadata_addr)
    var local_values = F64Ptr(unsafe_from_address=locals_addr)
    var stack = F64Ptr(unsafe_from_address=stack_addr)
    var result = F64Ptr(unsafe_from_address=return_addr)
    var pc = 0
    var sp = 0

    while True:
        if pc < 0 or pc >= code_count:
            return 100
        var op = Int(code[pc * 3])
        var a = Int(code[pc * 3 + 1])
        var b = Int(code[pc * 3 + 2])
        pc += 1

        if op == 0:
            result[0] = 0.0
            return 0
        elif op == 1:
            if a < 0 or a >= constants_count or sp >= stack_count:
                return 101
            stack[sp] = constants[a]
            sp += 1
        elif op == 2:
            if a < 0 or a >= local_count or sp >= stack_count:
                return 102
            stack[sp] = local_values[a]
            sp += 1
        elif op == 3:
            if a < 0 or a >= local_count or sp < 1:
                return 103
            sp -= 1
            local_values[a] = stack[sp]
        elif op == 4:
            if a < 0 or a >= array_count or sp < 1:
                return 104
            sp -= 1
            var index = Int(stack[sp])
            var length = Int(metadata[a * 4 + 2])
            if Int(metadata[a * 4 + 1]) == 2:
                length *= Int(metadata[a * 4 + 3])
            if index < 0:
                index += length
            if index < 0 or index >= length or Int(arrays[a]) == 0:
                return 105
            stack[sp] = load_value(Int(arrays[a]), Int(metadata[a * 4]), index)
            sp += 1
        elif op == 5:
            if a < 0 or a >= array_count or sp < 2:
                return 106
            sp -= 1
            var value = stack[sp]
            sp -= 1
            var index = Int(stack[sp])
            var length = Int(metadata[a * 4 + 2])
            if Int(metadata[a * 4 + 1]) == 2:
                length *= Int(metadata[a * 4 + 3])
            if index < 0:
                index += length
            if index < 0 or index >= length or Int(arrays[a]) == 0:
                return 107
            store_value(Int(arrays[a]), Int(metadata[a * 4]), index, value)
        elif op == 6:
            if a < 0 or a >= array_count or sp < 2:
                return 108
            sp -= 1
            var col = Int(stack[sp])
            sp -= 1
            var row = Int(stack[sp])
            var rows = Int(metadata[a * 4 + 2])
            var cols = Int(metadata[a * 4 + 3])
            if row < 0:
                row += rows
            if col < 0:
                col += cols
            if (
                Int(metadata[a * 4 + 1]) != 2
                or row < 0
                or row >= rows
                or col < 0
                or col >= cols
                or Int(arrays[a]) == 0
            ):
                return 109
            var offset = row * Int(metadata[a * 4 + 3]) + col
            stack[sp] = load_value(Int(arrays[a]), Int(metadata[a * 4]), offset)
            sp += 1
        elif op == 7:
            if a < 0 or a >= array_count or sp < 3:
                return 110
            sp -= 1
            var value = stack[sp]
            sp -= 1
            var col = Int(stack[sp])
            sp -= 1
            var row = Int(stack[sp])
            var rows = Int(metadata[a * 4 + 2])
            var cols = Int(metadata[a * 4 + 3])
            if row < 0:
                row += rows
            if col < 0:
                col += cols
            if (
                Int(metadata[a * 4 + 1]) != 2
                or row < 0
                or row >= rows
                or col < 0
                or col >= cols
                or Int(arrays[a]) == 0
            ):
                return 111
            var offset = row * Int(metadata[a * 4 + 3]) + col
            store_value(Int(arrays[a]), Int(metadata[a * 4]), offset, value)
        elif op == 8:
            if a < 0 or a >= array_count or b < 0 or b >= Int(metadata[a * 4 + 1]) or sp >= stack_count:
                return 112
            stack[sp] = Float64(metadata[a * 4 + 2 + b])
            sp += 1
        elif op == 9:
            if a < 0 or a >= array_count or sp >= stack_count:
                return 113
            var size = Int(metadata[a * 4 + 2])
            if Int(metadata[a * 4 + 1]) == 2:
                size *= Int(metadata[a * 4 + 3])
            stack[sp] = Float64(size)
            sp += 1
        elif op == 10:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] += stack[sp]
        elif op == 11:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] -= stack[sp]
        elif op == 12:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] *= stack[sp]
        elif op == 13:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] /= stack[sp]
        elif op == 14:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = floor(stack[sp - 1] / stack[sp])
        elif op == 15:
            if sp < 2:
                return 114
            sp -= 1
            var quotient = floor(stack[sp - 1] / stack[sp])
            stack[sp - 1] -= quotient * stack[sp]
        elif op == 16:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = pow(stack[sp - 1], stack[sp])
        elif op == 17:
            if sp < 1:
                return 115
            stack[sp - 1] = -stack[sp - 1]
        elif op == 18:
            if sp < 1:
                return 115
            stack[sp - 1] = abs(stack[sp - 1])
        elif op == 20:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = 1.0 if stack[sp - 1] < stack[sp] else 0.0
        elif op == 21:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = 1.0 if stack[sp - 1] <= stack[sp] else 0.0
        elif op == 22:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = 1.0 if stack[sp - 1] > stack[sp] else 0.0
        elif op == 23:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = 1.0 if stack[sp - 1] >= stack[sp] else 0.0
        elif op == 24:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = 1.0 if stack[sp - 1] == stack[sp] else 0.0
        elif op == 25:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = 1.0 if stack[sp - 1] != stack[sp] else 0.0
        elif op == 26:
            if sp < 1:
                return 115
            stack[sp - 1] = 1.0 if stack[sp - 1] == 0.0 else 0.0
        elif op == 27:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = 1.0 if stack[sp - 1] != 0.0 and stack[sp] != 0.0 else 0.0
        elif op == 28:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = 1.0 if stack[sp - 1] != 0.0 or stack[sp] != 0.0 else 0.0
        elif op == 30:
            if sp < 1:
                return 115
            stack[sp - 1] = sin(stack[sp - 1])
        elif op == 31:
            if sp < 1:
                return 115
            stack[sp - 1] = cos(stack[sp - 1])
        elif op == 32:
            if sp < 1:
                return 115
            stack[sp - 1] = exp(stack[sp - 1])
        elif op == 33:
            if sp < 1:
                return 115
            stack[sp - 1] = log(stack[sp - 1])
        elif op == 34:
            if sp < 1:
                return 115
            stack[sp - 1] = sqrt(stack[sp - 1])
        elif op == 35:
            if sp < 1:
                return 115
            stack[sp - 1] = tanh(stack[sp - 1])
        elif op == 36:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = min(stack[sp - 1], stack[sp])
        elif op == 37:
            if sp < 2:
                return 114
            sp -= 1
            stack[sp - 1] = max(stack[sp - 1], stack[sp])
        elif op == 40:
            if a < 0 or a >= code_count:
                return 116
            pc = a
        elif op == 41:
            if sp < 1 or a < 0 or a >= code_count:
                return 117
            sp -= 1
            if stack[sp] == 0.0:
                pc = a
        elif op == 42:
            if sp < 1:
                return 118
            sp -= 1
        elif op == 43:
            if sp < 1:
                return 119
            result[0] = stack[sp - 1]
            return 0
        else:
            return op
