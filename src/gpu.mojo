"""Optional GPU matrix multiplication for high-intensity square kernels."""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime TILE = 16
comptime MAX_N = 8192


def matmul_kernel(a: FPtr, b: FPtr, dst: FPtr, n_value: Int64):
    var n = Int(n_value)
    var row = Int(block_idx.y) * TILE + Int(thread_idx.y)
    var col = Int(block_idx.x) * TILE + Int(thread_idx.x)
    if row >= n or col >= n:
        return
    var total = 0.0
    for k in range(n):
        total += a[row * n + k] * b[k * n + col]
    dst[row * n + col] = total


@export("mn_gpu_available")
def mn_gpu_available() abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        _ = ctx.name()
        return 1
    except:
        return 0


@export("mn_gpu_matmul_square_f64")
def mn_gpu_matmul_square_f64(
    a_addr: Int, b_addr: Int, dst_addr: Int, n: Int
) abi("C") -> Int:
    if a_addr == 0 or b_addr == 0 or dst_addr == 0 or n <= 0 or n > MAX_N:
        return -2
    try:
        var ctx = DeviceContext()
        var count = n * n
        var da = ctx.enqueue_create_buffer[DType.float64](count)
        var db = ctx.enqueue_create_buffer[DType.float64](count)
        var ddst = ctx.enqueue_create_buffer[DType.float64](count)
        ctx.enqueue_copy(da, FPtr(unsafe_from_address=a_addr))
        ctx.enqueue_copy(db, FPtr(unsafe_from_address=b_addr))
        ctx.enqueue_function[matmul_kernel](
            da,
            db,
            ddst,
            Int64(n),
            grid_dim=((n + TILE - 1) // TILE, (n + TILE - 1) // TILE),
            block_dim=(TILE, TILE),
        )
        ctx.enqueue_copy(FPtr(unsafe_from_address=dst_addr), ddst)
        ctx.synchronize()
        return 0
    except:
        return -1
