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
