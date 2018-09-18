import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def calculate_rates(double r0, double tau0, unsigned int[:, :] donors, unsigned int [:, :] acceptors, double[:, :, :, :] grid):
    # grid: column, row, primitive, coord
    # donors: column, row, primitive
    # acceptors: column, row, primitive

    cdef unsigned int n_donors = donors.shape[0]
    cdef unsigned int n_acceptors = acceptors.shape[0]
    cdef unsigned int l, k
    cdef double d_x, d_y, d_z
    cdef double a_x, a_y, a_z
    cdef double da

    cdef np.ndarray[np.float64_t, ndim=1] d_rates = np.zeros(n_donors, dtype=np.float64)

    for l in prange(n_donors, nogil=True):
        d_x = grid[donors[l, 0], donors[l, 1], donors[l, 2], 0]
        d_y = grid[donors[l, 0], donors[l, 1], donors[l, 2], 1]
        d_z = grid[donors[l, 0], donors[l, 1], donors[l, 2], 2]
        for k in range(n_acceptors):
            a_x = grid[acceptors[k, 0], acceptors[k, 1], acceptors[k, 2], 0]
            a_y = grid[acceptors[k, 0], acceptors[k, 1], acceptors[k, 2], 1]
            a_z = grid[acceptors[k, 0], acceptors[k, 1], acceptors[k, 2], 2]
            da = sqrt((d_x - a_x)**2 + (d_y - a_y)**2 + (d_z - a_z)**2)
            d_rates[l] += 1./tau0 * (r0/da)**6
    return d_rates
