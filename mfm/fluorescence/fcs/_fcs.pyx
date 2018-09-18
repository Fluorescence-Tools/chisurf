import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdint cimport uint32_t, uint64_t
from libc.math cimport pow



@cython.boundscheck(False)
@cython.cdivision(True)
def correlate(uint32_t n, uint32_t B, uint64_t[:, :] t, uint64_t[:] taus, float[:] corr, float[:, :] w):
    cdef uint64_t b, pw
    cdef uint64_t ta, ti
    cdef uint64_t ca, ci, pa, pi, shift
    cdef uint64_t j

    for b in prange(B, nogil=True):
        j = (n * B + b)
        shift = taus[j]/(<int> pow(2.0, <double> (j / B)))
        # STARTING CHANNEL
        ca = 0 if t[0, 1] < t[1, 1] else 1  # currently active correlation channel
        ci = 1 if t[0, 1] < t[1, 1] else 0  # currently inactive correlation channel
        # POSITION ON ARRAY
        pa, pi = 0, 1  # position on active (pa), previous (pp) and inactive (pi) channel
        while pa < t[ca, 0] and pi <= t[ci, 0]:
            pa += 1
            if ca == 1:
                ta = t[ca, pa] + shift
                ti = t[ci, pi]
            else:
                ta = t[ca, pa]
                ti = t[ci, pi] + shift
            if ta >= ti:
                if ta == ti:
                    corr[j] += (w[ci, pi] * w[ca, pa])
                ca, ci = ci, ca
                pa, pi = pi, pa
