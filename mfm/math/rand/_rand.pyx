import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, pow, fabs
from libc.stdlib cimport rand, RAND_MAX


cdef extern from "mtrandom.h":
    cdef cppclass MTrandoms:
        void seedMT()
        double random0i1i() nogil
        double random0i1e() nogil

cdef MTrandoms rmt
rmt.seedMT()

cdef double eps = 244.14062E-6

@cython.cdivision(True)
cdef long random_c(long max) nogil:
    return rand() % max

@cython.cdivision(True)
cdef double cranf() nogil:
    return <double> rmt.random0i1e()

def ranf():
    return rmt.random0i1e()

@cython.boundscheck(False)
def weighted_choice(np.ndarray[np.double_t, ndim=1] weights, int n=1):
    """
    A weighted random number generator. The random number generator generates
    random numbers between zero and the length of the provided weight-array. The
    elements of the weight-arrays are proportional to the probability that the corresponding
    integer random number is generated.

    :param weights: array-like
    :param n: int
        number of random values to be generated
    :return: Returns an array containing the random values

    Examples
    --------

    >>> import numpy as np
    >>> weighted_choice(np.array([0.1, 0.5, 3]), 10)
    array([1, 2, 2, 2, 2, 2, 2, 2, 2, 1], dtype=uint32)

    http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
    """
    cdef double running_total = 0.0, rnd
    cdef int i, j, nWeights

    nWeights = weights.shape[0]

    cdef np.ndarray[np.uint32_t, ndim=1] r = np.empty(n, dtype=np.uint32)
    cdef np.ndarray[np.float64_t, ndim=1] totals = np.empty(nWeights, dtype=np.float64)

    for i in range(nWeights):
        running_total += weights[i]
        totals[i] = running_total

    for j in range(n):
        #rnd = cranf() * running_total
        rnd = rmt.random0i1e() * running_total
        for i in range(nWeights):
            if rnd <= totals[i]:
                r[j] = i
                break
    return r