import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, log

cdef double eps = 244.14062E-6

cdef extern from "mtrandom.h":
    cdef cppclass MTrandoms:
        void seedMT()
        double random0i1i() nogil
        double random0i1e() nogil

cdef MTrandoms rmt
rmt.seedMT()

@cython.cdivision(True)
cdef double ranf() nogil:
    return <double> rmt.random0i1e()


@cython.cdivision(True)
@cython.boundscheck(False)
def simulate_decay(unsigned int n_curves, double[:] decay, double dt_tac, float[:] k_quench, long[:] shift,
                   double t_step, double tau0):

    cdef int n_tac_channels = decay.shape[0]
    cdef long i, j, ki, shift_i
    cdef double k0 = 1. / tau0
    cdef double kQ_int
    cdef double time
    cdef long previous_frame, next_frame

    for i in range(n_curves):
        shift_i = shift[i]

        previous_frame = 0
        kQ_int = 0.0
        for j in range(n_tac_channels):
            time = j * dt_tac
            next_frame = <long> (time / t_step)
            if next_frame > previous_frame:
                for ki in range(previous_frame, next_frame):
                    kQ_int += k_quench[shift_i + ki]
                previous_frame = next_frame

            decay[j] += exp(-time * k0 - kQ_int * t_step)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_photons(float[:] dts, char[:] phs, unsigned long n_ph, float[:] k_quench, double t_step, float tau0,
                     int n_parallel, long[:] rand_shift):
    cdef unsigned long n_step
    cdef unsigned long i, j, shift_nbr

    # Radiation boundary condition

    for i in range(n_ph):
        # Look-up when photon was emitted

        dts[i] = <float> log(1./(ranf() + eps)) * tau0
        phs[i] = 1

        # count the number of collisions within dt
        n_step = <unsigned long> (dts[i] / t_step)
        shift_nbr = rand_shift[i]

        for j in range(shift_nbr, shift_nbr + n_step):
            if ranf() < k_quench[j] * t_step:
                phs[i] = 0
                break
