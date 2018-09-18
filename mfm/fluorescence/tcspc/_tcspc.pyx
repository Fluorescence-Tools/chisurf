import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, log, ceil, sqrt
from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from cython.parallel import prange


@cython.boundscheck(False)
@cython.cdivision(True)
def fconv(double[:] fit, double[:] x, double[:] irf, int stop, double dt):
    """
    :param fit: numpy-array
        this array is used to write the y-values after convolution
    :param x: vector of amplitdes and lifetimes in form: amplitude, lifetime
    :param irf:
    :param stop:
    :param dt:
    :return:
    """
    cdef int ne, i
    cdef int nExp = x.shape[0]/2
    cdef double delta_tHalf = dt/2
    cdef double currExp
    cdef double fitCurr
    for i in xrange(stop):
        fit[i] = 0.0
    for ne in xrange(nExp):
        currExp = exp(-dt/(x[2*ne+1]+1e-12))
        fitCurr = 0.0
        for i in xrange(stop):
            fitCurr = (fitCurr + delta_tHalf*irf[i-1])*currExp + delta_tHalf*irf[i]
            fit[i] += fitCurr*x[2*ne]
    return 0

@cython.boundscheck(False)
@cython.cdivision(True)
def rescale_w_bg(double[:] fit, double[:] decay, double[:] w_res, double bg, int start, int stop):
    cdef int i
    cdef double scale = 0.0
    cdef double sumnom = 0.0
    cdef double sumdenom = 0.0
    cdef double iwsq = 0.0
    for i in xrange(start, stop):
        iwsq = 1.0/(w_res[i]*w_res[i]+1e-12)
        if decay[i]!=0.0:
            sumnom += fit[i]*(decay[i]-bg)*iwsq
            sumdenom += fit[i]*fit[i]*iwsq
    if sumdenom!=0.0:
        scale = sumnom/sumdenom
    for i in xrange(start, stop):
        fit[i] *= scale
    return scale

@cython.boundscheck(False)
@cython.cdivision(True)
def fconv_per(double[:] fit, double[:] x, double[:] irf, int start, int stop,
              int n_points, double period, double dt):
    cdef int ne, i, irfStart = 0
    cdef int stop1, period_n = int(ceil(period/dt-0.5))
    cdef double fitCurr, expCurr, tail_a, dtHalf = dt*0.5
    cdef int nExp = x.shape[0]/2
    for i in xrange(stop):
        fit[i] = 0.0
    while irf[irfStart]==0:
        irfStart += 1
    stop1 = n_points-1 if (period_n+irfStart > n_points-1) else period_n+irfStart
    for ne in range(nExp):
        expCurr = exp(-dt/x[2*ne+1])
        tail_a = 1./(1.-exp(-period/x[2*ne+1]))
        fitCurr = 0
        for i in range(stop1):
            fitCurr=(fitCurr + dtHalf*irf[i-1])*expCurr + dtHalf*irf[i]
            fit[i] += fitCurr*x[2*ne]
        fitCurr *= exp(-(period_n - stop1 + start)*dt/x[2*ne+1])
        for i in xrange(start, stop):
            fitCurr *= expCurr
            fit[i] += fitCurr*x[2*ne]*tail_a
    return 0


@cython.boundscheck(False)
@cython.cdivision(True)
def pile_up(double n_excitation_pulse, double[:] data, double[:] model):
    """
    Add pile up effect to model function.

    Attention: This changes the scaling of the model function.

    :param n_excitation_pulse: double
        The number of excitation pulses
    :param data: numpy-array
        The array containing the experimental decay
    :param model: numpy-array
        The array containing the model function
    """
    cdef int i
    cdef double cum_d, s, a
    cdef int n_data_points = data.shape[0]

    cum_d = 0.0
    for i in range(n_data_points):
        if data[i] > 0.0 and n_excitation_pulse > cum_d:
            a = 1.0 - data[i] / (n_excitation_pulse - cum_d)
            s = -log(a)
            model[i] = model[i] / s * data[i]
            cum_d += data[i]

@cython.boundscheck(False)
@cython.cdivision(True)
def pile_up(double n_excitation_pulse, double[:] data, double[:] model):
    """
    Add pile up effect to model function.

    Attention: This changes the scaling of the model function.

    :param n_excitation_pulse: double
        The number of excitation pulses
    :param data: numpy-array
        The array containing the experimental decay
    :param model: numpy-array
        The array containing the model function
    """
    cdef int i
    cdef double cum_d, scaling_factor, rescaled_data
    cdef int n_data_points = data.shape[0]

    cum_d = 0.0
    for i in range(n_data_points):
        if data[i] > 0.0 and n_excitation_pulse > cum_d:
            rescaled_data = -log(1.0 - data[i] / (n_excitation_pulse - cum_d))
            scaling_factor = data[i] / rescaled_data
            model[i] = model[i] * scaling_factor
            cum_d += data[i]



@cython.boundscheck(False)
@cython.cdivision(True)
def fconv_per_cs(double[:] fit, double[:] x, double[:] lamp, int start, int stop,
              int n_points, double period, double dt, int conv_stop):
    """
    Fast convolution at high repetition rate with stop. Originally developed for
    Paris.

    :param fit: array of doubles
        Here the convolved fit is stored
    :param x: array of doubles
        Lifetime-spectrum of the form (amplitude, lifetime, amplitude, lifetime, ...)
    :param lamp: array-doubles
        The instrument response function
    :param start: int
        Start channel of convolution (position in array of IRF)
    :param stop: int
        Stop channel of convolution (position in array of IRF)
    :param n_points: int
        Number of points in fit and lamp
    :param period: double
        Period of repetition in nano-seconds
    :param dt: double
        Channel-width in nano-seconds
    :param conv_stop: int
        Stopping channel of convolution

     Remarks
     -------

     Fit and lamp have to have the same length.
     Seems to have problems with rising components (acceptor rise):_fconv_per works

    """
    cdef int ne, i, stop1
    cdef double fit_curr, exp_curr, tail_a
    cdef double lt_curr, x_curr
    cdef int n_exp = x.shape[0]/2

    cdef double dt_half = dt * 0.5
    cdef int period_n = <int>(ceil(period/dt-0.5))

    for i in range(0, stop):
        fit[i] = 0

    stop1 = n_points if period_n > n_points else period_n

    # convolution
    for ne in range(n_exp):
        lt_curr = x[2*ne+1]
        x_curr = x[2*ne]

        tail_a = 1./(1.-exp(-period/lt_curr))
        exp_curr = exp(-dt/lt_curr)

        fit_curr = 0.
        fit[0] += dt_half * lamp[0] * (exp_curr + 1.) * x_curr

        for i in range(conv_stop):
            fit_curr = (fit_curr + dt_half*lamp[i-1]) * exp_curr + dt_half * lamp[i]
            #fit_curr += dt_half*lamp[i-1]
            #fit_curr *= exp_curr + dt_half * lamp[i]
            fit[i] += fit_curr * x_curr

        for i in range(conv_stop, stop1):
            fit_curr *= exp_curr
            fit[i] += fit_curr * x_curr

        fit_curr *= exp(-(period_n - stop1)*dt/lt_curr)
        for i in range(stop):
            fit_curr *= exp_curr
            fit[i] += fit_curr * x_curr * tail_a


@cython.boundscheck(False)
@cython.cdivision(True)
def pddem(double[:] decayA, double[:] decayB, double[:] k, double[:] px, double[:] pm, double[:] pAB):
    """
    Electronic Energy Transfer within Asymmetric
    Pairs of Fluorophores: Partial Donor-Donor
    Energy Migration (PDDEM)
    Stanislav Kalinin
    http://www.diva-portal.org/smash/get/diva2:143149/FULLTEXT01


    Kalinin, S.V., Molotkovsky, J.G., and Johansson, L.B.
    Partial Donor-Donor Energy Migration (PDDEM) as a Fluorescence
    Spectroscopic Tool for Measuring Distances in Biomacromolecules.
    Spectrochim. Acta A, 58 (2002) 1087-1097.

    -> same results as Stas pddem code (pddem_t.c)

    :param decayA: decay A in form of [ampl lifetime, apml, lifetime...]
    :param decayB: decay B in form of [ampl lifetime, apml, lifetime...]
    :param k: rates of energy transfer [kAB, kBA]
    :param px: probabilities of excitation (pxA, pxB)
    :param pm: probabilities of emission (pmA, pmB)
    :param pAB: pure AB [0., 0]
    :return:
    """
    cdef double pxA, pxB, pmA, pmB, kAB, kBA
    cdef double cA, cB, ci
    cdef double itauA, itauB
    cdef double root, l1, l2
    cdef double piA, piB, piAB
    cdef int nA, nB, lenR, iA, iB
    cdef int n, i

    nA = decayA.shape[0] / 2
    nB = decayB.shape[0] / 2

    kAB, kBA = k[0], k[1]
    pxA, pxB = px[0], px[1]
    pmA, pmB = pm[0], pm[1]

    #tmp arrays for the return arguments
    lenR = 2 * nA * nB + nA + nB
    cdef double *c = <double *>malloc(lenR * sizeof(double))
    cdef double *tau = <double *>malloc(lenR * sizeof(double))

    ####  PDDEM-calculations ####
    # initial probabilities
    piA = (pAB[0] * (1.0 - pAB[1])) / (1.0 - pAB[0] * pAB[1])
    piB = (pAB[1] * (1.0 - pAB[0])) / (1.0 - pAB[0] * pAB[1])
    piAB = 1.0 - piA - piB

    n = 0
    for iA in range(nA):
        for iB in range(nB):
            cA = decayA[2 * iA]
            cB = decayB[2 * iB]
            itauA = 1. / decayA[2 * iA + 1]
            itauB = 1. / decayB[2 * iB + 1]
            root = sqrt((itauA - itauB + kAB - kBA) ** 2 + 4 * kAB * kBA)
            l1 = 0.5 * (-itauA - itauB - kAB - kBA + root)
            l2 = l1 - root

            ci = (pmA * (pxA * (-l2 - itauA - kAB) + pxB * kBA) + pmB * (pxA * kAB + pxB * (-l2 - itauB - kBA)))
            ci *= piAB * cA * cB / (l1 - l2)
            if abs(ci) > 1e-10:
                c[n] = ci
                tau[n] = -1 / l1
                n += 1
            ci = (pmA * (pxA * (l1 + itauA + kAB) - pxB * kBA) + pmB * (-pxA * kAB + pxB * (l1 + itauB + kBA)))
            ci *= piAB * cA * cB / (l1 - l2)
            if abs(ci) > 1e-10:
                c[n] = ci
                tau[n] = -1 / l2
                n += 1

    #  adding pureA, pureB
    for iA in range(nA):
        ci = pmA * pxA * piA * decayA[2 * iA]
        if abs(ci) > 1e-10:
            c[n] = ci
            tau[n] = decayA[2 * iA + 1]
            n += 1
    for iB in range(nB):
        ci = pmB * pxB * piB * decayB[2 * iB]
        if abs(ci) > 1e-10:
            c[n] = ci
            tau[n] = decayB[2 * iB + 1]
            n += 1
    cdef np.ndarray[np.float64_t, ndim=1] d = np.empty(2 * n, dtype=np.float64)
    for i in range(n):
        d[2 * i] = c[i]
        d[2 * i + 1] = tau[i]
    return d





















