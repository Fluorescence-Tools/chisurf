from __future__ import annotations

from math import exp, ceil
import numba as nb
import numpy as np


@nb.jit(nopython=True, nogil=True)
def fconv(
        decay: np.array,
        lifetime_spectrum: np.array,
        irf: np.array,
        stop: int,
        t: np.array
):
    """
    :param decay: numpy-array
        the content of this array is overwritten by the y-values after convolution
    :param lifetime_spectrum: vector of amplitdes and lifetimes in form: amplitude, lifetime
    :param irf:
    :param stop:
    :param dt:
    :return:
    """
    nExp = lifetime_spectrum.shape[0] / 2
    #delta_tHalf = dt/2.
    for i in range(stop):
        decay[i] = 0.0
    for ne in range(nExp):
        x_curr = lifetime_spectrum[2 * ne]
        if x_curr == 0.0:
            continue
        lt = (lifetime_spectrum[2 * ne + 1])
        if lt == 0.0:
            continue
        fitCurr = 0.0
        for i in range(stop):
            dt = t[i + 1] - t[i]
            delta_tHalf = dt / 2.
            currExp = exp(-dt / lt)
            fitCurr = (fitCurr + delta_tHalf*irf[i-1])*currExp + delta_tHalf*irf[i]
            decay[i] += fitCurr * x_curr


@nb.jit(nopython=True, nogil=True)
def fconv_per_cs(
        decay: np.array,
        lifetime_spectrum: np.array,
        irf: np.array,
        start: int,
        stop: int,
        n_points: int,
        period: float,
        dt: float,
        conv_stop: int
):
    """

    :param decay: array of doubles
        Here the convolved fit is stored
    :param lifetime_spectrum: array of doubles
        Lifetime-spectrum of the form (amplitude, lifetime, amplitude, lifetime, ...)
    :param irf: array-doubles
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
    """
    stop = min(stop, n_points - 1)
    start = max(start, 0)

    n_exp = lifetime_spectrum.shape[0] / 2
    dt_half = dt * 0.5
    period_n = ceil(period/dt-0.5)

    for i in range(start, stop):
        decay[i] = 0

    stop1 = min(n_points, period_n)

    # convolution
    for ne in range(n_exp):
        x_curr = lifetime_spectrum[2 * ne]
        if x_curr == 0.0:
            continue
        lt_curr = lifetime_spectrum[2 * ne + 1]
        if lt_curr == 0.0:
            continue

        tail_a = 1./(1.-exp(-period/lt_curr))
        exp_curr = exp(-dt/lt_curr)

        fit_curr = 0.
        decay[0] += dt_half * irf[0] * (exp_curr + 1.) * x_curr

        for i in range(conv_stop):
            fit_curr = (fit_curr + dt_half * irf[i - 1]) * exp_curr + dt_half * irf[i]
            decay[i] += fit_curr * x_curr

        for i in range(conv_stop, stop1):
            fit_curr *= exp_curr
            decay[i] += fit_curr * x_curr

        fit_curr *= exp(-(period_n - stop1)*dt/lt_curr)
        for i in range(stop):
            fit_curr *= exp_curr
            decay[i] += fit_curr * x_curr * tail_a


@nb.jit(nopython=True, nogil=True)
def fconv_per(
        decay: np.array,
        lifetime_spectrum: np.array,
        irf: np.array,
        start: int,
        stop: int,
        n_points: int,
        period: float,
        dt: float
):
    dt_half = dt * 0.5
    decay *= 0.0

    x = lifetime_spectrum
    n_exp = x.shape[0]/2

    period_n = int(ceil(period/dt-0.5))
    stop1 = min(period_n, n_points - 1)

    for ne in range(n_exp):
        x_curr = lifetime_spectrum[2 * ne]
        if x_curr == 0.0: continue
        lt_curr = lifetime_spectrum[2 * ne + 1]
        if lt_curr == 0.0: continue

        exp_curr = exp(-dt / lt_curr)
        tail_a = 1./(1.-exp(-period/lt_curr))

        fit_curr = 0.0
        for i in range(stop1):
            fit_curr = (fit_curr + dt_half * irf[i-1]) * exp_curr + dt_half * irf[i]
            decay[i] += fit_curr * x_curr
        fit_curr *= exp(-(period_n - stop1 + start)*dt/lt_curr)

        for i in range(start, stop):
            fit_curr *= exp_curr
            decay[i] += fit_curr * x_curr * tail_a
    return 0


@nb.jit(nopython=True, nogil=True)
def fconv_per_dt(
        decay,
        lifetime_spectrum,
        irf,
        start,
        stop,
        n_points,
        period,
        time
):
    # TODO: in future adaptive time-axis with increasing bin size
    x = lifetime_spectrum
    n_exp = x.shape[0]/2

    #period_n = int(ceil(period/dt-0.5))


    irf_start = 0
    while irf[irf_start] == 0:
        irf_start += 1

    decay *= 0.0
    #for i in range(stop):
    #    decay[i] = 0.0

    #stop1 = n_points - 1 if period_n + irf_start > n_points - 1 else period_n + irf_start
    #if period_n + irf_start > n_points - 1:
    #    stop1 = n_points - 1
    #else:
    #    stop1 = period_n + irf_start

    for ne in range(n_exp):
        x_curr = lifetime_spectrum[2 * ne]
        if x_curr == 0.0:
            continue
        lt_curr = lifetime_spectrum[2 * ne + 1]
        if lt_curr == 0.0:
            continue

        #tail_a = 1./(1.-exp(-period/lt_curr))

        fit_curr = 0.0
        for i in range(1, n_points):
            dt = (time[i] - time[i-1])
            dt_half = dt * 0.5
            exp_curr = exp(-dt / lt_curr)
            fit_curr = (fit_curr + dt_half * irf[i-1]) * exp_curr + dt_half * irf[i]
            decay[i] += fit_curr * x_curr
        #fit_curr *= exp(-(period_n - stop1 + start)*dt/lt_curr)

        #for i in range(start, stop):
        #    fit_curr *= exp_curr
        #    decay[i] += fit_curr * x_curr * tail_a
    return 0


@nb.jit(nopython=True, nogil=True)
def sconv(fit, p, irf, start, stop, dt):
    for i in range(start, stop):
        fit[i] = 0.5 * irf[0] * p[i]
        for j in range(1, i):
            fit[i] += irf[j] * p[i-j]
        fit[i] += 0.5 * irf[i] * p[0]
        fit[i] *= dt
    fit[0] = 0