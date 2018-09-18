from . import _tcspc
import numpy as np
from copy import deepcopy
from math import exp, ceil, sqrt, log
import numba as nb
import mfm


@nb.jit(nopython=True, nogil=True)
def get_scale_bg(fit, data, data_weight, bg, start, stop):
    """This function calculates a scaling factor for a given
    experimental histogram and model function. The scaling-factor
    scales the model function that the weighted photon counts
    agree

    :param fit:
    :param data:
    :param data_weight: 
    :param bg:
    :param start:
    :param stop:
    :return: scaling factor (float)
    """
    w = data_weight[start:stop]
    f = fit[start:stop]
    d = data[start:stop]

    w2 = w**2
    d_bg = np.maximum(d - bg, 0)

    sumnom = np.dot(d_bg * f, w2)
    sumdenom = np.dot(f * f, w2)
    scale = sumnom / sumdenom

    return scale


def bin_lifetime_spectrum(lifetime_spectrum, n_lifetimes, discriminate, discriminator=None):
    """Takes a interleaved lifetime spectrum

    :param lifetime_spectrum: interleaved lifetime spectrum
    :param dt_max: threshold below which two lifetimes are considered as identical
    :return: lifetime_spectrum
    """
    amplitudes, lifetimes = mfm.fluorescence.interleaved_to_two_columns(lifetime_spectrum, sort=False)
    lt, am = mfm.math.functions.datatools.histogram1D(lifetimes, amplitudes, n_lifetimes)
    if discriminate and discriminator is not None:
        lt, am = mfm.math.functions.datatools.discriminate(lt, am, discriminator)
    binned_lifetime_spectrum = mfm.fluorescence.two_column_to_interleaved(am, lt)
    return binned_lifetime_spectrum


@nb.jit(nopython=True, nogil=True)
def rescale_w_bg(fit, decay, w_res, bg, start, stop):
    scale = 0.0
    sumnom = 0.0
    sumdenom = 0.0
    for i in range(start, stop):
        iwsq = 1.0/(w_res[i]*w_res[i]+1e-12)
        if decay[i] != 0.0:
            sumnom += fit[i]*(decay[i]-bg)*iwsq
            sumdenom += fit[i]*fit[i]*iwsq
    if sumdenom != 0.0:
        scale = sumnom / sumdenom
    for i in range(start, stop):
        fit[i] *= scale
    return scale


@nb.jit(nopython=True, nogil=True)
def fconv(decay, lifetime_spectrum, irf, stop, t):
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
    for i in xrange(stop):
        decay[i] = 0.0
    for ne in xrange(nExp):
        x_curr = lifetime_spectrum[2 * ne]
        if x_curr == 0.0:
            continue
        lt = (lifetime_spectrum[2 * ne + 1])
        if lt == 0.0:
            continue
        fitCurr = 0.0
        for i in xrange(stop):
            dt = t[i + 1] - t[i]
            delta_tHalf = dt / 2.
            currExp = exp(-dt / lt)
            fitCurr = (fitCurr + delta_tHalf*irf[i-1])*currExp + delta_tHalf*irf[i]
            decay[i] += fitCurr * x_curr


@nb.jit(nopython=True, nogil=True)
def fconv_per_cs(decay, lifetime_spectrum, irf, start, stop, n_points, period, dt, conv_stop):
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
def fconv_per(decay, lifetime_spectrum, irf, start, stop, n_points, period, dt):
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
def fconv_per_dt(decay, lifetime_spectrum, irf, start, stop, n_points, period, time):
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
def pddem(decayA, decayB, k, px, pm, pAB):
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
    #return _tcspc.pddem(decayA, decayB, k, px, pm, pAB)

    nA = decayA.shape[0] / 2
    nB = decayB.shape[0] / 2

    kAB, kBA = k[0], k[1]
    pxA, pxB = px[0], px[1]
    pmA, pmB = pm[0], pm[1]

    #tmp arrays for the return arguments
    lenR = 2 * nA * nB + nA + nB
    c = np.empty(lenR, dtype=np.float64)
    tau = np.empty(lenR, dtype=np.float64)

    ####  PDDEM-calculations ####
    # initial probabilities
    piA = (pAB[0] * (1.0 - pAB[1])) / (1.0 - pAB[0] * pAB[1])
    piB = (pAB[1] * (1.0 - pAB[0])) / (1.0 - pAB[0] * pAB[1])
    piAB = 1.0 - piA - piB

    n = 0
    for iA in range(nA):
        for iB in range(nB):
            cA = decayA[2 * iA]
            if cA == 0.0:
                continue

            cB = decayB[2 * iB]
            if cB == 0.0:
                continue

            tauA = decayA[2 * iA + 1]
            if tauA == 0.0:
                continue
            itauA = 1. / tauA

            tauB = decayB[2 * iB + 1]
            if tauB == 0.0:
                continue
            itauB = 1. / tauB

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

    d = np.empty(2 * n, dtype=np.float64)
    for i in range(n):
        d[2 * i] = c[i]
        d[2 * i + 1] = tau[i]
    return d


@nb.jit(nopython=True, nogil=True)
def pile_up(data, model, rep_rate, dead_time, measurement_time, verbose=False):
    """
    Add pile up effect to model function.
    Attention: This changes the scaling of the model function.

    :param rep_rate: float
        The repetition-rate in MHz
    :param dead_time: float
        The dead-time of the system in nanoseconds
    :param measurement_time: float
        The measurement time in seconds
    :param data: numpy-array
        The array containing the experimental decay
    :param model: numpy-array
        The array containing the model function

    References
    ----------

    .. [1]  Coates, P.B., A fluorimetric attachment for an
            atomic-absorption spectrophotometer
            1968 J. Phys. E: Sci. Instrum. 1 878

    .. [2]  Walker, J.G., Iterative correction for pile-up in
            single-photon lifetime measurement
            2002 Optics Comm. 201 271-277

    """
    rep_rate *= 1e6
    dead_time *= 1e-9
    cum_sum = np.cumsum(data)
    n_pulse_detected = cum_sum[-1]
    total_dead_time = n_pulse_detected * dead_time
    live_time = measurement_time - total_dead_time
    n_excitation_pulses = max(live_time * rep_rate, n_pulse_detected)
    if verbose:
        print "------------------"
        print "rep. rate [Hz]: %s" % rep_rate
        print "live time [s]: %s" % live_time
        print "dead time per pulse [s]: %s" % dead_time
        print "n_pulse: %s" % n_excitation_pulses
        print "dead [s]: %s" % total_dead_time

    c = n_excitation_pulses - np.cumsum(data)
    rescaled_data = -np.log(1.0 - data / c)
    rescaled_data[rescaled_data == 0] = 1.0
    sf = data / rescaled_data
    sf = sf / np.sum(sf) * len(data)
    model *= sf


def dnl_table(y, window_length, window_function, xmin, xmax):
    """
    This function calculates a liberalization table for differential non-linearities given a measurement of
    uncorrelated light. The linearization table is smoothed to reduce the noise contained in the liberalization
    measurements.
    :return:
    """
    x2 = deepcopy(y)
    x2 /= x2[xmin:xmax].mean()
    mnx = np.ma.array(x2)
    mask2 = np.array([i < xmin or i > xmax for i in range(len(x2))])
    mnx.mask = mask2
    mnx.fill_value = 1.0
    mnx /= mnx.mean()
    yn = mnx.filled()
    return mfm.math.signal.window(yn, window_length, window_function)


@nb.jit(nopython=True, nogil=True)
def sconv(fit, p, irf, start, stop, dt):
    for i in range(start, stop):
        fit[i] = 0.5 * irf[0] * p[i]
        for j in range(1, i):
            fit[i] += irf[j] * p[i-j]
        fit[i] += 0.5 * irf[i] * p[0]
        fit[i] *= dt
    fit[0] = 0
