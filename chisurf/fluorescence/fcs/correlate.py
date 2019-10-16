"""

"""
from __future__ import annotations
from typing import Dict

from math import floor, pow
import numba as nb
import numpy as np


@nb.jit(nopython=True, nogil=True)
def correlate(
        n: int,
        B: int,
        t: np.ndarray,
        taus: np.ndarray,
        corr: np.ndarray,
        w: np.ndarray
) -> np.array:
    """

    :param n:
    :param B:
    :param t:
    :param taus:
    :param corr:
    :param w:
    :return:
    """
    for b in range(B):
        j = (n * B + b)
        shift = taus[j] // (pow(2.0, float(j / B)))
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
    return corr


def normalize(
        np1: int,
        np2: int,
        dt1: float,
        dt2: float,
        tau: np.array,
        corr: np.array,
        B: int
) -> float:
    """

    :param np1: number of photons in channel 1
    :param np2: number of photons in channel 2
    :param dt1: the total measurement time of the data contained in channel 1
    :param dt2: the total measurement time of the data contained in channel 2
    :param tau: the array containing the correlation times
    :param corr: the array containing the correlation values
    :param B:
    :return:
    """
    cr1 = float(np1) / float(dt1)
    cr2 = float(np2) / float(dt2)
    for j in range(corr.shape[0]):
        pw = 2.0 ** int(j / B)
        tCor = dt1 if dt1 < dt2 - tau[j] else dt2 - tau[j]
        corr[j] /= (tCor * float(pw))
        corr[j] /= (cr1 * cr2)
        tau[j] = tau[j] // pw * pw
    return float(min(cr1, cr2))


@nb.jit
def get_weights(
        rout: np.array,
        tac: np.array,
        wt: np.array,
        number_of_photons: int
):
    """

    :param rout:
    :param tac:
    :param wt:
    :param number_of_photons:
    :return:
    """
    w = np.zeros(number_of_photons, dtype=np.float32)
    for i in range(number_of_photons):
        w[i] = wt[rout[i], tac[i]]
    return w


@nb.jit
def count_rate_filter(
        mt: np.array,
        tw: int,
        n_ph_max: int,
        w: np.array,
        n_ph: int
):
    """

    :param mt:
    :param tw:
    :param n_ph_max:
    :param w:
    :param n_ph:
    :return:
    """
    i = 0
    while i < n_ph - 1:
        r = i
        i_ph = 0
        while (mt[r] - mt[i]) < tw and r < n_ph - 1:
            r += 1
            i_ph += 1
        if i_ph > n_ph_max:
            for k in range(i, r):
                w[k] = 0
        i = r


@nb.jit(nopython=True)
def make_fine(
        t: np.array,
        tac: np.array,
        number_of_tac_channels: int
):
    """

    :param t:
    :param tac:
    :param number_of_tac_channels:
    :return:
    """
    for i in range(1, t.shape[0]):
        t[i] = t[i] * number_of_tac_channels + tac[i]


@nb.jit
def count_photons(
        w: np.array
):
    """

    :param w:
    :return:
    """
    k = np.zeros(w.shape[0], dtype=np.uint64)
    for j in range(w.shape[0]):
        for i in range(w.shape[1]):
            if w[j, i] != 0.0:
                k[j] += 1
    return k


@nb.jit(nopython=True, nogil=True)
def compact(
        t: np.array,
        w: np.array,
        full: bool = False
) -> None:
    """

    :param t:
    :param w:
    :param full:
    :return:
    """
    for j in range(t.shape[0]):
        k = 1
        r = t.shape[1] if full else t[j, 0]
        for i in range(1, r):
            if t[j, k] != t[j, i] and w[j, i] != 0:
                k += 1
                t[j, k] = t[j, i]
                w[j, k] = w[j, i]
        t[j, 0] = k - 1


@nb.jit(nopython=True, nogil=True)
def coarsen(
        times: np.array,
        weights: np.array
) -> None:
    """

    :param times:
    :param weights:
    :return:
    """
    for j in range(times.shape[0]):
        times[j, 1] /= 2
        for i in range(2, times[j, 0]):
            times[j, i] /= 2
            if times[j, i - 1] == times[j, i]:
                weights[j, i - 1] += weights[j, i]
                weights[j, i] = 0.0
    compact(times, weights, False)


def log_corr(
        macro_times: np.array,
        tac_channels: np.array,
        rout: np.array,
        cr_filter: np.array,
        weights_1: np.array,
        weights_2: np.array,
        B: int,
        nc: int,
        fine: bool,
        number_of_tac_channels: int,
        verbose: bool = False
) -> Dict:
    """Correlate macros-times and micro-times using a logarit

    :param macro_times: the macros-time array
    :param tac_channels: the micro-time array
    :param rout: the array of routing channels
    :param cr_filter:
    :param weights_1:
    :param weights_2:
    :param B:
    :param nc:
    :param fine:
    :param number_of_tac_channels:
    :return:
    """

    # correlate with TAC
    if fine > 0:
        make_fine(macro_times, tac_channels, number_of_tac_channels)
    # make 2 corr-channels
    t = np.vstack([macro_times, macro_times])
    w = np.vstack([weights_1 * cr_filter, weights_2 * cr_filter])
    np1, np2 = count_photons(w)
    compact(t, w, True)
    # MACRO-Times
    mt1max, mt2max = t[0, t[0, 0]], t[1, t[1, 0]]
    mt1min, mt2min = t[0, 1], t[1, 1]
    dt1 = mt1max - mt1min
    dt2 = mt2max - mt2min
    # calculate tau axis
    taus = np.zeros(nc * B, dtype=np.uint64)
    corr = np.zeros(nc * B, dtype=np.float32)
    for j in range(1, nc * B):
        taus[j] = taus[j - 1] + pow(2.0, floor(j / B))
    # correlation
    for n in range(nc):
        if verbose:
            print("cascade %s\tnph1: %s\tnph2: %s" % (n, t[0, 0], t[1, 0]))
        corr = correlate(
            n=n,
            B=B,
            t=t,
            taus=taus,
            corr=corr,
            w=w
        )
        coarsen(t, w)
    results = {
        'number_of_photons_ch1': np1,
        'number_of_photons_ch2': np2,
        'measurement_time_ch1': dt1,
        'measurement_time_ch2': dt2,
        'correlation_time_axis': taus,
        'correlation_amplitude': corr
    }
    return results

