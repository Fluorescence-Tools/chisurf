from __future__ import annotations

from math import sqrt

import numba as nb
import numpy as np

import mfm
import mfm.math
import mfm.math.datatools


@nb.jit(nopython=True, nogil=True)
def get_scale_bg(
        fit: np.array,
        data: np.array,
        data_weight: np.array,
        bg: float,
        start: int,
        stop: int
) -> float:
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


def bin_lifetime_spectrum(
        lifetime_spectrum: np.array,
        n_lifetimes: int,
        discriminate: bool,
        discriminator=None
) -> np.array:
    """Takes a interleaved lifetime spectrum

    :param lifetime_spectrum: interleaved lifetime spectrum
    :param n_lifetimes:
    :param discriminate:
    :param discriminator:
    :return: lifetime_spectrum
    """
    amplitudes, lifetimes = mfm.math.datatools.interleaved_to_two_columns(lifetime_spectrum, sort=False)
    lt, am = mfm.math.datatools.histogram1D(lifetimes, amplitudes, n_lifetimes)
    if discriminate and discriminator is not None:
        lt, am = mfm.math.datatools.discriminate(lt, am, discriminator)
    binned_lifetime_spectrum = mfm.math.datatools.two_column_to_interleaved(am, lt)
    return binned_lifetime_spectrum


@nb.jit(nopython=True, nogil=True)
def rescale_w_bg(
        fit: np.array,
        decay: np.array,
        w_res: np.array,
        bg: float,
        start: int,
        stop: int
) -> float:
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


