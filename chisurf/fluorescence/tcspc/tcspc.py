from __future__ import annotations

from math import sqrt

import numba as nb
import numpy as np
import deprecation

import chisurf.math
import chisurf.math.datatools


# bin_lifetime_spectrum = skf.decay.rate_spectra.bin_lifetime_spectrum
def bin_lifetime_spectrum(
    lifetime_spectrum: np.array,
    n_lifetimes: int,
    discriminate: bool,
    discriminator=None
) -> np.array:
    """Takes an interleaved lifetime spectrum

    :param lifetime_spectrum: interleaved lifetime spectrum
    :param n_lifetimes:
    :param discriminate:
    :param discriminator:
    :return: lifetime_spectrum
    """
    amplitudes, lifetimes = chisurf.math.datatools.interleaved_to_two_columns(
        lifetime_spectrum,
        sort=False
    )
    print(lifetimes)
    print(amplitudes)
    lt, am = chisurf.math.datatools.histogram1D(
        values=lifetimes,
        weights=amplitudes,
        n_bins=n_lifetimes
    )
    if discriminate and discriminator is not None:
        lt, am = chisurf.math.datatools.discriminate(
            values=lt,
            weights=am,
            discriminator=discriminator
        )
    binned_lifetime_spectrum = chisurf.math.datatools.two_column_to_interleaved(
        x=am,
        t=lt
    )
    return binned_lifetime_spectrum


@nb.jit(nopython=True, nogil=True)
def rescale_w_bg(
        model_decay: np.array,
        experimental_decay: np.array,
        experimental_weights: np.array,
        experimental_background: float,
        start: int,
        stop: int
) -> float:
    """Computes a scaling factor that scales a model decay to an
    experimental decay on a defined range.

    Parameters
    ----------
    model_decay
    experimental_decay
    experimental_weights
    experimental_background
    start
    stop

    Returns
    -------
    float:
        The scaling factor that was used to scale the model function to the
        experimental decay.

    """
    scale = 0.0
    sum_nom = 0.0
    sum_denom = 0.0
    w = experimental_weights
    e = experimental_decay
    b = experimental_background
    m = model_decay
    for i in range(start, stop):
        if e[i] > 0.0:
            iwsq = 1.0 / (w[i] * w[i] + 1e-12)
            sum_nom += m[i] * (e[i] - b) * iwsq
            sum_denom += m[i] * m[i] * iwsq
    if sum_denom != 0.0:
        scale = sum_nom / sum_denom
    return scale


@nb.jit (nopython=True, nogil=True)
def pddem(
        decayA: np.ndarray,
        decayB: np.ndarray,
        k: np.ndarray,
        px: np.ndarray,
        pm: np.ndarray,
        pAB: np.ndarray
):
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

    :param decayA: model_decay A in form of [ampl lifetime, apml, lifetime...]
    :param decayB: model_decay B in form of [ampl lifetime, apml, lifetime...]
    :param k: rates of energy transfer [kAB, kBA]
    :param px: probabilities of excitation (pxA, pxB)
    :param pm: probabilities of emission (pmA, pmB)
    :param pAB: pure AB [0., 0]
    :return:
    """
    #return _tcspc.pddem(decayA, decayB, k, px, pm, pAB)
    eps = 1e-9

    nA = decayA.shape[0] // 2
    nB = decayB.shape[0] // 2

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
            ci *= piAB * cA * cB / (l1 - l2 + eps)
            if abs(ci) > 1e-10:
                c[n] = ci
                tau[n] = -1 / l1
                n += 1
            ci = (pmA * (pxA * (l1 + itauA + kAB) - pxB * kBA) + pmB * (-pxA * kAB + pxB * (l1 + itauB + kBA)))
            ci *= piAB * cA * cB / (l1 - l2 + eps)
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


