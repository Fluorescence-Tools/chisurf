"""
Module for multi-tau correlation routines.

This module implements several functions for processing time-correlated
single photon counting (TCSPC) data, including multi-tau correlation, photon
weight retrieval, count rate filtering, and data coarsening. The multi-tau
algorithm used here is based on the design described by Schätzel et al.,
Rev. Sci. Instrum. 66, 4276 (1995).
"""

from __future__ import annotations
from typing import Dict
import deprecation

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
    Compute the multi-tau correlation amplitude for a single cascade step.

    This function performs a multi-tau correlation calculation for a given cascade
    (specified by the parameter n) and channel block (B channels per cascade step).
    It uses photon arrival times and their weights to update the correlation amplitude
    array. The algorithm implements a cascade structure where each subsequent step
    coarsens the time resolution by a factor of 2.

    For further details on the multi-tau correlation algorithm, see:

      Schätzel, K., et al., "The design and implementation of a multi-tau correlator",
      Rev. Sci. Instrum. 66, 4276 (1995).

    Parameters
    ----------
    n : int
        The index of the current coarsening (cascade) step.
    B : int
        The number of correlation channels per cascade step.
    t : np.ndarray
        2D array of photon arrival times. The first element in each row (t[0,0] and t[1,0])
        holds the number of photons in that channel.
    taus : np.ndarray
        Array of correlation lag times.
    corr : np.ndarray
        Array holding the correlation amplitudes; updated in place.
    w : np.ndarray
        2D array of photon weights corresponding to the arrival times in t.

    Returns
    -------
    np.ndarray
        Updated correlation amplitude array.
    """
    for b in range(B):
        j = (n * B + b)
        shift = taus[j] // (pow(2.0, float(j // B)))
        # STARTING CHANNEL: determine active/inactive channels based on first two photon times.
        ca = 0 if t[0, 1] < t[1, 1] else 1  # currently active correlation channel
        ci = 1 if t[0, 1] < t[1, 1] else 0  # currently inactive correlation channel
        # POSITION ON ARRAY
        pa, pi = 0, 1  # positions on active and inactive channels
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


@deprecation.deprecated(
        deprecated_in="19.10.31",
        current_version="19.08.23",
        details="Correlation should be done using tttrlib"
    )
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
    Normalize the correlation amplitude array to yield a proper correlation function.

    The normalization procedure adjusts the raw correlation values based on the
    effective measurement time for each lag (taking into account the multi-tau
    binning) and the count rates in two channels.

    Parameters
    ----------
    np1 : int
        Number of photons in channel 1.
    np2 : int
        Number of photons in channel 2.
    dt1 : float
        Total measurement time for channel 1.
    dt2 : float
        Total measurement time for channel 2.
    tau : np.array
        Array of correlation lag times.
    corr : np.array
        Array of raw correlation amplitudes (to be normalized in place).
    B : int
        Number of correlation channels per coarsening step.

    Returns
    -------
    float
        The minimum of the two photon count rates (after normalization).

    Notes
    -----
    This function divides the correlation values by the effective measurement time
    for each lag and by the product of the count rates from the two channels.

    Deprecated
    ----------
    This function is deprecated since version 19.10.31. Use tttrlib for correlation
    computations instead.
    """
    cr1 = float(np1) / float(dt1)
    cr2 = float(np2) / float(dt2)
    for j in range(corr.shape[0]):
        pw = 2.0 ** int(j // B)
        tCor = dt1 if dt1 < dt2 - tau[j] else dt2 - tau[j]
        corr[j] /= (tCor * float(pw))
        corr[j] /= (cr1 * cr2)
        tau[j] = tau[j] / pw * pw
    return float(min(cr1, cr2))


@nb.jit(nopython=True)
def get_weights(
        routing_channels: np.ndarray,
        micro_times: np.ndarray,
        weights: np.ndarray,
        number_of_photons: int
) -> np.ndarray:
    """
    Retrieve photon weights based on routing channels and micro-times.

    For each photon, this function returns its weight by indexing the given weights
    array using the routing channel and micro-time of that photon.

    Parameters
    ----------
    routing_channels : np.ndarray
        Array of routing channel indices for each photon.
    micro_times : np.ndarray
        Array of micro-time (TAC) values for each photon.
    weights : np.ndarray
        2D array of weight values indexed by routing channel and micro-time.
    number_of_photons : int
        Total number of photons to process.

    Returns
    -------
    np.ndarray
        1D array of photon weights.
    """
    w = np.zeros(number_of_photons, dtype=np.float32)
    for i in range(number_of_photons):
        w[i] = weights[routing_channels[i], micro_times[i]]
    return w


@nb.jit(nopython=True)
def count_rate_filter(
        macro_times: np.ndarray,
        time_window: int,
        n_ph_max: int,
        weights: np.ndarray,
        n_ph: int
):
    """
    Apply a count-rate filter to photon weights.

    This function scans through the macro-times array and counts the number of photons
    within a given time window. If the number exceeds the maximum allowed (n_ph_max),
    the weights for those photons are set to zero to suppress contributions from periods
    of excessive count rate.

    Parameters
    ----------
    macro_times : np.ndarray
        Array of macro-time photon arrival times.
    time_window : int
        Time window (in the same units as macro_times) over which to count photons.
    n_ph_max : int
        Maximum allowed number of photons within a time window.
    weights : np.ndarray
        Array of photon weights (modified in place).
    n_ph : int
        Total number of photons.

    Returns
    -------
    None
    """
    i = 0
    while i < n_ph - 1:
        r = i
        i_ph = 0
        while (macro_times[r] - macro_times[i]) < time_window and r < n_ph - 1:
            r += 1
            i_ph += 1
        if i_ph > n_ph_max:
            for k in range(i, r):
                weights[k] = 0
        i = r


@nb.jit(nopython=True)
def make_fine(
        t: np.array,
        tac: np.array,
        number_of_tac_channels: int
):
    """
    Refine macro-time values using micro-time (TAC) information.

    This function adjusts the macro-time array by incorporating the TAC channel values.
    The refined time for each photon is computed as:

        refined_time = original_time * number_of_tac_channels + tac_value

    Parameters
    ----------
    t : np.array
        Array of macro-time photon arrival times.
    tac : np.array
        Array of micro-time (TAC) values for each photon.
    number_of_tac_channels : int
        Number of TAC channels used in the measurement.

    Returns
    -------
    None
        The function modifies the array t in place.
    """
    for i in range(1, t.shape[0]):
        t[i] = t[i] * number_of_tac_channels + tac[i]


@nb.jit(nopython=True)
def count_photons(
        w: np.array
):
    """
    Count the number of photons in each channel based on the weight array.

    For each row in the weight array, this function counts the number of nonzero
    weight entries, which correspond to detected photons.

    Parameters
    ----------
    w : np.array
        2D array of photon weights.

    Returns
    -------
    np.array
        1D array where each element is the photon count for the corresponding row.
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
    Compact the time and weight arrays by removing redundant entries.

    This function compresses the given time (t) and weight (w) arrays by eliminating
    consecutive duplicate time entries (when the corresponding weight is zero). The
    parameter 'full' determines whether to compact the entire row or only up to the
    photon count specified in t[j, 0].

    Parameters
    ----------
    t : np.array
        2D array of time values.
    w : np.array
        2D array of corresponding weights.
    full : bool, optional
        If True, compaction is applied to the entire row; otherwise, only up to
        the current photon count (default is False).

    Returns
    -------
    None
        The arrays t and w are modified in place.
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
    Coarsen the time and weight arrays for multi-tau correlation.

    This function reduces the resolution of the time and weight arrays by dividing
    the time values by 2 (except for the first element) in each row. When consecutive
    time values become equal after coarsening, their weights are merged. Finally, the
    arrays are compacted to remove zero-weight entries.

    Parameters
    ----------
    times : np.array
        2D array of time values.
    weights : np.array
        2D array of corresponding weights.

    Returns
    -------
    None
        The arrays are modified in place.
    """
    for j in range(times.shape[0]):
        times[j, 1] //= 2
        for i in range(2, times[j, 0]):
            times[j, i] //= 2
            if times[j, i - 1] == times[j, i]:
                weights[j, i - 1] += weights[j, i]
                weights[j, i] = 0.0
    compact(times, weights, False)


@deprecation.deprecated(
        deprecated_in="19.10.31",
        current_version="19.08.23",
        details="Correlation should be done using tttrlib"
    )
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
    """
    Perform logarithmic correlation on macro- and micro-time photon data.

    This function computes the correlation function of photon arrival times using a
    multi-tau algorithm with logarithmically spaced lag times. It optionally refines
    the macro-times using micro-time (TAC) data (if 'fine' is True) and constructs two
    correlation channels by stacking the macro-time and weight arrays. The correlation
    is then computed in a cascade of 'nc' coarsening steps (each with B channels)
    using the correlate() function. After each cascade, the arrays are coarsened.

    Parameters
    ----------
    macro_times : np.array
        Array of macro-time photon arrival times.
    tac_channels : np.array
        Array of micro-time (TAC) values.
    rout : np.array
        Array of routing channels for each photon.
    cr_filter : np.array
        Count rate filter array applied to weight the photon events.
    weights_1 : np.array
        Weight array for channel 1.
    weights_2 : np.array
        Weight array for channel 2.
    B : int
        Number of correlation channels per cascade.
    nc : int
        Number of cascades (coarsening steps) to perform.
    fine : bool
        If True, refine macro-times with TAC channel information.
    number_of_tac_channels : int
        Number of TAC channels.
    verbose : bool, optional
        If True, prints diagnostic messages.

    Returns
    -------
    Dict
        A dictionary containing:
          - 'number_of_photons_ch1': Photon count in channel 1.
          - 'number_of_photons_ch2': Photon count in channel 2.
          - 'measurement_time_ch1': Total measurement time for channel 1.
          - 'measurement_time_ch2': Total measurement time for channel 2.
          - 'correlation_time_axis': Array of correlation lag times.
          - 'correlation_amplitude': Array of computed correlation amplitudes.

    References
    ----------
    Schätzel, K., et al., "The design and implementation of a multi-tau correlator",
    Rev. Sci. Instrum. 66, 4276 (1995).

    Notes
    -----
    This function is deprecated. It is recommended to perform correlation calculations
    using tttrlib.
    """
    # Optionally refine macro-times using TAC channels.
    if fine > 0:
        make_fine(macro_times, tac_channels, number_of_tac_channels)
    # Stack macro-time arrays for two correlation channels.
    t = np.vstack([macro_times, macro_times])
    w = np.vstack([weights_1 * cr_filter, weights_2 * cr_filter])
    np1, np2 = count_photons(w)
    compact(t, w, True)
    # Determine measurement times from macro-time arrays.
    mt1max, mt2max = t[0, t[0, 0]], t[1, t[1, 0]]
    mt1min, mt2min = t[0, 1], t[1, 1]
    dt1 = mt1max - mt1min
    dt2 = mt2max - mt2min
    # Initialize tau axis and correlation amplitude array.
    taus = np.zeros(nc * B, dtype=np.uint64)
    corr = np.zeros(nc * B, dtype=np.float32)
    for j in range(1, nc * B):
        taus[j] = taus[j - 1] + pow(2.0, floor(j / B))
    # Perform cascaded correlation.
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
