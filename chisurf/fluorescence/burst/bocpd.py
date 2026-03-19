"""
Bayesian Online Changepoint Detection (BOCPD) for burst detection in fluorescence data.

This module implements the BOCPD algorithm for detecting bursts in fluorescence data.
The algorithm is based on the paper:
Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection.
arXiv preprint arXiv:0710.3742.

The implementation is optimized using numba for performance.
"""

import numpy as np
import tttrlib
from numba import njit, prange
from math import lgamma, log
from typing import List, Dict, Tuple, Optional, Union, Any


def bin_photons(donor: np.ndarray, acceptor: np.ndarray, dt: float = 1e-3, tmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin photon timestamps into count arrays.

    Parameters
    ----------
    donor : np.ndarray
        Array of donor photon timestamps.
    acceptor : np.ndarray
        Array of acceptor photon timestamps.
    dt : float, optional
        Bin width in seconds. Default is 1e-3 (1 ms).
    tmax : float, optional
        Maximum time value. If None, uses the maximum of donor and acceptor timestamps.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing (donor_counts, acceptor_counts, bin_edges).
    """
    if tmax is None:
        tmax = max(float(donor.max()), float(acceptor.max()))
    bins = np.arange(0.0, tmax + dt, dt)
    D, _ = np.histogram(donor, bins)
    A, _ = np.histogram(acceptor, bins)
    return D.astype(np.int64), A.astype(np.int64), bins


@njit(nopython=True)
def _logsumexp(a: np.ndarray) -> float:
    """
    Compute the log of the sum of exponentials of input elements in a numerically stable way.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    float
        Log of the sum of exponentials of input elements.
    """
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


@njit(nopython=True)
def _log_poisson_pmf(k: int, lam: float) -> float:
    """
    Compute the log of the Poisson probability mass function.

    Parameters
    ----------
    k : int
        Number of events.
    lam : float
        Expected number of events.

    Returns
    -------
    float
        Log of the Poisson probability mass function.
    """
    if lam <= 0.0:
        lam = 1e-300
    return k * log(lam) - lam - lgamma(k + 1.0)


@njit(nopython=True)
def compute_predictive(D_t: int, A_t: int, alphaD: np.ndarray, betaD: np.ndarray, 
                      alphaA: np.ndarray, betaA: np.ndarray, Rmax: int) -> np.ndarray:
    """
    Compute the predictive distribution for the current time step.

    Parameters
    ----------
    D_t : int
        Donor count at current time step.
    A_t : int
        Acceptor count at current time step.
    alphaD : np.ndarray
        Alpha parameters for donor Gamma prior.
    betaD : np.ndarray
        Beta parameters for donor Gamma prior.
    alphaA : np.ndarray
        Alpha parameters for acceptor Gamma prior.
    betaA : np.ndarray
        Beta parameters for acceptor Gamma prior.
    Rmax : int
        Maximum run length.

    Returns
    -------
    np.ndarray
        Predictive distribution for the current time step.
    """
    pred = np.empty(Rmax)
    for rl in prange(Rmax):
        lamD = alphaD[rl] / betaD[rl]
        lamA = alphaA[rl] / betaA[rl]
        pred[rl] = _log_poisson_pmf(D_t, lamD) + _log_poisson_pmf(A_t, lamA)
    return pred


@njit(nopython=True)
def bocpd_joint_poisson_optimized(D: np.ndarray, A: np.ndarray, prior_count: float = 1.0, 
                                 prior_duration: float = 1.0, changepoint_prob: float = 1e-3, 
                                 max_run: int = 500, 
                                 # Backward compatibility parameters
                                 alpha: float = None, beta: float = None, hazard: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bayesian Online Changepoint Detection with joint Poisson likelihood.

    This function implements the BOCPD algorithm with a joint Poisson likelihood
    for donor and acceptor channels. It is optimized using numba for performance.

    Parameters
    ----------
    D : np.ndarray
        Array of donor counts.
    A : np.ndarray
        Array of acceptor counts.
    prior_count : float, optional
        Prior photon count before data (alpha parameter for Gamma prior). Default is 1.0.
    prior_duration : float, optional
        Time window assumed for prior_count (beta parameter for Gamma prior). Default is 1.0.
    changepoint_prob : float, optional
        Probability of burst start in any bin (hazard rate). Default is 1e-3.
    max_run : int, optional
        Maximum run length. Default is 500.
    alpha : float, optional
        Deprecated: Use prior_count instead. Alpha parameter for Gamma prior.
    beta : float, optional
        Deprecated: Use prior_duration instead. Beta parameter for Gamma prior.
    hazard : float, optional
        Deprecated: Use changepoint_prob instead. Hazard rate (probability of changepoint).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (changepoints, run_length_map).
    """
    # Handle backward compatibility
    if alpha is not None:
        prior_count = alpha
    if beta is not None:
        prior_duration = beta
    if hazard is not None:
        changepoint_prob = hazard
    T = len(D)
    max_run = min(max_run, T)

    log_h = log(changepoint_prob)
    log_1h = np.log1p(-changepoint_prob)

    log_R_prev = np.full(max_run + 1, -np.inf)
    log_R_prev[0] = 0.0

    alphaD_buf = np.full((2, max_run + 1), prior_count)
    betaD_buf  = np.full((2, max_run + 1), prior_duration)
    alphaA_buf = np.full((2, max_run + 1), prior_count)
    betaA_buf  = np.full((2, max_run + 1), prior_duration)

    cps = []
    run_length_map = np.zeros(T, dtype=np.int64)

    for t in range(T):
        d = D[t]
        a = A[t]
        Rmax = min(t + 1, max_run)

        curr = t % 2
        prev = (t + 1) % 2

        pred = compute_predictive(d, a,
                                  alphaD_buf[prev], betaD_buf[prev],
                                  alphaA_buf[prev], betaA_buf[prev],
                                  Rmax)

        log_growth = log_R_prev[:Rmax] + pred + log_1h
        log_cp = _logsumexp(log_R_prev[:Rmax] + pred + log_h)

        log_R = np.full(max_run + 1, -np.inf)
        log_R[0] = log_cp
        log_R[1:Rmax + 1] = log_growth
        log_R[:Rmax + 1] -= _logsumexp(log_R[:Rmax + 1])

        alphaD_buf[curr][0] = prior_count + d
        betaD_buf[curr][0]  = prior_duration + 1.0
        alphaA_buf[curr][0] = prior_count + a
        betaA_buf[curr][0]  = prior_duration + 1.0

        for rl in range(1, Rmax + 1):
            alphaD_buf[curr][rl] = alphaD_buf[prev][rl - 1] + d
            betaD_buf[curr][rl]  = betaD_buf[prev][rl - 1] + 1.0
            alphaA_buf[curr][rl] = alphaA_buf[prev][rl - 1] + a
            betaA_buf[curr][rl]  = betaA_buf[prev][rl - 1] + 1.0

        rl_map = np.argmax(log_R[:Rmax + 1])
        run_length_map[t] = rl_map
        if rl_map == 0:
            cps.append(t)

        log_R_prev[:max_run + 1] = log_R

    return np.array(cps, dtype=np.int64), run_length_map


def extract_bursts(D: np.ndarray, A: np.ndarray, bins: np.ndarray, 
                  changepoints: np.ndarray, min_counts: int = 20) -> List[Dict[str, Any]]:
    """
    Extract bursts from changepoints.

    Parameters
    ----------
    D : np.ndarray
        Array of donor counts.
    A : np.ndarray
        Array of acceptor counts.
    bins : np.ndarray
        Bin edges.
    changepoints : np.ndarray
        Array of changepoint indices.
    min_counts : int, optional
        Minimum number of counts (donor + acceptor) for a burst. Default is 20.

    Returns
    -------
    List[Dict[str, Any]]
        List of burst dictionaries, each containing:
        - start_bin: Start bin index
        - end_bin: End bin index
        - start: Start time
        - end: End time
        - donor: Number of donor photons
        - acceptor: Number of acceptor photons
        - FRET: FRET efficiency (acceptor / (donor + acceptor))
    """
    cps = [0] + list(changepoints) + [len(D)]
    bursts = []
    for i in range(len(cps) - 1):
        s, e = cps[i], cps[i + 1]
        nd, na = int(D[s:e].sum()), int(A[s:e].sum())
        if nd + na >= min_counts:
            bursts.append({
                'start_bin': s,
                'end_bin': e,
                'start': bins[s],
                'end': bins[e - 1] if e - 1 < len(bins) else bins[-1],
                'donor': nd,
                'acceptor': na,
                'FRET': na / (nd + na) if (nd + na) > 0 else np.nan
            })
    return bursts


def bocpd_burst_detection(donor_timestamps: np.ndarray, acceptor_timestamps: np.ndarray, 
                         dt: float = 1e-3, prior_count: float = 1.0, prior_duration: float = 1.0, 
                         changepoint_prob: float = 0.1, max_run: int = 256, 
                         min_counts: int = 20,
                         # Backward compatibility parameters
                         alpha: float = None, beta: float = None, hazard: float = None) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform BOCPD burst detection on donor and acceptor timestamps.

    This is the main function to be called from other modules. It performs the full
    BOCPD burst detection pipeline: binning photons, running BOCPD, and extracting bursts.

    Parameters
    ----------
    donor_timestamps : np.ndarray
        Array of donor photon timestamps.
    acceptor_timestamps : np.ndarray
        Array of acceptor photon timestamps.
    dt : float, optional
        Bin width in seconds. Default is 1e-3 (1 ms).
    prior_count : float, optional
        Prior photon count before data (alpha parameter for Gamma prior). Default is 1.0.
    prior_duration : float, optional
        Time window assumed for prior_count (beta parameter for Gamma prior). Default is 1.0.
    changepoint_prob : float, optional
        Probability of burst start in any bin (hazard rate). Default is 0.1.
    max_run : int, optional
        Maximum run length. Default is 256.
    min_counts : int, optional
        Minimum number of counts (donor + acceptor) for a burst. Default is 20.
    alpha : float, optional
        Deprecated: Use prior_count instead. Alpha parameter for Gamma prior.
    beta : float, optional
        Deprecated: Use prior_duration instead. Beta parameter for Gamma prior.
    hazard : float, optional
        Deprecated: Use changepoint_prob instead. Hazard rate (probability of changepoint).

    Returns
    -------
    Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - List of burst dictionaries
        - Bin edges
        - Donor counts
        - Acceptor counts
    """
    # Handle backward compatibility
    if alpha is not None:
        prior_count = alpha
    if beta is not None:
        prior_duration = beta
    if hazard is not None:
        changepoint_prob = hazard
        
    # Bin photons
    D_counts, A_counts, bins = bin_photons(donor_timestamps, acceptor_timestamps, dt=dt)
    
    # Run BOCPD
    cps, rl_map = bocpd_joint_poisson_optimized(
        D_counts, A_counts,
        prior_count=prior_count, prior_duration=prior_duration,
        changepoint_prob=changepoint_prob,
        max_run=max_run
    )
    
    # Extract bursts
    bursts = extract_bursts(D_counts, A_counts, bins, cps, min_counts=min_counts)
    
    return bursts, bins, D_counts, A_counts


def create_burst_mask(bursts: List[Dict[str, Any]], tttr_length: int) -> np.ndarray:
    """
    Create a binary mask for bursts.

    Parameters
    ----------
    bursts : List[Dict[str, Any]]
        List of burst dictionaries from extract_bursts.
    tttr_length : int
        Length of the TTTR object.

    Returns
    -------
    np.ndarray
        Binary mask with 1s for photons in bursts and 0s elsewhere.
    """
    mask = np.zeros(tttr_length, dtype=np.uint8)
    
    for burst in bursts:
        start_idx = int(burst['start'] / tttr_length * len(mask))
        end_idx = int(burst['end'] / tttr_length * len(mask))
        mask[start_idx:end_idx+1] = 1
        
    return mask


def bin_photons_multi(timestamps_list: List[np.ndarray], dt: float = 1e-3, tmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin photon timestamps from multiple channels into count arrays.
    
    Parameters
    ----------
    timestamps_list : List[np.ndarray]
        List of arrays containing photon timestamps in seconds for each channel
    dt : float, optional
        Bin width in seconds. Default is 1e-3 (1 ms).
    tmax : float, optional
        Maximum time to consider. If None, uses the maximum timestamp across all channels.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (counts, bin_edges) where:
        - counts is a 2D array of shape (T, n_channels) containing counts per bin for each channel
        - bin_edges is an array of bin edges
    """
    if not timestamps_list:
        return np.array([]), np.array([])
    
    # Find maximum timestamp across all channels
    if tmax is None:
        tmax = 0
        for timestamps in timestamps_list:
            if len(timestamps) > 0:
                tmax = max(tmax, timestamps.max())
    
    n_bins = int(np.ceil(tmax / dt))
    n_channels = len(timestamps_list)
    
    # Initialize counts array
    counts = np.zeros((n_bins, n_channels), dtype=np.int64)
    
    # Bin timestamps for each channel
    for i, timestamps in enumerate(timestamps_list):
        if len(timestamps) > 0:
            counts[:, i], bin_edges = np.histogram(timestamps, bins=n_bins, range=(0, tmax))
        else:
            bin_edges = np.linspace(0, tmax, n_bins + 1)
    
    return counts, bin_edges


def extract_bursts_multi(counts: np.ndarray, bin_edges: np.ndarray, changepoints: np.ndarray, min_counts: int = 20) -> List[Dict[str, Any]]:
    """
    Extract bursts from changepoints for multiple channels.

    Parameters
    ----------
    counts : np.ndarray
        2D array of shape (T, n_channels) containing counts per bin for each channel
    bin_edges : np.ndarray
        Bin edges
    changepoints : np.ndarray
        Array of changepoint indices
    min_counts : int, optional
        Minimum number of counts (across all channels) for a burst. Default is 20.

    Returns
    -------
    List[Dict[str, Any]]
        List of burst dictionaries, each containing:
        - start_bin: Start bin index
        - end_bin: End bin index
        - start_time: Start time
        - end_time: End time
        - counts: List of counts for each channel
        - total_counts: Total counts across all channels
        - donor/acceptor/FRET: Only if exactly 2 channels (for backward compatibility)
    """
    cps = [0] + list(changepoints) + [len(counts)]
    bursts = []
    n_channels = counts.shape[1]
    
    for i in range(len(cps) - 1):
        s, e = cps[i], cps[i + 1]
        
        # Get counts for each channel in this burst
        channel_counts = np.sum(counts[s:e, :], axis=0)
        total_counts = np.sum(channel_counts)
        
        if total_counts >= min_counts:
            burst_dict = {
                'start_bin': s,
                'end_bin': e,
                'start': bin_edges[s],
                'end': bin_edges[e - 1] if e - 1 < len(bin_edges) else bin_edges[-1],
                'counts': channel_counts.tolist(),
                'total_counts': total_counts
            }
            
            # For backward compatibility, if we have exactly 2 channels, add donor/acceptor fields
            if n_channels == 2:
                burst_dict['donor'] = channel_counts[0]
                burst_dict['acceptor'] = channel_counts[1]
                burst_dict['FRET'] = channel_counts[1] / total_counts if total_counts > 0 else np.nan
                
            bursts.append(burst_dict)
            
    return bursts


def bocpd_burst_detection_multi(
    timestamps_list: List[np.ndarray],
    dt: float = 1e-3,
    prior_count: float = 1.0,
    prior_duration: float = 1.0,
    changepoint_prob: float = 0.1,
    max_run: int = 256,
    min_counts: int = 20,
    # Backward compatibility parameters
    alpha: float = None,
    beta: float = None,
    hazard: float = None
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Perform BOCPD burst detection on multiple channels.
    
    This function extends the original BOCPD algorithm to work with an arbitrary number of channels.
    For 2 channels, it uses the original algorithm. For more than 2 channels, it processes each pair
    of channels and combines the results.
    
    Parameters
    ----------
    timestamps_list : List[np.ndarray]
        List of arrays containing photon timestamps in seconds for each channel
    dt : float, optional
        Bin width in seconds. Default is 1e-3 (1 ms).
    prior_count : float, optional
        Prior photon count before data (alpha parameter for Gamma prior). Default is 1.0.
    prior_duration : float, optional
        Time window assumed for prior_count (beta parameter for Gamma prior). Default is 1.0.
    changepoint_prob : float, optional
        Probability of burst start in any bin (hazard rate). Default is 0.1.
    max_run : int, optional
        Maximum run length. Default is 256.
    min_counts : int, optional
        Minimum number of counts (across all channels) for a burst. Default is 20.
    alpha : float, optional
        Deprecated: Use prior_count instead. Alpha parameter for Gamma prior.
    beta : float, optional
        Deprecated: Use prior_duration instead. Beta parameter for Gamma prior.
    hazard : float, optional
        Deprecated: Use changepoint_prob instead. Hazard rate (probability of changepoint).
        
    Returns
    -------
    Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
        Tuple containing:
        - List of burst dictionaries
        - Bin edges
        - Filtered rates (or counts if not available)
        - Innovation distances (or None if not available)
        - Additional result information (or None if not available)
    """
    # Handle backward compatibility
    if alpha is not None:
        prior_count = alpha
    if beta is not None:
        prior_duration = beta
    if hazard is not None:
        changepoint_prob = hazard
    
    # Check if we have any channels
    if not timestamps_list:
        return [], np.array([]), np.array([]), np.array([]), None
    
    n_channels = len(timestamps_list)
    
    # Special case: exactly 2 channels - use the original algorithm
    if n_channels == 2:
        bursts, bin_edges, D_counts, A_counts = bocpd_burst_detection(
            timestamps_list[0],
            timestamps_list[1],
            dt=dt,
            prior_count=prior_count,
            prior_duration=prior_duration,
            changepoint_prob=changepoint_prob,
            max_run=max_run,
            min_counts=min_counts
        )
        
        # Convert to the multi-channel format
        counts = np.stack([D_counts, A_counts], axis=1)
        
        return bursts, bin_edges, counts, np.array([]), None
    
    # Special case: only 1 channel - use a simplified approach
    if n_channels == 1:
        # Bin the timestamps
        counts, bin_edges = bin_photons_multi(timestamps_list, dt)
        
        # For a single channel, we can use a simple threshold approach
        # This is a placeholder - in a real implementation, you might want to use
        # a more sophisticated approach for single-channel burst detection
        threshold = np.mean(counts) + 2 * np.std(counts)  # Simple threshold
        over_threshold = counts[:, 0] > threshold
        
        # Find bursts as contiguous regions above threshold
        changepoints = []
        in_burst = False
        for i in range(len(over_threshold)):
            if over_threshold[i] and not in_burst:
                changepoints.append(i)
                in_burst = True
            elif not over_threshold[i] and in_burst:
                changepoints.append(i)
                in_burst = False
        
        # Extract bursts
        bursts = extract_bursts_multi(counts, bin_edges, np.array(changepoints), min_counts)
        
        return bursts, bin_edges, counts, np.array([]), None
    
    # General case: more than 2 channels
    # Bin all timestamps
    counts, bin_edges = bin_photons_multi(timestamps_list, dt)
    
    # Process each pair of channels and collect all changepoints
    all_changepoints = set()
    
    # Process each pair of channels
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            # Skip pairs where either channel has no photons
            if len(timestamps_list[i]) == 0 or len(timestamps_list[j]) == 0:
                continue
                
            # Run BOCPD on this pair
            cps_ij, _ = bocpd_joint_poisson_optimized(
                counts[:, i], counts[:, j],
                prior_count=prior_count,
                prior_duration=prior_duration,
                changepoint_prob=changepoint_prob,
                max_run=max_run
            )
            
            # Add changepoints to the set
            all_changepoints.update(cps_ij)
    
    # Convert to sorted array
    changepoints = np.array(sorted(all_changepoints), dtype=np.int64)
    
    # Extract bursts
    bursts = extract_bursts_multi(counts, bin_edges, changepoints, min_counts)
    
    return bursts, bin_edges, counts, np.array([]), None


def convert_bursts_to_start_stop(bursts: List[Dict[str, Any]], tttr: tttrlib.TTTR) -> np.ndarray:
    """
    Convert burst dictionaries to start-stop indices for the TTTR object.

    Parameters
    ----------
    bursts : List[Dict[str, Any]]
        List of burst dictionaries from extract_bursts.
    tttr : tttrlib.TTTR
        TTTR object.

    Returns
    -------
    np.ndarray
        Array of start-stop indices with shape (n_bursts, 2).
    """
    if not bursts:
        return np.array([], dtype=np.uint64).reshape(0, 2)
    
    macro_times = tttr.macro_times
    time_unit = tttr.header.macro_time_resolution
    
    start_stop = []
    for burst in bursts:
        start_time = burst['start']
        end_time = burst['end']
        
        # Convert times to macro_time units
        start_macro = start_time / time_unit
        end_macro = end_time / time_unit
        
        # Find indices of photons within the burst
        indices = np.where((macro_times >= start_macro) & (macro_times <= end_macro))[0]
        
        if len(indices) > 0:
            start_stop.append([indices[0], indices[-1]])
    
    if not start_stop:
        return np.array([], dtype=np.uint64).reshape(0, 2)
    
    return np.array(start_stop, dtype=np.uint64)
