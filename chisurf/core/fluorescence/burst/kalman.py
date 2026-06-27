"""
Kalman filter–based burst detection for (multi-channel) photon count data.
Includes loading from TTTR files using tttrlib and constructing binned time series.

Author: ChatGPT
License: MIT
"""

from __future__ import annotations
import numpy as np
import tttrlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union, Any
from numba import njit


@njit(cache=True, inline='always')
def _inv2x2(A: np.ndarray) -> np.ndarray:
    """Compute the analytic inverse of a 2x2 matrix.

    Parameters
    ----------
    A : np.ndarray
        2×2 matrix to invert.

    Returns
    -------
    np.ndarray
        Inverse of A.

    Notes
    -----
    Uses the analytic formula: inv([[a,b],[c,d]]) = [[d,-b],[-c,a]] / (ad - bc)
    Avoids LAPACK overhead for the common 2-channel case.
    """
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if det == 0.0:
        det = 1e-300
    inv = np.empty((2, 2))
    inv[0, 0] =  A[1, 1] / det
    inv[0, 1] = -A[0, 1] / det
    inv[1, 0] = -A[1, 0] / det
    inv[1, 1] =  A[0, 0] / det
    return inv


@njit(cache=True)
def _kalman_filter_loop(
    y: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    Q: np.ndarray,
    dt: float,
    r_scale: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JIT-compiled Kalman filter recursion assuming A = H = I (identity).

    Parameters
    ----------
    y : np.ndarray
        Observed count rates, shape (T, dim).
    x0 : np.ndarray
        Initial state estimate, shape (dim,).
    P0 : np.ndarray
        Initial state covariance, shape (dim, dim).
    Q : np.ndarray
        Process noise covariance, shape (dim, dim).
    dt : float
        Bin width in seconds.
    r_scale : float
        Measurement noise scaling parameter.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (x_filt, P_filt, D_mahal) — filtered states (T, dim),
        filtered covariances (T, dim, dim), Mahalanobis distances (T,).
    """
    T, dim = y.shape
    x_filt = np.zeros((T, dim))
    P_filt = np.zeros((T, dim, dim))
    D_mahal = np.zeros(T)

    x = x0.copy()
    P = P0.copy()
    I = np.eye(dim)
    use_analytic = (dim == 2)

    for t in range(T):
        # Predict: A = I, so x_pred = x, P_pred = P + Q
        P_pred = P + Q

        # Build diagonal measurement noise R from Poisson statistics
        R = np.zeros((dim, dim))
        for i in range(dim):
            rate_i = x[i] if x[i] > 1e-12 else 1e-12
            R[i, i] = r_scale * (rate_i / dt)

        v = y[t] - x           # innovation
        S = P_pred + R          # innovation covariance

        # Kalman gain K = P_pred @ S^{-1} (H = I, so K = P_pred S^{-1})
        if use_analytic:
            S_inv = _inv2x2(S)
        else:
            S_inv = np.linalg.inv(S)

        K = P_pred @ S_inv

        # Update
        x = x + K @ v
        P = (I - K) @ P_pred

        x_filt[t] = x
        P_filt[t] = P

        # Mahalanobis distance: sqrt(v^T S^{-1} v)
        Sv = S_inv @ v
        val = 0.0
        for i in range(dim):
            val += v[i] * Sv[i]
        if val < 0.0:
            val = 0.0
        D_mahal[t] = np.sqrt(val)

    return x_filt, P_filt, D_mahal



@dataclass
class Burst:
    start: int
    end: int  # inclusive
    peak_score: float


@dataclass
class KalmanBurstResult:
    x_filt: np.ndarray  # (T, dim) filtered rates
    P_filt: np.ndarray  # (T, dim, dim) filtered covariances
    innovation_mahal: np.ndarray  # (T,) Mahalanobis innovation distances D_t
    z_thresh: float
    bursts: List[Burst]


@dataclass
class KalmanBurstDetector:
    dim: int
    dt: float
    q: float = 1.0
    r_scale: float = 1.0
    z_thresh: float = 5.0
    min_len: int = 5
    merge_gap: int = 1
    x0: Optional[np.ndarray] = None
    P0: Optional[np.ndarray] = None

    _A: np.ndarray = field(init=False, repr=False)
    _Q: np.ndarray = field(init=False, repr=False)
    _H: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize Kalman filter matrices and default states."""
        self._A = np.eye(self.dim)
        self._H = np.eye(self.dim)
        self._Q = np.eye(self.dim) * self.q
        if self.x0 is None:
            self.x0 = np.zeros(self.dim)
        if self.P0 is None:
            self.P0 = np.eye(self.dim) * 1e6

    def _adaptive_R(self, y_rate: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        """Build an adaptive measurement noise covariance matrix.

        The noise is proportional to the predicted count rate to model
        Poissonian photon-counting statistics.

        Parameters
        ----------
        y_rate : np.ndarray
            Observed count rates at the current time step.
        x_pred : np.ndarray
            Predicted state (count rates) from the Kalman filter.

        Returns
        -------
        np.ndarray
            Diagonal measurement noise covariance matrix.
        """
        rate = np.maximum(x_pred, 1e-12)
        var_rate = rate / self.dt
        R = np.diag(self.r_scale * var_rate)
        return R

    def detect(self, counts: np.ndarray) -> KalmanBurstResult:
        """Run Kalman filter burst detection on binned photon counts.

        Parameters
        ----------
        counts : np.ndarray
            Array of shape (T, dim) with binned photon counts per channel.

        Returns
        -------
        KalmanBurstResult
            Dataclass containing filtered rates, covariances, innovation
            distances, and detected bursts.
        """
        counts = np.asarray(counts, dtype=float)
        assert counts.ndim == 2 and counts.shape[1] == self.dim, "counts must be of shape (T, dim)"
        y = counts / self.dt

        x_filt, P_filt, D_mahal = _kalman_filter_loop(
            y, self.x0, self.P0, self._Q, self.dt, self.r_scale
        )

        bursts = self._extract_bursts(D_mahal)

        return KalmanBurstResult(x_filt=x_filt, P_filt=P_filt, innovation_mahal=D_mahal, z_thresh=self.z_thresh, bursts=bursts)

    def _extract_bursts(self, D: np.ndarray) -> List[Burst]:
        """Extract burst intervals from the innovation distance trace.

        Parameters
        ----------
        D : np.ndarray
            Array of Mahalanobis innovation distances.

        Returns
        -------
        List[Burst]
            List of detected burst intervals.
        """
        over = D > self.z_thresh
        if not np.any(over):
            return []

        # Find start and end indices of contiguous True regions
        padded = np.empty(over.size + 2, dtype=np.bool_)
        padded[0] = False
        padded[1:-1] = over
        padded[-1] = False
        
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1  # inclusive
        
        lengths = ends - starts + 1
        valid = lengths >= self.min_len
        starts = starts[valid]
        ends = ends[valid]
        
        bursts: List[Burst] = []
        for s, e in zip(starts, ends):
            peak = np.max(D[s:e+1])
            bursts.append(Burst(int(s), int(e), float(peak)))

        if self.merge_gap > 0 and len(bursts) > 1:
            merged: List[Burst] = [bursts[0]]
            for b in bursts[1:]:
                last = merged[-1]
                if b.start - last.end - 1 <= self.merge_gap:
                    merged[-1] = Burst(start=last.start, end=b.end, peak_score=max(last.peak_score, b.peak_score))
                else:
                    merged.append(b)
            bursts = merged

        return bursts


def bin_photons(donor: np.ndarray, acceptor: np.ndarray, dt: float = 1e-3, tmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin photon timestamps into count arrays.
    
    Args:
        donor: Array of donor photon timestamps in seconds
        acceptor: Array of acceptor photon timestamps in seconds
        dt: Bin width in seconds
        tmax: Maximum time to consider. If None, uses the maximum timestamp.
        
    Returns:
        Tuple of (D_counts, A_counts, bin_edges) where:
        - D_counts is an array of donor counts per bin
        - A_counts is an array of acceptor counts per bin
        - bin_edges is an array of bin edges
    """
    if tmax is None:
        tmax = max(donor.max() if len(donor) > 0 else 0, 
                  acceptor.max() if len(acceptor) > 0 else 0)
    
    n_bins = int(np.ceil(tmax / dt))
    
    D_counts, bin_edges = np.histogram(donor, bins=n_bins, range=(0, tmax))
    A_counts, _ = np.histogram(acceptor, bins=n_bins, range=(0, tmax))
    
    return D_counts, A_counts, bin_edges


def bin_photons_multi(timestamps_list: List[np.ndarray], dt: float = 1e-3, tmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin photon timestamps from multiple channels into count arrays.
    
    Args:
        timestamps_list: List of arrays containing photon timestamps in seconds for each channel
        dt: Bin width in seconds
        tmax: Maximum time to consider. If None, uses the maximum timestamp across all channels.
        
    Returns:
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


def kalman_burst_detection(
    donor_timestamps: np.ndarray, 
    acceptor_timestamps: np.ndarray,
    dt: float = 1e-3,
    q: float = 20.0,
    r_scale: float = 1.0,
    z_thresh: float = 3.0,
    min_len: int = 2,
    merge_gap: int = 20,
    min_counts: int = 20
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, KalmanBurstResult]:
    """
    Perform Kalman filter-based burst detection on donor and acceptor photon timestamps.
    
    Args:
        donor_timestamps: Array of donor photon timestamps in seconds
        acceptor_timestamps: Array of acceptor photon timestamps in seconds
        dt: Bin width in seconds
        q: Process noise parameter
        r_scale: Measurement noise scaling parameter
        z_thresh: Threshold for burst detection
        min_len: Minimum burst length in bins
        merge_gap: Maximum gap between bursts to merge them
        min_counts: Minimum number of photons (donor + acceptor) for a burst
        
    Returns:
        Tuple of (bursts, bin_edges, filtered_rates, innovation_distances, result) where:
        - bursts is a list of dictionaries with keys 'start_bin', 'end_bin', 'peak_score'
        - bin_edges is an array of bin edges
        - filtered_rates is a (T, 2) array of filtered rates [donor_rate, acceptor_rate]
        - innovation_distances is a (T,) array of Mahalanobis innovation distances
        - result is the KalmanBurstResult object
    """
    # Bin the photon timestamps
    D_counts, A_counts, bin_edges = bin_photons(donor_timestamps, acceptor_timestamps, dt)
    
    # Stack counts for Kalman filter
    counts = np.stack([D_counts, A_counts], axis=1)
    
    # Create and run the Kalman filter detector
    detector = KalmanBurstDetector(
        dim=2,
        dt=dt,
        q=q,
        r_scale=r_scale,
        z_thresh=z_thresh,
        min_len=min_len,
        merge_gap=merge_gap
    )
    
    result = detector.detect(counts)
    
    # Convert bursts to the format expected by the rest of the code
    bursts = []
    for b in result.bursts:
        # Calculate the total counts in this burst
        start_bin = b.start
        end_bin = b.end
        donor_counts = np.sum(counts[start_bin:end_bin+1, 0])
        acceptor_counts = np.sum(counts[start_bin:end_bin+1, 1])
        total_counts = donor_counts + acceptor_counts
        
        # Skip bursts with too few photons
        if total_counts < min_counts:
            continue
            
        bursts.append({
            'start_bin': start_bin,
            'end_bin': end_bin,
            'peak_score': b.peak_score,
            'start_time': bin_edges[start_bin],
            'end_time': bin_edges[end_bin + 1] if end_bin + 1 < len(bin_edges) else bin_edges[-1],
            'duration': bin_edges[end_bin + 1] - bin_edges[start_bin] if end_bin + 1 < len(bin_edges) else bin_edges[-1] - bin_edges[start_bin],
            'donor': donor_counts,
            'acceptor': acceptor_counts,
            'FRET': acceptor_counts / total_counts if total_counts > 0 else np.nan
        })
    
    return bursts, bin_edges, result.x_filt, result.innovation_mahal, result


def kalman_burst_detection_multi(
    timestamps_list: List[np.ndarray],
    dt: float = 1e-3,
    q: float = 20.0,
    r_scale: float = 1.0,
    z_thresh: float = 3.0,
    min_len: int = 2,
    merge_gap: int = 20,
    min_counts: int = 20
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, KalmanBurstResult]:
    """
    Perform Kalman filter-based burst detection on multiple channels.
    
    Args:
        timestamps_list: List of arrays containing photon timestamps in seconds for each channel
        dt: Bin width in seconds
        q: Process noise parameter
        r_scale: Measurement noise scaling parameter
        z_thresh: Threshold for burst detection
        min_len: Minimum burst length in bins
        merge_gap: Maximum gap between bursts to merge them
        min_counts: Minimum number of photons (across all channels) for a burst
        
    Returns:
        Tuple of (bursts, bin_edges, filtered_rates, innovation_distances, result) where:
        - bursts is a list of dictionaries with keys 'start_bin', 'end_bin', 'peak_score', 'counts'
        - bin_edges is an array of bin edges
        - filtered_rates is a (T, n_channels) array of filtered rates
        - innovation_distances is a (T,) array of Mahalanobis innovation distances
        - result is the KalmanBurstResult object
    """
    # Check if we have any channels
    if not timestamps_list:
        return [], np.array([]), np.array([]), np.array([]), None
    
    # Bin the photon timestamps
    counts, bin_edges = bin_photons_multi(timestamps_list, dt)
    
    # Get the number of channels
    n_channels = counts.shape[1]
    
    # Create and run the Kalman filter detector
    detector = KalmanBurstDetector(
        dim=n_channels,
        dt=dt,
        q=q,
        r_scale=r_scale,
        z_thresh=z_thresh,
        min_len=min_len,
        merge_gap=merge_gap
    )
    
    result = detector.detect(counts)
    
    # Convert bursts to the format expected by the rest of the code
    bursts = []
    for b in result.bursts:
        # Calculate the total counts in this burst
        start_bin = b.start
        end_bin = b.end
        
        # Get counts for each channel in this burst
        channel_counts = np.sum(counts[start_bin:end_bin+1, :], axis=0)
        total_counts = np.sum(channel_counts)
        
        # Skip bursts with too few photons
        if total_counts < min_counts:
            continue
        
        # Create a burst dictionary with counts for each channel
        burst_dict = {
            'start_bin': start_bin,
            'end_bin': end_bin,
            'peak_score': b.peak_score,
            'start_time': bin_edges[start_bin],
            'end_time': bin_edges[end_bin + 1] if end_bin + 1 < len(bin_edges) else bin_edges[-1],
            'duration': bin_edges[end_bin + 1] - bin_edges[start_bin] if end_bin + 1 < len(bin_edges) else bin_edges[-1] - bin_edges[start_bin],
            'counts': channel_counts.tolist(),  # Store counts for each channel
            'total_counts': total_counts
        }
        
        # For backward compatibility, if we have exactly 2 channels, add donor/acceptor fields
        if n_channels == 2:
            burst_dict['donor'] = channel_counts[0]
            burst_dict['acceptor'] = channel_counts[1]
            burst_dict['FRET'] = channel_counts[1] / total_counts if total_counts > 0 else np.nan
            
        bursts.append(burst_dict)
    
    return bursts, bin_edges, result.x_filt, result.innovation_mahal, result


def convert_bursts_to_start_stop(bursts: List[Dict[str, Any]], tttr: tttrlib.TTTR) -> np.ndarray:
    """
    Convert bursts to start-stop indices for the TTTR data.
    
    Parameters
    ----------
    bursts : List[Dict[str, Any]]
        List of burst dictionaries with 'start_time' and 'end_time' keys.
    tttr : tttrlib.TTTR
        TTTR object.
        
    Returns
    -------
    np.ndarray
        Array of shape (n_bursts, 2) with [start_idx, end_idx] for each burst.
    """
    if not bursts:
        return np.array([], dtype=np.uint64).reshape(0, 2)
    
    macro_times = tttr.macro_times
    time_unit = tttr.header.macro_time_resolution
    
    starts_time = np.array([burst['start_time'] for burst in bursts])
    ends_time = np.array([burst['end_time'] for burst in bursts])
    
    starts_mt = (starts_time / time_unit).astype(np.int64)
    ends_mt = (ends_time / time_unit).astype(np.int64)
    
    start_indices = np.searchsorted(macro_times, starts_mt)
    end_indices = np.searchsorted(macro_times, ends_mt, side='right') - 1
    
    valid = start_indices <= end_indices
    if not np.any(valid):
        return np.array([], dtype=np.uint64).reshape(0, 2)
        
    return np.stack([start_indices[valid], end_indices[valid]], axis=1).astype(np.uint64)
