"""
Utility functions for 2D-FCS data handling and visualization.

This module provides helper functions for data import/export using tttrlib,
preprocessing, and visualization utilities for the 2D-FCS plugin.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union
from pathlib import Path

# ChiSurf logging
from chisurf import logging

import pyqtgraph as pg
import tttrlib


def load_tttr_data(file_path: Union[str, Path], filetype: str = 'PTU') -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load TTTR (Time-Tagged Time-Resolved) data using tttrlib.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the TTTR data file
    filetype : str
        File type for tttrlib (e.g., 'PTU', 'PT3', 'HT3', 'HDF5')
        If provided, will be used; otherwise auto-detection is used
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict]
        (macro_times, micro_times, metadata)
        
    Notes
    -----
    Uses tttrlib for robust TTTR data loading across various formats.
    """
    file_path = Path(file_path)
    logger = logging.getLogger(__name__)
    
    if not file_path.exists():
        raise FileNotFoundError(f"TTTR file not found: {file_path}")
    
    try:
        logger.info(f"2D-FLCS: Loading TTTR data with auto-detection (filetype parameter ignored)")
        # Always use auto-detection like intensity trace plugin
        tttr = tttrlib.TTTR(str(file_path))  # No filetype parameter
        
        # Extract macro and micro times
        header = tttr.get_header()
        macro_res = getattr(header, "macro_time_resolution", None)
        micro_res = getattr(header, "micro_time_resolution", None)
        logger.info(f"2D-FLCS: Header macro_time_resolution={macro_res}, micro_time_resolution={micro_res}")
        
        logger.info("2D-FLCS: Extracting macro_times (raw ticks)...")
        macro_times = tttr.macro_times
        logger.info(f"2D-FLCS: macro_times shape: {macro_times.shape if hasattr(macro_times, 'shape') else len(macro_times)}")
        logger.info(f"2D-FLCS: macro_times sample ticks: {macro_times[:5] if len(macro_times) > 0 else 'empty'}")
        
        logger.info("2D-FLCS: Extracting micro_times (raw bins)...")
        micro_times = tttr.micro_times
        logger.info(f"2D-FLCS: micro_times shape: {micro_times.shape if hasattr(micro_times, 'shape') else len(micro_times)}")
        logger.info(f"2D-FLCS: micro_times sample bins: {micro_times[:5] if len(micro_times) > 0 else 'empty'}")
        
        # Extract metadata
        logger.info(f"2D-FLCS: Extracting metadata...")
        metadata = {
            'file_type': 'auto-detected',  # Always auto-detected
            'n_photons': len(tttr),
            'duration': getattr(tttr, 'get_duration', lambda: 'Unknown')(),
            'resolution': tttr.get_resolution() if hasattr(tttr, 'get_resolution') else None,
            'sync_period': tttr.get_sync_period() if hasattr(tttr, 'get_sync_period') else None,
            'channels': tttr.get_number_of_channels() if hasattr(tttr, 'get_number_of_channels') else 1,
            'file_size': file_path.stat().st_size
        }
        metadata = {
            'file_type': 'auto-detected',
            'n_photons': len(tttr),
            'duration': getattr(tttr, 'get_duration', lambda: 'Unknown')(),
            'resolution': tttr.get_resolution() if hasattr(tttr, 'get_resolution') else None,
            'sync_period': tttr.get_sync_period() if hasattr(tttr, 'get_sync_period') else None,
            'channels': tttr.get_number_of_channels() if hasattr(tttr, 'get_number_of_channels') else 1,
            'macro_time_resolution': macro_res,
            'micro_time_resolution': micro_res,
            'file_size': file_path.stat().st_size
        }
        logger.info(
            "2D-FLCS: Metadata extracted: "
            f"macro_res={macro_res}, micro_res={micro_res}, duration={metadata['duration']}, photons={metadata['n_photons']}"
        )
        
        logger.info(f"2D-FLCS: Successfully loaded {len(tttr)} photons from {file_path}")
        return macro_times, micro_times, metadata
        
    except Exception as e:
        logger.error(f"2D-FLCS: FAILED to load TTTR data from {file_path}")
        logger.error(f"2D-FLCS: Specified filetype was: {filetype}")
        logger.error(f"2D-FLCS: File exists: {file_path.exists()}")
        logger.error(f"2D-FLCS: File size: {file_path.stat().st_size if file_path.exists() else 'N/A'}")
        logger.error(f"2D-FLCS: Exception type: {type(e).__name__}")
        logger.error(f"2D-FLCS: Exception message: {str(e)}")
        import traceback
        logger.error(f"2D-FLCS: Full traceback:\n{traceback.format_exc()}")
        raise


def get_tttr_filetype(file_path: Union[str, Path]) -> str:
    """
    Determine the appropriate tttrlib filetype based on file extension.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the TTTR file
        
    Returns
    -------
    str
        Filetype string for tttrlib
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    filetype_map = {
        '.ptu': 'PTU',
        '.pt3': 'PT3', 
        '.ht3': 'HT3',
        '.hdf5': 'HDF5',
        '.h5': 'HDF5',
        '.spc': 'SPC-130',  # Becker & Hickl SPC files
        '.bin': 'BIN'
    }
    
    return filetype_map.get(suffix, 'PTU')  # Default to PTU


def create_exponential_curve(
    tau_values: np.ndarray,
    time_axis: np.ndarray,
    instrument_response: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create exponential decay curves for given tau values.
    
    Parameters
    ----------
    tau_values : np.ndarray
        Array of lifetime values
    time_axis : np.ndarray  
        Time axis for the curves
    instrument_response : np.ndarray, optional
        Instrument response function for convolution
        
    Returns
    -------
    np.ndarray
        Exponential curves matrix (n_taus x n_times)
    """
    curves = np.zeros((len(tau_values), len(time_axis)))
    
    for i, tau in enumerate(tau_values):
        if tau > 0:
            curves[i, :] = np.exp(-time_axis / tau)
        else:
            curves[i, :] = np.zeros_like(time_axis)
    
    # Convolve with IRF if provided
    if instrument_response is not None:
        for i in range(len(tau_values)):
            curves[i, :] = np.convolve(curves[i, :], instrument_response, mode='same')
    
    return curves


def create_mi_matrix(
    tau_values: np.ndarray,
    n_states: int,
    method: str = 'uniform'
) -> np.ndarray:
    """
    Create MI (Maximum Entropy) matrix for regularization.
    
    Parameters
    ----------
    tau_values : np.ndarray
        Array of lifetime values
    n_states : int
        Number of states
    method : str
        Method for creating MI matrix ('uniform', 'gaussian', 'exponential')
        
    Returns
    -------
    np.ndarray
        MI matrix (n_taus x n_states)
    """
    n_taus = len(tau_values)
    mi_matrix = np.zeros((n_taus, n_states))
    
    if method == 'uniform':
        mi_matrix[:, :] = 1.0
    elif method == 'gaussian':
        # Create Gaussian distributions centered at different tau values
        centers = np.linspace(tau_values[0], tau_values[-1], n_states)
        width = (tau_values[-1] - tau_values[0]) / (2 * n_states)
        
        for i, center in enumerate(centers):
            mi_matrix[:, i] = np.exp(-0.5 * ((tau_values - center) / width) ** 2)
    elif method == 'exponential':
        # Create exponentially spaced distributions
        for i in range(n_states):
            mi_matrix[:, i] = np.exp(-tau_values / tau_values[-1] * (i + 1))
    else:
        raise ValueError(f"Unknown MI matrix method: {method}")
    
    # Normalize columns
    for i in range(n_states):
        if np.sum(mi_matrix[:, i]) > 0:
            mi_matrix[:, i] /= np.sum(mi_matrix[:, i])
    
    return mi_matrix


def preprocess_tttr_data(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    time_window: Optional[Tuple[float, float]] = None,
    micro_time_range: Optional[Tuple[float, float]] = None,
    remove_afterpulses: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess TTTR data by filtering and cleaning.
    
    Parameters
    ----------
    macro_times : np.ndarray
        Macro time array
    micro_times : np.ndarray
        Micro time array
    time_window : Tuple[float, float], optional
        Time window to keep (start, end) in seconds
    micro_time_range : Tuple[float, float], optional
        Micro time range to keep (min, max) in nanoseconds
    remove_afterpulses : bool
        Whether to remove afterpulses
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Filtered macro_times and micro_times
    """
    logger = logging.getLogger(__name__)
    
    # Start with all photons
    mask = np.ones(len(macro_times), dtype=bool)
    
    # Filter by time window
    if time_window is not None:
        start_time, end_time = time_window
        mask &= (macro_times >= start_time) & (macro_times <= end_time)
    
    # Filter by micro time range
    if micro_time_range is not None:
        min_time, max_time = micro_time_range
        mask &= (micro_times >= min_time) & (micro_times <= max_time)
    
    # Remove afterpulses (simple implementation)
    if remove_afterpulses:
        # Remove photons that arrive too quickly after previous photon
        min_separation = 1e-9  # 1 ns minimum separation
        time_diffs = np.diff(macro_times[mask])
        afterpulse_mask = np.ones(len(time_diffs) + 1, dtype=bool)
        afterpulse_mask[1:] = time_diffs >= min_separation
        mask = np.where(mask)[0][afterpulse_mask]
    
    filtered_macro = macro_times[mask]
    filtered_micro = micro_times[mask]
    
    logger.info(f"Preprocessing: {len(macro_times)} -> {len(filtered_macro)} photons")
    
    return filtered_macro, filtered_micro


def visualize_2d_matrix(
    matrix: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    title: str = "2D Matrix",
    colormap: str = 'viridis',
    log_scale: bool = False
) -> pg.ImageItem:
    """
    Create visualization of 2D matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        2D matrix to visualize
    x_axis : np.ndarray
        X-axis values
    y_axis : np.ndarray
        Y-axis values
    title : str
        Plot title
    colormap : str
        Colormap name
    log_scale : bool
        Whether to use logarithmic scale
        
    Returns
    -------
    pg.ImageItem
        Image item for adding to plot
    """
    if log_scale:
        display_matrix = np.log10(matrix + 1)
    else:
        display_matrix = matrix
    
    img = pg.ImageItem(display_matrix)
    
    # Set colormap
    cmap = pg.colormap.get(colormap)
    img.setLookupTable(cmap.getLookupTable())
    
    return img


def export_2d_fdc_data(
    fdc_data: Dict,
    file_path: Union[str, Path],
    format: str = 'npz'
) -> None:
    """
    Export 2D-FDC data to file.
    
    Parameters
    ----------
    fdc_data : Dict
        Dictionary containing 2D-FDC data
    file_path : Union[str, Path]
        Output file path
    format : str
        Export format ('npz', 'txt', 'hdf5')
    """
    file_path = Path(file_path)
    
    if format == 'npz':
        np.savez_compressed(file_path, **fdc_data)
    elif format == 'txt':
        # Export as text files
        base_name = file_path.stem
        for key, value in fdc_data.items():
            if isinstance(value, np.ndarray):
                np.savetxt(file_path.parent / f"{base_name}_{key}.txt", value)
    elif format == 'hdf5':
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 export")
        
        with h5py.File(file_path, 'w') as f:
            for key, value in fdc_data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                else:
                    f.attrs[key] = value
    else:
        raise ValueError(f"Unsupported export format: {format}")


def import_2d_fdc_data(file_path: Union[str, Path]) -> Dict:
    """
    Import 2D-FDC data from file.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        File path to import
        
    Returns
    -------
    Dict
        Dictionary containing 2D-FDC data
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        data = np.load(file_path)
        return {key: data[key] for key in data.files}
    elif file_path.suffix == '.hdf5':
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 import")
        
        data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
            for key in f.attrs:
                data[key] = f.attrs[key]
        return data
    else:
        raise ValueError(f"Unsupported import format: {file_path.suffix}")


class TwoDMatrixPlotItem(pg.PlotItem):
    """Custom plot item for 2D matrix visualization with enhanced features."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item = None
        self.color_bar = None
        
        # Setup axes
        self.setLabel('left', 'Y-axis')
        self.setLabel('bottom', 'X-axis')
        
        # Enable mouse interactions
        self.setMenuEnabled(True)
    
    def set_2d_data(
        self,
        matrix: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        title: str = "2D Matrix",
        colormap: str = 'viridis',
        log_scale: bool = False
    ):
        """Set 2D matrix data for visualization."""
        # Clear previous data
        self.clear()
        
        # Create image item
        self.image_item = visualize_2d_matrix(
            matrix, x_axis, y_axis, title, colormap, log_scale
        )
        self.addItem(self.image_item)
        
        # Set title
        self.setTitle(title)
        
        # Add color bar
        if self.color_bar is None:
            self.color_bar = pg.ColorBarItem(values=(matrix.min(), matrix.max()))
            self.color_bar.setImageItem(self.image_item)
        
        # Set axis ranges
        self.setRange(xRange=[x_axis[0], x_axis[-1]], yRange=[y_axis[0], y_axis[-1]])
    
        return h_section, v_section


def split_tttr_data(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    n_chunks: int = 5
) -> list:
    """
    Split TTTR data into N sequential chunks for variance estimation.
    
    Parameters
    ----------
    macro_times : np.ndarray
    micro_times : np.ndarray
    n_chunks : int
        Number of chunks to split data into
        
    Returns
    -------
    list of Tuples (macro, micro)
    """
    chunk_size = len(macro_times) // n_chunks
    chunks = []
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else len(macro_times)
        chunks.append((macro_times[start:end], micro_times[start:end]))
        
    return chunks


def bootstrap_tttr_data(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    n_realizations: int = 1,
    fraction: float = 1.0
) -> list:
    """
    Create bootstrap realizations by random resampling of photons.
    
    Parameters
    ----------
    macro_times : np.ndarray
    micro_times : np.ndarray
    n_realizations : int
    fraction : float
        Fraction of photons to sample in each realization
        
    Returns
    -------
    list of Tuples (macro, micro)
    """
    n_photons = len(macro_times)
    sample_size = int(n_photons * fraction)
    realizations = []
    
    for _ in range(n_realizations):
        indices = np.random.choice(n_photons, size=sample_size, replace=True)
        # Sort indices to maintain chronological order in macro_times (preferred for some correlators)
        indices.sort()
        realizations.append((macro_times[indices], micro_times[indices]))
        
    return realizations
    return realizations


def scale_correlations(
    mat_g: np.ndarray,
    mat_a: np.ndarray,
    tau_values: np.ndarray
) -> np.ndarray:
    """
    Scale raw correlations by state brightness (Amplitude * Tau).
    
    Equivalent to Mat_PhotonSum scaling in MATLAB script 'TK_MyMain_Analyze.m'.
    
    Parameters
    ----------
    mat_g : np.ndarray
        Correlation matrix between states [n_states x n_states].
        If 3D [n_states x n_states x n_dt], scales each slice.
    mat_a : np.ndarray
        State lifetime distribution [n_tau x n_states].
    tau_values : np.ndarray
        Lifetime grid [n_tau].
        
    Returns
    -------
    np.ndarray
        Photon-weighted correlation matrix.
    """
    n_states = mat_a.shape[1]
    photon_sum = np.zeros(n_states)
    
    for i in range(n_states):
        # Sum of (Amplitude * Tau) for state i
        photon_sum[i] = np.sum(mat_a[:, i] * tau_values)
        
    s_mat = np.diag(photon_sum)
    
    if mat_g.ndim == 2:
        return s_mat.T @ mat_g @ s_mat
    elif mat_g.ndim == 3:
        scaled = np.zeros_like(mat_g)
        for t in range(mat_g.shape[2]):
            scaled[:, :, t] = s_mat.T @ mat_g[:, :, t] @ s_mat
        return scaled
    else:
        raise ValueError("mat_g must be 2D or 3D")


def calculate_correlation_ratios(
    mat_m: np.ndarray,
    tau_values: np.ndarray,
    state_windows: list,
    func_type: int = 2
) -> np.ndarray:
    """
    Integrate 2D distribution over state windows and calculate ratios.
    
    Parameters
    ----------
    mat_m : np.ndarray
        2D amplitude distribution [n_tau x n_tau].
        If 3D [n_tau x n_tau x n_dt], processes each slice.
    tau_values : np.ndarray
        Lifetime grid [n_tau].
    state_windows : list of tuples
        Lifetime range (min, max) in physical units for each state.
    func_type : int
        1: gij / gii
        2: (gij + gji) / (gii + gjj)
        
    Returns
    -------
    np.ndarray
        Matrix of correlation ratios [n_states x n_states] or [n_states x n_states x n_dt].
    """
    n_states = len(state_windows)
    
    # Pre-calculate window indices
    window_indices = []
    for w_min, w_max in state_windows:
        idx = np.where((tau_values >= w_min) & (tau_values <= w_max))[0]
        window_indices.append(idx)

    def process_2d(m_slice):
        correlations = np.zeros((n_states, n_states))
        for i in range(n_states):
            idx_i = window_indices[i]
            for j in range(n_states):
                idx_j = window_indices[j]
                if len(idx_i) > 0 and len(idx_j) > 0:
                    sub_mat = m_slice[np.ix_(idx_i, idx_j)]
                    correlations[i, j] = np.sum(sub_mat)
        
        ratios = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                if func_type == 1:
                    if correlations[i, i] > 0:
                        ratios[i, j] = correlations[i, j] / correlations[i, i]
                elif func_type == 2:
                    denom = correlations[i, i] + correlations[j, j]
                    if denom > 0:
                        ratios[i, j] = (correlations[i, j] + correlations[j, i]) / denom
        return ratios

    if mat_m.ndim == 2:
        return process_2d(mat_m)
    elif mat_m.ndim == 3:
        n_dt = mat_m.shape[2]
        all_ratios = np.zeros((n_states, n_states, n_dt))
        for t in range(n_dt):
            all_ratios[:, :, t] = process_2d(mat_m[:, :, t])
        return all_ratios
    else:
        raise ValueError("mat_m must be 2D or 3D")
