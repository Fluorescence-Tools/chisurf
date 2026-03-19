# %% [markdown]
# ## Environment Setup
# 
# **Important**: This notebook requires `tttrlib` to be installed.
# 
# ### Installation Instructions:
# 
# #### macOS/Linux:
# ```bash
# conda install -c bioconda tttrlib
# ```
# 
# #### Windows:
# tttrlib is not available via conda on Windows. You have several options:
# 
# 1. **Install via pip** (easiest method):
#    ```bash
#    pip install tttrlib
#    ```
#    Note: This requires Visual Studio Build Tools or Visual Studio Community Edition to be installed for compilation.
# 
# 2. **Use WSL (Windows Subsystem for Linux)**:
#    - Install WSL2 (Windows 10/11): https://docs.microsoft.com/en-us/windows/wsl/install
#    - Once in WSL, follow the Linux installation instructions:
#      ```bash
#      conda install -c bioconda tttrlib
#      ```
#    - This provides a native Linux environment on Windows without dual-booting
# 
# 3. **Compile from source using conda recipe**:
#    - A conda recipe for Windows is available in the tttrlib repository
#    - Clone the repository: https://github.com/fluorescence-tools/tttrlib
#    - Navigate to the conda recipe directory
#    - Build using: `conda build conda-recipe`
#    - Install the built package
# 
# 4. **Manual compilation** (requires Visual Studio Community Edition):
#    - Install Visual Studio Community Edition (free) with C++ build tools
#    - Clone the repository: https://github.com/fluorescence-tools/tttrlib
#    - Follow the build instructions in the repository
#    - Use CMake to configure and build the project
# 
# 5. **Use ChiSurf environment** (recommended for full functionality):
#    - Install ChiSurf which includes tttrlib
#    - **Download**: https://peulen.xyz/downloads/chisurf
#    - **GitHub**: https://github.com/fluorescence-tools/chisurf
# 
# **Note**: For Windows compilation, Visual Studio Community Edition is free and provides all necessary build tools. Make sure to select "Desktop development with C++" during installation.
# 
# ### Running this notebook:
# 1. Open Anaconda Prompt or terminal
# 2. Activate the environment: `conda activate chisurf-env` (or your environment with tttrlib)
# 3. Launch Jupyter: `jupyter notebook` or `jupyter lab`
# 4. Open this notebook
# 
# Alternatively, you can select the appropriate kernel from the Jupyter kernel selector.
# %% [markdown]
# # Burst Selection Analysis - ALEX Configuration
# 
# This notebook replicates the **basic functionality** of the chisurf burst_selection plugin using only tttrlib and standard Python libraries.
# 
# **Note:** For more complex burst analysis workflows, interactive visualization, and advanced filtering options, use the full [ChiSurf](https://github.com/fluorescence-tools/chisurf) application. This notebook provides a simplified, standalone script for batch processing and basic burst selection.
# 
# ## Data Type: ALEX (Alternating Laser Excitation), single color excitation, PIE (Pulsed Interleaved Excitation)
# - **Green photons**: Routing channel 0
# - **Red photons**: Routing channel 1
# - **No microtime separation**: All photons from each channel are used
# 
# ## Features
# - Load multiple HDF5 TTTR files from a folder
# - Apply burst detection with full parameter control
# - Support for count rate and burst filter modes
# - Channel selection and delta-time filtering
# - Gap filling for burst merging
# - Export results to .bur files and HDF5 format
# - Compatible with chisurf burst_selection plugin output format
# %%
# Import required libraries
import os
import pathlib
import json
import time
import zipfile
import shutil
from datetime import datetime
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import tttrlib
from scipy.optimize import least_squares
from scipy.stats import norm

# Configure matplotlib for inline plotting
# %matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

print("Libraries imported successfully!")
print(f"tttrlib version: {tttrlib.__version__}")


@dataclass
class Parameter:
    """A simple standalone fitting parameter (chisurf-like, but script-local)."""

    value: float
    fixed: bool = False
    bounds_on: bool = False
    lb: float = -np.inf
    ub: float = np.inf
    name: str = ""

    def __post_init__(self) -> None:
        if self.bounds_on and self.lb > self.ub:
            raise ValueError(f"Invalid bounds for {self.name or 'parameter'}: lb={self.lb} > ub={self.ub}")
        if self.bounds_on and not (self.lb <= float(self.value) <= self.ub):
            raise ValueError(
                f"Initial value for {self.name or 'parameter'} out of bounds: value={self.value}, lb={self.lb}, ub={self.ub}"
            )


@dataclass
class GaussianComponent:
    """A single 1D Gaussian mixture component.

    Parameters
    ----------
    mu, sigma, population : Parameter
        Parameters for the Gaussian mean, width (sigma), and mixture population.
        If at least one population is not fixed, free populations are
        constrained such that all populations sum to 1.
    """

    mu: Parameter
    sigma: Parameter
    population: Parameter

    def copy(self) -> "GaussianComponent":
        return GaussianComponent(
            mu=Parameter(**self.mu.__dict__),
            sigma=Parameter(**self.sigma.__dict__),
            population=Parameter(**self.population.__dict__),
        )


class GaussianMixtureHistogramFitter:
    """Fit a user-specified Gaussian mixture to a 1D histogram.

    This is intended as a lightweight, script-friendly alternative to an
    automated GMM fit. The user provides initial parameters and per-parameter
    fixed/free flags.
    """

    def __init__(self, components: List[GaussianComponent]):
        if not components:
            raise ValueError("At least one GaussianComponent is required")
        self.components = [c.copy() for c in components]

    def pdf(self, x_vals: np.ndarray) -> np.ndarray:
        x_vals = np.asarray(x_vals, dtype=float)
        weights = np.array([c.population.value for c in self.components], dtype=float)
        pdf_total = np.zeros_like(x_vals, dtype=float)
        for w, c in zip(weights, self.components):
            pdf_total += float(w) * norm.pdf(x_vals, loc=float(c.mu.value), scale=float(c.sigma.value))
        return pdf_total

    def component_pdfs(self, x_vals: np.ndarray) -> List[np.ndarray]:
        x_vals = np.asarray(x_vals, dtype=float)
        weights = np.array([c.population.value for c in self.components], dtype=float)
        return [float(w) * norm.pdf(x_vals, loc=float(c.mu.value), scale=float(c.sigma.value)) for w, c in zip(weights, self.components)]

    def fit_to_histogram(
        self,
        data: np.ndarray,
        bins: int,
        data_range: Tuple[float, float],
        min_sigma: float = 1e-4,
    ) -> List[GaussianComponent]:
        data = np.asarray(data, dtype=float).ravel()
        xmin, xmax = map(float, data_range)

        hist_counts, bin_edges = np.histogram(data, bins=bins, range=(xmin, xmax))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = float(abs((xmax - xmin) / bins))
        n_events = float(data.size)

        # Build initial parameter vector and bounds
        x0: List[float] = []
        lo: List[float] = []
        hi: List[float] = []
        free_pop_components: List[GaussianComponent] = []

        def p_bounds(p: Parameter) -> Tuple[float, float]:
            if p.bounds_on:
                return float(p.lb), float(p.ub)
            return -np.inf, np.inf

        for c in self.components:
            if not c.mu.fixed:
                x0.append(float(c.mu.value))
                lb, ub = p_bounds(c.mu)
                lo.append(lb)
                hi.append(ub)

            if not c.sigma.fixed:
                x0.append(float(c.sigma.value))
                lb, ub = p_bounds(c.sigma)
                lo.append(lb)
                hi.append(ub)

            if not c.population.fixed:
                free_pop_components.append(c)

        for c in free_pop_components:
            x0.append(float(c.population.value))
            lo.append(0.0)
            hi.append(1.0)

        x0_arr = np.array(x0, dtype=float)
        bounds = (np.array(lo, dtype=float), np.array(hi, dtype=float))
        denom = np.sqrt(hist_counts.astype(float) + 1.0)

        def apply_x(xvec: np.ndarray) -> None:
            xvec = np.asarray(xvec, dtype=float).ravel()
            k = 0

            for c in self.components:
                if not c.mu.fixed:
                    c.mu.value = float(xvec[k])
                    k += 1
                if not c.sigma.fixed:
                    c.sigma.value = float(xvec[k])
                    k += 1

            if free_pop_components:
                free_vals = xvec[k : k + len(free_pop_components)]
                fixed_sum = float(sum(float(c.population.value) for c in self.components if c.population.fixed))
                remaining = 1.0 - fixed_sum
                free_pops = (free_vals / float(free_vals.sum())) * remaining
                for c, p in zip(free_pop_components, free_pops):
                    c.population.value = float(p)

        def residuals(xvec: np.ndarray) -> np.ndarray:
            apply_x(xvec)
            pred_counts = n_events * bin_width * self.pdf(bin_centers)
            return (pred_counts - hist_counts.astype(float)) / denom

        res = least_squares(residuals, x0=x0_arr, bounds=bounds, method="trf")
        apply_x(res.x)

        return [c.copy() for c in self.components]


# %% [markdown]
# ## Configuration Parameters
# 
# Set all burst selection parameters here. These match the chisurf plugin parameters.
# %%
proximity_gaussians = [
    GaussianComponent(
        mu=Parameter(name="mu1", value=0.20, fixed=False, bounds_on=True, lb=-0.05, ub=1.05),
        sigma=Parameter(name="sigma1", value=0.05, fixed=False, bounds_on=True, lb=1e-4, ub=0.50),
        population=Parameter(name="p1", value=0.33, fixed=False, bounds_on=False),
    ),
    GaussianComponent(
        mu=Parameter(name="mu2", value=0.50, fixed=False, bounds_on=True, lb=-0.05, ub=1.05),
        sigma=Parameter(name="sigma2", value=0.05, fixed=False, bounds_on=True, lb=1e-4, ub=0.50),
        population=Parameter(name="p2", value=0.33, fixed=False, bounds_on=False),
    ),
    GaussianComponent(
        mu=Parameter(name="mu3", value=0.80, fixed=False, bounds_on=True, lb=-0.05, ub=1.05),
        sigma=Parameter(name="sigma3", value=0.05, fixed=False, bounds_on=True, lb=1e-4, ub=0.50),
        population=Parameter(name="p3", value=0.34, fixed=False, bounds_on=False),
    ),
]
n_gaussians = len(proximity_gaussians)
# %%
# ==============================================================================
# INPUT/OUTPUT CONFIGURATION
# ==============================================================================
input_folder = r"Q:\tttr-data\hdf\Brick-Mic\DNA_Alexa488-ATTO543-10sec"
file_type = 'PHOTON-HDF5'
output_folder_name = "burstwise_analysis"

# ==============================================================================
# DETECTOR CONFIGURATION - ALEX MODE
# ==============================================================================
# For ALEX data: photons are separated by routing channel
# Each tttr_channeldefinition gets ALL photons from its channel (no microtime filtering)
detectors = {
    "green": {
        "chs": [0],              # Routing channel for green tttr_channeldefinition
        "micro_time_ranges": []  # Empty = use ALL photons from this channel
    },
    "red": {
        "chs": [1],              # Routing channel for red tttr_channeldefinition
        "micro_time_ranges": []  # Empty = use ALL photons from this channel
    }
}

# PIE windows for analysis (microtime ranges)
# For ALEX without microtime separation, use full range
windows = {
    "prompt": (0, 65500)  # Full microtime range
}

# Channels to use for burst detection (empty = use all channels)
channels = []  # Empty = use all channels [0, 1]

# ==============================================================================
# BURST DETECTION PARAMETERS
# ==============================================================================

# Delta time filtering (milliseconds) - applied BEFORE burst detection
dt_min = 0.0001  # Minimum inter-photon time (ms)
dt_max = 0.25    # Maximum inter-photon time (ms)
use_dt_min = False
use_dt_max = True

# Burst search parameters (maps to tttrlib.TTTR.burst_search)
min_ph = 30        # Minimum number of photons per burst (minimum_number_of_photons)
ph_window = 10      # Number of consecutive photons to check (minimum_window)
time_window = dt_max / 1000.0  # Maximum duration in seconds (maximum_duration)

# Count rate filter parameters (when filter_mode = 'count_rate')
count_rate_window_ms = 1.0
invert_filter = False

# ==============================================================================
# GAP FILLING
# ==============================================================================
use_gap_fill = True
max_gap = 4  # Maximum gap size to fill (photons)

# ==============================================================================
# BINNING PARAMETERS
# ==============================================================================
trace_bin_width = 1.0  # Bin width for intensity trace (ms)

# ==============================================================================
# OUTPUT OPTIONS
# ==============================================================================
save_bur = True   # Save .bur files (tab-separated burst data)
save_hdf5 = True  # Save combined HDF5 file
zip_output = False
remove_folder_after_zip = False

print("Configuration loaded successfully!")
print(f"Input folder: {input_folder}")
print(f"  Green detector: Channel {detectors['green']['chs']}")
print(f"  Red detector: Channel {detectors['red']['chs']}")
print(f"  Using tttrlib.TTTR.burst_search() with:")
print(f"    min_ph = {min_ph}")
print(f"    ph_window = {ph_window}")
print(f"    time_window = {time_window:.6f} seconds")
# %%
def find_bursts(selection_mask, max_gap=0):
    """
    Find burst start and stop indices from a boolean selection mask.
    
    Parameters:
    -----------
    selection_mask : np.ndarray
        Boolean array where True indicates selected photons
    max_gap : int
        Maximum gap size to bridge between bursts
        
    Returns:
    --------
    np.ndarray
        Array of shape (n_bursts, 2) with [start, stop] indices
    """
    if len(selection_mask) == 0:
        return np.array([], dtype=np.uint64).reshape(0, 2)
    
    # Fill small gaps if requested
    if max_gap > 0:
        selection_mask = fill_small_gaps(selection_mask, max_gap)
    
    # Find transitions
    padded = np.pad(selection_mask.astype(np.int8), (1, 1), mode='constant', constant_values=0)
    diff = np.diff(padded)
    
    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0]
    
    if len(starts) == 0 or len(stops) == 0:
        return np.array([], dtype=np.uint64).reshape(0, 2)
    
    return np.column_stack((starts, stops)).astype(np.uint64)


def fill_small_gaps(arr, max_gap=4):
    """
    Fill small gaps in a boolean array.
    
    Parameters:
    -----------
    arr : np.ndarray
        Boolean array
    max_gap : int
        Maximum gap size to fill
        
    Returns:
    --------
    np.ndarray
        Array with small gaps filled
    """
    if max_gap <= 0:
        return arr
    
    result = arr.copy()
    in_burst = False
    gap_start = 0
    
    for i in range(len(arr)):
        if arr[i]:
            if in_burst and i - gap_start <= max_gap:
                # Fill the gap
                result[gap_start:i] = True
            in_burst = True
        else:
            if in_burst:
                gap_start = i
                in_burst = False
    
    return result


def create_array_with_ones(start_stop, length):
    """
    Create a boolean array with ones at specified ranges.
    
    Parameters:
    -----------
    start_stop : np.ndarray
        Array of shape (n_bursts, 2) with [start, stop] indices
    length : int
        Length of output array
        
    Returns:
    --------
    np.ndarray
        Boolean array with True in burst regions
    """
    result = np.zeros(length, dtype=bool)
    for start, stop in start_stop:
        if 0 <= start < length and 0 <= stop <= length and start < stop:
            result[start:stop] = True
    return result


# %%
def apply_channel_filter(tttr, channels):
    """
    Apply channel filter using tttrlib.TTTRMask.
    
    Parameters:
    -----------
    tttr : tttrlib.TTTR
        TTTR object
    channels : list
        List of channel numbers to select
        
    Returns:
    --------
    np.ndarray
        Boolean mask of selected photons
    """
    if not channels or len(channels) == 0:
        # No channel filter - select all
        return np.ones(len(tttr), dtype=bool)
    
    mask = tttrlib.TTTRMask()
    mask.select_channels(tttr, channels, mask=True)
    return mask.get_mask().astype(bool)


def apply_microtime_filter(tttr, microtime_ranges):
    """
    Apply microtime range filter using tttrlib.TTTRMask.
    
    Parameters:
    -----------
    tttr : tttrlib.TTTR
        TTTR object
    microtime_ranges : list of tuples
        List of (start, stop) microtime ranges to EXCLUDE
        
    Returns:
    --------
    np.ndarray
        Boolean mask of selected photons
    """
    if not microtime_ranges or len(microtime_ranges) == 0:
        return np.ones(len(tttr), dtype=bool)
    
    mask = tttrlib.TTTRMask()
    mask.select_microtime_ranges(tttr, microtime_ranges)
    mask.flip()  # Flip to exclude the ranges
    return mask.get_mask().astype(bool)


def count_rate_filter(tttr, n_ph_max, time_window, invert=False):
    """
    Apply count rate filter to TTTR data.
    This is a simplified implementation - chisurf uses a more complex version.
    
    Parameters:
    -----------
    tttr : tttrlib.TTTR
        TTTR object
    n_ph_max : int
        Maximum number of photons in time window
    time_window : float
        Time window in seconds
    invert : bool
        Invert the selection
        
    Returns:
    --------
    np.ndarray
        Boolean mask of selected photons
    """
    macro_times = tttr.macro_times
    res = tttr.header.macro_time_resolution
    n_photons = len(macro_times)
    
    if n_photons == 0:
        return np.array([], dtype=bool)
    
    # Convert time window to macro time units
    window_mt = int(time_window / res)
    
    # Create selection mask
    selection = np.zeros(n_photons, dtype=bool)
    
    # Sliding window count rate filter (optimized with searchsorted)
    for i in range(n_photons):
        t_start = macro_times[i]
        t_end = t_start + window_mt
        
        # Use binary search to find photons in window
        j = np.searchsorted(macro_times[i:], t_end, side='left') + i
        count = j - i
        
        # Apply threshold
        if count <= n_ph_max:
            selection[i] = True
    
    if invert:
        selection = ~selection
    
    return selection


def burst_filter(tttr, min_ph, ph_window, time_window):
    """
    Apply burst filter using tttrlib's burst_search method.
    
    This is the correct implementation matching chisurf.fluorescence.burst.burst_filter
    
    Parameters:
    -----------
    tttr : tttrlib.TTTR
        TTTR object
    min_ph : int
        Minimum photons per burst
    ph_window : int
        Number of consecutive photons to check for burst detection
    time_window : float
        Maximum time window in seconds
        
    Returns:
    --------
    np.ndarray
        Boolean mask of selected photons
    """
    # Call tttrlib's burst_search method with POSITIONAL arguments
    # Signature: burst_search(min_ph, ph_window, time_window)
    start_stop = tttr.burst_search(min_ph, ph_window, time_window)
    
    # Reshape to (n_bursts, 2) array
    start_stop = np.array(start_stop).reshape((-1, 2))
    
    # Create boolean mask from start-stop indices
    n_photons = len(tttr)
    selection = create_array_with_ones(start_stop, n_photons)
    
    return selection


# %% [markdown]
# ## Main Processing Function
# %%
def process_tttr_file(tttr_file, config):
    """
    Process a single TTTR file and return burst dataframe.
    
    This follows the chisurf WizardTTTRPhotonFilter workflow:
    1. Apply channel filter (using TTTRMask)
    2. Apply microtime filter (using TTTRMask)
    3. Apply dT filtering
    4. Apply burst/count rate filter
    5. Apply gap filling
    6. Find burst start/stop indices
    7. Generate burst dataframe
    
    Parameters:
    -----------
    tttr_file : pathlib.Path
        Path to TTTR file
    config : dict
        Configuration dictionary with all parameters
        
    Returns:
    --------
    pd.DataFrame or None
        Burst dataframe or None if processing failed
    """
    try:
        print(f"Processing: {tttr_file.name}")
        
        # Load TTTR data
        print(f"TTTR file: {tttr_file}")
        tttr = tttrlib.TTTR(str(tttr_file), 'PHOTON-HDF5')
        
        if len(tttr) == 0:
            print(f"  Warning: No photons in {tttr_file.name}")
            return None
        
        n_photons = len(tttr)
        print(f"  Total photons: {n_photons:,}")
        
        # Step 1: Initialize selection mask (all photons selected)
        selection = np.ones(n_photons, dtype=bool)
        
        # Step 2: Apply channel filter using TTTRMask
        if config['CHANNELS']:
            channel_mask = apply_channel_filter(tttr, config['CHANNELS'])
            selection = np.logical_and(selection, channel_mask)
            print(f"  After channel filter: {np.sum(selection):,} photons")
        
        # Step 3: Apply microtime filter using TTTRMask (if configured)
        # Note: This would exclude photons in certain microtime ranges
        # For now, we skip this as it's not in the basic config
        
        # Step 4: Apply delta macro time (dT) filtering
        if config['USE_DT_MIN'] or config['USE_DT_MAX']:
            macro_times = tttr.macro_times
            res = tttr.header.macro_time_resolution
            
            # Calculate dT (time between consecutive photons)
            dT = np.diff(macro_times, prepend=0) * res * 1000  # Convert to ms
            
            if config['USE_DT_MIN']:
                selection = np.logical_and(selection, dT >= config['DT_MIN'])
            if config['USE_DT_MAX']:
                selection = np.logical_and(selection, dT <= config['DT_MAX'])
            
            print(f"  After dT filter: {np.sum(selection):,} photons")
        
        # Step 5: Apply burst/count rate filter
        filter_mode = config['FILTER_MODE']
        
        if filter_mode == 'count_rate':
            print(f"  Applying count rate filter (n_ph_max={config['PHOTON_THRESHOLD']}, "
                  f"window={config['COUNT_RATE_WINDOW_MS']}ms)...")
            filter_mask = count_rate_filter(
                tttr,
                n_ph_max=config['PHOTON_THRESHOLD'],
                time_window=config['COUNT_RATE_WINDOW_MS'] / 1000.0,
                invert=config['INVERT_FILTER']
            )
            selection = np.logical_and(selection, filter_mask)
            print(f"  After count rate filter: {np.sum(selection):,} photons")
            
        # Use tttrlib's burst_search method
        filter_mask = burst_filter(
            tttr,
            min_ph=config['PHOTON_THRESHOLD'],
            ph_window=config['PH_WINDOW'],
            time_window=config['DT_MAX'] / 1000.0
        )
        selection = np.logical_and(selection, filter_mask)
        print(f"  After burst filter: {np.sum(selection):,} photons")
        
        # Step 6: Apply gap filling
        if config['USE_GAP_FILL'] and config['MAX_GAP'] > 0:
            selection = fill_small_gaps(selection, config['MAX_GAP'])
            print(f"  After gap filling (max_gap={config['MAX_GAP']}): {np.sum(selection):,} photons")
        
        # Step 7: Find burst start/stop indices from the selection mask
        # This is the key step - we find continuous regions of True values
        burst_start_stop = find_bursts(selection, max_gap=0)
        
        print(f"  Found {len(burst_start_stop)} bursts")
        
        if len(burst_start_stop) == 0:
            print(f"  Warning: No bursts found in {tttr_file.name}")
            return None
        
        # Print burst statistics
        burst_sizes = burst_start_stop[:, 1] - burst_start_stop[:, 0]
        print(f"  Burst size: min={burst_sizes.min()}, max={burst_sizes.max()}, "
              f"mean={burst_sizes.mean():.1f}, median={np.median(burst_sizes):.1f}")
        
        # Step 8: Generate burst dataframe
        df = generate_burst_dataframe(
            start_stop=burst_start_stop,
            filename=tttr_file,
            tttr=tttr,
            windows=config['WINDOWS'],
            detectors=config['DETECTORS'],
            include_interleaved_zeros=config['SAVE_BUR']
        )
        
        print(f"  Generated dataframe with {len(df)} rows")
        
        return df
        
    except Exception as e:
        print(f"  Error processing {tttr_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# %%
def generate_burst_dataframe(start_stop, filename, tttr, windows, detectors, include_interleaved_zeros=True):
    """
    Generate a DataFrame with burst summary information.
    Compatible with chisurf burst_selection plugin output format.
    
    Parameters:
    -----------
    start_stop : np.ndarray
        Array of shape (n_bursts, 2) with [start, stop] indices
    filename : str or pathlib.Path
        Path to the TTTR file
    tttr : tttrlib.TTTR
        TTTR object
    windows : dict
        Dictionary {window_name: (r_start, r_stop)}
    detectors : dict
        Dictionary {det_name: {"chs": [...], "micro_time_ranges": [(mt_start, mt_stop), ...]}}
    include_interleaved_zeros : bool
        Whether to include interleaved zero rows (for .bur format compatibility)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing burst summary information
    """
    file_name_only = pathlib.Path(filename).name
    
    # Unpack TTTR data
    macro = tttr.macro_times
    micro = tttr.micro_times
    rout = tttr.routing_channel
    res = tttr.header.macro_time_resolution
    n_ph = len(tttr)
    
    # Build column list
    static_cols = [
        "First Photon", "Last Photon", "Duration (ms)", "Mean Macro Time (ms)",
        "Number of Photons", "Count Rate (KHz)", "First File", "Last File",
    ]
    det_cols = []
    for d in detectors:
        det_cols += [
            f"First Photon ({d})", f"Last Photon ({d})",
            f"Duration ({d}) (ms)", f"Mean Macrotime ({d}) (ms)",
            f"Number of Photons ({d})", f"{d.capitalize()} Count Rate (KHz)",
        ]
    win_cols = []
    for w, (r0, r1) in windows.items():
        for d in detectors:
            # Remove window range suffix to match YAML expectations
            win_cols.append(f"S {w} {d} (kHz)")
    
    # Extra blank column for compatibility
    cols = static_cols + det_cols + win_cols + [""]
    
    # Map column to index for fast assignment
    idx = {c: i for i, c in enumerate(cols)}
    n_cols = len(cols)
    
    # Precompute global masks for detectors
    det_global = {}
    for d, info in detectors.items():
        # Channel mask
        chm = np.isin(rout, info["chs"])
        
        # Microtime mask
        if len(info["micro_time_ranges"]) > 0:
            # If microtime ranges specified, filter by them
            mtm = np.zeros(n_ph, bool)
            for r0, r1 in info["micro_time_ranges"]:
                mtm |= (micro >= r0) & (micro < r1)
            det_global[d] = chm & mtm
        else:
            # If no microtime ranges, use all photons from this channel (ALEX mode)
            det_global[d] = chm
    
    # Precompute global masks for windows
    win_global = {}
    for w, (r0, r1) in windows.items():
        # Only apply window filter if microtime data exists (not all zeros)
        if micro.max() > 0:
            win_global[w] = (micro >= r0) & (micro < r1)
        else:
            # No microtime data - include all photons
            win_global[w] = np.ones(n_ph, bool)
    
    # Helper zero-row
    zero_row = [0] * n_cols
    zero_row[-1] = ""  # Last column is blank string
    
    out = []
    
    # Add leading zero row if there are bursts and interleaved zeros requested
    has_bursts = len(start_stop) > 0
    if has_bursts and include_interleaved_zeros:
        out.append(zero_row.copy())
    
    # Process each burst
    for start, stop in start_stop:
        # Convert numpy integers to Python int for indexing
        start = int(start)
        stop = int(stop)
        
        if stop <= start or stop > n_ph or start < 0:
            continue
        
        # Allocate a fresh row
        row = zero_row.copy()
        
        # Static stats
        dur = (macro[stop - 1] - macro[start]) * res * 1e3  # Duration in ms
        meanm = ((macro[stop - 1] + macro[start]) / 2) * res * 1e3  # Mean time in ms
        npix = stop - start
        crate = (npix / (dur / 1e3)) / 1e3 if dur > 0 else np.nan  # Count rate in KHz
        
        row[idx["First Photon"]] = start
        row[idx["Last Photon"]] = stop
        row[idx["Duration (ms)"]] = dur
        row[idx["Mean Macro Time (ms)"]] = meanm
        row[idx["Number of Photons"]] = npix
        row[idx["Count Rate (KHz)"]] = crate
        row[idx["First File"]] = file_name_only
        row[idx["Last File"]] = file_name_only
        
        # Slice views for this burst
        sl = slice(start, stop)
        
        # Per-tttr_channeldefinition stats
        for d in detectors:
            mask = det_global[d][sl]
            idxs = np.nonzero(mask)[0]
            col0 = f"First Photon ({d})"
            
            if idxs.size == 0:
                row[idx[col0]] = -1
                row[idx[f"Last Photon ({d})"]] = -1
                row[idx[f"Duration ({d}) (ms)"]] = -1.0
                row[idx[f"Mean Macrotime ({d}) (ms)"]] = -1.0
                row[idx[f"Number of Photons ({d})"]] = 0
                row[idx[f"{d.capitalize()} Count Rate (KHz)"]] = -1.0
            else:
                i0, i1 = int(idxs[0]), int(idxs[-1])
                abs0, abs1 = start + i0, start + i1
                d_ms = (macro[abs1] - macro[abs0]) * res * 1e3
                m_ms = ((macro[abs1] + macro[abs0]) / 2) * res * 1e3
                rate = (idxs.size / (d_ms / 1e3)) / 1e3 if d_ms > 0 else np.nan
                
                row[idx[col0]] = abs0
                row[idx[f"Last Photon ({d})"]] = abs1
                row[idx[f"Duration ({d}) (ms)"]] = d_ms
                row[idx[f"Mean Macrotime ({d}) (ms)"]] = m_ms
                row[idx[f"Number of Photons ({d})"]] = idxs.size
                row[idx[f"{d.capitalize()} Count Rate (KHz)"]] = rate
        
        # Per-window, per-tttr_channeldefinition stats
        for w in windows:
            wmask = win_global[w][sl]
            for d in detectors:
                combined = det_global[d][sl] & wmask
                idxs = np.nonzero(combined)[0]
                # Column name without window range suffix
                key = f"S {w} {d} (kHz)"
                
                if idxs.size == 0:
                    row[idx[key]] = -1.0
                else:
                    abs0, abs1 = start + int(idxs[0]), start + int(idxs[-1])
                    d_ms = (macro[abs1] - macro[abs0]) * res * 1e3
                    row[idx[key]] = (idxs.size / (d_ms / 1e3)) / 1e3 if d_ms > 0 else np.nan
        
        # Blank column already set to ""
        out.append(row)
        
        # Add trailing zero row if interleaved zeros requested
        if include_interleaved_zeros:
            out.append(zero_row.copy())
    
    # Build DataFrame
    return pd.DataFrame(out, columns=cols)

# %%
def process_tttr_file(tttr_file, config):
    """
    Process a single TTTR file and return burst dataframe.
    
    This follows the chisurf WizardTTTRPhotonFilter workflow:
    1. Apply channel filter (using TTTRMask)
    2. Apply microtime filter (using TTTRMask)
    3. Apply dT filtering
    4. Apply burst/count rate filter
    5. Apply gap filling
    6. Find burst start/stop indices
    7. Generate burst dataframe
    
    Parameters:
    -----------
    tttr_file : pathlib.Path
        Path to TTTR file
    config : dict
        Configuration dictionary with all parameters
        
    Returns:
    --------
    pd.DataFrame or None
        Burst dataframe or None if processing failed
    """
    try:
        print(f"Processing: {tttr_file.name}")
        
        # Load TTTR data
        print(f"TTTR file: {tttr_file}")
        tttr = tttrlib.TTTR(str(tttr_file), file_type)
        
        if len(tttr) == 0:
            print(f"  Warning: No photons in {tttr_file.name}")
            return None
        
        n_photons = len(tttr)
        print(f"  Total photons: {n_photons:,}")
        
        # Step 1: Initialize selection mask (all photons selected)
        selection = np.ones(n_photons, dtype=bool)
        
        # Step 2: Apply channel filter using TTTRMask
        if config['CHANNELS']:
            channel_mask = apply_channel_filter(tttr, config['CHANNELS'])
            selection = np.logical_and(selection, channel_mask)
            print(f"  After channel filter: {np.sum(selection):,} photons")
        
        # Step 3: Apply microtime filter using TTTRMask (if configured)
        # Note: This would exclude photons in certain microtime ranges
        # For now, we skip this as it's not in the basic config
        
        # Step 4: Apply delta macro time (dT) filtering
        if config['USE_DT_MIN'] or config['USE_DT_MAX']:
            macro_times = tttr.macro_times
            res = tttr.header.macro_time_resolution
            
            # Calculate dT (time between consecutive photons)
            dT = np.diff(macro_times, prepend=0) * res * 1000  # Convert to ms
            
            if config['USE_DT_MIN']:
                selection = np.logical_and(selection, dT >= config['DT_MIN'])
            if config['USE_DT_MAX']:
                selection = np.logical_and(selection, dT <= config['DT_MAX'])
            
            print(f"  After dT filter: {np.sum(selection):,} photons")
        
        # Step 5: Apply burst/count rate filter
        filter_mode = config['FILTER_MODE']
        
        if filter_mode == 'count_rate':
            print(f"  Applying count rate filter (n_ph_max={config['MIN_PH']}, "
                  f"window={config['COUNT_RATE_WINDOW_MS']}ms)...")
            filter_mask = count_rate_filter(
                tttr,
                n_ph_max=config['MIN_PH'],
                time_window=config['COUNT_RATE_WINDOW_MS'] / 1000.0,
                invert=config['INVERT_FILTER']
            )
            selection = np.logical_and(selection, filter_mask)
            print(f"  After count rate filter: {np.sum(selection):,} photons")
            
        # Use tttrlib's burst_search method
        print(f"  Running tttrlib.TTTR.burst_search():")
        print(f"    minimum_number_of_photons (MIN_PH) = {config['MIN_PH']}")
        print(f"    minimum_window (PH_WINDOW) = {config['PH_WINDOW']}")
        print(f"    maximum_duration (TIME_WINDOW) = {config['TIME_WINDOW']:.6f} seconds")
            
        filter_mask = burst_filter(
            tttr,
            min_ph=config['MIN_PH'],
            ph_window=config['PH_WINDOW'],
            time_window=config['TIME_WINDOW']
        )
        selection = np.logical_and(selection, filter_mask)
        print(f"  After burst filter: {np.sum(selection):,} photons")
        
        # Step 6: Apply gap filling
        if config['USE_GAP_FILL'] and config['MAX_GAP'] > 0:
            selection = fill_small_gaps(selection, config['MAX_GAP'])
            print(f"  After gap filling (max_gap={config['MAX_GAP']}): {np.sum(selection):,} photons")
        
        # Step 7: Find burst start/stop indices from the selection mask
        # This is the key step - we find continuous regions of True values
        burst_start_stop = find_bursts(selection, max_gap=0)
        
        n_bursts = len(burst_start_stop)
        print(f"  Found {n_bursts} bursts")
        
        if n_bursts == 0:
            print(f"  Warning: No bursts found in {tttr_file.name}")
            return None
        
        # Step 8: Generate burst dataframe
        df = generate_burst_dataframe(
            burst_start_stop,
            tttr_file,
            tttr,
            config['WINDOWS'],
            config['DETECTORS'],
            include_interleaved_zeros=True
        )
        
        print(f"  Generated dataframe with {len(df)} rows (includes zero rows)")
        return df
        
    except Exception as e:
        print(f"  Error processing {tttr_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

# %%
def process_tttr_file(tttr_file, config):
    """
    Process a single TTTR file and return burst dataframe.
    
    This follows the chisurf WizardTTTRPhotonFilter workflow:
    1. Apply channel filter (using TTTRMask)
    2. Apply microtime filter (using TTTRMask)
    3. Apply dT filtering
    4. Apply burst/count rate filter
    5. Apply gap filling
    6. Find burst start/stop indices
    7. Generate burst dataframe
    
    Parameters:
    -----------
    tttr_file : pathlib.Path
        Path to TTTR file
    config : dict
        Configuration dictionary with all parameters
        
    Returns:
    --------
    pd.DataFrame or None
        Burst dataframe or None if processing failed
    """
    print(f"Processing: {tttr_file.name}")
    
    # Load TTTR data
    print(f"TTTR file: {tttr_file}")
    tttr = tttrlib.TTTR(str(tttr_file), file_type)
    
    if len(tttr) == 0:
        print(f"  Warning: No photons in {tttr_file.name}")
        return None
    
    n_photons = len(tttr)
    print(f"  Total photons: {n_photons:,}")
    
    # Step 1: Initialize selection mask (all photons selected)
    selection = np.ones(n_photons, dtype=bool)
    
    # Step 2: Apply channel filter using TTTRMask
    if config['channels']:
        channel_mask = apply_channel_filter(tttr, config['channels'])
        selection = np.logical_and(selection, channel_mask)
        print(f"  After channel filter: {np.sum(selection):,} photons")
    
    # Step 3: Apply microtime filter using TTTRMask (if configured)
    # Note: This would exclude photons in certain microtime ranges
    # For now, we skip this as it's not in the basic config
    
    # Step 4: Apply delta macro time (dT) filtering
    if config['use_dt_min'] or config['use_dt_max']:
        macro_times = tttr.macro_times
        res = tttr.header.macro_time_resolution
        
        # Calculate dT (time between consecutive photons)
        dT = np.diff(macro_times, prepend=0) * res * 1000  # Convert to ms
        
        if config['use_dt_min']:
            selection = np.logical_and(selection, dT >= config['dt_min'])
        if config['use_dt_max']:
            selection = np.logical_and(selection, dT <= config['dt_max'])
        
        print(f"  After dT filter: {np.sum(selection):,} photons")
    
        # Use tttrlib's burst_search method
        print(f"  Running tttrlib.TTTR.burst_search():")
        print(f"    min_ph = {config['min_ph']}")
        print(f"    ph_window = {config['ph_window']}")
        print(f"    time_window = {config['time_window']:.6f} seconds")
        
        filter_mask = burst_filter(
            tttr,
            min_ph=config['min_ph'],
            ph_window=config['ph_window'],
            time_window=config['time_window']
        )
        selection = np.logical_and(selection, filter_mask)
        print(f"  After burst filter: {np.sum(selection):,} photons")
        
        # Step 6: Apply gap filling
        if config['use_gap_fill'] and config['max_gap'] > 0:
            selection = fill_small_gaps(selection, config['max_gap'])
            print(f"  After gap filling (max_gap={config['max_gap']}): {np.sum(selection):,} photons")
        
        # Step 7: Find burst start/stop indices from the selection mask
        # This is the key step - we find continuous regions of True values
        burst_start_stop = find_bursts(selection, max_gap=0)
        
        n_bursts = len(burst_start_stop)
        print(f"  Found {n_bursts} bursts")
        
        if n_bursts == 0:
            print(f"  Warning: No bursts found in {tttr_file.name}")
            return None
        
        # Step 8: Generate burst dataframe
        df = generate_burst_dataframe(
            burst_start_stop,
            tttr_file,
            tttr,
            config['windows'],
            config['detectors'],
            include_interleaved_zeros=True
        )
        
        print(f"  Generated dataframe with {len(df)} rows (includes zero rows)")
        return df
        

# %%
# Collect configuration into a dictionary
config = {
    'detectors': detectors,
    'windows': windows,
    'channels': channels,
    'dt_min': dt_min,
    'dt_max': dt_max,
    'use_dt_min': use_dt_min,
    'use_dt_max': use_dt_max,
    'min_ph': min_ph,
    'ph_window': ph_window,
    'time_window': time_window,
    'count_rate_window_ms': count_rate_window_ms,
    'invert_filter': invert_filter,
    'use_gap_fill': use_gap_fill,
    'max_gap': max_gap,
    'trace_bin_width': trace_bin_width,
    'save_bur': save_bur,
    'save_hdf5': save_hdf5,
}

# Find all HDF5 files in input folder
input_path = pathlib.Path(input_folder)
hdf5_files = sorted(input_path.glob('*.h5')) + sorted(input_path.glob('*.hdf5'))

print(f"Found {len(hdf5_files)} HDF5 files in {input_folder}")
for f in hdf5_files:
    print(f"  - {f.name}")

# Process all files
all_dataframes = []
for hdf5_file in hdf5_files:
    df = process_tttr_file(hdf5_file, config)
    if df is not None:
        all_dataframes.append(df)

print(f"\nSuccessfully processed {len(all_dataframes)} files")
# %%
def make_fret_panels(
    all_dataframes,
    output_path,
    gaussians: Optional[List[GaussianComponent]] = None,
    n_gaussians=3,
    bins_prox=50,
    bins_dt=50,
    bins_rate=50,
    bins_da=50,
    prox_range=(0.0, 1.0),
    dt_range_hist=(-1.0, 1.0),
    dt_axis_range=(-1.1, 1.1),
    rate_range=(0.0, 200.0),
    rate_range_da=(0.0, 200.0),
):
    """Generate a manual Gaussian-mixture fit (or optional GMM) and FRET-style panels from burst data.

    Parameters
    ----------
    all_dataframes : list of pd.DataFrame
        List of burst dataframes from all TTTR files.
    output_path : pathlib.Path
        Base output directory where the 'for_plot' subfolder will be created.
    gaussians : list[GaussianComponent]
        User-specified Gaussian components for the Proximity ratio histogram.
        The number of Gaussians is inferred from the list length.
        Parameters can be fixed via the boolean flags on each component.
    bins_prox, bins_dt, bins_rate, bins_da : int
        Number of bins for Proximity ratio, dT, count-rate vs E*, and donor/acceptor rate histograms.
    prox_range : (float, float)
        Range for Apparent FRET / Proximity ratio.
    dt_range_hist : (float, float)
        Range used for dT binning.
    dt_axis_range : (float, float)
        Axis limits for the dT panel.
    rate_range : (float, float)
        Axis limits for the Acceptor count-rate vs E* panel.
    rate_range_da : (float, float)
        Axis limits for the donor vs acceptor count-rate panel.
    """
    for_plot_dir = output_path / "for_plot"
    for_plot_dir.mkdir(exist_ok=True, parents=True)

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df = combined_df.loc[~(combined_df.select_dtypes(include=["number"]) == 0).all(axis=1)]

    prox_min, prox_max = prox_range

    # ------------------------------------------------------------------
    # Proximity ratio and GMM fit
    # ------------------------------------------------------------------
    if "Number of Photons (green)" in combined_df.columns and "Number of Photons (red)" in combined_df.columns:
        total_ph = combined_df["Number of Photons (red)"] + combined_df["Number of Photons (green)"]
        prox_series = combined_df["Number of Photons (red)"] / total_ph
        prox_series = prox_series.replace([np.inf, -np.inf], np.nan)
        combined_df["Proximity Ratio"] = prox_series

        prox_valid_mask = total_ph > 0
        prox_valid_mask &= prox_series.notna()
        prox_valid_mask &= (prox_series >= prox_min) & (prox_series <= prox_max)
        prox_valid = prox_series[prox_valid_mask]
    else:
        prox_valid = pd.Series([], dtype=float)

    if len(prox_valid) == 0:
        print("No valid Proximity Ratio data available for GMM fitting and plotting.")
        return

    bins_x = bins_prox
    bin_width = abs((prox_max - prox_min) / bins_x) if prox_max != prox_min else 1.0

    if not isinstance(gaussians, list) or len(gaussians) == 0:
        raise ValueError("gaussians must be a non-empty list of GaussianComponent")

    n_components = len(gaussians)
    fitter = GaussianMixtureHistogramFitter([g.copy() for g in gaussians])
    fitted = fitter.fit_to_histogram(
        data=prox_valid.to_numpy(),
        bins=bins_x,
        data_range=(prox_min, prox_max),
    )

    weights = np.array([c.population.value for c in fitted], dtype=float)
    means = np.array([c.mu.value for c in fitted], dtype=float)
    sigmas = np.array([c.sigma.value for c in fitted], dtype=float)

    print("Manual Gaussian mixture parameters (Proximity Ratio):")
    for i in range(n_components):
        print(f"  Component {i + 1}: weight={weights[i]:.4f}, mean={means[i]:.4f}, sigma={sigmas[i]:.4f}")

    n_points = 1000
    x = np.linspace(prox_min, prox_max, n_points)
    pdf_components = fitter.component_pdfs(x)
    pdf_total = np.sum(pdf_components, axis=0)

    scale_factor = len(prox_valid) * bin_width
    count_components = [pdf_components[i] * scale_factor for i in range(n_components)]
    count_total = pdf_total * scale_factor

    # Save GMM curves (counts) to CSV
    gmm_df = pd.DataFrame({"proximity_ratio": x, "gmm_total_counts": count_total})
    for i in range(n_components):
        gmm_df[f"component_{i + 1}_counts"] = count_components[i]
    gmm_csv_path = for_plot_dir / "gmm_proximity_ratio_curves.csv"
    gmm_df.to_csv(gmm_csv_path, index=False)

    # 1D marginal histogram (counts)
    hist_counts, bin_edges = np.histogram(prox_valid.to_numpy(), bins=bins_x, range=(prox_min, prox_max))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_df = pd.DataFrame({"proximity_ratio_center": bin_centers, "counts": hist_counts})
    hist_csv_path = for_plot_dir / "hist_proximity_ratio.csv"
    hist_df.to_csv(hist_csv_path, index=False)

    # ------------------------------------------------------------------
    # 2D histograms: dT vs E*, Acceptor rate vs E*, Acceptor vs Donor rate
    # ------------------------------------------------------------------
    has_dt = "Duration (green) (ms)" in combined_df.columns and "Duration (red) (ms)" in combined_df.columns
    has_rates = "Green Count Rate (KHz)" in combined_df.columns and "Red Count Rate (KHz)" in combined_df.columns

    hist2d_dt = None
    hist2d_acc_fret = None
    hist2d_da = None

    if has_dt:
        Tg_minus_Tr = combined_df["Duration (green) (ms)"] - combined_df["Duration (red) (ms)"]
        Tg_minus_Tr = Tg_minus_Tr.replace([np.inf, -np.inf], np.nan)
        dt_valid_mask = prox_valid_mask & Tg_minus_Tr.notna()
        prox_dt = prox_series[dt_valid_mask]
        dt_vals = Tg_minus_Tr[dt_valid_mask]

        bins_y_dt = bins_dt
        hist2d_dt, xedges_dt, yedges_dt = np.histogram2d(
            prox_dt.to_numpy(),
            dt_vals.to_numpy(),
            bins=[bins_x, bins_y_dt],
            range=[list(prox_range), list(dt_range_hist)],
        )

        x_centers_dt = 0.5 * (xedges_dt[:-1] + xedges_dt[1:])
        y_centers_dt = 0.5 * (yedges_dt[:-1] + yedges_dt[1:])
        Xc_dt, Yc_dt = np.meshgrid(x_centers_dt, y_centers_dt, indexing="ij")
        hist2d_dt_df = pd.DataFrame({
            "proximity_ratio_center": Xc_dt.ravel(),
            "delta_t_center_ms": Yc_dt.ravel(),
            "counts": hist2d_dt.ravel(),
        })
        hist2d_dt_csv_path = for_plot_dir / "hist2d_prox_vs_delta_t.csv"
        hist2d_dt_df.to_csv(hist2d_dt_csv_path, index=False)

    if has_rates:
        cr_g = combined_df["Green Count Rate (KHz)"]
        cr_r = combined_df["Red Count Rate (KHz)"]
        cr_g = cr_g.replace([np.inf, -np.inf], np.nan)
        cr_r = cr_r.replace([np.inf, -np.inf], np.nan)

        # Acceptor rate vs E* (top-3 figure)
        acc_valid_mask = prox_valid_mask & cr_r.notna()
        prox_acc = prox_series[acc_valid_mask]
        acc_vals = cr_r[acc_valid_mask]

        bins_y_acc = bins_rate
        hist2d_acc_fret, xedges_acc, yedges_acc = np.histogram2d(
            prox_acc.to_numpy(),
            acc_vals.to_numpy(),
            bins=[bins_x, bins_y_acc],
            range=[list(prox_range), list(rate_range)],
        )

        x_centers_acc = 0.5 * (xedges_acc[:-1] + xedges_acc[1:])
        y_centers_acc = 0.5 * (yedges_acc[:-1] + yedges_acc[1:])
        Xc_acc, Yc_acc = np.meshgrid(x_centers_acc, y_centers_acc, indexing="ij")
        hist2d_acc_df = pd.DataFrame({
            "proximity_ratio_center": Xc_acc.ravel(),
            "acceptor_rate_center_khz": Yc_acc.ravel(),
            "counts": hist2d_acc_fret.ravel(),
        })
        hist2d_acc_csv_path = for_plot_dir / "hist2d_prox_vs_acceptor_rate.csv"
        hist2d_acc_df.to_csv(hist2d_acc_csv_path, index=False)

        # Acceptor vs Donor count rate (separate figure)
        da_valid_mask = cr_g.notna() & cr_r.notna()
        donor_vals = cr_g[da_valid_mask]
        acceptor_vals = cr_r[da_valid_mask]

        bins_donor = bins_da
        bins_acceptor = bins_da
        hist2d_da, xedges_da, yedges_da = np.histogram2d(
            donor_vals.to_numpy(),
            acceptor_vals.to_numpy(),
            bins=[bins_donor, bins_acceptor],
            range=[list(rate_range_da), list(rate_range_da)],
        )

        x_centers_da = 0.5 * (xedges_da[:-1] + xedges_da[1:])
        y_centers_da = 0.5 * (yedges_da[:-1] + yedges_da[1:])
        Xc_da, Yc_da = np.meshgrid(x_centers_da, y_centers_da, indexing="ij")
        hist2d_da_df = pd.DataFrame({
            "donor_rate_center_khz": Xc_da.ravel(),
            "acceptor_rate_center_khz": Yc_da.ravel(),
            "counts": hist2d_da.ravel(),
        })
        hist2d_da_csv_path = for_plot_dir / "hist2d_donor_vs_acceptor_rate.csv"
        hist2d_da_df.to_csv(hist2d_da_csv_path, index=False)

    # ------------------------------------------------------------------
    # Figure 1: top three panels (marginal + dT + Acceptor vs E*)
    # ------------------------------------------------------------------
    svg_path_top = None
    png_path_top = None
    if hist2d_dt is not None and hist2d_acc_fret is not None:
        base_cmap = cm.get_cmap("viridis", 256)
        newcolors = base_cmap(np.linspace(0, 1, 256))
        newcolors[0, -1] = 0.0
        transparent_viridis = colors.ListedColormap(newcolors)

        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(3.5, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.2, 3.0, 3.0]},
        )

        # Panel 1: marginal + GMM
        ax1.bar(
            bin_centers,
            hist_counts,
            width=(prox_max - prox_min) / bins_x,
            color="lightgray",
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.plot(x, count_total, color="black", linewidth=1.5, zorder=2)
        for i in range(n_components):
            ax1.plot(
                x,
                count_components[i],
                color="darkmagenta",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
                zorder=3,
            )
        ax1.set_xlim(prox_min, prox_max)
        ax1.set_ylabel("Events", labelpad=4)
        ax1.tick_params(axis="x", labelbottom=False)

        # Panel 2: dT vs E*
        pcm_dt = ax2.pcolormesh(
            xedges_dt,
            yedges_dt,
            hist2d_dt.T,
            cmap=transparent_viridis,
            shading="auto",
            linewidth=0.0,
            antialiased=False,
        )
        ax2.set_xlim(prox_min, prox_max)
        ax2.set_ylim(dt_axis_range)
        ax2.set_ylabel("dT (Donor, Acceptor) [ms]", labelpad=4)
        ax2.tick_params(axis="x", labelbottom=False)
        ax2.set_box_aspect(1.0)

        # Panel 3: Acceptor rate vs E*
        pcm_acc = ax3.pcolormesh(
            xedges_acc,
            yedges_acc,
            hist2d_acc_fret.T,
            cmap=transparent_viridis,
            shading="auto",
            linewidth=0.0,
            antialiased=False,
        )
        ax3.set_xlim(prox_min, prox_max)
        ax3.set_ylim(rate_range)
        ax3.set_ylabel("Acceptor Count Rate [kHz]", labelpad=4)
        ax3.set_xlabel(r"Apparent FRET $E^*$", labelpad=6)
        ax3.set_box_aspect(1.0)

        for ax in [ax1, ax2, ax3]:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")

        fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.10, hspace=0.0)

        svg_path_top = for_plot_dir / "fret_panel_top3.svg"
        png_path_top = for_plot_dir / "fret_panel_top3.png"
        fig.savefig(svg_path_top, format="svg")
        fig.savefig(png_path_top, format="png", dpi=300)
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: donor vs acceptor count rate
    # ------------------------------------------------------------------
    svg_path_da = None
    png_path_da = None
    if hist2d_da is not None:
        base_cmap = cm.get_cmap("viridis", 256)
        newcolors = base_cmap(np.linspace(0, 1, 256))
        newcolors[0, -1] = 0.0
        transparent_viridis_da = colors.ListedColormap(newcolors)

        fig2 = plt.figure(figsize=(3.5, 3.5))
        ax_da = fig2.add_subplot(1, 1, 1)
        pcm_da = ax_da.pcolormesh(
            xedges_da,
            yedges_da,
            hist2d_da.T,
            cmap=transparent_viridis_da,
            shading="auto",
            linewidth=0.0,
            antialiased=False,
        )
        ax_da.set_xlim(rate_range_da)
        ax_da.set_ylim(rate_range_da)
        ax_da.set_xlabel("Donor Count Rate [kHz]", labelpad=6)
        ax_da.set_ylabel("Acceptor Count Rate [kHz]", labelpad=4)
        ax_da.set_box_aspect(1.0)

        for label in ax_da.get_xticklabels() + ax_da.get_yticklabels():
            label.set_fontweight("bold")

        fig2.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.12)

        svg_path_da = for_plot_dir / "donor_acceptor_panel.svg"
        png_path_da = for_plot_dir / "donor_acceptor_panel.png"
        fig2.savefig(svg_path_da, format="svg")
        fig2.savefig(png_path_da, format="png", dpi=300)
        plt.show()
        plt.close(fig2)

    # Info file
    info_path = for_plot_dir / "info.txt"
    with open(info_path, "w") as f:
        f.write(f"Number of bursts used for proximity-ratio fit: {len(prox_valid)}\n")
        requested_n = len(gaussians) if gaussians is not None else n_gaussians
        f.write(f"Number of Gaussian components requested: {requested_n}\n")
        f.write(f"Number of Gaussian components used: {n_components}\n")
        for i in range(n_components):
            f.write(
                f"Component {i + 1}: weight={weights[i]:.6f}, "
                f"mean={means[i]:.6f}, sigma={sigmas[i]:.6f}\n"
            )
        f.write(f"Histogram CSV (Proximity Ratio): {hist_csv_path.name}\n")
        f.write(f"GMM curves CSV: {gmm_csv_path.name}\n")
        if has_dt:
            f.write(f"2D histogram CSV (E* vs dT): {hist2d_dt_csv_path.name}\n")
        if has_rates:
            f.write(f"2D histogram CSV (E* vs Acceptor rate): {hist2d_acc_csv_path.name}\n")
            f.write(f"2D histogram CSV (Donor vs Acceptor rates): {hist2d_da_csv_path.name}\n")
        if svg_path_top is not None:
            f.write(f"Top-3 overview SVG figure: {svg_path_top.name}\n")
        if png_path_top is not None:
            f.write(f"Top-3 overview PNG figure: {png_path_top.name}\n")
        if svg_path_da is not None:
            f.write(f"Donor/Acceptor SVG figure: {svg_path_da.name}\n")
        if png_path_da is not None:
            f.write(f"Donor/Acceptor PNG figure: {png_path_da.name}\n")

    print(f"GMM fit and panel figures saved in: {for_plot_dir}")
# %%
# Create output directory
output_path = input_path / output_folder_name
output_path.mkdir(exist_ok=True, parents=True)

print(f"Output directory: {output_path}")

# Save .bur files
if save_bur:
    bur_dir = output_path / 'bi4_bur'
    bur_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nSaving .bur files to {bur_dir}")
    for i, (hdf5_file, df) in enumerate(zip(hdf5_files, all_dataframes)):
        bur_file = bur_dir / f"{hdf5_file.stem}.bur"
        df.to_csv(bur_file, sep='\t', index=False)
        print(f"  Saved: {bur_file.name}")

# Save combined HDF5 file
if save_hdf5:
    hdf5_dir = output_path / 'hdf5'
    hdf5_dir.mkdir(exist_ok=True, parents=True)
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Remove interleaved zero rows for HDF5
    combined_df = combined_df.loc[~(combined_df.select_dtypes(include=['number']) == 0).all(axis=1)]
    
    # Drop empty columns
    combined_df = combined_df.dropna(axis=1, how='all')
    
    # Optimize data types
    for c in combined_df.select_dtypes(include=["int"]).columns:
        if (combined_df[c] >= 0).all():
            combined_df[c] = pd.to_numeric(combined_df[c], downcast="unsigned")
        else:
            combined_df[c] = pd.to_numeric(combined_df[c], downcast="integer")
    
    for c in combined_df.select_dtypes(include=["float"]).columns:
        combined_df[c] = combined_df[c].astype(np.float32)
    
    # Save to HDF5
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    h5_file = hdf5_dir / f"burst_data_{timestamp}.h5"
    
    print(f"\nSaving combined HDF5 file to {h5_file}")
    combined_df.to_hdf(h5_file, key='results', mode='w', format='fixed', index=False)
    print(f"  Saved {len(combined_df)} bursts to HDF5")

    # Save parameters info
    info_dir = output_path / 'Info'
    info_dir.mkdir(exist_ok=True, parents=True)
    
    params = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dt_min': dt_min,
        'dt_max': dt_max,
        'use_dt_min': use_dt_min,
        'use_dt_max': use_dt_max,
        'min_ph': min_ph,
        'ph_window': ph_window,
        'time_window': time_window,
        'count_rate_window_ms': count_rate_window_ms,
        'invert_filter': invert_filter,
        'use_gap_fill': use_gap_fill,
        'max_gap': max_gap,
        'trace_bin_width': trace_bin_width,
        'channels': channels,
        'detectors': detectors,
        'windows': windows,
        'files': [f.name for f in hdf5_files]
    }
    
    with open(info_dir / 'photon_selection_parameters.json', 'w') as f:
        json.dump(params, f, indent=4)
    
    with open(info_dir / 'datetime.txt', 'w') as f:
        now = datetime.now()
        f.write(f"Date: {now.strftime('%Y-%m-%d')}\nTime: {now.strftime('%H:%M:%S')}\n")
    
    print(f"\nSaved parameter files to {info_dir}")
    
    # Optionally zip the output
    if zip_output:
        zip_file = output_path.parent / f"{output_path.name}.zip"
        print(f"\nCreating ZIP archive: {zip_file}")
        
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    file_path = pathlib.Path(root) / file
                    rel_path = file_path.relative_to(output_path)
                    zipf.write(file_path, rel_path)
        
        print(f"  ZIP archive created")
        
        if remove_folder_after_zip:
            shutil.rmtree(output_path)
            print(f"  Removed folder: {output_path}")
    
    print("\n✓ Processing complete!")
else:
    print("No data to save.")
# %% [markdown]
# ## Diagnostic: Check Routing Channels
# 
# Let's check what routing channels are actually present in your data files.
# %%
# Check the first file to verify ALEX configuration
if len(hdf5_files) > 0:
    test_file = hdf5_files[0]
    print(f"Analyzing: {test_file.name}\n")
    
    tttr = tttrlib.TTTR(str(test_file), 'PHOTON-HDF5')
    
    # Check routing channels
    routing_channels = tttr.routing_channel
    unique_channels = np.unique(routing_channels)
    
    print("=" * 60)
    print("ROUTING CHANNELS (ALEX Configuration)")
    print("=" * 60)
    print(f"Unique routing channels found: {unique_channels.tolist()}")
    print(f"\nPhoton counts per channel:")
    for ch in unique_channels:
        count = np.sum(routing_channels == ch)
        percentage = (count / len(routing_channels)) * 100
        print(f"  Channel {ch}: {count:,} photons ({percentage:.1f}%)")
    
    # Check microtime range
    micro_times = tttr.micro_times
    print(f"\n" + "=" * 60)
    print("MICROTIME RANGE")
    print("=" * 60)
    print(f"Microtime min: {micro_times.min()}")
    print(f"Microtime max: {micro_times.max()}")
    print(f"Microtime range: 0 - {micro_times.max()}")
    
    # Verify configuration matches data
    print(f"\n" + "=" * 60)
    print("CONFIGURATION VERIFICATION")
    print("=" * 60)
    print(f"✓ Data type: ALEX (separate routing channels)")
    print(f"✓ Green detector: Channel {detectors['green']['chs']} (no microtime filter)")
    print(f"✓ Red detector: Channel {detectors['red']['chs']} (no microtime filter)")
    print(f"✓ Analysis window: {windows['prompt']} (full microtime range)")
    
    # Check if configuration matches data
    green_ch = detectors['green']['chs'][0]
    red_ch = detectors['red']['chs'][0]
    
    if green_ch in unique_channels and red_ch in unique_channels:
        print(f"\n✓ Configuration matches data!")
        print(f"  Green channel {green_ch}: {np.sum(routing_channels == green_ch):,} photons")
        print(f"  Red channel {red_ch}: {np.sum(routing_channels == red_ch):,} photons")
    else:
        print(f"\n⚠️  WARNING: Configuration mismatch!")
        print(f"  Expected channels: {green_ch} (green), {red_ch} (red)")
        print(f"  Found channels: {unique_channels.tolist()}")
        print(f"\n  Please update DETECTORS configuration to match your data.")
else:
    print("No files to analyze")
# %% [markdown]
# ## Intensity Traces Visualization
# 
# Display green and red intensity traces for the first file with 1ms binning.
# %%
IDX = 0 # File index

# Load the first file
first_file = hdf5_files[IDX]
print(f"Creating intensity traces for: {first_file.name}")


tttr = tttrlib.TTTR(str(first_file), 'PHOTON-HDF5')


# Get green and red channel indices
green_ch = detectors['green']['chs']
red_ch = detectors['red']['chs']


print(f"Green channels: {green_ch}")
print(f"Red channels: {red_ch}")


# Select photons by channel
green_indices = tttr.get_selection_by_channel(green_ch)
red_indices = tttr.get_selection_by_channel(red_ch)


print(f"Green photons: {len(green_indices):,}")
print(f"Red photons: {len(red_indices):,}")


# Create intensity traces with 1ms binning
bin_width_ms = 1.0  # 1 millisecond
bin_width_s = bin_width_ms / 1000.0  # Convert to seconds


# Get intensity traces for green and red channels
intensity_trace_green = tttr[green_indices].get_intensity_trace(bin_width_s)
intensity_trace_red = tttr[red_indices].get_intensity_trace(bin_width_s)


print(f"Trace length: {len(intensity_trace_green)} bins ({len(intensity_trace_green) * bin_width_ms:.1f} ms)")


# Create time axis in milliseconds
time_axis = np.arange(len(intensity_trace_green)) * bin_width_ms


# Plot the traces
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)


# Green trace
axes[0].plot(time_axis, intensity_trace_green, 'g-', linewidth=0.5, label='Green')
axes[0].set_ylabel('Intensity (counts/ms)', fontsize=10)
axes[0].set_title(f'Intensity Traces - {first_file.name} (1ms binning)', fontsize=12)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)


# Red trace (inverted)
axes[1].plot(time_axis, intensity_trace_red, 'r-', linewidth=0.5, label='Red')
axes[1].invert_yaxis()
axes[1].set_ylabel('Intensity (counts/ms)', fontsize=10)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)


# Combined trace
axes[2].plot(time_axis, intensity_trace_green, 'g-', linewidth=0.5, alpha=0.7, label='Green')
axes[2].plot(time_axis, intensity_trace_red, 'r-', linewidth=0.5, alpha=0.7, label='Red')
axes[2].set_xlabel('Time (ms)', fontsize=10)
axes[2].set_ylabel('Intensity (counts/ms)', fontsize=10)
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)


plt.tight_layout()
plt.show()


print("\nIntensity traces displayed")

# --- Save trace used in this visualization to CSV in for_plot ---
for_plot_dir = output_path / "for_plot"
for_plot_dir.mkdir(exist_ok=True, parents=True)

_intensity_df = pd.DataFrame({
    "time_ms": time_axis,
    "intensity_green": intensity_trace_green,
    "intensity_red": intensity_trace_red,
})

_intensity_csv_path = for_plot_dir / f"intensity_trace_vis_{first_file.stem}.csv"
_intensity_df.to_csv(_intensity_csv_path, index=False)

print(f"Saved intensity trace CSV for visualization to: {_intensity_csv_path}")
# %% [markdown]
# ## Burst Analysis Visualization
# 
# The output can be further analyzed and visualized using **ndxplorer** (part of ChiSurf).
# 
# Here we show a basic 2D histogram plot of Total Green vs Total Red photons with Proximity Ratio, similar to ndxplorer's visualization.
# %%
# Combine all dataframes
combined_df = pd.concat(all_dataframes, ignore_index=True)

# Remove interleaved zero rows
combined_df = combined_df.loc[~(combined_df.select_dtypes(include=['number']) == 0).all(axis=1)]


Sg = combined_df['Green Count Rate (KHz)']
Sr = combined_df['Red Count Rate (KHz)']

# Calculate Proximity Ratio: Sr / (Sg + Sr)
prox_ratio = Sr / (Sg + Sr)

# Calculate Tg-Tr(ms): Duration difference
Tg_minus_Tr = combined_df['Duration (green) (ms)'] - combined_df['Duration (red) (ms)']

# Filter out invalid values
valid_mask = (Sg > 0) & (Sr > 0) & np.isfinite(prox_ratio) & np.isfinite(Tg_minus_Tr)
prox_valid = prox_ratio[valid_mask]
Tg_minus_Tr_valid = Tg_minus_Tr[valid_mask]

print(f"Valid bursts for plotting: {len(prox_valid)}")

# Create figure with marginal distributions
fig = plt.figure(figsize=(10, 10))

# Define grid for subplots
gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05,
                        height_ratios=[1, 4, 0.2], width_ratios=[4, 1, 0.2])

# Main 2D histogram
ax_main = fig.add_subplot(gs[1, 0])

# Marginal distributions
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

# Colorbar axis
ax_cbar = fig.add_subplot(gs[1, 2])

# 2D histogram (Proximity Ratio vs Tg-Tr(ms))
bins_x = 50  # Proximity ratio bins
bins_y = 50  # Tg-Tr bins

h, xedges, yedges, im = ax_main.hist2d(
    prox_valid, Tg_minus_Tr_valid,
    bins=[bins_x, bins_y],
    range=[[0, 1], [-0.5, 0.5]],  # Fixed ranges for both axes
    cmap='viridis',
    cmin=0
)

ax_main.set_xlabel('Proximity Ratio', fontsize=12)
ax_main.set_ylabel('Tg-Tr (ms)', fontsize=12)
ax_main.set_xlim(0, 1)  # Proximity ratio ranges from 0 to 1
ax_main.set_ylim(-0.5, 0.5)  # Fixed Tg-Tr range
ax_main.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
ax_main.grid(True, alpha=0.3)

# Colorbar
cbar = plt.colorbar(im, cax=ax_cbar)
cbar.set_label('Counts', fontsize=10)

# Top marginal (Proximity Ratio distribution)
ax_top.hist(prox_valid, bins=bins_x, range=(0, 1), color='steelblue', alpha=0.7, edgecolor='black')
ax_top.set_ylabel('Counts', fontsize=10)
ax_top.tick_params(labelbottom=False)
ax_top.grid(True, alpha=0.3, axis='y')

# Right marginal (Tg-Tr distribution)
ax_right.hist(Tg_minus_Tr_valid, bins=bins_y, range=(-0.5, 0.5), orientation='horizontal', 
                color='steelblue', alpha=0.7, edgecolor='black')
ax_right.set_xlabel('Counts', fontsize=10)
ax_right.tick_params(labelleft=False)
ax_right.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax_right.grid(True, alpha=0.3, axis='x')

# Title
fig.suptitle('Burst Analysis: Tg-Tr vs Proximity Ratio', fontsize=14, y=0.98)

plt.show()

# %%

# Combine all dataframes for visualization
combined_df = pd.concat(all_dataframes, ignore_index=True)

# Remove zero rows
combined_df = combined_df.loc[~(combined_df.select_dtypes(include=['number']) == 0).all(axis=1)]

# Calculate proximity ratio if we have green and red detectors
if 'Number of Photons (green)' in combined_df.columns and 'Number of Photons (red)' in combined_df.columns:
    combined_df['Proximity Ratio'] = combined_df.apply(
        lambda row: row['Number of Photons (red)'] / 
                    (row['Number of Photons (red)'] + row['Number of Photons (green)'])
        if (row['Number of Photons (red)'] + row['Number of Photons (green)']) > 0 else 0,
        axis=1
    )

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Burst duration histogram
ax = axes[0, 0]
durations = combined_df['Duration (ms)'].dropna()
if len(durations) > 0:
    ax.hist(durations, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Burst Duration (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Burst Duration Distribution (n={len(durations)})')
    ax.grid(True, alpha=0.3)

# Plot 2: Number of photons per burst
ax = axes[0, 1]
n_photons = combined_df['Number of Photons'].dropna()
if len(n_photons) > 0:
    ax.hist(n_photons, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Photons')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Photons per Burst (n={len(n_photons)})')
    ax.grid(True, alpha=0.3)

# Plot 3: Count rate histogram
ax = axes[1, 0]
count_rates = combined_df['Count Rate (KHz)'].dropna()
if len(count_rates) > 0:
    ax.hist(count_rates, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Count Rate (KHz)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Count Rate Distribution (n={len(count_rates)})')
    ax.grid(True, alpha=0.3)

# Plot 4: Proximity ratio (if available)
ax = axes[1, 1]
if 'Proximity Ratio' in combined_df.columns:
    prox_ratio = combined_df['Proximity Ratio'].dropna()
    if len(prox_ratio) > 0:
        ax.hist(prox_ratio, bins=50, edgecolor='black', alpha=0.7, range=(0, 1))
        ax.set_xlabel('Proximity Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Proximity Ratio Distribution (n={len(prox_ratio)})')
        ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Proximity Ratio\\nNot Available', 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# %%
make_fret_panels(
    all_dataframes=all_dataframes,
    output_path=output_path,
    gaussians=proximity_gaussians,
    n_gaussians=n_gaussians,
    bins_prox=81,
    bins_dt=81,
    bins_rate=81,
    bins_da=81,
    prox_range=(-0.05, 1.05),
    dt_range_hist=(-1.05, 1.05),
    dt_axis_range=(-1.05, 1.05),
    rate_range=(-10.0, 210.0),
    rate_range_da=(-10.0, 210.0),
)
# %%
