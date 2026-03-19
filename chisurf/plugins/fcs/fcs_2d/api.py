"""
High-level API for 2D-FLCS analysis.

This module provides a functional interface to the 2D-FLCS computational core,
making it easy to perform analysis from Python scripts and Jupyter notebooks.
"""

from typing import Tuple, Dict, Any, Optional, Callable
import numpy as np

from .core import TwoDFDCreator
from .fit import TwoDMEMFitter, OneDMEMFitter, GlobalTwoDMEMFitter
from .utils import scale_correlations, calculate_correlation_ratios

def correlate_tttr(
    macro_times: np.ndarray,
    micro_times: np.ndarray,
    dT: float = 0.1,
    ddT: float = 0.05,
    tMin: float = 1.0,
    tMax: float = 12.0,
    logt_imax: int = 100,
    progress_callback: Optional[callable] = None
) -> Dict[str, np.ndarray]:
    """
    Create 2D-FDC matrices from TTTR arrival times.
    
    Args:
        macro_times: Macro-time ticks.
        micro_times: Micro-time ticks.
        dT: Correlation delay (ticks or physical).
        ddT: Integration window (ticks or physical).
        tMin: Minimum microtime (ticks or physical).
        tMax: Maximum microtime (ticks or physical).
        logt_imax: Points for log-scale matrix.
        progress_callback: Optional callback for progress updates.
        
    Returns:
        Dict containing 'mat_lin', 'mat_lin_t', 'mat_log', 'mat_log_t'.
    """
    creator = TwoDFDCreator()
    mat_lin, mat_lin_t, mat_log, mat_log_t = creator.create_2d_fdc(
        macro_times, micro_times, dT, ddT, tMin, tMax, logt_imax, progress_callback
    )
    return {
        'mat_lin': mat_lin,
        'mat_lin_t': mat_lin_t,
        'mat_log': mat_log,
        'mat_log_t': mat_log_t
    }

def fit_mem_2d(
    fdc_matrix: np.ndarray,
    fdc_time_axis: np.ndarray,
    n_components: int = 3,
    tau_range: Tuple[float, float] = (0.1, 10.0),
    tau_step: float = 0.05,
    regulator: float = 0.1,
    y0_initial: float = 1000.0,
    use_1d_init: bool = True,
    max_iterations: int = 10000,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Perform 2D-MEM fitting on a linear 2D-FDC matrix.
    
    Args:
        fdc_matrix: Linear 2D-FDC matrix (mat_lin).
        fdc_time_axis: Time axis for the matrix (mat_lin_t).
        n_components: Number of components/states to fit.
        tau_range: Range of lifetimes (min, max) in physical units.
        tau_step: Resolution of lifetime grid.
        regulator: Regularization constant.
        y0_initial: Initial baseline estimate.
        max_iterations: Maximum optimization evaluations.
        progress_callback: Optional callback for progress updates.
        
    Returns:
        Dict containing fit results, model, chi2, etc.
    """
    fitter = TwoDMEMFitter()
    return fitter.fit_2d_mem_wrapper(
        mat_2dfdc=fdc_matrix,
        mat_2dfdc_cor=fdc_matrix,  # Assuming raw data for corrected in basic API
        mat_2dfdc_it=fdc_time_axis,
        n_components=n_components,
        tau_range=tau_range,
        tau_step=tau_step,
        regulator=regulator,
        y0_initial=y0_initial,
        use_1d_init=use_1d_init,
        max_iterations=max_iterations,
        progress_callback=progress_callback
    )

def fit_1d_mem(
    decay_data: np.ndarray,
    time_axis: np.ndarray,
    tau_values: np.ndarray,
    exp_curve: np.ndarray,
    regulator: float = 0.1,
    y0_initial: float = 1.0,
    irf: Optional[np.ndarray] = None,
    irf_params: Optional[Dict[str, int]] = None,
    max_iterations: int = 10000
) -> Dict[str, Any]:
    """
    Perform 1D-MEM fitting on a fluorescence decay.
    
    Args:
        ...
        irf: Instrument Response Function.
        irf_params: Dict with 'rise_fl', 'rise_irf', 'min_irf', 'max_irf'.
    """
    fitter = OneDMEMFitter()
    n_tau = len(tau_values)
    initial_mat_a = np.ones((n_tau, 1)) * (np.mean(decay_data) / n_tau)
    mi_matrix = np.ones((n_tau, 1))
    
    return fitter.fit_1d_mem(
        initial_mat_a=initial_mat_a,
        initial_y0=y0_initial,
        fix_mat_a=0, fix_y0=0,
        regulator_const=regulator,
        mat_1dfdc_it=time_axis,
        mat_1dfdc=decay_data,
        mat_1dfdc_cor=decay_data,
        tau_values=tau_values,
        exp_curve=exp_curve,
        mi_matrix=mi_matrix,
        irf=irf,
        irf_params=irf_params,
        max_iterations=max_iterations
    )

def fit_global_2d_mem(
    fdc_matrices: List[np.ndarray],
    fdc_time_axis: np.ndarray,
    n_states: int = 2,
    tau_range: Tuple[float, float] = (0.1, 10.0),
    tau_step: float = 0.05,
    regulator: float = 0.1,
    max_iterations: int = 10000
) -> Dict[str, Any]:
    """
    Globally fit multiple 2D-FDC matrices sharing a common state distribution.
    
    Args:
        fdc_matrices: List of linear 2D-FDC matrices.
        fdc_time_axis: Common time axis.
        n_states: Number of states/components.
        tau_range: Lifetime range (ns).
        tau_step: Lifetime resolution.
        regulator: Regularization constant.
        max_iterations: Max evaluations.
    """
    fitter = GlobalTwoDMEMFitter()
    
    # Prepare tau values
    tau_values = np.arange(tau_range[0], tau_range[1] + tau_step, tau_step)
    n_tau = len(tau_values)
    
    # Create exponential curve matrix
    exp_curve = np.zeros((len(fdc_time_axis), n_tau))
    for i, t in enumerate(fdc_time_axis):
        exp_curve[i, :] = np.exp(-t / tau_values)
        
    # Initial guesses
    initial_mat_a = np.ones((n_tau, n_states)) * (1.0 / n_tau)
    mi_matrix = np.ones((n_tau, n_states))
    initial_mat_g_list = [np.eye(n_states) * 1000.0 for _ in fdc_matrices]
    initial_y0_list = [100.0 for _ in fdc_matrices]
    
    return fitter.fit_global_2d_mem(
        initial_mat_a=initial_mat_a,
        initial_mat_g_list=initial_mat_g_list,
        initial_y0_list=initial_y0_list,
        fix_mat_a=0, fix_mat_g=0, fix_y0=0,
        regulator_const=regulator,
        mat_2dfdc_it=fdc_time_axis,
        mat_2dfdc_list=fdc_matrices,
        mat_2dfdc_cor_list=fdc_matrices,
        tau_values=tau_values,
        exp_curve=exp_curve,
        mi_matrix=mi_matrix,
        max_iterations=max_iterations
    )

def fit_rate_matrix(
    dT_values: np.ndarray,
    correlation_data: np.ndarray,
    initial_rates: np.ndarray,
    initial_brightness: Optional[np.ndarray] = None,
    max_iterations: int = 10000
) -> Dict[str, Any]:
    """
    Fit population kinetics (rate matrix) to correlation decay data.
    
    Args:
        dT_values: Array of correlation delays (dT).
        correlation_data: 3D array [n_states x n_states x n_dT].
        initial_rates: Initial rate matrix [n_states x n_states].
        initial_brightness: Relative brightness of each state.
    """
    from .fit import RateMatrixFitter
    fitter = RateMatrixFitter()
    
    n_states = initial_rates.shape[0]
    if initial_brightness is None:
        initial_brightness = np.ones(n_states)
        
    initial_y0 = np.zeros((n_states, n_states))
    
    # All free except diagonal (which is determined by off-diagonals in M)
    fix_rates = np.zeros((n_states, n_states))
    np.fill_diagonal(fix_rates, 1) 
    
    return fitter.fit_rate_matrix(
        xdata=dT_values,
        ydata=correlation_data,
        state_assign=np.arange(n_states),
        initial_rate_matrix=initial_rates,
        initial_brightness=initial_brightness,
        initial_y0=initial_y0,
        fix_rates=fix_rates,
        fix_brightness=np.zeros(n_states),
        fix_y0=np.zeros((n_states, n_states)),
        max_iterations=max_iterations
    )

def get_scaled_correlations(
    mat_g: np.ndarray,
    mat_a: np.ndarray,
    tau_values: np.ndarray
) -> np.ndarray:
    """Scale raw correlations by state brightness (Amplitude * Tau)."""
    return scale_correlations(mat_g, mat_a, tau_values)

def get_correlation_ratios(
    mat_m: np.ndarray,
    tau_values: np.ndarray,
    state_windows: list,
    func_type: int = 2
) -> np.ndarray:
    """Integrate 2D distribution over state windows and calculate ratios."""
    return calculate_correlation_ratios(mat_m, tau_values, state_windows, func_type)
