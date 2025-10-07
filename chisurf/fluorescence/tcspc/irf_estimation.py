"""
Instrument Response Function Estimation

Estimate instrument response functions (IRFs) from time-series fluorescence decay data
using truncated exponential fitting and Richardson-Lucy deconvolution.

This module provides functionality to automatically extract IRFs from TCSPC measurements
without requiring separate IRF measurements. The approach uses:
1. Savitzky-Golay filtering to identify decay regions
2. Truncated exponential fitting to model the decay
3. Richardson-Lucy deconvolution to extract the IRF

This is a blind IRF estimation method that infers the instrument response directly
from the measured fluorescence decay data.

References
----------
- Adrián Gómez-Sánchez et al., "Blind instrument response function identification 
  from fluorescence decays", Biophysical Reports, 2024.
  https://doi.org/10.1016/j.bpr.2024.100155
- Richardson, W. H. (1972). "Bayesian-Based Iterative Method of Image Restoration"
- Lucy, L. B. (1974). "An iterative technique for the rectification of observed distributions"
- Savitzky, A.; Golay, M. J. E. (1964). "Smoothing and Differentiation of Data by 
  Simplified Least Squares Procedures"
"""

from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, Union
import warnings


def pad_array(
    x: np.ndarray,
    pad_left: int,
    pad_right: int,
    axis: int,
    mode: str = "reflect"
) -> np.ndarray:
    """
    Pad a numpy array along one axis.

    Parameters
    ----------
    x : np.ndarray
        Input array to pad
    pad_left : int
        Number of elements to pad before the data
    pad_right : int
        Number of elements to pad after the data
    axis : int
        Axis along which to pad
    mode : str, optional
        Padding mode: 'reflect', 'edge', or 'constant' (default: 'reflect')

    Returns
    -------
    np.ndarray
        Padded array
    """
    if pad_left == 0 and pad_right == 0:
        return x

    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (pad_left, pad_right)

    if mode == "reflect":
        return np.pad(x, pad_width, mode='reflect')
    elif mode == "edge":
        return np.pad(x, pad_width, mode='edge')
    elif mode == "constant":
        return np.pad(x, pad_width, mode='constant', constant_values=0)
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")


def median_filter_nd(
    x: np.ndarray,
    window_size: Union[int, list, tuple] = 3,
    axes: Optional[Union[list, tuple]] = None,
    mode: str = "reflect"
) -> np.ndarray:
    """
    Apply N-dimensional median filter over specified axes.

    Parameters
    ----------
    x : np.ndarray
        Input array
    window_size : int or list/tuple of ints, optional
        Window size(s) for the filter. If int, same size for all axes.
        If list/tuple, must match len(axes) (default: 3)
    axes : list/tuple of ints, optional
        Axes to filter along. If None, all axes are filtered (default: None)
    mode : str, optional
        Padding mode: 'reflect', 'constant', 'nearest', 'mirror', 'wrap' (default: 'reflect')

    Returns
    -------
    np.ndarray
        Median-filtered array of the same shape as x

    Raises
    ------
    ValueError
        If window_size is not odd or does not match axes length
    """
    if axes is None:
        axes = list(range(x.ndim))

    if isinstance(window_size, int):
        window_size = [window_size] * len(axes)
    elif len(window_size) != len(axes):
        raise ValueError("window_size must be scalar or match len(axes)")

    # Check for odd values
    for w in window_size:
        if w % 2 == 0:
            raise ValueError(f"All window sizes must be odd, got {w}")

    # Build size tuple for scipy median_filter
    size = [1] * x.ndim
    for ax, w in zip(axes, window_size):
        size[ax] = w

    return scipy_median_filter(x, size=size, mode=mode)


def generate_truncated_exponential(
    t: np.ndarray,
    params: Dict[str, float]
) -> np.ndarray:
    """
    Generate a truncated exponential curve from fit parameters.

    Model:
        y = A * exp(-(t - t0) * k) + C, for t >= t0
        y = C, for t < t0

    Parameters
    ----------
    t : np.ndarray
        1D array of time points
    params : dict
        Dictionary with keys {"A", "k", "C", "t0"}
        - A: amplitude
        - k: decay rate (1/lifetime)
        - C: constant offset
        - t0: start time of decay

    Returns
    -------
    np.ndarray
        Model values for each time point in t
    """
    A = params["A"]
    k = params["k"]
    C = params["C"]
    t0 = params["t0"]

    y = np.where(
        t >= t0,
        A * np.exp(-(t - t0) * k) + C,
        C
    )

    return y


def estimate_lifetime(
    x: np.ndarray,
    y: np.ndarray,
    t0: int,
    t1: int
) -> float:
    """
    Estimate the decay lifetime (tau) from the centroid of the baseline-subtracted
    signal between t0 and t1.

    Parameters
    ----------
    x : np.ndarray
        1D array of time points
    y : np.ndarray
        1D array of signal values
    t0 : int
        Start index for decay region
    t1 : int
        End index for decay region

    Returns
    -------
    float
        Estimated lifetime tau
    """
    x_region = x[t0:t1+1].astype(np.float32)
    x_region = x_region - x_region.min()  # shift to start at 0

    # Baseline subtraction and clamping
    y_region = y[t0:t1+1].astype(np.float32)
    y_clamped = np.maximum(y_region - y_region.min(), 0.0)

    # Centroid-based lifetime estimation
    tau = np.sum(x_region * y_clamped) / np.sum(y_clamped)

    return tau


def partial_convolution_fft(
    signal: np.ndarray,
    kernel: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """
    Perform 1D convolution along a specified axis using FFT.
    
    This implements circular convolution via FFT, which is appropriate for
    Richardson-Lucy deconvolution when the kernel is properly normalized.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array
    kernel : np.ndarray
        Convolution kernel (1D)
    axis : int, optional
        Axis along which to convolve (default: 0)

    Returns
    -------
    np.ndarray
        Convolved result
    """
    # Move axis to position 0 for easier processing
    signal = np.moveaxis(signal, axis, 0)
    original_shape = signal.shape

    # Reshape to (n_time, n_channels)
    n_time = signal.shape[0]
    signal_2d = signal.reshape(n_time, -1)
    n_channels = signal_2d.shape[1]

    # Perform FFT convolution for each channel
    result = np.zeros_like(signal_2d)
    kernel_fft = np.fft.fft(kernel, n=n_time)

    for c in range(n_channels):
        signal_fft = np.fft.fft(signal_2d[:, c])
        conv_fft = signal_fft * kernel_fft
        conv = np.fft.ifft(conv_fft)
        # Take real part (circular convolution via FFT)
        result[:, c] = np.real(conv)

    # Reshape back to original shape
    result = result.reshape(original_shape)
    result = np.moveaxis(result, 0, axis)

    return result


class IRFEstimator:
    """
    Estimate instrument response functions (IRFs) from time-series fluorescence decay data.

    This class implements a blind IRF estimation algorithm which uses truncated exponential 
    fitting and Richardson-Lucy deconvolution to extract IRFs from fluorescence decay 
    measurements without requiring separate IRF measurements.
    
    The algorithm is based on the method described in:
    Gómez-Sánchez et al., "Blind instrument response function identification from 
    fluorescence decays", Biophysical Reports, 2024.
    https://doi.org/10.1016/j.bpr.2024.100155

    Attributes
    ----------
    data : np.ndarray
        Input time-series data (shape: [num_samples, num_channels])
    dt : float
        Time step between samples
    time : np.ndarray
        Time vector
    num_samples : int
        Number of time points
    num_channels : int
        Number of channels
    t0 : np.ndarray or None
        Per-channel start index of decay
    t1 : np.ndarray or None
        Per-channel end index of decay
    params : dict or None
        Fitted parameters ("A", "C", "k")
    data_fit : np.ndarray or None
        Fitted exponential curves
    kernel : np.ndarray or None
        Kernel for deconvolution
    irf : np.ndarray or None
        Deconvolved IRFs

    Examples
    --------
    >>> import numpy as np
    >>> # Create synthetic decay data
    >>> time = np.linspace(0, 50, 500)
    >>> data = np.exp(-time/4.0) + 0.1
    >>> data = data.reshape(-1, 1)  # Single channel
    >>> 
    >>> # Estimate IRF
    >>> estimator = IRFEstimator(data, dt=time[1]-time[0])
    >>> irf = estimator.run()
    """

    def __init__(self, data: np.ndarray, dt: float = 1.0):
        """
        Initialize IRFEstimator with time-series data and time step.

        Parameters
        ----------
        data : np.ndarray
            1D or 2D array of time-series data. If 1D, promoted to shape (num_samples, 1)
        dt : float, optional
            Sampling interval (default: 1.0)

        Raises
        ------
        ValueError
            If data has more than 2 dimensions
        """
        data = np.asarray(data, dtype=np.float32)

        if data.ndim == 1:
            self.data = data.reshape(-1, 1)
        elif data.ndim == 2:
            self.data = data
        else:
            raise ValueError("data must be 1D or 2D array")

        self.dt = dt
        self.time = np.arange(self.data.shape[0]) * dt

        self.num_samples, self.num_channels = self.data.shape
        self.t0 = None   # shape (num_channels,)
        self.t1 = None   # shape (num_channels,)
        self.params = None  # dict with A, k, C
        self.data_fit = None  # shape (num_samples, num_channels)
        self.kernel = None  # shape (num_samples,)
        self.irf = None  # shape (num_samples, num_channels)

    def find_t0_t1(
        self,
        window_length: int = 11,
        polyorder: int = 3,
        persistence: int = 5,
        threshold: float = 0.05
    ) -> None:
        """
        Estimate per-channel start (t0) and end (t1) indices of decay using
        Savitzky-Golay derivative filtering.

        Parameters
        ----------
        window_length : int, optional
            Length of the Savitzky-Golay filter window (must be odd) (default: 11)
        polyorder : int, optional
            Polynomial order for Savitzky-Golay filter (default: 3)
        persistence : int, optional
            Number of consecutive positive derivative samples for t1 detection (default: 5)
        threshold : float, optional
            Minimum amplitude threshold for t1 detection (fraction of channel range) (default: 0.05)

        Raises
        ------
        ValueError
            If window_length is not odd
        """
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd.")

        t0s, t1s = [], []
        y_range = self.data.max(axis=0) - self.data.min(axis=0)

        for c in range(self.num_channels):
            y = self.data[:, c]
            dy = savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=1, delta=self.dt)

            # t0: global minimum of derivative
            t0 = int(np.argmin(dy))

            # t1: first point after t0 with persistent positive derivative
            t1 = len(dy) - 1  # fallback to end
            for i in range(t0 + 1, len(dy) - persistence):
                avg_diff = dy[i:i + persistence].mean()
                amplitude = max(0, self.data[i + persistence, c] - self.data[:, c].min())
                if avg_diff > 0 and amplitude > threshold * y_range[c]:
                    t1 = i
                    break

            t0s.append(t0)
            t1s.append(t1)

        self.t0 = np.array(t0s, dtype=int)
        self.t1 = np.array(t1s, dtype=int)

    def fit_exponential(
        self,
        offset: int = 0,
        method: str = 'L-BFGS-B',
        max_iter: int = 1000
    ) -> None:
        """
        Fit per-channel truncated exponential curves to the data between t0 and t1
        using scipy optimization.

        Parameters
        ----------
        offset : int, optional
            Shift applied to t0 when selecting data for fitting (default: 0)
        method : str, optional
            Optimization method for scipy.optimize.minimize (default: 'L-BFGS-B')
        max_iter : int, optional
            Maximum number of optimization iterations (default: 1000)

        Raises
        ------
        RuntimeError
            If t0 or t1 have not been computed
        """
        if self.t0 is None or self.t1 is None:
            raise RuntimeError("Run find_t0_t1 first.")

        # Initialize parameters
        # For C_init, estimate background from the late decay region (near t1 but before it)
        C_init = np.zeros(self.num_channels)
        for c in range(self.num_channels):
            # Use data in the late decay region (80-100% of t0-t1 range)
            decay_length = self.t1[c] - self.t0[c]
            if decay_length > 20:
                # Take last 20% of decay region as background estimate
                bg_start = self.t0[c] + int(0.8 * decay_length)
                bg_end = self.t1[c]
                bg_data = self.data[bg_start:bg_end, c]
                # Use median of this region as background
                C_init[c] = np.median(bg_data) if len(bg_data) > 0 else 0.0
            else:
                # For short decay regions, use minimum of decay region
                decay_data = self.data[self.t0[c]:self.t1[c], c]
                C_init[c] = np.min(decay_data) if len(decay_data) > 0 else 0.0
        
        # Ensure C_init is non-negative
        C_init = np.maximum(C_init, 0.0)
        
        A_init = (self.data.max(axis=0) - C_init)

        # Estimate initial lifetime from central channel
        cc = self.num_channels // 2
        tau = estimate_lifetime(self.time, self.data[:, cc], self.t0[cc], self.t1[cc])
        k_init = 1.0 / tau

        # Optimize all parameters jointly
        def loss_function(params):
            A = params[:self.num_channels]
            C = params[self.num_channels:2*self.num_channels]
            k = params[-1]

            total_loss = 0.0
            for c in range(self.num_channels):
                y_true = self.data[self.t0[c]+offset:self.t1[c]+1, c]
                x = np.arange(len(y_true)) * self.dt
                y_pred = A[c] * np.exp(-k * x) + C[c]
                total_loss += np.mean((y_true - y_pred) ** 2)

            return total_loss / self.num_channels

        # Initial parameter vector
        x0 = np.concatenate([A_init, C_init, [k_init]])

        # Bounds: A > 0, C >= 0, k > 0
        bounds = [(1e-6, None)] * self.num_channels + \
                 [(0, None)] * self.num_channels + \
                 [(1e-6, None)]

        # Optimize
        result = minimize(
            loss_function,
            x0,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iter}
        )

        # Extract optimized parameters
        A_opt = result.x[:self.num_channels]
        C_opt = result.x[self.num_channels:2*self.num_channels]
        k_opt = result.x[-1]

        self.params = {
            "A": A_opt,
            "C": C_opt,
            "k": k_opt
        }

    def generate_data_fit(self) -> None:
        """
        Generate fitted truncated exponential curves for all channels using
        parameters in self.params.

        Raises
        ------
        RuntimeError
            If self.params is None
        """
        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")

        exp_curves = np.zeros_like(self.data)

        for c in range(self.num_channels):
            params = {
                "A": self.params["A"][c],
                "C": self.params["C"][c],
                "k": self.params["k"],
                "t0": int(self.t0[c]) * self.dt,
            }
            exp_curves[:, c] = generate_truncated_exponential(self.time, params)

        self.data_fit = exp_curves

    def generate_kernel(self) -> None:
        """
        Build a normalized, positive kernel from a truncated exponential with
        unit amplitude and zero offset.

        Raises
        ------
        RuntimeError
            If self.params is None
        """
        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")

        params = {
            "A": 1.0,
            "C": 0.0,
            "k": self.params["k"],
            "t0": 0.0,
        }

        exp_curve = generate_truncated_exponential(self.time, params)
        exp_curve = np.maximum(exp_curve, 0)  # enforce positivity
        exp_curve /= exp_curve.sum()  # normalize kernel
        self.kernel = exp_curve

    def richardson_lucy_deconvolution(
        self,
        iterations: int = 30,
        eps: float = 1e-4,
        regularization: int = 3
    ) -> None:
        """
        Perform Richardson-Lucy deconvolution channel-wise using FFT-based
        convolutions and a precomputed kernel.

        Parameters
        ----------
        iterations : int, optional
            Number of RL iterations (default: 30)
        eps : float, optional
            Small value to avoid division by zero (default: 1e-4)
        regularization : int, optional
            Median filter window size for regularization (default: 3)

        Raises
        ------
        RuntimeError
            If self.kernel is None
        """
        if self.kernel is None:
            raise RuntimeError("Run generate_kernel() first or provide a convolution kernel manually.")

        # Initialize output
        x_est = np.ones_like(self.data)

        # Prepare kernels
        kernel = self.kernel.copy()
        kernel_t = self.kernel[::-1].copy()  # time-reversed kernel

        # Subtract offset
        y = self.data.copy() - self.params['C'].reshape(1, -1)
        y = np.maximum(y, 0)

        # RL deconvolution
        for _ in range(iterations):
            conv = partial_convolution_fft(x_est, kernel, axis=0)
            conv = np.maximum(conv, eps)  # avoid div by 0
            relative_blur = y / conv
            correction = partial_convolution_fft(relative_blur, kernel_t, axis=0)
            x_est = x_est * correction
            x_est = np.maximum(x_est, 0)  # enforce positivity

            if regularization > 1:
                x_est = median_filter_nd(x_est, window_size=regularization, axes=[0], mode='reflect')
        
        # Remove any DC offset from IRF (IRF should integrate to a finite value, not have constant background)
        # Estimate DC component from the tail of the IRF
        tail_length = min(50, len(x_est) // 10)
        if tail_length > 0:
            for c in range(self.num_channels):
                dc_offset = np.median(x_est[-tail_length:, c])
                x_est[:, c] = np.maximum(x_est[:, c] - dc_offset, 0)

        self.irf = x_est

    def run(
        self,
        window_length: int = 11,
        polyorder: int = 3,
        persistence: int = 5,
        threshold: float = 0.05,
        fit_method: str = 'L-BFGS-B',
        fit_max_iter: int = 1000,
        rl_iterations: int = 500,
        regularization: int = 3
    ) -> np.ndarray:
        """
        Execute the full IRF estimation pipeline:
            1. Find t0, t1 per channel using Savitzky-Golay filtering
            2. Fit truncated exponential curves
            3. Generate fitted exponential curves
            4. Generate deconvolution kernel
            5. Perform Richardson-Lucy deconvolution

        Parameters
        ----------
        window_length : int, optional
            Length of Savitzky-Golay filter window (must be odd) (default: 11)
        polyorder : int, optional
            Polynomial order for SG filter (default: 3)
        persistence : int, optional
            Number of consecutive positive derivative samples for t1 (default: 5)
        threshold : float, optional
            Minimum amplitude threshold for t1 (default: 0.05)
        fit_method : str, optional
            Optimization method for exponential fit (default: 'L-BFGS-B')
        fit_max_iter : int, optional
            Maximum iterations for exponential fit (default: 1000)
        rl_iterations : int, optional
            Number of RL deconvolution iterations (default: 500)
        regularization : int, optional
            Median filter size for RL deconvolution (default: 3)

        Returns
        -------
        np.ndarray
            Estimated IRFs (shape: [num_samples, num_channels])
        """
        self.find_t0_t1(
            window_length=window_length,
            polyorder=polyorder,
            persistence=persistence,
            threshold=threshold
        )
        self.fit_exponential(method=fit_method, max_iter=fit_max_iter)
        self.generate_data_fit()
        self.generate_kernel()
        self.richardson_lucy_deconvolution(
            iterations=rl_iterations,
            regularization=regularization
        )

        return self.irf

    def plot_raw_and_fit(self, ax=None):
        """
        Plot raw data points and fitted exponential curves for each channel.
        Also draws vertical dashed lines at t0 and t1 for each channel.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or array of Axes, optional
            Axes to plot on. If None, new figure and axes are created

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : np.ndarray of matplotlib.axes.Axes
            Array of axes objects

        Raises
        ------
        RuntimeError
            If self.data_fit is None
        """
        if self.data_fit is None:
            raise RuntimeError("Run generate_data_fit() first.")

        import matplotlib.pyplot as plt

        if ax is None:
            nrows = int(np.ceil(np.sqrt(self.num_channels)))
            ncols = int(np.ceil(self.num_channels / nrows))
            fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=True)
            ax = np.array(ax).reshape(-1)
        else:
            fig = ax[0].figure if isinstance(ax, np.ndarray) else ax.figure
            ax = np.array(ax).reshape(-1)

        for c in range(self.num_channels):
            ax[c].plot(self.time, self.data[:, c], 'k.', label='Raw', markersize=2)
            ax[c].plot(self.time, self.data_fit[:, c], 'r-', label='Fit')
            ax[c].axvline(self.time[int(self.t0[c])], color='grey', linestyle='--', alpha=0.5, label='Fitting interval')
            ax[c].axvline(self.time[int(self.t1[c])], color='grey', linestyle='--', alpha=0.5)
            ax[c].set_title(f"Channel {c}")
            ax[c].set_xlabel('Time (ns)')
            ax[c].set_ylabel('Intensity')

        # Hide unused subplots
        for c in range(self.num_channels, len(ax)):
            ax[c].axis('off')

        fig.legend(['Raw', 'Fit', 'Fitting interval'], loc='upper right', bbox_to_anchor=(0.98, 0.95))
        fig.tight_layout()

        return fig, ax

    def plot_forward_model(self, ax=None):
        """
        Convolve estimated IRFs with the fitted exponential kernel and plot the
        forward model against the measured data for each channel.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or array of Axes, optional
            Axes to plot on. If None, new figure and axes are created

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : np.ndarray of matplotlib.axes.Axes
            Array of axes objects

        Raises
        ------
        RuntimeError
            If self.irf is None
        """
        if self.irf is None:
            raise RuntimeError("Run richardson_lucy_deconvolution() first.")

        import matplotlib.pyplot as plt

        if ax is None:
            nrows = int(np.ceil(np.sqrt(self.num_channels)))
            ncols = int(np.ceil(self.num_channels / nrows))
            fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=True)
            ax = np.array(ax).reshape(-1)
        else:
            fig = ax[0].figure if isinstance(ax, np.ndarray) else ax.figure
            ax = np.array(ax).reshape(-1)

        forward = partial_convolution_fft(self.irf, self.kernel, axis=0)
        forward += self.params['C'].reshape(1, -1)

        for c in range(self.num_channels):
            ax[c].plot(self.time, self.data[:, c], 'k.', label='Measured', markersize=2)
            ax[c].plot(self.time, forward[:, c], 'g-', label='IRF ⊗ Exp')
            ax[c].set_title(f"Channel {c}")
            ax[c].set_xlabel('Time (ns)')
            ax[c].set_ylabel('Intensity')

        # Hide unused subplots
        for c in range(self.num_channels, len(ax)):
            ax[c].axis('off')

        fig.legend(['Measured', 'IRF ⊗ Exp'], loc='upper right', bbox_to_anchor=(0.95, 0.95))
        fig.tight_layout()

        return fig, ax
