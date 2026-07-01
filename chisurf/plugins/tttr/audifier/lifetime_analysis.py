"""
Lifetime Domain Visualization for TTTR Audifier

This module implements inverse Laplace transform (ILT) based lifetime analysis
for TTTR data, providing lifetime waterfall plots as a diagnostic tool.

The implementation uses regularized non-negative least squares for ILT,
with log-spaced lifetime grids suitable for fluorescence lifetime analysis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def lifetime_spectrum_ilt(
    t: np.ndarray,
    y: np.ndarray,
    *,
    tau_min: float,
    tau_max: float,
    n_tau: int = 200,
    lam: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a microtime decay into a lifetime spectrum using regularized
    inverse Laplace transform (ILT).

    Parameters
    ----------
    t : np.ndarray
        Time axis in seconds (microtime)
    y : np.ndarray
        Decay histogram (counts)
    tau_min, tau_max : float
        Lifetime search range (seconds)
    n_tau : int, default 200
        Number of lifetime grid points
    lam : float, default 1e-2
        Regularization parameter (higher = smoother)

    Returns
    -------
    tau : np.ndarray
        Lifetime grid (log-spaced, seconds)
    a : np.ndarray
        Lifetime spectrum amplitudes (non-negative)
    """
    # Input validation
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if t.shape != y.shape:
        raise ValueError("t and y must have the same shape")
    if np.any(y < 0):
        raise ValueError("y must be non-negative (counts)")
    if tau_min <= 0 or tau_max <= 0 or tau_min >= tau_max:
        raise ValueError("tau_min < tau_max and both must be positive")
    if n_tau < 2:
        raise ValueError("n_tau must be >= 2")
    if lam < 0:
        raise ValueError("lam must be non-negative")

    # Remove zero-time points and ensure proper ordering
    mask = t > 0
    t = t[mask]
    y = y[mask]
    if t.size == 0:
        raise ValueError("No valid time points (t > 0)")

    # Sort by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    y = y[sort_idx]

    # Create log-spaced lifetime grid
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)

    # Build the kernel matrix K[i, j] = exp(-t[i] / tau[j])
    # This is the forward Laplace transform kernel
    K = np.exp(-t[:, np.newaxis] / tau[np.newaxis, :])

    # Regularized non-negative least squares
    # Solve: min ||K @ a - y||^2 + lam * ||L @ a||^2
    # subject to a >= 0

    # Simple regularization: penalize roughness (second derivative)
    # For log-spaced tau, this approximates smoothness in log(tau)
    n = n_tau
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i] = 1
        L[i, i + 1] = -2
        L[i, i + 2] = 1

    # Build augmented system [K; sqrt(lam)*L] @ a = [y; 0]
    K_aug = np.vstack([K, np.sqrt(lam) * L])
    y_aug = np.concatenate([y, np.zeros(n - 2)])

    # Non-negative least squares using scipy if available, otherwise simple iterative method
    try:
        from scipy.optimize import nnls

        a, _ = nnls(K_aug, y_aug)
    except ImportError:
        # Simple projected gradient descent for NNLS
        a = _nnls_simple(K_aug, y_aug, max_iter=1000, tol=1e-6)

    return tau, a


def _nnls_simple(
    A: np.ndarray, b: np.ndarray, max_iter: int = 1000, tol: float = 1e-6
) -> np.ndarray:
    """
    Simple non-negative least squares solver using projected gradient descent.
    Fallback when scipy.optimize.nnls is not available.
    """
    m, n = A.shape
    x = np.zeros(n, dtype=np.float64)

    # Compute gradient step size (1/Lipschitz constant)
    # L = largest eigenvalue of A.T @ A
    AtA = A.T @ A
    L = np.max(np.linalg.eigvals(AtA).real) + 1e-12
    step_size = 1.0 / L

    for iteration in range(max_iter):
        # Gradient: A.T @ (A @ x - b)
        grad = A.T @ (A @ x - b)

        # Gradient descent step
        x_new = x - step_size * grad

        # Project onto non-negative orthant
        x_new = np.maximum(x_new, 0.0)

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x


def compute_lifetime_waterfall(
    data: TTTRData,
    channel: int,
    *,
    macro_bin_width_s: float,
    micro_gate: tuple[int, int] | None,
    tau_min: float,
    tau_max: float,
    n_tau: int = 200,
    lam: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a lifetime waterfall plot by applying ILT to microtime histograms
    across macro time bins.

    Parameters
    ----------
    data : TTTRData
        TTTR photon data
    channel : int
        Routing channel to analyze
    macro_bin_width_s : float
        Width of macro time bins in seconds
    micro_gate : Optional[Tuple[int, int]]
        Microtime gate as (min_bin, max_bin), or None for no gate
    tau_min, tau_max : float
        Lifetime range in seconds
    n_tau : int, default 200
        Number of lifetime grid points
    lam : float, default 1e-2
        ILT regularization parameter

    Returns
    -------
    A : np.ndarray
        2D array (macro_time_bin × lifetime) of lifetime amplitudes
    macro_t_s : np.ndarray
        Macro time axis (seconds)
    tau : np.ndarray
        Lifetime axis (seconds)
    """
    # Import TTTRData from core module to avoid circular imports
    try:
        from .core import TTTRData
    except ImportError:
        # Fallback for direct execution
        from core import TTTRData

    # Validate inputs
    if not isinstance(data, TTTRData):
        raise TypeError("data must be a TTTRData instance")

    # Filter photons by channel and microtime gate
    routing = data.routing
    macro = data.macro_ticks
    micro = data.micro_bins

    mask = routing == channel
    if micro_gate is not None:
        g0, g1 = micro_gate
        mask &= (micro >= g0) & (micro < g1)

    if not np.any(mask):
        raise ValueError(f"No photons found for channel {channel} with specified gate")

    macro_filtered = macro[mask]
    micro_filtered = micro[mask]

    # Create macro time bins
    t0 = int(macro_filtered.min())
    t1 = int(macro_filtered.max()) + 1
    ticks_per_bin = max(1, int(round(macro_bin_width_s / data.macro_time_unit_s)))
    n_macro = int(math.ceil((t1 - t0) / ticks_per_bin))
    macro_edges = t0 + np.arange(n_macro + 1, dtype=np.int64) * ticks_per_bin

    # Convert macro edges to seconds
    macro_t_s = (macro_edges - macro_edges[0]) * data.macro_time_unit_s
    macro_centers = 0.5 * (macro_edges[:-1] + macro_edges[1:])

    # Initialize output array
    A = np.zeros((n_macro, n_tau), dtype=np.float64)

    # Process each macro bin
    for i in range(n_macro):
        # Find photons in this macro bin
        bin_mask = ((macro_filtered - t0) // ticks_per_bin) == i
        if not np.any(bin_mask):
            continue

        micro_in_bin = micro_filtered[bin_mask]

        # Build microtime histogram
        if data.micro_time_unit_s is not None:
            # Use actual microtime timing if available
            micro_max = int(micro_in_bin.max()) + 1
            hist, bin_edges = np.histogram(micro_in_bin, bins=range(micro_max + 1))
            micro_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            t_axis = micro_centers * data.micro_time_unit_s
        else:
            # Use bin indices as time units
            micro_max = int(micro_in_bin.max()) + 1
            hist, bin_edges = np.histogram(micro_in_bin, bins=range(micro_max + 1))
            micro_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            t_axis = micro_centers  # in bin units

        # Apply ILT to get lifetime spectrum
        try:
            tau, a = lifetime_spectrum_ilt(
                t_axis,
                hist.astype(np.float64),
                tau_min=tau_min,
                tau_max=tau_max,
                n_tau=n_tau,
                lam=lam,
            )
            A[i, :] = a
        except Exception as e:
            # If ILT fails, leave as zeros
            print(f"Warning: ILT failed for macro bin {i}: {e}")
            continue

    return A, macro_centers, tau


def plot_lifetime_waterfall(
    A: np.ndarray,
    macro_t_s: np.ndarray,
    tau: np.ndarray,
    *,
    log_tau: bool = True,
    log_amplitude: bool = True,
    title: str = "Lifetime Waterfall",
) -> None:
    """
    Plot a lifetime waterfall with explicit axis semantics.

    Parameters
    ----------
    A : np.ndarray
        2D array (macro_time_bin × lifetime) of lifetime amplitudes
    macro_t_s : np.ndarray
        Macro time axis (seconds)
    tau : np.ndarray
        Lifetime axis (seconds)
    log_tau : bool, default True
        Use logarithmic scale for lifetime axis
    log_amplitude : bool, default True
        Use logarithmic scaling for amplitudes
    title : str, default "Lifetime Waterfall"
        Plot title
    """
    # Validate inputs
    A = np.asarray(A, dtype=np.float64)
    macro_t_s = np.asarray(macro_t_s, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be 2D")
    if A.shape[0] != macro_t_s.shape[0]:
        raise ValueError("A and macro_t_s must have matching first dimension")
    if A.shape[1] != tau.shape[0]:
        raise ValueError("A and tau must have matching second dimension")

    # Prepare data for plotting
    plot_data = A.copy()

    # Apply amplitude scaling
    if log_amplitude:
        # Use log1p for better handling of zeros
        plot_data = np.log1p(plot_data)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Set extent for imshow: [left, right, bottom, top]
    # x-axis: lifetime (tau), y-axis: macro time
    extent = [tau[0], tau[-1], macro_t_s[-1], macro_t_s[0]]

    # Use a perceptually uniform colormap
    im = plt.imshow(plot_data, aspect="auto", extent=extent, cmap="viridis", origin="upper")

    # Set axis labels with units
    plt.xlabel("Lifetime τ (s)")
    plt.ylabel("Macro time (s)")
    plt.title(title)

    # Use log scale for lifetime axis if requested
    if log_tau:
        plt.xscale("log")

    # Add colorbar with label
    cbar = plt.colorbar(im)
    if log_amplitude:
        cbar.set_label("Log amplitude (log₁₁(counts))")
    else:
        cbar.set_label("Amplitude (counts)")

    plt.tight_layout()
    plt.show()


def plot_lifetime_waterfall_multichannel(
    data: TTTRData,
    channels: list[int],
    *,
    macro_bin_width_s: float,
    micro_gates: dict[int, tuple[int, int]],
    tau_min: float,
    tau_max: float,
    n_tau: int = 200,
    lam: float = 1e-2,
) -> None:
    """
    Plot lifetime waterfalls for multiple routing channels as separate panels.

    Parameters
    ----------
    data : TTTRData
        TTTR photon data
    channels : List[int]
        List of routing channels to plot
    macro_bin_width_s : float
        Width of macro time bins in seconds
    micro_gates : Dict[int, Tuple[int, int]]
        Microtime gates for each channel
    tau_min, tau_max : float
        Lifetime range in seconds
    n_tau : int, default 200
        Number of lifetime grid points
    lam : float, default 1e-2
        ILT regularization parameter
    """
    # Import TTTRData to avoid circular imports
    try:
        from .core import TTTRData
    except ImportError:
        # Fallback for direct execution
        from core import TTTRData

    # Validate inputs
    if not isinstance(data, TTTRData):
        raise TypeError("data must be a TTTRData instance")

    channels = list(channels)
    if len(channels) == 0:
        raise ValueError("No channels specified")

    # Check that all channels have gates
    for ch in channels:
        if ch not in micro_gates:
            raise ValueError(f"Microtime gate not specified for channel {ch}")

    # Create subplots (stacked vertically)
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 3 * n_channels), sharex=True, sharey=True)

    if n_channels == 1:
        axes = [axes]  # Ensure axes is always a list

    # Process each channel
    all_tau = None
    all_macro_t_s = None
    vmax = 0.0

    # First pass: compute all data to determine shared color scale
    channel_data = []
    for i, ch in enumerate(channels):
        try:
            A, macro_t_s, tau = compute_lifetime_waterfall(
                data=data,
                channel=ch,
                macro_bin_width_s=macro_bin_width_s,
                micro_gate=micro_gates[ch],
                tau_min=tau_min,
                tau_max=tau_max,
                n_tau=n_tau,
                lam=lam,
            )
            channel_data.append((A, macro_t_s, tau))

            # Store reference axes for sharing
            if all_tau is None:
                all_tau = tau
                all_macro_t_s = macro_t_s

            # Track maximum amplitude for color scaling
            vmax = max(vmax, np.max(np.log1p(A)))

        except Exception as e:
            print(f"Warning: Failed to process channel {ch}: {e}")
            channel_data.append(None)

    # Second pass: plot with shared color scale
    if all_tau is None:
        raise ValueError("No channels could be processed successfully")

    # Create shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

    for i, ch in enumerate(channels):
        ax = axes[i]
        data_tuple = channel_data[i]

        if data_tuple is None:
            ax.text(
                0.5,
                0.5,
                f"Channel {ch}\n(processing failed)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_ylabel("Macro time (s)")
            continue

        A, macro_t_s, tau = data_tuple
        plot_data = np.log1p(A)

        # Plot with shared extent
        extent = [tau[0], tau[-1], macro_t_s[-1], macro_t_s[0]]
        im = ax.imshow(
            plot_data,
            aspect="auto",
            extent=extent,
            cmap="viridis",
            origin="upper",
            vmin=0,
            vmax=vmax,
        )

        ax.set_ylabel(f"Channel {ch}\nMacro time (s)")
        ax.set_xscale("log")

        # Add channel label
        ax.text(
            0.02,
            0.98,
            f"Ch {ch}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Set common x-axis label
    axes[-1].set_xlabel("Lifetime τ (s)")

    # Add colorbar
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Log amplitude (log₁₁(counts))")

    plt.suptitle("Lifetime Waterfall - Multi-Channel", fontsize=14)
    plt.tight_layout()
    plt.show()


# Import math for ceil function
import math
