from __future__ import annotations

import numpy as np
import chisurf.core.fluorescence.fcs.correlate

from chisurf import typing

weightCalculations = ['Koppel', 'none']
correlationMethods = ['tp']


def noise(
        times: np.ndarray,
        correlation: np.ndarray,
        measurement_duration: float,
        mean_count_rate: float,
        weight_type: str = 'suren',
        skip_points: int = 0,
        correlation_amplitude_range: typing.Tuple[int, int] = (0, 16),
        time_upper: float = 10,
        z0_w0: float = 3.5,
        starchev_a1: float = 2.e-3,
        starchev_a2: float = 1.8e-1,
        starchev_c1: float = 1.0e-4,
        diffusion_time: float = None,
        verbose: bool = False
) -> np.array:
    """
    Calculate noise weights for fluorescence correlation spectroscopy (FCS)
    correlation curves.

    This function computes the noise weights (standard deviations) associated
    with each point in an FCS correlation curve using different weighting schemes.
    These weights can be used in subsequent fitting or analysis of the correlation
    data. The diffusion time, if not provided, is estimated as the time when the
    correlation curve falls to half of its (background-corrected) amplitude. To
    avoid early-time artefacts such as afterpulsing, the first few points can be
    skipped using the parameter `skip_points`.

    Weighting methods:

    - **suren**: Applies a noise model that accounts for shot noise and fluctuations
      in the correlation function. (Publication details are unclear for this method.)
    - **starchev**: Implements the noise model described by Starchev in:

          Starchev, O. Y. (2001). Noise on fluorescence correlation spectroscopy.
          *Journal of Colloid and Interface Science, 233*, 50–55.

      The parameters `starchev_a1`, `starchev_a2`, and `starchev_c1` are empirical
      fitting parameters that depend on the correlator settings.
    - **uniform**: Assigns a uniform weight (i.e. all ones).

    Parameters
    ----------
    times : np.ndarray
        Array of correlation times in milliseconds.
    correlation : np.ndarray
        Array of correlation amplitudes.
    measurement_duration : float
        Total measurement duration in seconds.
    mean_count_rate : float
        Mean count rate in kHz.
    weight_type : str, optional
        Type of weighting to use. Options include 'suren', 'starchev', and 'uniform'.
        Default is 'starchev'.
    skip_points : int, optional
        Number of initial data points to skip (e.g., to avoid afterpulsing effects).
        Default is 0.
    correlation_amplitude_range : tuple of int, optional
        Tuple (lb, ub) defining the index range used to calculate the correlation offset
        and the mean correlation amplitude. Default is (0, 16).
    time_upper : float, optional
        Upper time limit (in ms) below which the standard weighting is applied.
        For times above this threshold, weights are scaled down. Default is 10.
    z0_w0 : float, optional
        Shape factor (z0/w0) for the 3D Gaussian detection volume. Default is 3.5.
    starchev_a1 : float, optional
        Empirical fitting parameter a1 for the Starchev method. Default is 2.e-3.
    starchev_a2 : float, optional
        Empirical fitting parameter a2 for the Starchev method. Default is 1.8e-1.
    starchev_c1 : float, optional
        Empirical fitting parameter c1 for the Starchev method. Default is 1.0e-4.
    diffusion_time : float, optional
        Characteristic diffusion time in milliseconds. If not provided, it is estimated
        as the time at which the (background-corrected) correlation curve falls to half
        its maximum value.
    verbose : bool, optional
        If True, prints diagnostic messages. Default is False.

    Returns
    -------
    np.ndarray
        Array of noise weights (standard deviations) corresponding to each point in the
        correlation curve.

    References
    ----------
    Starchev, O. Y. (2001). Noise on fluorescence correlation spectroscopy.
    Journal of Colloid and Interface Science, 233, 50–55.

    Notes
    -----
    - The diffusion time is estimated by finding the first time point where the correlation
      falls below half of its mean amplitude (after background subtraction). If no such point
      is found, the middle of the time array is used.
    - The 'suren' weighting method is mentioned in the literature, but its publication details
      remain unclear.
    """
    if verbose:
        print("Calculating FCS weights")
        print("Using method: %s" % weight_type)
        print("measurement_duration [s]: %s" % measurement_duration)
        print("mean_count_rate [kHz]: %s" % mean_count_rate)
        print("skip_points: %s" % skip_points)

    sd = np.ones_like(correlation)
    if skip_points > 0:
        times = times[skip_points:]
        correlation = correlation[skip_points:]

    lb, ub = correlation_amplitude_range

    correlation_offset = np.mean(correlation[-lb:-ub])
    mean_correlation_amplitude = np.mean(correlation[lb:ub]) - correlation_offset
    if mean_correlation_amplitude == 0:
        print("WARNING: the mean correlation amplitude seems to be zero!")
        mean_correlation_amplitude = 1.0
    if diffusion_time is None:
        try:
            imaxhalf = np.min(np.nonzero(correlation < mean_correlation_amplitude / 2.0 + correlation_offset))
            diffusion_time = times[imaxhalf]
        except ValueError:
            diffusion_time = times[len(times) // 2]

    if weight_type == 'suren':
        dt = np.diff(times)
        dt = np.hstack([dt, dt[-1]])
        ns = measurement_duration * 1000. / dt
        na = dt * mean_count_rate

        A = np.exp(-2 * dt / diffusion_time)
        B = np.exp(-2 * times / diffusion_time)
        m = times / dt
        S = (mean_correlation_amplitude ** 2 / ns *
             ((1 + A) * (1 + B) + 2 * m * (1 - A) * B) / (1 - A) +
             2 * mean_correlation_amplitude / ns ** 2.0 * (1 + B) +
             (1 + mean_correlation_amplitude * np.sqrt(B)) / (ns * na ** 2))
        S *= (times < time_upper) + (times >= time_upper) * 10 ** (-np.log(times + 1e-12) / np.log(10) + 1)
        S = np.sqrt(np.abs(S))
        sd[skip_points:] = S
    elif weight_type == 'starchev':
        # Noise on fluorescence correlation spectroscopy - Starchev method
        # Reference: Starchev, O. Y. (2001). Noise on fluorescence correlation spectroscopy.
        # Journal of Colloid and Interface Science, 233, 50–55.
        tc = diffusion_time
        N = 1. / mean_correlation_amplitude
        a1 = starchev_a1
        a2 = starchev_a2
        c1 = starchev_c1
        p = z0_w0  # Shape factor of 3D Gaussian detection volume.
        i = mean_count_rate
        var_g = 1. / N ** 3.0 * (a1 / i + a2 / i ** 2) / (1 + p ** 2 * times / tc) + \
                c1 / (N ** 2.0) / (1 + p ** 2 * times / tc) ** 0.33
        sd[skip_points:] = np.sqrt(var_g)
    elif weight_type == 'uniform':
        sd = np.ones_like(correlation)

    return sd


def background_factor_ac(signal_cr_khz: float, background_cr_khz: float) -> float:
    """Return background attenuation factor for an autocorrelation model.

    The correlation amplitude of a fluorophore in the presence of a constant
    background is reduced by a factor

    .. math::

        k_\mathrm{bg} = \left(\frac{S - B}{S}\right)^2 ,

    where ``S`` is the total detected countrate (signal + background) and
    ``B`` is the background countrate. This factor multiplies the *amplitude*
    of a background-free model curve while leaving its baseline unchanged, so
    that

    ``G_meas(τ) - b = k_bg * (G_true(τ) - b)``.

    Parameters
    ----------
    signal_cr_khz : float
        Total detected countrate S in kHz.
    background_cr_khz : float
        Background countrate B in kHz.

    Returns
    -------
    float
        Attenuation factor ``k_bg`` in the range (0, 1]. If the inputs are
        non-finite or ``S <= B`` or ``S <= 0``, ``1.0`` is returned and no
        correction should be applied.
    """

    try:
        S = float(signal_cr_khz)
        B = float(background_cr_khz)
    except Exception:
        return 1.0

    if not np.isfinite(S) or not np.isfinite(B):
        return 1.0
    if S <= 0.0 or B < 0.0 or B >= S:
        return 1.0

    ratio = (S - B) / S
    return float(ratio * ratio)


def background_factor_cc(
        signal1_cr_khz: float,
        background1_cr_khz: float,
        signal2_cr_khz: float,
        background2_cr_khz: float,
) -> float:
    """Return background attenuation factor for a cross-correlation model.

    For cross-correlation between two channels with total countrates ``S1``,
    ``S2`` and backgrounds ``B1``, ``B2``, the amplitude reduction is

    .. math::

        k_\mathrm{bg} = \frac{S_1 - B_1}{S_1} \cdot \frac{S_2 - B_2}{S_2} .

    As for :func:`background_factor_ac`, this factor is meant to be applied to
    the *amplitude* of a background-free model curve while keeping the
    baseline ``b`` unchanged.

    Parameters
    ----------
    signal1_cr_khz, signal2_cr_khz : float
        Total detected countrates ``S1`` and ``S2`` in kHz.
    background1_cr_khz, background2_cr_khz : float
        Background countrates ``B1`` and ``B2`` in kHz.

    Returns
    -------
    float
        Attenuation factor ``k_bg`` in the range (0, 1]. If any input is
        non-finite or inconsistent (e.g. ``Si <= Bi`` or ``Si <= 0``), ``1.0``
        is returned and no correction should be applied.
    """

    try:
        S1 = float(signal1_cr_khz)
        B1 = float(background1_cr_khz)
        S2 = float(signal2_cr_khz)
        B2 = float(background2_cr_khz)
    except Exception:
        return 1.0

    if not (np.isfinite(S1) and np.isfinite(B1) and np.isfinite(S2) and np.isfinite(B2)):
        return 1.0
    if S1 <= 0.0 or S2 <= 0.0:
        return 1.0
    if B1 < 0.0 or B2 < 0.0 or B1 >= S1 or B2 >= S2:
        return 1.0

    r1 = (S1 - B1) / S1
    r2 = (S2 - B2) / S2
    return float(r1 * r2)


def _spline_local_residual_std(
        times: np.ndarray,
        correlation: np.ndarray,
        knot_count: int = 5,
        window: int = 3,
) -> np.ndarray:
    """Estimate local noise via residuals to a smooth spline on log10(tau).

    This is inspired by PyCorrFit's spline-based weighting: a smooth function is
    fitted to ``G(τ)`` on a log-time axis, and the local standard deviation of
    the residuals within a sliding window is used as a noise estimate.

    The implementation is deliberately conservative:

    - If SciPy is available, a cubic B-spline with ``knot_count`` interior
      knots is used (``scipy.interpolate.splrep/splev``).
    - If SciPy is not available or fitting fails, a simple moving-average
      smoother is used instead.
    - Any non-finite or degenerate cases fall back to unit standard deviations.
    """

    t = np.asarray(times, dtype=float).ravel()
    g = np.asarray(correlation, dtype=float).ravel()
    n = g.size
    if n == 0:
        return np.ones_like(g, dtype=float)

    # Work on positive, finite lag times only; others get unit variance.
    mask = np.isfinite(t) & np.isfinite(g) & (t > 0.0)
    if not np.any(mask):
        return np.ones_like(g, dtype=float)

    t_valid = t[mask]
    g_valid = g[mask]
    if t_valid.size < max(knot_count + 2, 8):
        # Not enough points for a meaningful spline fit.
        return np.ones_like(g, dtype=float)

    # Use log10 time axis for smoother behavior across decades.
    x = np.log10(t_valid)

    # Build a smooth approximation g_smooth(x).
    g_smooth = None
    try:
        try:
            import scipy.interpolate as _spintp  # type: ignore[import]
        except Exception:
            _spintp = None  # type: ignore[assignment]

        if _spintp is not None:
            # Interior knots between min/max of the log-time axis.
            k = int(max(1, knot_count))
            knots = np.linspace(x[1], x[-1], k + 2)[1:-1]
            tck = _spintp.splrep(x, g_valid, s=0.0, k=3, t=knots)
            g_smooth = _spintp.splev(x, tck, der=0)
    except Exception:
        g_smooth = None

    if g_smooth is None:
        # Fallback: simple moving average on linear time.
        half = int(max(1, window))
        width = 2 * half + 1
        kernel = np.ones(width, dtype=float) / float(width)
        g_pad = np.pad(g_valid, (half, half), mode="edge")
        g_smooth = np.convolve(g_pad, kernel, mode="valid")

    g_smooth = np.asarray(g_smooth, dtype=float).ravel()
    if g_smooth.size != g_valid.size:
        return np.ones_like(g, dtype=float)

    resid = g_valid - g_smooth

    # Sliding-window standard deviation of residuals.
    w = max(1, int(window))
    sd_local = np.empty_like(resid, dtype=float)
    for i in range(resid.size):
        lo = max(0, i - w)
        hi = min(resid.size, i + w + 1)
        block = resid[lo:hi]
        if block.size < 2 or not np.any(np.isfinite(block)):
            sd_local[i] = 1.0
        else:
            s = float(np.nanstd(block))
            sd_local[i] = s if np.isfinite(s) and s > 0.0 else 1.0

    # Map back into full-size array; invalid/zero-time entries get unit sd.
    sd_full = np.ones_like(g, dtype=float)
    sd_full[mask] = sd_local
    return sd_full


def compute_weights(
        times: np.ndarray,
        correlation: np.ndarray,
        acquisition_time_s: float,
        mean_count_rate_khz: float,
        mode: str | None = None,
        existing_weights: np.ndarray | None = None,
        noise_kwargs: dict | None = None,
) -> np.ndarray:
    """Return correlation-amplitude weights for FCS curves.

    This helper produces the *weights* (typically ``1/sigma``) used by
    :mod:`chisurf` for FCS fitting. It is a thin convenience wrapper around
    :func:`noise` and is intended for use when reading FCS data.

    Parameters
    ----------
    times : np.ndarray
        Correlation lag times (ms).
    correlation : np.ndarray
        Correlation amplitudes ``G(τ)``.
    acquisition_time_s : float
        Total acquisition time in seconds.
    mean_count_rate_khz : float
        Mean countrate in kHz.
    mode : {None, "file", "none", "uniform", "suren", "starchev", "photon_noise"}, optional
        Weighting mode. ``None`` or ``"file"`` keeps ``existing_weights`` if
        available. ``"none"`` / ``"uniform"`` produce unit weights. The
        remaining modes call :func:`noise` with the corresponding
        ``weight_type``.
    existing_weights : np.ndarray, optional
        Pre-existing weights (``1/sigma``) as stored in an
        ``FCSDataset['correlation_amplitude_weights']``. Used when
        ``mode is None`` or ``mode == "file"``.
    noise_kwargs : dict, optional
        Additional keyword arguments forwarded to :func:`noise`.

    Returns
    -------
    np.ndarray
        Weights ``w = 1/sigma`` with the same shape as ``correlation``.
    """

    if noise_kwargs is None:
        noise_kwargs = {}

    times = np.asarray(times, dtype=float).ravel()
    correlation = np.asarray(correlation, dtype=float).ravel()
    if times.size == 0 or correlation.size == 0:
        return np.ones_like(correlation, dtype=float)

    if existing_weights is not None:
        existing_weights = np.asarray(existing_weights, dtype=float).ravel()
        if existing_weights.size != correlation.size:
            existing_weights = None

    # Default behaviour: keep weights provided by the reader
    if mode is None or str(mode).lower() == "file":
        if existing_weights is not None:
            return existing_weights
        # Fall back to a reasonable photon-noise estimate
        mode = "suren"

    m = str(mode).lower()

    if m in ("none", "uniform"):
        return np.ones_like(correlation, dtype=float)

    # PyCorrFit-style spline-based local variance: weight_type "splineX".
    if m.startswith("spline"):
        # Optional knot count in the suffix (e.g. "spline5").
        suffix = m[len("spline"):]
        knots = None
        if suffix:
            try:
                knots = int(suffix)
            except Exception:
                knots = None
        if knots is None:
            try:
                knots = int(noise_kwargs.get("spline_knots", 5))
            except Exception:
                knots = 5

        try:
            spread = int(noise_kwargs.get("weight_spread", 3))
        except Exception:
            spread = 3

        try:
            sd = _spline_local_residual_std(
                times=times,
                correlation=correlation,
                knot_count=knots,
                window=spread,
            )
        except Exception:
            # Fallback: uniform if spline-based estimation fails.
            sd = np.ones_like(correlation, dtype=float)

        # Convert standard deviations to weights later in the common tail.
        sd = np.asarray(sd, dtype=float).ravel()
        if sd.size != correlation.size:
            return np.ones_like(correlation, dtype=float)

        tiny = 1e-12
        w = np.empty_like(sd, dtype=float)
        for i, v in enumerate(sd):
            if not np.isfinite(v) or abs(v) < tiny:
                w[i] = 1.0
            else:
                w[i] = 1.0 / float(v)
        return w

    if m == "photon_noise":
        weight_type = "suren"
    elif m in ("suren", "starchev"):
        weight_type = m
    else:
        # Unknown mode: keep existing weights if any, otherwise uniform
        if existing_weights is not None:
            return existing_weights
        return np.ones_like(correlation, dtype=float)

    try:
        sd = noise(
            times=times,
            correlation=correlation,
            measurement_duration=float(acquisition_time_s),
            mean_count_rate=float(mean_count_rate_khz),
            weight_type=weight_type,
            **noise_kwargs,
        )
    except Exception:
        # On failure, fall back to uniform weights
        return np.ones_like(correlation, dtype=float)

    sd = np.asarray(sd, dtype=float).ravel()
    if sd.size != correlation.size:
        # Shape mismatch: do not attempt to broadcast, just use uniform
        return np.ones_like(correlation, dtype=float)

    # Convert standard deviations to weights = 1/sigma, guarding against
    # zeros and non-finite values.
    tiny = 1e-12
    w = np.empty_like(sd, dtype=float)
    for i, v in enumerate(sd):
        if not np.isfinite(v) or abs(v) < tiny:
            w[i] = 1.0
        else:
            w[i] = 1.0 / float(v)
    return w

