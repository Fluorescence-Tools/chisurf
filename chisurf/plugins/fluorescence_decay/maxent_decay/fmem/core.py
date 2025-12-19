import math
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from numba import njit

try:
    from tqdm import trange as _mem_trange
except Exception:  # tqdm is optional; fall back to a plain range
    _mem_trange = range


MIN_PROB = 1e-12


def load_tcspc_two_column(path: str) -> np.ndarray:
    """Load a two-column TCSPC text file ("Chan  Data") like extract2c.c.

    This function scans all lines, attempts to parse two numbers from the
    beginning of each line, and returns the second column (counts) as a
    1D float array. Header/comment lines are ignored.
    """
    chan = []
    data = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            chan.append(x)
            data.append(y)
    if not data:
        return np.zeros(0, dtype=float)
    return np.asarray(data, dtype=float)


def autofitrange(lamp: np.ndarray, decay: np.ndarray, threshold: float = 10.0) -> Tuple[int, int]:
    """Port of gfit/autofitrange.m.

    Parameters
    ----------
    lamp, decay : array_like
        1D arrays of instrument response and decay counts.
    threshold : float
        Intensity threshold for the decay tail (default 100, as in MATLAB).

    Returns
    -------
    start, stop : int
        0-based inclusive indices defining the fit range.
    """
    lamp = np.asarray(lamp, dtype=float).ravel()
    decay = np.asarray(decay, dtype=float).ravel()
    if lamp.size == 0 or decay.size == 0:
        return 0, max(0, min(lamp.size, decay.size) - 1)

    start = int(np.argmax(lamp))
    peak = int(np.argmax(decay))

    below = np.nonzero(decay < threshold)[0]
    if below.size == 0:
        candidates = np.array([decay.size - 1], dtype=int)
    else:
        candidates = np.concatenate((below, np.array([decay.size - 1], dtype=int)))

    greater = candidates[candidates > peak]
    if greater.size == 0:
        stop = int(candidates[-1])
    else:
        stop = int(greater[0])

    if stop < start:
        stop = start
    return start, stop


def auto_fit_range_tcspc(
    decay: np.ndarray,
    count_threshold: float = 10.0,
    area: float = 0.999,
    start_fraction: float = 0.9,
    start_at_peak: bool = True,
    skip_first: int = 0,
    skip_last: int = 0,
) -> Tuple[int, int]:
    """Auto fit range similar to chisurf.fluorescence.tcspc.initial_fit_range.

    Parameters
    ----------
    decay : array_like
        1D decay counts.
    count_threshold : float
        Minimum counts to consider a channel as part of the fit.
    area : float
        Fraction of total counts (from the start index onward) to include.
    start_fraction : float
        Fraction of the peak at which to start when ``start_at_peak`` is True.
    start_at_peak : bool
        If True, start where ``decay > start_fraction * max(decay)``; otherwise
        start where ``decay > count_threshold``.
    skip_first, skip_last : int
        Extra channels to skip at the beginning / end of the fit range.
    """

    decay = np.asarray(decay, dtype=float).ravel()
    n = decay.size
    if n == 0:
        return 0, 0

    if start_at_peak and decay.max() > 0.0:
        lvl = start_fraction * float(decay.max())
        idx = np.nonzero(decay > lvl)[0]
    else:
        idx = np.nonzero(decay > count_threshold)[0]
    if idx.size == 0:
        start = 0
    else:
        start = int(idx[0])

    if start >= n - 1:
        return max(0, n - 1), max(0, n - 1)

    tail = decay[start:]
    s_total = float(np.sum(tail))
    if s_total <= 0.0:
        stop = n - 1
    else:
        cumsum = np.cumsum(tail, dtype=float)
        area_idx = np.nonzero(cumsum >= area * s_total)[0]
        if area_idx.size == 0:
            stop = n - 1
        else:
            stop = int(start + area_idx[0])

    stop = min(max(stop, start), n - 1)

    if decay[stop] < count_threshold:
        rev = np.nonzero(decay[::-1] >= count_threshold)[0]
        if rev.size > 0:
            stop = n - int(rev[0]) - 1

    while (stop + 1) < n and decay[stop] >= count_threshold:
        stop += 1

    start = min(start + int(skip_first), n - 1)
    stop = min(max(stop - int(skip_last), start), n - 1)
    return start, stop


def e1te2(e1: Sequence[float], e2: Sequence[float]) -> np.ndarray:
    """Port of gfit/e1te2.m.

    e1, e2 are [c1, tau1, c2, tau2, ...]. The result is all pairwise
    products with parallel combination of lifetimes.
    """
    a1 = np.asarray(e1, dtype=float).ravel()
    a2 = np.asarray(e2, dtype=float).ravel()
    if a1.size % 2 != 0 or a2.size % 2 != 0:
        raise ValueError("e1 and e2 must contain amplitude/tau pairs")

    n1 = a1.size // 2
    n2 = a2.size // 2
    out = np.empty(2 * n1 * n2, dtype=float)
    k = 0
    for i in range(n1):
        c1 = a1[2 * i]
        t1 = a1[2 * i + 1]
        for j in range(n2):
            c2 = a2[2 * j]
            t2 = a2[2 * j + 1]
            out[2 * k] = c1 * c2
            out[2 * k + 1] = 1.0 / (1.0 / t1 + 1.0 / t2)
            k += 1
    return out[: 2 * k]


@njit(cache=True)
def _shift_lamp(lamp: np.ndarray, ts_channels: float) -> np.ndarray:
    """Numba port of shift_lamp from fsconv2.c.

    Parameters
    ----------
    lamp : 1D float64 array
        Original instrument response.
    ts_channels : float
        Shift in channel units (can be fractional).

    Returns
    -------
    lampsh : 1D float64 array
        Shifted IRF, same length as ``lamp``.
    """
    n_points = lamp.shape[0]
    lampsh = np.zeros(n_points, dtype=np.float64)

    tsint = int(math.floor(ts_channels))
    tsdbl = ts_channels - float(tsint)

    out_left = 0
    out_right = 0
    if tsint < 0:
        out_left = -tsint
    if tsint + 1 > 0:
        out_right = tsint + 1

    if out_left > n_points:
        out_left = n_points
    if out_right > n_points:
        out_right = n_points

    for j in range(out_left):
        lampsh[j] = 0.0

    limit = n_points - out_right
    if limit < out_left:
        limit = out_left

    for j in range(out_left, limit):
        idx = j + tsint
        if idx < 0 or idx + 1 >= n_points:
            lampsh[j] = 0.0
        else:
            lampsh[j] = lamp[idx] * (1.0 - tsdbl) + lamp[idx + 1] * tsdbl

    start_tail = n_points - out_right
    if start_tail < 0:
        start_tail = 0
    for j in range(start_tail, n_points):
        lampsh[j] = 0.0

    return lampsh


@njit(cache=True)
def _fconv_single_shot(
    lampsh: np.ndarray,
    dt: float,
    amps: np.ndarray,
    taus: np.ndarray,
    stop: int,
) -> np.ndarray:
    """Numba port of fconv (single-shot convolution) from fsconv2.c.

    Convolves a sum of exponentials with a shifted IRF.
    """
    n_points = lampsh.shape[0]
    if stop >= n_points:
        stop = n_points - 1
    if stop < 1:
        stop = 1

    fit = np.zeros(n_points, dtype=np.float64)
    deltathalf = 0.5 * dt

    nexp = amps.shape[0]
    for k in range(nexp):
        amp = float(amps[k])
        tau = float(taus[k])
        if tau <= 0.0 or amp == 0.0:
            continue
        expcurr = math.exp(-dt / tau)
        fitcurr = 0.0
        for i in range(1, stop + 1):
            fitcurr = (fitcurr + deltathalf * lampsh[i - 1]) * expcurr + deltathalf * lampsh[i]
            fit[i] += fitcurr * amp

    return fit


@njit(cache=True)
def _fconv_periodic(
    lampsh: np.ndarray,
    dt: float,
    amps: np.ndarray,
    taus: np.ndarray,
    start: int,
    stop: int,
    period: float,
) -> np.ndarray:
    """Numba approximation of fconv_per from fsconv2.c.

    For very large ``period`` this effectively reduces to the single-shot case.
    """
    n_points = lampsh.shape[0]
    if stop >= n_points:
        stop = n_points - 1
    if start < 0:
        start = 0
    if start > stop:
        start = stop

    fit = np.zeros(n_points, dtype=np.float64)
    if period <= 0.0:
        return _fconv_single_shot(lampsh, dt, amps, taus, stop)

    lamp_start = 0
    while lamp_start < n_points and lampsh[lamp_start] == 0.0:
        lamp_start += 1

    period_n = int(math.ceil(period / dt - 0.5))
    stop1 = period_n + lamp_start
    if stop1 > n_points - 1:
        stop1 = n_points - 1

    deltathalf = 0.5 * dt
    nexp = amps.shape[0]

    for k in range(nexp):
        amp = float(amps[k])
        tau = float(taus[k])
        if tau <= 0.0 or amp == 0.0:
            continue

        expcurr = math.exp(-dt / tau)
        tail_a = 1.0 / (1.0 - math.exp(-period / tau))
        fitcurr = 0.0

        for i in range(1, stop1 + 1):
            fitcurr = (fitcurr + deltathalf * lampsh[i - 1]) * expcurr + deltathalf * lampsh[i]
            fit[i] += fitcurr * amp

        steps_to_start = period_n - stop1 + start
        if steps_to_start > 0:
            fitcurr *= math.exp(-steps_to_start * dt / tau)

        for i in range(start, stop + 1):
            fitcurr *= expcurr
            fit[i] += fitcurr * amp * tail_a

    return fit


def _build_Fi_distances(
    decay: np.ndarray,
    lamp: np.ndarray,
    dt: float,
    R: np.ndarray,
    tau0: float,
    R0: float,
    donly: np.ndarray,
    x_donly: float,
    timeshift: float,
    background: float,
    lamp_scatter: float,
    fitstart: int,
    fitstop: int,
    period: float,
    irf_background: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct Fi, y, sigma for the distance MEM (me_vin4_E.m analogue)."""
    decay = np.asarray(decay, dtype=float).ravel()
    lamp = np.asarray(lamp, dtype=float).ravel()
    R = np.asarray(R, dtype=float).ravel()
    donly = np.asarray(donly, dtype=float).ravel()

    if decay.size == 0 or lamp.size == 0 or R.size == 0 or donly.size == 0:
        raise ValueError("decay, lamp, R and donly must be non-empty")
    if donly.size % 2 != 0:
        raise ValueError("donly must contain amplitude/tau pairs")

    n = R.size

    if fitstop >= decay.size:
        fitstop = decay.size - 1
    if fitstop >= lamp.size:
        fitstop = lamp.size - 1
    if fitstart < 0:
        fitstart = 0
    if fitstart > fitstop:
        fitstart = fitstop

    y = decay[fitstart : fitstop + 1]
    sigma = np.sqrt(y) + (y == 0.0)

    M = y.size
    Fi = np.empty((M, n), dtype=float)

    irf_bg = float(irf_background)
    lamp_corr = lamp - irf_bg
    lamp_corr[lamp_corr < 0.0] = 0.0
    # NOTE: ``timeshift`` is expressed in detector channels (samples) to match
    # ChiSurf's Curve shifting semantics (Curve.__lshift__). It may be
    # fractional.
    ts_channels = float(timeshift)
    lampsh = _shift_lamp(lamp_corr.astype(np.float64), ts_channels)

    amps_donly = donly[0::2].astype(np.float64)
    taus_donly = donly[1::2].astype(np.float64)
    if period > 0.0:
        donor_full = _fconv_periodic(
            lampsh,
            float(dt),
            amps_donly,
            taus_donly,
            int(fitstart),
            int(fitstop),
            float(period),
        )
    else:
        donor_full = _fconv_single_shot(
            lampsh,
            float(dt),
            amps_donly,
            taus_donly,
            int(fitstop),
        )
    donor_seg = donor_full[fitstart : fitstop + 1]
    lamp_scatter_seg = lampsh[fitstart : fitstop + 1].astype(float)

    x_d = float(x_donly)
    if x_d < 0.0:
        x_d = 0.0
    if x_d > 1.0:
        x_d = 1.0
    x_fret = 1.0 - x_d
    fit_additive = float(background) + float(lamp_scatter) * lamp_scatter_seg

    for j in range(n):
        Rj = float(R[j])
        if Rj <= 0.0:
            raise ValueError("R grid must be positive")

        kfret = (1.0 / float(tau0)) * (float(R0) / Rj) ** 6
        e2 = np.array([1.0, 1.0 / kfret], dtype=float)
        d = e1te2(donly, e2)
        amps = d[0::2].astype(np.float64)
        taus = d[1::2].astype(np.float64)

        if period > 0.0:
            fit_full = _fconv_periodic(lampsh, float(dt), amps, taus, int(fitstart), int(fitstop), float(period))
        else:
            fit_full = _fconv_single_shot(lampsh, float(dt), amps, taus, int(fitstop))

        col = x_fret * fit_full[fitstart : fitstop + 1] + x_d * donor_seg

        Fi[:, j] = col / sigma

    return Fi, y, sigma, fit_additive


def _build_Fi_lifetimes(
    decay: np.ndarray,
    lamp: np.ndarray,
    dt: float,
    tau: np.ndarray,
    timeshift: float,
    background: float,
    lamp_scatter: float,
    fitstart: int,
    fitstop: int,
    period: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct Fi, y, sigma for the lifetime MEM (me_vin4.m analogue)."""

    decay = np.asarray(decay, dtype=float).ravel()
    lamp = np.asarray(lamp, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()

    if decay.size == 0 or lamp.size == 0 or tau.size == 0:
        raise ValueError("decay, lamp and tau must be non-empty")

    tau = tau[tau > 0.0]
    if tau.size == 0:
        raise ValueError("tau grid must contain positive values")

    if fitstop >= decay.size:
        fitstop = decay.size - 1
    if fitstop >= lamp.size:
        fitstop = lamp.size - 1
    if fitstart < 0:
        fitstart = 0
    if fitstart > fitstop:
        fitstart = fitstop

    y = decay[fitstart : fitstop + 1]
    sigma = np.sqrt(y) + (y == 0.0)

    M = y.size
    n = tau.size
    Fi = np.empty((M, n), dtype=float)

    # NOTE: ``timeshift`` is expressed in detector channels (samples) to match
    # ChiSurf's Curve shifting semantics (Curve.__lshift__). It may be
    # fractional.
    ts_channels = float(timeshift)
    lampsh = _shift_lamp(lamp.astype(np.float64), ts_channels)

    lamp_seg = lampsh[fitstart : fitstop + 1].astype(float)
    fit_additive = np.full_like(y, float(background), dtype=float) + float(lamp_scatter) * lamp_seg

    for j in range(n):
        amps = np.array([1.0], dtype=float)
        taus = np.array([float(tau[j])], dtype=float)

        if period > 0.0:
            fit_full = _fconv_periodic(
                lampsh,
                float(dt),
                amps,
                taus,
                int(fitstart),
                int(fitstop),
                float(period),
            )
        else:
            fit_full = _fconv_single_shot(
                lampsh,
                float(dt),
                amps,
                taus,
                int(fitstop),
            )

        col = fit_full[fitstart : fitstop + 1]
        Fi[:, j] = col / sigma

    return Fi, y, sigma, fit_additive


def _quadpr_bound(C: np.ndarray, d: np.ndarray, lower_bound: float) -> np.ndarray:
    """Small bound-constrained QP solver ``min 0.5 x^T C x + d^T x``.

    The only constraints are lower bounds ``x >= lower_bound``. We use a
    simple active-set strategy: repeatedly solve the unconstrained system on
    the currently-free variables and clamp any components that violate the
    bound until the active set stops growing.
    """

    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    n = d.size
    if C.shape != (n, n):
        raise ValueError("C must be square with shape (n, n)")

    # Ensure symmetry; the quadratic form only depends on the symmetric part.
    C = 0.5 * (C + C.T)
    lb = float(lower_bound)

    x = np.zeros(n, dtype=float)
    active = np.zeros(n, dtype=bool)

    for _ in range(50):
        free = ~active
        if np.any(free):
            C_ff = C[np.ix_(free, free)]
            d_f = d[free]
            try:
                x_f = -np.linalg.solve(C_ff, d_f)
            except np.linalg.LinAlgError:
                x_f = -np.linalg.lstsq(C_ff, d_f, rcond=None)[0]
            x[free] = x_f

        # Enforce the bound on the active set explicitly.
        x[active] = lb

        viol = x < lb
        new_active = viol & ~active
        if not np.any(new_active):
            break
        active |= viol

    x = np.maximum(x, lb)
    return x


def _run_mem(
    H: np.ndarray,
    g0: np.ndarray,
    m: np.ndarray,
    const_chi2: float,
    nu: float,
    progress_cb: Optional[Callable[[int, float, float, float, float], None]] = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    min_prob: float = MIN_PROB,
) -> Dict[str, Any]:
    """MEM optimizer following the structure of me_vin4_E.m."""
    H = np.asarray(H, dtype=float)
    g0 = np.asarray(g0, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()

    n = m.size
    if H.shape != (n, n):
        raise ValueError("H must have shape (n, n) with n = len(m)")

    H = 0.5 * (H + H.T)

    H_eps = H + np.diag(np.diag(H) * 1e-12)
    p_esm = _quadpr_bound(H_eps, -g0, lower_bound=min_prob)
    chisq_esm = 0.5 * p_esm @ H @ p_esm - g0 @ p_esm + const_chi2
    L_esm = np.log(np.clip(p_esm / m, a_min=1e-300, a_max=None))
    S_esm = (-L_esm + 1.0) @ p_esm - float(np.sum(m))
    Q_esm = float(chisq_esm - 0.5 * nu * S_esm)

    p = m.copy()
    L = np.log(np.clip(p / m, a_min=1e-300, a_max=None))
    chisq = 0.5 * p @ H @ p - g0 @ p + const_chi2
    S = (-L + 1.0) @ p - float(np.sum(m))
    Q = float(chisq - 0.5 * nu * S)

    dgrad = 1.0
    history: Iterable[Tuple[float, float, float, float]] = []  # type: ignore
    hist_list = []

    for niter in _mem_trange(1, max_iter + 1):
        if dgrad <= tol:
            break

        Delta_diag = 0.5 / np.maximum(p, min_prob)
        C_eff = H + np.diag(nu * Delta_diag)
        d_eff = -g0 + 0.5 * nu * (L - 1.0)

        p = _quadpr_bound(C_eff, d_eff, lower_bound=min_prob)
        chisq = 0.5 * p @ H @ p - g0 @ p + const_chi2
        L = np.log(np.clip(p / m, a_min=1e-300, a_max=None))
        S = (-L + 1.0) @ p - float(np.sum(m))
        Q = float(chisq - 0.5 * nu * S)

        grad_chi2 = H @ p - g0
        grad_S = -L
        mask = p > -1.1 * min_prob
        grad_chi2 = grad_chi2 * mask
        grad_S = grad_S * mask
        norm_chi2 = float(np.linalg.norm(grad_chi2))
        norm_S = float(np.linalg.norm(grad_S))
        if norm_chi2 == 0.0 or norm_S == 0.0:
            dgrad = 0.0
        else:
            diff = grad_chi2 / norm_chi2 - grad_S / norm_S
            dgrad = 0.5 * float(np.linalg.norm(diff))

        hist_list.append((float(chisq), float(S), float(Q), float(dgrad)))

        if progress_cb is not None:
            progress_cb(int(niter), float(chisq), float(S), float(Q), float(dgrad))

    history = hist_list

    return {
        "p": p,
        "chisq": float(chisq),
        "S": float(S),
        "Q": float(Q),
        "history": history,
        "p_esm": p_esm,
        "chisq_esm": float(chisq_esm),
        "S_esm": float(S_esm),
        "Q_esm": Q_esm,
        "nu": float(nu),
        "niter": len(history),
    }


def solve_lifetime_mem(
    decay: Sequence[float],
    lamp: Sequence[float],
    dt: float,
    tau: Optional[Sequence[float]] = None,
    timeshift: float = 0.0,
    background: float = 0.0,
    lamp_scatter: float = 0.0,
    fitrange: Optional[Tuple[int, int]] = None,
    irf_background: Optional[float] = None,
    fit_start_fraction: float = 0.9,
    nu: float = 1e-5,
    progress_cb: Optional[Callable[[int, float, float, float, float], None]] = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    period: Optional[float] = None,
    optimize_nuisance: bool = False,
    nuisance_max_iter: int = 20,
    nuisance_step_timeshift: Optional[float] = None,
    nuisance_step_background: Optional[float] = None,
    nuisance_step_irf_background: Optional[float] = None,
    nuisance_step_x_donly: Optional[float] = None,
    nuisance_param_tol: float = 1e-3,
    prior: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """MEM analysis of TCSPC lifetimes (Python analogue of me_vin4.m).

    The default lifetime grid spans 1.0–4.0 ns with 0.01 ns spacing, i.e.
    ``tau = np.arange(1.0, 4.0 + 1e-9, 0.01)``.

    Parameters
    ----------
    timeshift:
        IRF shift relative to the decay in detector channels (samples).
    """
    decay_arr = np.asarray(decay, dtype=float).ravel()
    lamp_arr = np.asarray(lamp, dtype=float).ravel()

    if fitrange is None:
        try:
            if isinstance(lamp_scatter, (tuple, list)) and len(lamp_scatter) == 2:
                fitrange = (int(lamp_scatter[0]), int(lamp_scatter[1]))
                lamp_scatter = 0.0
        except Exception:
            pass

    # IRF background correction: subtract constant offset from lamp and clamp
    if irf_background is None:
        tail_len_lamp = min(500, lamp_arr.size)
        if tail_len_lamp > 0:
            irf_bg_val = float(np.median(lamp_arr[-tail_len_lamp:]))
        else:
            irf_bg_val = 0.0
    else:
        irf_bg_val = float(irf_background)

    lamp_corr = lamp_arr - irf_bg_val
    lamp_corr[lamp_corr < 0.0] = 0.0

    if tau is None:
        tau_arr = np.arange(1.0, 4.0 + 1e-9, 0.01, dtype=float)
    else:
        tau_arr = np.asarray(tau, dtype=float).ravel()

    tau_arr = tau_arr[tau_arr > 0.0]
    if tau_arr.size == 0:
        raise ValueError("tau grid must contain positive values")

    if fitrange is None:
        fitstart, fitstop = auto_fit_range_tcspc(
            decay_arr,
            count_threshold=100.0,
            area=0.999,
            start_fraction=float(fit_start_fraction),
            start_at_peak=True,
        )
    else:
        fitstart, fitstop = int(fitrange[0]), int(fitrange[1])

    if period is None:
        period_val = 0.0
    else:
        period_val = float(period)

    tail_len_decay = min(500, decay_arr.size)
    if tail_len_decay > 0:
        decay_bg_median = float(np.median(decay_arr[-tail_len_decay:]))
    else:
        decay_bg_median = 0.0

    bg0 = float(background) if background > 0.0 else decay_bg_median
    ts0 = float(timeshift)
    irf_bg0 = float(irf_bg_val)

    if prior is None:
        prior_vec = np.ones_like(tau_arr, dtype=float)
        prior_vec /= float(np.sum(prior_vec))
    else:
        prior_vec = np.asarray(prior, dtype=float).ravel()
        if prior_vec.size != tau_arr.size:
            raise ValueError("prior must have same length as tau grid")
        prior_vec[prior_vec <= 0.0] = MIN_PROB
        prior_vec /= float(np.sum(prior_vec))

    def _eval_mem_lifetime_single(ts_val: float, bg_val: float, irf_bg_single: float) -> Dict[str, Any]:
        if irf_bg_single == irf_bg_val:
            lamp_corr_single = lamp_corr
        else:
            lamp_corr_single = lamp_arr - irf_bg_single
            lamp_corr_single[lamp_corr_single < 0.0] = 0.0

        Fi_single, y_single, sigma_single, fit_additive = _build_Fi_lifetimes(
            decay_arr,
            lamp_corr_single,
            float(dt),
            tau_arr,
            float(ts_val),
            float(bg_val),
            float(lamp_scatter),
            int(fitstart),
            int(fitstop),
            period_val,
        )

        y_eff = y_single - fit_additive
        y_w = y_eff / sigma_single

        M_single = float(y_single.size)
        H_single = (2.0 / M_single) * (Fi_single.T @ Fi_single)
        g0_single = (2.0 / M_single) * (y_w @ Fi_single)
        const_chi2_single = float(np.sum(y_w * y_w) / M_single)

        m_single = prior_vec

        res_single = _run_mem(
            H_single,
            g0_single,
            m_single,
            const_chi2_single,
            float(nu),
            progress_cb=progress_cb,
            max_iter=int(max_iter),
            tol=float(tol),
            min_prob=MIN_PROB,
        )

        res_single.update(
            {
                "tau": tau_arr,
                "fitrange": (int(fitstart), int(fitstop)),
                "dt": float(dt),
                "timeshift": float(ts_val),
                "background": float(bg_val),
                "lamp_scatter": float(lamp_scatter),
                "fit_additive": fit_additive,
                "irf_background": float(irf_bg_single),
                "nu_input": float(nu),
                "H": H_single,
                "g0": g0_single,
                "y": y_single,
                "sigma": sigma_single,
                "Fi": Fi_single,
                "prior": prior_vec,
            }
        )
        return res_single

    if not optimize_nuisance:
        result = _eval_mem_lifetime_single(ts0, float(background), irf_bg0)
        result["nuisance_optimized"] = False
        return result

    if nuisance_step_timeshift is None:
        # timeshift is in channels (samples)
        step_ts = 0.5
    else:
        step_ts = float(abs(nuisance_step_timeshift))

    if nuisance_step_background is None:
        step_bg = 0.5 * max(bg0, 1.0)
    else:
        step_bg = float(abs(nuisance_step_background))

    if nuisance_step_irf_background is None:
        step_irf = 0.5 * max(irf_bg0, 1.0)
    else:
        step_irf = float(abs(nuisance_step_irf_background))

    steps = np.array([step_ts, step_bg, step_irf], dtype=float)

    max_shift_channels = 20.0
    ts_min = -max_shift_channels
    ts_max = max_shift_channels
    lower_bounds = np.array([ts_min, 0.0, 0.0], dtype=float)
    upper_bounds = np.array([ts_max, np.inf, np.inf], dtype=float)

    eval_history = []

    def _objective(x_vec: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        x_clipped = np.minimum(np.maximum(x_vec, lower_bounds), upper_bounds)
        res_loc = _eval_mem_lifetime_single(
            float(x_clipped[0]), float(x_clipped[1]), float(x_clipped[2])
        )
        Q_loc = float(res_loc["Q"])
        chisq_loc = float(res_loc["chisq"])
        eval_history.append(
            {
                "params": x_clipped.copy(),
                "Q": Q_loc,
                "chisq": chisq_loc,
            }
        )
        return Q_loc, res_loc

    x_best = np.array([ts0, bg0, irf_bg0], dtype=float)
    Q_best, res_best = _objective(x_best)
    steps_cur = steps.copy()
    param_tol = float(nuisance_param_tol)

    for _ in _mem_trange(int(nuisance_max_iter)):
        improved = False
        for i in range(3):
            for sign in (1.0, -1.0):
                trial = x_best.copy()
                trial[i] += sign * steps_cur[i]
                Q_trial, res_trial = _objective(trial)
                if Q_trial < Q_best:
                    Q_best = Q_trial
                    x_best = trial
                    res_best = res_trial
                    improved = True
                    break
            if improved:
                continue
        if not improved:
            steps_cur *= 0.5
            if np.all(steps_cur < param_tol):
                break

    res_best["nuisance_optimized"] = True
    res_best["nuisance_result"] = {
        "initial": np.array([ts0, bg0, irf_bg0], dtype=float),
        "best": x_best,
        "steps_final": steps_cur,
        "eval_history": eval_history,
    }
    return res_best


def solve_fret_mem(
    decay: Sequence[float],
    lamp: Sequence[float],
    dt: float,
    R: Optional[Sequence[float]] = None,
    tau0: float = 4.1,
    R0: float = 52.0,
    donly: Optional[Sequence[float]] = None,
    x_donly: float = 0.0,
    timeshift: float = 0.0,
    background: float = 0.0,
    lamp_scatter: float = 0.0,
    fitrange: Optional[Tuple[int, int]] = None,
    irf_background: Optional[float] = None,
    fit_start_fraction: float = 0.9,
    nu: float = 1e-5,
    progress_cb: Optional[Callable[[int, float, float, float, float], None]] = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    period: Optional[float] = None,
    optimize_nuisance: bool = False,
    nuisance_max_iter: int = 20,
    nuisance_step_timeshift: Optional[float] = None,
    nuisance_step_background: Optional[float] = None,
    nuisance_step_irf_background: Optional[float] = None,
    nuisance_step_x_donly: Optional[float] = None,
    nuisance_param_tol: float = 1e-3,
    prior: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Python plugin port of me_vin4_E.m using Numba instead of MEX/C.

    Parameters
    ----------
    decay : array_like
        Measured decay counts vs time.
    lamp : array_like
        Instrument response function on the same time grid.
    dt : float
        Time step per channel (same units as ``tau0``).
    R : array_like, optional
        Distance grid. Default is 18:0.5:120 as in me_vin4_E.m.
    tau0 : float, optional
        Donor lifetime in the absence of FRET.
    R0 : float, optional
        Förster radius.
    donly : array_like, optional
        Donor-only decay as [c1, tau1, c2, tau2, ...]. Default [1, tau0].
    timeshift : float, optional
        IRF shift relative to the decay in detector channels (samples).
    background : float, optional
        Constant background added to the model.
    lamp_scatter : float, optional
        Lamp scatter coefficient multiplied by the IRF.
    fitrange : (int, int), optional
        0-based inclusive start/stop indices for the fit range. If None,
        use :func:`autofitrange` on ``lamp`` and ``decay``.
    nu : float, optional
        Entropy Lagrange multiplier (regularization strength).
    max_iter : int, optional
        Maximum number of MEM iterations.
    tol : float, optional
        Stopping criterion on the gradient angle (dgrad).
    period : float, optional
        Excitation period. If None or <= 0, a single-shot convolution is used.

    Returns
    -------
    result : dict
        Dictionary with keys ``p`` (distance distribution), ``R``, ``chisq``,
        ``S``, ``Q``, ``history`` and additional diagnostic information.
    """

    decay_arr = np.asarray(decay, dtype=float).ravel()
    lamp_arr = np.asarray(lamp, dtype=float).ravel()

    if R is None:
        R_arr = np.arange(18.0, 120.0 + 1e-9, 0.5, dtype=float)
    else:
        R_arr = np.asarray(R, dtype=float).ravel()

    if donly is None:
        donly_arr = np.array([1.0, float(tau0)], dtype=float)
    else:
        donly_arr = np.asarray(donly, dtype=float).ravel()

    # IRF background correction: subtract constant offset from lamp and clamp
    if irf_background is None:
        tail_len_lamp = min(500, lamp_arr.size)
        if tail_len_lamp > 0:
            irf_bg_val = float(np.median(lamp_arr[-tail_len_lamp:]))
        else:
            irf_bg_val = 0.0
    else:
        irf_bg_val = float(irf_background)

    if fitrange is None:
        fitstart, fitstop = auto_fit_range_tcspc(
            decay_arr,
            count_threshold=100.0,
            area=0.999,
            start_fraction=float(fit_start_fraction),
            start_at_peak=True,
        )
    else:
        fitstart, fitstop = int(fitrange[0]), int(fitrange[1])

    if period is None:
        period_val = 0.0
    else:
        period_val = float(period)

    if prior is None:
        prior_vec = np.ones_like(R_arr, dtype=float)
        prior_vec /= float(np.sum(prior_vec))
    else:
        prior_vec = np.asarray(prior, dtype=float).ravel()
        if prior_vec.size != R_arr.size:
            raise ValueError("prior must have same length as R grid")
        prior_vec[prior_vec <= 0.0] = MIN_PROB
        prior_vec /= float(np.sum(prior_vec))

    def _eval_mem_distance_single(
        ts_val: float,
        bg_val: float,
        irf_bg_single: float,
        x_donly_val: float,
    ) -> Dict[str, Any]:
        Fi, y, sigma, fit_additive = _build_Fi_distances(
            decay_arr,
            lamp_arr,
            float(dt),
            R_arr,
            float(tau0),
            float(R0),
            donly_arr,
            float(x_donly_val),
            float(ts_val),
            float(bg_val),
            float(lamp_scatter),
            int(fitstart),
            int(fitstop),
            period_val,
            float(irf_bg_single),
        )
        y_eff = y - fit_additive

        # Weighted data vector y/sigma for the least-squares terms.
        y_w = y_eff / sigma

        M = y.size
        H = (2.0 / M) * (Fi.T @ Fi)
        g0 = (2.0 / M) * (y_w @ Fi)
        const_chi2 = float(np.sum(y_w * y_w) / M)

        m = prior_vec

        res_single = _run_mem(
            H,
            g0,
            m,
            const_chi2,
            float(nu),
            progress_cb=progress_cb,
            max_iter=int(max_iter),
            tol=float(tol),
            min_prob=MIN_PROB,
        )

        res_single.update(
            {
                "R": R_arr,
                "tau0": float(tau0),
                "R0": float(R0),
                "donly": donly_arr,
                "x_donly": float(x_donly_val),
                "fit_additive": fit_additive,
                "fitrange": (int(fitstart), int(fitstop)),
                "dt": float(dt),
                "timeshift": float(ts_val),
                "background": float(bg_val),
                "lamp_scatter": float(lamp_scatter),
                "irf_background": float(irf_bg_single),
                "nu_input": float(nu),
                "Fi": Fi,
                "H": H,
                "g0": g0,
                "y": y,
                "sigma": sigma,
                "prior": prior_vec,
            }
        )
        return res_single

    if not optimize_nuisance:
        result = _eval_mem_distance_single(float(timeshift), float(background), float(irf_bg_val), float(x_donly))
        result["nuisance_optimized"] = False
        return result

    ts0 = float(timeshift)
    bg0 = float(background)
    if bg0 <= 0.0:
        tail_len_decay = min(500, decay_arr.size)
        if tail_len_decay > 0:
            bg0 = float(np.median(decay_arr[-tail_len_decay:]))
        else:
            bg0 = 0.0
    irf_bg0 = float(irf_bg_val)

    x0 = float(x_donly)
    if x0 < 0.0:
        x0 = 0.0
    if x0 > 1.0:
        x0 = 1.0

    if nuisance_step_timeshift is None:
        # timeshift is in channels (samples)
        step_ts = 0.5
    else:
        step_ts = float(abs(nuisance_step_timeshift))

    if nuisance_step_background is None:
        step_bg = 0.5 * max(bg0, 1.0)
    else:
        step_bg = float(abs(nuisance_step_background))

    if nuisance_step_irf_background is None:
        step_irf = 0.5 * max(irf_bg0, 1.0)
    else:
        step_irf = float(abs(nuisance_step_irf_background))

    if nuisance_step_x_donly is None:
        step_x = 0.05
    else:
        step_x = float(abs(nuisance_step_x_donly))

    steps = np.array([step_ts, step_bg, step_irf, step_x], dtype=float)

    max_shift_channels = 20.0
    ts_min = -max_shift_channels
    ts_max = max_shift_channels
    lower_bounds = np.array([ts_min, 0.0, 0.0, 0.0], dtype=float)
    upper_bounds = np.array([ts_max, np.inf, np.inf, 1.0], dtype=float)

    eval_history = []

    def _objective(x_vec: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        x_clipped = np.minimum(np.maximum(x_vec, lower_bounds), upper_bounds)
        res_loc = _eval_mem_distance_single(
            float(x_clipped[0]), float(x_clipped[1]), float(x_clipped[2]), float(x_clipped[3])
        )
        Q_loc = float(res_loc["Q"])
        chisq_loc = float(res_loc["chisq"])
        eval_history.append(
            {
                "params": x_clipped.copy(),
                "Q": Q_loc,
                "chisq": chisq_loc,
            }
        )
        return Q_loc, res_loc

    x_best = np.array([ts0, bg0, irf_bg0, x0], dtype=float)
    Q_best, res_best = _objective(x_best)
    steps_cur = steps.copy()
    param_tol = float(nuisance_param_tol)

    for _ in _mem_trange(int(nuisance_max_iter)):
        improved = False
        for i in range(4):
            for sign in (1.0, -1.0):
                trial = x_best.copy()
                trial[i] += sign * steps_cur[i]
                Q_trial, res_trial = _objective(trial)
                if Q_trial < Q_best:
                    Q_best = Q_trial
                    x_best = trial
                    res_best = res_trial
                    improved = True
                    break
            if improved:
                continue
        if not improved:
            steps_cur *= 0.5
            if np.all(steps_cur < param_tol):
                break

    res_best["nuisance_optimized"] = True
    res_best["nuisance_result"] = {
        "initial": np.array([ts0, bg0, irf_bg0, x0], dtype=float),
        "best": x_best,
        "steps_final": steps_cur,
        "eval_history": eval_history,
    }
    return res_best


__all__ = [
    "MIN_PROB",
    "load_tcspc_two_column",
    "auto_fit_range_tcspc",
    "solve_lifetime_mem",
    "solve_fret_mem",
]
