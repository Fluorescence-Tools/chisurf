"""

"""
from __future__ import annotations
from chisurf import typing

import numpy as np
import warnings
from typing import TYPE_CHECKING

import chisurf as cs
import chisurf.core.math.statistics


if TYPE_CHECKING:
    from chisurf.core.fitting.fit import Fit

EPS = 1e-15


def scan_parameter(
        fit: Fit,
        parameter_name: str,
        scan_range=(None, None),
        rel_range: float = 0.2,
        n_steps: int = 30
) -> typing.Dict:
    """Performs a chi2-scan for the parameter

    :param fit: the fit of type 'fitting.Fit'
    :param parameter_name: the name of the parameter (in the parameter dictionary)
    :param scan_range: the range within the parameter is scanned if not provided 'rel_range' is used
    :param rel_range: defines +/- values for scanning
    :param n_steps: number of steps between +/-
    :return:
    """
    # Store initial values before varying the parameter
    initial_parameter_values = fit.model.parameter_values

    varied_parameter = fit.model.parameters_all_dict[parameter_name]
    is_fixed = varied_parameter.fixed

    varied_parameter.fixed = True
    chi2r_array = np.empty(n_steps, dtype=float)

    # Determine range within the parameter is varied
    parameter_value = varied_parameter.value
    p_min, p_max = scan_range
    if p_min is None or p_max is None:
        p_min = parameter_value * (1. - rel_range)
        p_max = parameter_value * (1. + rel_range)
    parameter_array = np.linspace(p_min, p_max, n_steps)

    for i, p in enumerate(parameter_array):
        varied_parameter.fixed = is_fixed
        fit.model.parameter_values = initial_parameter_values
        varied_parameter.fixed = True
        varied_parameter.value = p
        fit.run()
        chi2r_array[i] = fit.chi2r

    varied_parameter.fixed = is_fixed
    fit.model.parameter_values = initial_parameter_values
    fit.update()

    return {
        'chi2r': chi2r_array,
        'parameter_values': parameter_array,
        'parameter_names': [parameter_name]
    }


def _eval_scan_point(
        fit: Fit,
        parameter,
        p_value: float,
        initial_parameter_values,
        is_fixed: bool
) -> float:
    """Helper: fix param to p_value, refit, return chi2r."""
    parameter.fixed = is_fixed
    fit.model.parameter_values = initial_parameter_values
    parameter.fixed = True
    parameter.value = p_value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit.run()
    return fit.chi2r


def _finite_parameter_bound(value) -> typing.Optional[float]:
    """Return a finite parameter bound or ``None``.

    Parameters
    ----------
    value : object
        Candidate bound value.

    Returns
    -------
    float or None
        Finite bound converted to float, otherwise ``None``.
    """
    try:
        bound = float(value)
    except Exception:
        return None
    return bound if np.isfinite(bound) else None


def _interpolate_threshold_crossing(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        threshold: float
) -> typing.Optional[float]:
    """Interpolate an x-position where a segment crosses a threshold.

    Parameters
    ----------
    x0, y0, x1, y1 : float
        Segment endpoints.
    threshold : float
        Target y-value.

    Returns
    -------
    float or None
        Interpolated x crossing if the segment spans the threshold.
    """
    try:
        x0 = float(x0)
        y0 = float(y0)
        x1 = float(x1)
        y1 = float(y1)
        threshold = float(threshold)
    except Exception:
        return None
    if not all(np.isfinite(v) for v in (x0, y0, x1, y1, threshold)):
        return None
    d0 = y0 - threshold
    d1 = y1 - threshold
    if d0 == 0.0:
        return x0
    if d1 == 0.0:
        return x1
    if d0 * d1 > 0.0:
        return None
    denom = y1 - y0
    if abs(denom) <= EPS:
        return None
    return x0 + (threshold - y0) * (x1 - x0) / denom


def _find_side_crossing(
        values,
        chi2r,
        threshold: float,
        v0: float,
        sign: int
) -> typing.Optional[float]:
    """Find the first threshold crossing on one side of the minimum.

    Parameters
    ----------
    values : array_like
        Parameter values from a completed scan.
    chi2r : array_like
        Reduced chi-squared values for ``values``.
    threshold : float
        F-test threshold.
    v0 : float
        Best-fit parameter value.
    sign : int
        ``-1`` for the lower side, ``+1`` for the upper side.

    Returns
    -------
    float or None
        First threshold crossing away from ``v0``.
    """
    pairs = []
    for x, y in zip(values, chi2r):
        try:
            x = float(x)
            y = float(y)
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y):
            pairs.append((x, y))
    if sign < 0:
        side = sorted((p for p in pairs if p[0] <= v0), key=lambda p: p[0], reverse=True)
    else:
        side = sorted((p for p in pairs if p[0] >= v0), key=lambda p: p[0])
    for first, second in zip(side, side[1:]):
        crossing = _interpolate_threshold_crossing(
            first[0], first[1], second[0], second[1], threshold
        )
        if crossing is not None:
            return float(crossing)
    return None


def _scan_one_side(
        fit: Fit,
        parameter,
        v0: float,
        chi2r_min: float,
        sign: int,
        threshold: float,
        initial_parameter_values,
        is_fixed: bool,
        max_points: int,
        target_dchi2: float,
        p_boundary: float = None
):
    """Scan in one direction from v0, using adaptive step sizes.

    Expands outward until the chi² crosses the F-test threshold.
    When *p_boundary* is provided, the scan is clamped to that
    boundary as a hard limit.  Without a boundary the scan is
    unbounded — it only stops on threshold crossing, max_points
    exhaustion, or numerical failure.

    Step sizes start small and grow to keep the curve smooth.
    A short warmup with geometric growth is followed by
    delta-chi² guided adaptation (step sized to produce roughly
    *target_dchi2* change in chi² per step).

    Parameters
    ----------
    sign : +1 for increasing values, -1 for decreasing.
    target_dchi2 : desired chi2r increment per step (controls smoothness).
    p_boundary : float or None, optional
        Hard limit in the scan direction.  None = unbounded.
    """
    min_step = max(abs(v0) * 1e-10, 1e-14)
    xs = [v0]
    ys = [chi2r_min]
    v_cross = None

    span = max(abs(v0) * 0.02, 1e-4, min_step * 10.0)

    segment_points = max(int(max_points), 25)
    max_expansions = 1 if p_boundary is not None else 10
    previous_x = v0
    previous_y = chi2r_min

    for expansion in range(max_expansions):
        if p_boundary is not None:
            end_x = float(p_boundary)
        else:
            end_x = v0 + sign * span * (2.0 ** expansion)

        if sign < 0 and end_x < 0.0 <= v0:
            end_x = 0.0
        if abs(end_x - previous_x) < EPS:
            break

        values = np.linspace(previous_x, end_x, segment_points + 1)[1:]
        for v_next in values:
            try:
                chi2_next = _eval_scan_point(
                    fit, parameter, float(v_next), initial_parameter_values, is_fixed
                )
            except Exception:
                return xs, ys, v_cross
            if not np.isfinite(chi2_next):
                return xs, ys, v_cross

            crossing = _interpolate_threshold_crossing(
                previous_x, previous_y, float(v_next), float(chi2_next), threshold
            )
            if crossing is not None:
                xs.append(float(crossing))
                ys.append(float(threshold))
                v_cross = float(crossing)
                return xs, ys, v_cross

            xs.append(float(v_next))
            ys.append(float(chi2_next))
            previous_x = float(v_next)
            previous_y = float(chi2_next)

        if p_boundary is not None:
            break

    return xs, ys, v_cross


def adaptive_scan_parameter(
        fit: Fit,
        parameter_name: str,
        scan_range: typing.Tuple[float, float] = (None, None),
        p_value: float = 0.99,
        max_points_per_side: int = 50
) -> typing.Dict:
    """Adaptive F-test-driven chi² scan for a parameter.

    Starts from the best-fit value and expands outward in both
    directions until the chi² crosses the F-test threshold for the
    given p-value. Step sizes are adjusted dynamically so that the
    resulting curve is smooth (roughly equal chi² increments per
    step). A secant (Newton-like) step refines the exact threshold
    crossing.

    Parameters
    ----------
    fit : Fit
        The fit object.
    parameter_name : str
        Name of the parameter to scan.
    scan_range : tuple, optional
        (p_min, p_max) absolute range.  When both entries are finite
        the scan is clamped to these limits.  Pass (None, None)
        (default) for an unbounded scan that expands until the
        threshold is crossed.
    p_value : float, optional
        F-test p-value threshold (default 0.99).
    max_points_per_side : int, optional
        Soft cap on the number of evaluations per direction (default 50).

    Returns
    -------
    dict with keys:
        chi2r, parameter_values, parameter_names,
        threshold, chi2r_min, v0, p_value, crossings, nu, n_extra_params
    """
    initial_parameter_values = fit.model.parameter_values

    varied_parameter = fit.model.parameters_all_dict[parameter_name]
    is_fixed = varied_parameter.fixed
    v0 = varied_parameter.value

    chi2r_min = fit.chi2r
    n_points = fit.model.n_points
    n_free = fit.model.n_free
    nu = n_points - n_free - 1

    threshold = cs.core.math.statistics.chi2_threshold(
        chi2r_min, n_extra_params=1, nu=nu, p_value=p_value
    )

    # Desired chi² increment per step for smoothness. The UI point count is a
    # density request, not only a stop limit.
    target_dchi2 = max((threshold - chi2r_min) / max(float(max_points_per_side), 10.0), 1e-10)

    # Parse optional scan_range: (None, None) means unbounded
    p_min, p_max = scan_range if scan_range is not None else (None, None)
    if bool(getattr(varied_parameter, 'bounds_on', False)) and isinstance(getattr(varied_parameter, 'bounds', None), (tuple, list)):
        bounds = getattr(varied_parameter, 'bounds', None)
        if len(bounds) == 2:
            p_min = _finite_parameter_bound(bounds[0]) if p_min is None else p_min
            p_max = _finite_parameter_bound(bounds[1]) if p_max is None else p_max

    if p_min is None and v0 >= 0:
        p_min = max(0.0, v0 * 1e-6)

    neg_boundary = p_min if p_min is not None and p_min < v0 else None
    pos_boundary = p_max if p_max is not None and p_max > v0 else None

    # Scan in both directions
    neg_xs, neg_ys, neg_cross = _scan_one_side(
        fit, varied_parameter, v0, chi2r_min, -1, threshold,
        initial_parameter_values, is_fixed,
        max_points_per_side, target_dchi2, p_boundary=neg_boundary
    )
    pos_xs, pos_ys, pos_cross = _scan_one_side(
        fit, varied_parameter, v0, chi2r_min, +1, threshold,
        initial_parameter_values, is_fixed,
        max_points_per_side, target_dchi2, p_boundary=pos_boundary
    )

    # Restore state
    varied_parameter.fixed = is_fixed
    fit.model.parameter_values = initial_parameter_values
    fit.update()

    # Merge: neg (excl v0) + v0 + pos (excl v0)
    all_x = list(reversed(neg_xs[1:])) + [v0] + pos_xs[1:]
    all_y = list(reversed(neg_ys[1:])) + [chi2r_min] + pos_ys[1:]
    neg_cross = _find_side_crossing(all_x, all_y, threshold, v0, -1)
    pos_cross = _find_side_crossing(all_x, all_y, threshold, v0, +1)

    return {
        'chi2r': np.array(all_y),
        'parameter_values': np.array(all_x),
        'parameter_names': [parameter_name],
        'threshold': threshold,
        'chi2r_min': chi2r_min,
        'v0': v0,
        'p_value': p_value,
        'crossings': (neg_cross, pos_cross),
        'nu': nu,
        'n_extra_params': 1,
    }
