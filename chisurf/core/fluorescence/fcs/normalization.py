from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any


def resolve_total_mean_count_rate(meta: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract the total mean count rate from metadata.

    Handles both ``mean_count_rate_total`` and ``mean_count_rate`` with
    a ``per_detector`` semantic.

    Parameters
    ----------
    meta : dict or None
        Data metadata dictionary.

    Returns
    -------
    float or None
        Total mean count rate in kHz, or ``None`` if not available.
    """
    if not isinstance(meta, dict):
        return None

    total = meta.get("mean_count_rate_total")
    if total is not None:
        try:
            v = float(total)
            if np.isfinite(v):
                return v
        except Exception:
            pass

    mean_cr = meta.get("mean_count_rate")
    if mean_cr is None:
        return None

    try:
        mean_cr_val = float(mean_cr)
        if not np.isfinite(mean_cr_val):
            return None
    except Exception:
        return None

    semantics = str(meta.get("mean_count_rate_semantics", "")).strip().lower()
    if "per_detector" in semantics:
        detector_count = meta.get("detector_count")
        if detector_count is not None:
            try:
                det_count = float(detector_count)
                if det_count > 1.0 and np.isfinite(det_count):
                    return mean_cr_val * det_count
            except Exception:
                pass

    return mean_cr_val


def diffusion_reference_component(
    x: np.ndarray,
    td: float,
    s: Optional[float] = None
) -> Optional[np.ndarray]:
    """Return a 2D or 3D Gaussian diffusion component.

    Parameters
    ----------
    x : array_like
        Correlation-time axis.
    td : float
        Diffusion time.
    s : float, optional
        Structure parameter for 3D diffusion. If omitted, 2D diffusion is
        returned.

    Returns
    -------
    numpy.ndarray or None
        Diffusion component, or ``None`` if parameters are invalid.
    """
    try:
        x = np.asarray(x, dtype=float)
        td = float(td)
    except Exception:
        return None
    if not (np.isfinite(td) and td != 0.0):
        return None
    if not np.all(np.isfinite(x)):
        return None

    term = (1.0 + x / td) ** (-1.0)
    if s is None:
        return term

    try:
        s = float(s)
    except Exception:
        return None
    if not (np.isfinite(s) and s != 0.0):
        return None
    return term * (1.0 + x / (s * s * td)) ** (-0.5)


def fcs_diffusion_reference(
    x: np.ndarray,
    params: Dict[str, float],
    b_default: float = 1.0
) -> Optional[np.ndarray]:
    """Return the diffusion-only FCS reference curve including ``1/N``.

    The baseline ``b`` is intentionally *not* included. Plotting callbacks
    subtract ``b`` before dividing by this reference, so including ``b`` here
    would leave residual diffusion curvature in reference-normalized curves.

    Parameters
    ----------
    x : array_like
        Correlation-time axis.
    params : dict
        Fitted parameter values with keys 'N', 'td', 's'.
    b_default : float, optional
        Accepted for API compatibility; not used in the reference.

    Returns
    -------
    numpy.ndarray or None
        ``Gdiff`` curve without baseline, or ``None`` when required parameters
        are not available.
    """
    n = params.get("N")
    if n is None or n == 0.0:
        return None

    try:
        n = float(n)
        if not np.isfinite(n) or n == 0.0:
            return None
    except Exception:
        return None

    td = params.get("td")
    s = params.get("s")

    component = diffusion_reference_component(x, td, s=s)
    if component is None:
        return None

    return 1.0 / abs(n) * component


def normalize_fcs_curve(
    y: np.ndarray,
    reference: np.ndarray,
    b: float = 1.0
) -> np.ndarray:
    """Normalize FCS curve as ``(G - b) / Gdiff``.

    Parameters
    ----------
    y : array_like
        Curve values.
    reference : array_like
        Diffusion reference values (Gdiff).
    b : float, optional
        Baseline value to subtract.

    Returns
    -------
    numpy.ndarray
        Normalized curve values.
    """
    y = np.asarray(y, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return (y - b) / reference


def compute_cpm(
    mean_count_rate_total: float,
    N: float
) -> Optional[float]:
    """Compute CPM (counts per molecule) from total count rate and N.

    Parameters
    ----------
    mean_count_rate_total : float
        Total mean count rate in kHz.
    N : float
        Fitted particle number.

    Returns
    -------
    float or None
        CPM value, or None if inputs are invalid.
    """
    if not (N > 0.0 and np.isfinite(N) and np.isfinite(mean_count_rate_total)):
        return None
    return mean_count_rate_total / N


def compute_cpm_all(
    mean_count_rate_total: float,
    N: float,
    bunch_params: Dict[str, float]
) -> Optional[float]:
    """Compute CPM for all molecules (including dark) from bunching parameters.

    Parameters
    ----------
    mean_count_rate_total : float
        Total mean count rate in kHz.
    N : float
        Fitted particle number (bright molecules).
    bunch_params : dict
        Dictionary of bunching parameters (keys starting with 'ba').

    Returns
    -------
    float or None
        CPM for all molecules, or None if inputs are invalid.
    """
    if not (N > 0.0 and np.isfinite(N) and np.isfinite(mean_count_rate_total)):
        return None

    bunch_sum = 0.0
    for name, v in bunch_params.items():
        if not name.startswith("ba"):
            continue
        try:
            v = float(v)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        v = abs(v)
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        bunch_sum += v

    bright_fraction = 1.0 - bunch_sum
    if bright_fraction <= 0.0 or not np.isfinite(bright_fraction):
        return None

    N_all = N / bright_fraction
    if not (N_all > 0.0 and np.isfinite(N_all)):
        return None

    return mean_count_rate_total / N_all