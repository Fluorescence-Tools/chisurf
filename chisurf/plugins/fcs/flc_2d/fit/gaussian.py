"""Multi-Gaussian peak fit of a recovered lifetime distribution.

Port of ``TK_FitF_GaussianMulti`` / ``TK_MyMain_Fit_GaussianMulti``: after the MEM/ILT
returns a smooth lifetime distribution ``A(tau)``, this fits a sum of Gaussians to extract
discrete lifetime components (peak position ``x0``, width ``dx``, amplitude ``A``)::

    f(x) = y0 + sum_k  A_k * exp(-((x - x0_k) / dx_k)^2)

The MATLAB code optimizes with repeated ``fminsearch``; here it is a single bounded
least-squares (``scipy.optimize.least_squares``), which is faster and more robust, with the
same parameter layout (``y0`` then ``A, x0, dx`` per component). Components can be seeded
automatically from the largest peaks of the distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GaussianComponent", "GaussianFitResult", "fit_gaussian_multi"]


@dataclass
class GaussianComponent:
    """One fitted Gaussian peak."""

    amplitude: float  # A
    center: float  # x0 (lifetime, ns)
    width: float  # dx (1/e half-width in the exp(-((x-x0)/dx)^2) convention)


@dataclass
class GaussianFitResult:
    """Result of a multi-Gaussian fit to a lifetime distribution."""

    offset: float  # y0
    components: list[GaussianComponent]
    model: np.ndarray  # (n_data,) fitted curve
    sse: float  # mean squared error (matches the MATLAB normalization)

    @property
    def centers(self) -> np.ndarray:
        """Peak centers (ns), sorted ascending."""
        return np.array(sorted(c.center for c in self.components))


def _model(params: np.ndarray, x: np.ndarray, n_comp: int) -> np.ndarray:
    y = np.full_like(x, params[0])
    for k in range(n_comp):
        amp, x0, dx = params[1 + 3 * k : 4 + 3 * k]
        y = y + amp * np.exp(-(((x - x0) / dx) ** 2))
    return y


def _seed_components(x: np.ndarray, y: np.ndarray, n_comp: int) -> list[tuple[float, float, float]]:
    """Pick ``n_comp`` initial (A, x0, dx) guesses from the strongest local maxima."""
    peaks = []
    for i in range(y.size):
        left = y[i - 1] if i > 0 else -np.inf
        right = y[i + 1] if i < y.size - 1 else -np.inf
        if y[i] >= left and y[i] >= right and y[i] > 0:
            peaks.append(i)
    peaks.sort(key=lambda i: y[i], reverse=True)
    span = float(x.max() - x.min()) or 1.0
    default_dx = span / (4 * n_comp)
    seeds: list[tuple[float, float, float]] = []
    for j in range(n_comp):
        if j < len(peaks):
            i = peaks[j]
            seeds.append((float(y[i]), float(x[i]), default_dx))
        else:
            seeds.append(
                (float(y.max()) * 0.1, float(x.min() + (j + 0.5) * span / n_comp), default_dx)
            )
    seeds.sort(key=lambda s: s[1])  # ascending center
    return seeds


def fit_gaussian_multi(
    x: np.ndarray,
    y: np.ndarray,
    n_components: int,
    *,
    initial: np.ndarray | None = None,
    fit_offset: bool = False,
) -> GaussianFitResult:
    """Fit ``y0 + sum_k A_k exp(-((x - x0_k)/dx_k)^2)`` to ``(x, y)``.

    Parameters
    ----------
    x, y
        Lifetime axis (ns) and distribution values (e.g. an ILT/MEM result).
    n_components
        Number of Gaussian peaks to fit.
    initial
        Optional flat parameter vector ``[y0, A1, x0_1, dx_1, A2, ...]``. When omitted the
        peaks are seeded from the largest local maxima of ``y``.
    fit_offset
        Fit the baseline ``y0`` (otherwise held at 0, matching the MATLAB default
        ``Fix1orNot0(1) == 1``).
    """
    from scipy.optimize import least_squares

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    if initial is None:
        p0 = [0.0]
        for amp, x0, dx in _seed_components(x, y, n_components):
            p0 += [amp, x0, dx]
        p0 = np.array(p0, dtype=float)
    else:
        p0 = np.asarray(initial, dtype=float)
        if p0.size != 1 + 3 * n_components:
            raise ValueError("initial must have length 1 + 3*n_components")

    span = float(x.max() - x.min()) or 1.0
    y0_fixed = float(p0[0])

    # Optimize y0 only when requested; otherwise fit the 3*n_comp peak params and hold y0.
    if fit_offset:
        var0 = p0
        lo = np.full(var0.size, -np.inf)
        hi = np.full(var0.size, np.inf)
        off = 1
    else:
        var0 = p0[1:]
        lo = np.full(var0.size, -np.inf)
        hi = np.full(var0.size, np.inf)
        off = 0
    x_lo, x_hi = float(x.min()), float(x.max())
    for k in range(n_components):
        lo[off + 3 * k] = 0.0  # A >= 0
        lo[off + 1 + 3 * k] = x_lo  # x0 within the data range
        hi[off + 1 + 3 * k] = x_hi
        lo[off + 2 + 3 * k] = span * 1e-3  # dx > 0
        hi[off + 2 + 3 * k] = span  # dx <= full span
    # keep seeds strictly inside the bounds
    var0 = np.clip(var0, lo + 1e-9, hi - 1e-9)

    def _resid(v):
        full = v if fit_offset else np.concatenate([[y0_fixed], v])
        return _model(full, x, n_components) - y

    res = least_squares(_resid, var0, bounds=(lo, hi), method="trf", max_nfev=5000)
    params = res.x if fit_offset else np.concatenate([[y0_fixed], res.x])
    model = _model(params, x, n_components)
    sse = float(np.sum((model - y) ** 2) / y.size)
    comps = [
        GaussianComponent(
            float(params[1 + 3 * k]), float(params[2 + 3 * k]), float(params[3 + 3 * k])
        )
        for k in range(n_components)
    ]
    return GaussianFitResult(float(params[0]), comps, model, sse)
