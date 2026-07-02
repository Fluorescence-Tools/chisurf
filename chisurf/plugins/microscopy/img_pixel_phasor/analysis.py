"""Pure-``numpy``/``scipy`` phasor analysis (PRD-55 / PRD-56).

Qt-free and tttrlib-free phasor-space operations that consume calibrated ``g,s``
maps (as produced by
:func:`chisurf.core.fluorescence.imaging.pixel_maps.phasor_maps`) and return derived
maps, cursor masks, and reference-geometry polylines. These functions are the single
source of truth for the phasor math exposed over the ``phasor.*`` RPC namespace
(``backend/services.py``) and consumed by ndXplorer (PRD-56).

The formulas follow the PhasorPy reference (``thirdparty/phasorpy``, read-only); nothing
is imported from it and it is **not** a dependency. Angular frequency uses the MHz→ns
convention ``omega = 2*pi*f*1e-3`` so that lifetimes come out in nanoseconds when the
frequency is given in MHz.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import ndimage

__all__ = [
    "angular_frequency",
    "phasor_to_apparent_lifetime",
    "phasor_filter_median",
    "phasor_filter_gaussian",
    "phasor_component_fraction",
    "phasor_unmix",
    "mask_from_circular_cursor",
    "mask_from_elliptic_cursor",
    "pseudo_color",
    "universal_semicircle_polyline",
    "lifetime_to_phasor",
    "lifetime_tick_markers",
    "iso_lifetime_contours",
    "fret_trajectory",
    "build_overlays",
]

#: Overlay-set names understood by :func:`build_overlays`.
OVERLAY_SETS = (
    "semicircle", "lifetime_grid", "lifetime_ticks", "fret", "component_line",
    "polar_grid", "components", "cursor",
)

#: Default reference lifetimes (ns) for lifetime ticks / grid overlays.
_DEFAULT_TAUS = (0.5, 1.0, 2.0, 4.0, 8.0)

#: MHz → ns unit-conversion factor for ``omega = 2*pi*f*unit_conversion``.
_UNIT_CONVERSION = 1e-3


def angular_frequency(frequency_mhz: float) -> float:
    """Angular frequency ``omega`` (rad/ns) for a modulation frequency in MHz."""
    return 2.0 * math.pi * float(frequency_mhz) * _UNIT_CONVERSION


# --------------------------------------------------------------------------------------
# Apparent lifetime
# --------------------------------------------------------------------------------------
def phasor_to_apparent_lifetime(
    g: np.ndarray,
    s: np.ndarray,
    frequency_mhz: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Apparent phase and modulation lifetimes from phasor coordinates.

    With :math:`\omega = 2\pi f`:

    .. math::

        \tau_{\phi} = \omega^{-1}\, S / G
        \qquad
        \tau_{M} = \omega^{-1}\, \sqrt{1/(G^2 + S^2) - 1}

    Parameters
    ----------
    g, s : np.ndarray
        Real (``G``) and imaginary (``S``) phasor coordinate maps.
    frequency_mhz : float
        Modulation frequency in MHz.

    Returns
    -------
    tau_phi : np.ndarray
        Apparent phase lifetime (ns).
    tau_m : np.ndarray
        Apparent modulation lifetime (ns).

    Notes
    -----
    Coordinates outside the universal semicircle give non-physical (negative or
    ``NaN``) lifetimes; these are returned as-is rather than clipped. Division by zero
    yields ``inf`` (semicircle endpoints), matching the PhasorPy reference.
    """
    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    omega = angular_frequency(frequency_mhz)
    if omega == 0.0:
        raise ValueError("frequency_mhz must be non-zero")
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_phi = (s / g) / omega
        radius_sq = g * g + s * s
        tau_m = np.sqrt(1.0 / radius_sq - 1.0) / omega
    return tau_phi, tau_m


# --------------------------------------------------------------------------------------
# Denoising filters
# --------------------------------------------------------------------------------------
def _nan_safe_filter(a: np.ndarray, filt) -> np.ndarray:
    """Apply ``filt`` while ignoring ``NaN`` (normalized-convolution style)."""
    a = np.asarray(a, dtype=np.float64)
    nan = np.isnan(a)
    if not nan.any():
        return filt(a)
    filled = np.where(nan, 0.0, a)
    weight = (~nan).astype(np.float64)
    num = filt(filled)
    den = filt(weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[den == 0.0] = np.nan
    return out


def phasor_filter_median(
    g: np.ndarray,
    s: np.ndarray,
    size: int = 3,
    repeat: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-safe median filter applied ``repeat`` times to each of ``g`` and ``s``.

    Parameters
    ----------
    g, s : np.ndarray
        Phasor coordinate maps.
    size : int
        Square median-filter footprint size.
    repeat : int
        Number of successive median passes.

    Returns
    -------
    tuple of np.ndarray
        Filtered ``(g, s)``. The intensity / mean map is untouched (not an input).
    """
    size = int(size)
    repeat = int(repeat)

    def one(a: np.ndarray) -> np.ndarray:
        out = np.asarray(a, dtype=np.float64)
        for _ in range(max(0, repeat)):
            out = _nan_safe_filter(out, lambda x: ndimage.median_filter(x, size=size))
        return out

    return one(g), one(s)


def phasor_filter_gaussian(
    g: np.ndarray,
    s: np.ndarray,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-safe Gaussian filter of ``g`` and ``s`` (sibling of the median filter)."""
    sigma = float(sigma)

    def one(a: np.ndarray) -> np.ndarray:
        return _nan_safe_filter(a, lambda x: ndimage.gaussian_filter(x, sigma=sigma))

    return one(g), one(s)


# --------------------------------------------------------------------------------------
# Component fraction / unmixing
# --------------------------------------------------------------------------------------
def phasor_component_fraction(
    g: np.ndarray,
    s: np.ndarray,
    c1: Sequence[float],
    c2: Sequence[float],
) -> np.ndarray:
    r"""Return the fraction of component 1 by projection onto the two-component line.

    Each phasor is projected onto the line between ``c1=(g1,s1)`` and ``c2=(g2,s2)``;
    the normalized projection (clipped to ``[0, 1]``) is the fraction of ``c1``.

    .. math::

        f = \frac{(g - g_2)(g_1 - g_2) + (s - s_2)(s_1 - s_2)}
                 {(g_1 - g_2)^2 + (s_1 - s_2)^2}
    """
    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    g1, s1 = float(c1[0]), float(c1[1])
    g2, s2 = float(c2[0]), float(c2[1])
    denom = (g1 - g2) ** 2 + (s1 - s2) ** 2
    if denom == 0.0:
        raise ValueError("component endpoints c1 and c2 must differ")
    f = ((g - g2) * (g1 - g2) + (s - s2) * (s1 - s2)) / denom
    return np.clip(f, 0.0, 1.0)


def phasor_unmix(
    g: np.ndarray,
    s: np.ndarray,
    components: Sequence[Sequence[float]],
) -> list[np.ndarray]:
    r"""Return non-negative, sum-to-one fractions for ``N >= 2`` component phasors.

    Solves, per pixel, the constrained least-squares system

    .. math:: \min_f \lVert A f - b \rVert,\quad f \ge 0,\ \sum_i f_i = 1

    where the rows of ``A`` are the component ``g``, ``s`` and a sum-to-one row (the
    latter weighted heavily), via :func:`scipy.optimize.nnls`. For ``N == 2`` this is
    equivalent to :func:`phasor_component_fraction`.

    Parameters
    ----------
    g, s : np.ndarray
        Phasor coordinate maps.
    components : sequence of (g_i, s_i)
        Endpoint phasor of each of the ``N`` components.

    Returns
    -------
    list of np.ndarray
        One fraction map per component, each with the shape of ``g``. ``NaN`` pixels
        (in ``g`` or ``s``) yield ``NaN`` fractions.
    """
    from scipy.optimize import nnls

    comps = np.asarray(components, dtype=np.float64)
    if comps.ndim != 2 or comps.shape[1] != 2:
        raise ValueError("components must have shape (N, 2)")
    n = comps.shape[0]
    if n < 2:
        raise ValueError("phasor_unmix requires at least 2 components")

    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    shape = g.shape
    gf = g.ravel()
    sf = s.ravel()

    sum_weight = 1e3  # enforce sum-to-one strongly relative to the g/s residuals
    a = np.vstack([comps[:, 0], comps[:, 1], np.full(n, sum_weight)])  # (3, N)

    fractions = np.full((n, gf.size), np.nan, dtype=np.float64)
    valid = np.isfinite(gf) & np.isfinite(sf)
    for idx in np.nonzero(valid)[0]:
        b = np.array([gf[idx], sf[idx], sum_weight])
        sol, _ = nnls(a, b)
        total = sol.sum()
        if total > 0:
            sol = sol / total
        fractions[:, idx] = sol
    return [fractions[i].reshape(shape) for i in range(n)]


# --------------------------------------------------------------------------------------
# Cursors / pseudo-color
# --------------------------------------------------------------------------------------
def mask_from_circular_cursor(
    g: np.ndarray,
    s: np.ndarray,
    center: Sequence[float],
    radius: float,
) -> np.ndarray:
    """Boolean mask of pixels whose ``(g, s)`` fall within a circular cursor."""
    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    cg, cs = float(center[0]), float(center[1])
    return (g - cg) ** 2 + (s - cs) ** 2 <= float(radius) ** 2


def mask_from_elliptic_cursor(
    g: np.ndarray,
    s: np.ndarray,
    center: Sequence[float],
    radii: Sequence[float],
    angle: float = 0.0,
) -> np.ndarray:
    """Boolean mask of pixels within an elliptic cursor (``angle`` in radians)."""
    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    cg, cs = float(center[0]), float(center[1])
    rg, rs = float(radii[0]), float(radii[1])
    if rg == 0.0 or rs == 0.0:
        raise ValueError("elliptic cursor radii must be non-zero")
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dg, ds = g - cg, s - cs
    u = dg * cos_a + ds * sin_a
    v = -dg * sin_a + ds * cos_a
    return (u / rg) ** 2 + (v / rs) ** 2 <= 1.0


def pseudo_color(
    masks: Sequence[np.ndarray],
    colors: Sequence[Sequence[float]] | None = None,
    intensity: np.ndarray | None = None,
) -> np.ndarray:
    """Colorize a stack of boolean masks into an RGB label image.

    Parameters
    ----------
    masks : sequence of np.ndarray
        Boolean masks sharing a common shape. Later masks overwrite earlier ones where
        they overlap.
    colors : sequence of (r, g, b), optional
        One RGB triple (in ``[0, 1]``) per mask. Defaults to a simple categorical
        palette cycled as needed.
    intensity : np.ndarray, optional
        If given, the RGB output is modulated by this normalized intensity map so
        selections fade with photon count.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` float RGB image in ``[0, 1]``.
    """
    masks = [np.asarray(m, dtype=bool) for m in masks]
    if not masks:
        raise ValueError("pseudo_color requires at least one mask")
    shape = masks[0].shape
    if colors is None:
        colors = _DEFAULT_PALETTE
    rgb = np.zeros(shape + (3,), dtype=np.float64)
    for i, m in enumerate(masks):
        color = np.asarray(colors[i % len(colors)], dtype=np.float64)
        rgb[m] = color
    if intensity is not None:
        inten = np.asarray(intensity, dtype=np.float64)
        finite = inten[np.isfinite(inten)]
        vmax = finite.max() if finite.size and finite.max() > 0 else 1.0
        scale = np.clip(np.nan_to_num(inten) / vmax, 0.0, 1.0)
        rgb *= scale[..., None]
    return rgb


_DEFAULT_PALETTE = (
    (0.90, 0.10, 0.10),
    (0.10, 0.60, 0.90),
    (0.20, 0.80, 0.20),
    (0.95, 0.75, 0.10),
    (0.60, 0.30, 0.80),
    (0.30, 0.85, 0.80),
)


# --------------------------------------------------------------------------------------
# Reference geometry (overlay polylines)
# --------------------------------------------------------------------------------------
def universal_semicircle_polyline(n_points: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return the universal semicircle (center ``(0.5, 0)``, radius ``0.5``) as ``(x, y)``."""
    theta = np.linspace(0.0, math.pi, int(n_points))
    return 0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta)


def lifetime_to_phasor(
    tau: np.ndarray | float,
    frequency_mhz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Phasor coordinate(s) of a single-exponential lifetime ``tau`` (ns)."""
    omega = angular_frequency(frequency_mhz)
    wt = omega * np.asarray(tau, dtype=np.float64)
    g = 1.0 / (1.0 + wt * wt)
    s = wt / (1.0 + wt * wt)
    return g, s


def lifetime_tick_markers(
    frequency_mhz: float,
    taus: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Semicircle tick coordinates for a set of reference lifetimes (ns)."""
    g, s = lifetime_to_phasor(np.asarray(taus, dtype=np.float64), frequency_mhz)
    return g, s


def iso_lifetime_contours(
    frequency_mhz: float,
    taus: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
    n_points: int = 64,
) -> list[dict]:
    """Iso-lifetime grid lines as labelled polylines.

    For each reference lifetime ``tau`` this returns two polylines:

    - an **iso-phase** radial line from the origin through the phasor point (all points
      with apparent phase lifetime ``tau``), and
    - an **iso-modulation** arc at constant modulation radius ``M = 1/sqrt(1+(wt)^2)``
      from the ``G`` axis up to the phase angle (all points with apparent modulation
      lifetime ``tau``).

    Returns
    -------
    list of dict
        Each entry is ``{"name", "kind", "tau", "x", "y"}`` where ``kind`` is
        ``"iso_phase"`` or ``"iso_modulation"`` and ``x``/``y`` are lists.
    """
    omega = angular_frequency(frequency_mhz)
    out: list[dict] = []
    for tau in taus:
        wt = omega * float(tau)
        phi = math.atan2(wt, 1.0)  # phase angle of the phasor point
        modulation = 1.0 / math.sqrt(1.0 + wt * wt)
        # iso-phase: radial line from origin to the semicircle at angle phi
        r = np.linspace(0.0, 1.0, int(n_points))
        out.append(
            {
                "name": f"tau_phi={tau:g} ns",
                "kind": "iso_phase",
                "tau": float(tau),
                "x": (r * math.cos(phi)).tolist(),
                "y": (r * math.sin(phi)).tolist(),
            }
        )
        # iso-modulation: arc of radius `modulation` from angle 0 to phi
        ang = np.linspace(0.0, phi, int(n_points))
        out.append(
            {
                "name": f"tau_m={tau:g} ns",
                "kind": "iso_modulation",
                "tau": float(tau),
                "x": (modulation * np.cos(ang)).tolist(),
                "y": (modulation * np.sin(ang)).tolist(),
            }
        )
    return out


def fret_trajectory(
    frequency_mhz: float,
    tau_d0: float,
    e_range: Sequence[float] | None = None,
    n_points: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Quenched-donor FRET trajectory on the semicircle.

    Traces the phasor of a mono-exponential donor whose lifetime shortens with FRET
    efficiency ``E`` as ``tau_DA = tau_D0 * (1 - E)``.

    Parameters
    ----------
    frequency_mhz : float
        Modulation frequency (MHz).
    tau_d0 : float
        Donor-only lifetime (ns).
    e_range : sequence of float, optional
        ``(E_min, E_max)`` FRET-efficiency range; defaults to ``(0.0, 0.99)``.
    n_points : int
        Number of samples along the trajectory.

    Returns
    -------
    tuple of np.ndarray
        ``(x, y)`` phasor coordinates along the trajectory.
    """
    e_min, e_max = (0.0, 0.99) if e_range is None else (float(e_range[0]), float(e_range[1]))
    e = np.linspace(e_min, e_max, int(n_points))
    tau_da = float(tau_d0) * (1.0 - e)
    return lifetime_to_phasor(tau_da, frequency_mhz)


def polar_grid_polylines(
    radii: Sequence[float] | None = None,
    angles: Sequence[float] | int | None = None,
    n_points: int = 128,
) -> list[dict]:
    """Polar coordinate grid (concentric circles + radial spokes) as polylines.

    Ported from phasorpy's ``PhasorPlot.polar_grid`` (geometry only). Circles are
    centred at the origin; the unit circle (radius ``1``) is flagged ``major`` in
    the returned dict so a renderer can draw it more boldly. Radial spokes run
    from the origin to the unit circle at the requested ``angles``.

    Parameters
    ----------
    radii : sequence of float, optional
        Circle radii in ``(0, 1]``; defaults to ``(1/3, 2/3, 1.0)``.
    angles : sequence of float or int, optional
        Spoke angles in radians, or an integer count of equidistant spokes over
        ``[0, 2*pi)``; defaults to ``12`` equidistant spokes.
    n_points : int
        Vertices per circle.

    Returns
    -------
    list of dict
        Each entry is ``{"name", "kind": "curve", "x", "y", "major": bool}``.
    """
    radii_list = [1.0 / 3.0, 2.0 / 3.0, 1.0] if radii is None else [float(r) for r in radii]
    if angles is None:
        angle_list = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False).tolist()
    elif isinstance(angles, int):
        angle_list = np.linspace(0.0, 2.0 * math.pi, int(angles), endpoint=False).tolist()
    else:
        angle_list = [float(a) for a in angles]

    out: list[dict] = []
    theta = np.linspace(0.0, 2.0 * math.pi, int(n_points))
    for r in radii_list:
        if r <= 1e-3 or r > 1.0 + 1e-9:
            continue
        major = abs(r - 1.0) < 1e-3
        out.append(
            {
                "name": "unit circle" if major else f"r={r:g}",
                "kind": "curve",
                "x": (r * np.cos(theta)).tolist(),
                "y": (r * np.sin(theta)).tolist(),
                "major": bool(major),
            }
        )
    for a in angle_list:
        out.append(
            {
                "name": f"angle={math.degrees(a):g}",
                "kind": "curve",
                "x": [0.0, math.cos(a)],
                "y": [0.0, math.sin(a)],
                "major": False,
            }
        )
    return out


def _sort_by_angle(pts: np.ndarray) -> np.ndarray:
    """Return ``pts`` (shape ``(n, 2)``) ordered counter-clockwise around their centroid."""
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(ang)]


def component_mixing(
    components: Sequence[Sequence[float]],
    fractions: Sequence[float] | None = None,
) -> list[dict]:
    """Mixing geometry for ``n`` component phasors (ported from ``PhasorPlot.components``).

    Two modes, matching phasorpy:

    - **No fractions** — outline the polygon of all possible linear combinations
      (the mixing region), returned as a closed ``curve`` plus a ``scatter`` of the
      component vertices.
    - **With fractions** — draw a line from each component to the fraction-weighted
      average (the mixture point), plus a ``scatter`` marking components and mixture.

    Parameters
    ----------
    components : sequence of (g, s)
        Component phasor coordinates (``n >= 2``).
    fractions : sequence of float, optional
        Per-component weights; if given, they define the weighted mixture point.

    Returns
    -------
    list of dict
        Overlay dicts (``curve`` / ``scatter``) with ``labels`` on the scatter.
    """
    pts = np.asarray(components, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError("components must have shape (n>=2, 2)")
    labels = [f"C{i + 1}" for i in range(pts.shape[0])]
    out: list[dict] = []
    if fractions is None:
        ring = _sort_by_angle(pts)
        ring = np.vstack([ring, ring[:1]])  # close the polygon
        out.append(
            {"name": "mixing region", "kind": "curve",
             "x": ring[:, 0].tolist(), "y": ring[:, 1].tolist(),
             "style": {"color": "#888888", "width": 1, "dash": True}}
        )
        out.append(
            {"name": "components", "kind": "scatter",
             "x": pts[:, 0].tolist(), "y": pts[:, 1].tolist(),
             "labels": labels, "style": {"color": "#50c0ff", "symbol": "o"}}
        )
        return out
    w = np.asarray(fractions, dtype=float)
    if w.shape[0] != pts.shape[0]:
        raise ValueError("fractions length must match number of components")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("fractions must sum to a positive value")
    mix = np.average(pts, axis=0, weights=w)
    for i, (g, s) in enumerate(pts):
        out.append(
            {"name": f"mix line {i + 1}", "kind": "curve",
             "x": [float(g), float(mix[0])], "y": [float(s), float(mix[1])],
             "style": {"color": "#888888", "width": 1}}
        )
    out.append(
        {"name": "components", "kind": "scatter",
         "x": pts[:, 0].tolist(), "y": pts[:, 1].tolist(),
         "labels": labels, "style": {"color": "#50c0ff", "symbol": "o"}}
    )
    out.append(
        {"name": "mixture", "kind": "scatter",
         "x": [float(mix[0])], "y": [float(mix[1])],
         "labels": ["mix"], "style": {"color": "#ff5050", "symbol": "x"}}
    )
    return out


def cursor_polyline(
    center: Sequence[float],
    kind: str = "circular",
    radius: float = 0.05,
    radii: Sequence[float] | None = None,
    angle: float = 0.0,
    n_points: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Outline of a gating cursor as a closed ``(x, y)`` polyline.

    Ported from phasorpy's ``PhasorPlot.cursor``. ``kind="circular"`` draws a
    circle of ``radius`` about ``center``; ``kind="elliptic"`` draws an ellipse
    with semi-axes ``radii`` rotated by ``angle`` (radians). This is the visual
    companion to :func:`mask_from_circular_cursor` / :func:`mask_from_elliptic_cursor`.
    """
    cg, cs = float(center[0]), float(center[1])
    t = np.linspace(0.0, 2.0 * math.pi, int(n_points))
    if kind == "circular":
        return cg + float(radius) * np.cos(t), cs + float(radius) * np.sin(t)
    if kind == "elliptic":
        if radii is None:
            raise ValueError("elliptic cursor requires 'radii'")
        a, b = float(radii[0]), float(radii[1])
        ex, ey = a * np.cos(t), b * np.sin(t)
        ca, sa = math.cos(float(angle)), math.sin(float(angle))
        return cg + ex * ca - ey * sa, cs + ex * sa + ey * ca
    raise ValueError(f"unknown cursor kind: {kind!r}")


def density_contours(
    density: np.ndarray,
    g_range: Sequence[float],
    s_range: Sequence[float],
    levels: int | Sequence[float] = 5,
) -> list[dict]:
    """Iso-density contour lines of a 2-D phasor histogram as polylines.

    Ported from phasorpy's ``PhasorPlot.contour``. Uses ``contourpy`` (a matplotlib
    dependency, already present) to extract the contour geometry without any GUI, so
    the result is a plain LineSet drawable by the calculator, ndXplorer or the RPC
    layer. ``density`` is indexed ``[g_bin, s_bin]`` (axis 0 = g), matching the
    phasor density image; ``g_range`` / ``s_range`` give the physical extent.

    Returns
    -------
    list of dict
        One ``{"name", "kind": "curve", "x", "y", "level"}`` per contour segment.
    """
    from contourpy import contour_generator

    arr = np.asarray(density, dtype=float)
    if arr.ndim != 2:
        raise ValueError("density must be 2-D")
    ng, ns = arr.shape
    g = np.linspace(float(g_range[0]), float(g_range[1]), ng)
    s = np.linspace(float(s_range[0]), float(s_range[1]), ns)
    # contourpy expects z[row, col] over (x=cols, y=rows); density is [g, s] so g
    # indexes columns (x) and s indexes rows (y) once transposed.
    gen = contour_generator(x=g, y=s, z=arr.T)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []
    vmax = float(finite.max())
    if isinstance(levels, int):
        level_values = np.linspace(vmax / (levels + 1), vmax, int(levels))
    else:
        level_values = np.asarray(levels, dtype=float)
    out: list[dict] = []
    for lv in level_values:
        for seg in gen.lines(float(lv)):
            seg = np.asarray(seg, dtype=float)
            if seg.shape[0] < 2:
                continue
            out.append(
                {"name": f"contour {lv:g}", "kind": "curve",
                 "x": seg[:, 0].tolist(), "y": seg[:, 1].tolist(),
                 "level": float(lv), "style": {"color": "#39ff14", "width": 1}}
            )
    return out


def build_overlays(
    frequency_mhz: float,
    harmonic: int = 1,
    sets: Sequence[str] | None = None,
    taus: Sequence[float] | None = None,
    c1: Sequence[float] | None = None,
    c2: Sequence[float] | None = None,
    tau_d0: float = 4.0,
    e_range: Sequence[float] | None = None,
    n_points: int = 256,
    components: Sequence[Sequence[float]] | None = None,
    fractions: Sequence[float] | None = None,
    cursors: Sequence[dict] | None = None,
    polar_radii: Sequence[float] | None = None,
    polar_angles: Sequence[float] | int | None = None,
) -> list[dict]:
    """Assemble reference-geometry overlays as ``[{name, kind, x, y, style}, ...]``.

    The single source of truth shared by the ``phasor.overlays`` RPC handler and the
    in-process phasor calculator. ``kind`` is ``"curve"`` (polyline), ``"scatter"``
    (markers, with optional ``labels``), or the iso-line kinds from
    :func:`iso_lifetime_contours`.
    """
    freq = float(frequency_mhz) * int(harmonic)
    wanted = list(sets) if sets is not None else ["semicircle", "lifetime_grid", "lifetime_ticks"]
    tau_list = list(taus) if taus is not None else list(_DEFAULT_TAUS)
    overlays: list[dict] = []

    if "semicircle" in wanted:
        x, y = universal_semicircle_polyline(n_points=int(n_points))
        overlays.append(
            {"name": "universal semicircle", "kind": "curve", "x": x.tolist(), "y": y.tolist(),
             "style": {"color": "w", "width": 1}}
        )
    if "lifetime_grid" in wanted:
        overlays.extend(
            {**c, "style": {"color": "#888888", "width": 1, "dash": True}}
            for c in iso_lifetime_contours(freq, taus=tau_list)
        )
    if "lifetime_ticks" in wanted:
        gx, sy = lifetime_tick_markers(freq, taus=tau_list)
        overlays.append(
            {"name": "lifetime ticks", "kind": "scatter", "x": gx.tolist(), "y": sy.tolist(),
             "labels": [f"{t:g} ns" for t in tau_list], "style": {"color": "y", "symbol": "o"}}
        )
    if "fret" in wanted:
        fx, fy = fret_trajectory(freq, tau_d0=float(tau_d0), e_range=e_range)
        overlays.append(
            {"name": "FRET trajectory", "kind": "curve", "x": fx.tolist(), "y": fy.tolist(),
             "style": {"color": "#ff5050", "width": 2}}
        )
    if "component_line" in wanted and c1 is not None and c2 is not None:
        overlays.append(
            {"name": "component line", "kind": "curve",
             "x": [float(c1[0]), float(c2[0])], "y": [float(c1[1]), float(c2[1])],
             "style": {"color": "#50c0ff", "width": 2}}
        )
    if "polar_grid" in wanted:
        for g in polar_grid_polylines(radii=polar_radii, angles=polar_angles):
            major = g.pop("major", False)
            g["style"] = {"color": "#666666", "width": 1.5 if major else 1,
                          "dash": not major}
            overlays.append(g)
    if "components" in wanted and components is not None:
        overlays.extend(component_mixing(components, fractions=fractions))
    if "cursor" in wanted and cursors:
        for cur in cursors:
            cx, cy = cursor_polyline(
                cur["center"], kind=cur.get("kind", "circular"),
                radius=float(cur.get("radius", 0.05)), radii=cur.get("radii"),
                angle=float(cur.get("angle", 0.0)),
            )
            overlays.append(
                {"name": cur.get("name", "cursor"), "kind": "curve",
                 "x": cx.tolist(), "y": cy.tolist(),
                 "style": {"color": cur.get("color", "#ffb000"), "width": 2}}
            )
    return overlays
