from __future__ import annotations

import numpy as np

from chisurf.core.math.functions.distributions import normal_distribution


def mask_zero_photon_bins(fit, xmin: int, wres: np.ndarray) -> np.ndarray:
    """Return residuals with 2D PDA bins with zero experimental photons masked.

    Parameters
    ----------
    fit : chisurf.core.fitting.fit.Fit
        Fit providing the experimental DataCurve.
    xmin : int
        Starting index of the residual window.
    wres : numpy.ndarray
        Weighted residuals over the selected window.
    """
    try:
        if wres is None:
            return wres
        data_obj = getattr(fit, "data", None)
        y_data = getattr(data_obj, "y", None)
        if y_data is None:
            return wres
        y_arr = np.asarray(y_data, dtype=float)

        # Cache the zero-photon mask on the data object so the expensive
        # comparison y_arr > 0.0 is only performed once per dataset/size.
        try:
            nonzero_full = getattr(data_obj, "_pda_nonzero_mask", None)
        except Exception:
            nonzero_full = None
        if nonzero_full is None or getattr(nonzero_full, "size", 0) != y_arr.size:
            nonzero_full = y_arr > 0.0
            try:
                data_obj._pda_nonzero_mask = nonzero_full
            except Exception:
                pass

        n_points = wres.size
        start = int(max(0, xmin))
        stop = start + n_points
        nonzero = nonzero_full[start:stop]
        if not nonzero.size or not n_points:
            return wres
        mlen = min(nonzero.size, n_points)
        out = np.array(wres, copy=True)
        out[:mlen][~nonzero[:mlen]] = 0.0
        return out
    except Exception:
        return wres


def get_pda_distance_distribution(fit) -> list:
    """Return the P(R) distance-distribution curves for a Gaussian-distance PDA fit.

    Qt-free accessor for the data-driven (AutoForm) ``distribution`` plot. The
    first curve is the summed distribution; one curve per Gaussian component
    follows (restoring the per-component overlay of the legacy widget).

    Parameters
    ----------
    fit : chisurf.core.fitting.fit.Fit
        Fit whose ``model.distances`` provides the Gaussian components.

    Returns
    -------
    list
        ``[[p_sum, r], [p_1, r], ...]`` (each ``[y, x]``); empty if unavailable.
    """
    model = getattr(fit, "model", None)
    distances = getattr(model, "distances", None)
    if distances is None:
        return []
    try:
        dist = np.asarray(distances.distribution, dtype=float)
        if dist.ndim != 2 or dist.shape[1] == 0:
            return []
        r, p_sum = dist[0], dist[1]
        curves = [[p_sum, r]]
        means = distances.means
        sigmas = distances.sigmas
        amplitudes = distances.amplitudes
        if getattr(distances, "limited_width", False) and means.size:
            sigmas = (sigmas / 100.0) * means
        for mean, sigma, amp in zip(means, sigmas, amplitudes):
            if sigma <= 0.0 or amp <= 0.0:
                continue
            y = amp * normal_distribution(x=r, loc=float(mean), scale=float(sigma), norm=False)
            curves.append([y, r])
        return curves
    except Exception:
        return []


def get_pda_residual_image(fit_group, weighted: bool = True):
    """Return a 2D S1S2 residual image ``(image, x_axis, y_axis)`` for a PDA fit.

    Qt-free accessor for the data-driven (AutoForm) ``residual2d`` plot. Compares
    the experimental S1S2 histogram (``fit.data.pda['s1s2']``) with the model
    S1S2 matrix (``fit.model.pda.s1s2``). With ``weighted=True`` the residual is
    counting-noise weighted, ``(data - model) / sqrt(max(data, 1))``.

    Parameters
    ----------
    fit_group : object
        A fit group (with ``selected_fit``) or a plain fit.
    weighted : bool
        Whether to weight the residual by counting shot noise.

    Returns
    -------
    tuple
        ``(image, x_axis, y_axis)`` or ``(None, None, None)`` if unavailable.
    """
    fit = getattr(fit_group, "selected_fit", fit_group)
    data_pda = getattr(getattr(fit, "data", None), "pda", None)
    model_obj = getattr(getattr(fit, "model", None), "pda", None)
    if data_pda is None or model_obj is None:
        return None, None, None
    try:
        data_2d = np.asarray(data_pda.get("s1s2"), dtype=float)
        model_2d = np.asarray(getattr(model_obj, "s1s2"), dtype=float)
    except Exception:
        return None, None, None
    if data_2d.ndim != 2 or model_2d.ndim != 2:
        return None, None, None

    n0 = min(data_2d.shape[0], model_2d.shape[0])
    n1 = min(data_2d.shape[1], model_2d.shape[1])
    d = data_2d[:n0, :n1]
    m = model_2d[:n0, :n1]
    if weighted:
        img = (d - m) / np.sqrt(np.maximum(d, 1.0))
    else:
        img = d - m
    return img, np.arange(n1, dtype=float), np.arange(n0, dtype=float)


def apply_lightpath_to_nuisance(
    nuisance,
    lightpath_result: dict,
    donor: str,
    acceptor: str,
    green_detector: str,
    red_detector: str,
    green_laser: str | None = None,
) -> None:
    """Feed a light-path simulation result into a PDA FRET nuisance group.

    Bridge between the light-path simulator plugin and PDA models. Accepts
    either the full result of
    ``chisurf.plugins.core.lightpath_simulator.core.workflow.simulate_lightpath``
    (which nests the matrices under ``"crosstalk_matrices"``) or a bare
    ``crosstalk_matrices`` dict, and delegates to
    :meth:`PdaFretNuisance.apply_lightpath_matrices`.

    Parameters
    ----------
    nuisance : PdaFretNuisance
        Target nuisance group (updated in place).
    lightpath_result : dict
        Light-path simulation output or its ``crosstalk_matrices`` sub-dict.
    donor, acceptor : str
        Dye labels.
    green_detector, red_detector : str
        Detector labels for the green/red channels.
    green_laser : str, optional
        Donor-excitation laser label.
    """
    matrices = lightpath_result
    if isinstance(lightpath_result, dict) and "crosstalk_matrices" in lightpath_result:
        matrices = lightpath_result["crosstalk_matrices"]
    nuisance.apply_lightpath_matrices(
        matrices,
        donor=donor,
        acceptor=acceptor,
        green_detector=green_detector,
        red_detector=red_detector,
        green_laser=green_laser,
    )


def green_probability_from_efficiency(E, nuisance) -> np.ndarray:
    """Return the per-photon green (channel-1) probability for FRET efficiency ``E``.

    Uses the same excitation/emission/crosstalk description as
    :class:`~chisurf.core.models.pda.pdagauss.PdaGaussianDistanceModel`: absolute
    excitation probabilities (ExDG/ExAG), per-channel detector efficiencies
    (gG/gR), the 2x2 emission-detection crosstalk matrix (cGD/cGA/cRD/cRA) and
    the donor/acceptor quantum yields (QYD/QYA).

    Parameters
    ----------
    E : array_like
        FRET efficiency (or grid of efficiencies).
    nuisance : PdaFretNuisance
        Nuisance group supplying the correction parameters.

    Returns
    -------
    numpy.ndarray
        Probability that a photon is detected in the green channel.
    """
    eps = 1e-12
    E = np.clip(np.asarray(E, dtype=float), eps, 1.0 - eps)
    n = nuisance
    ExDG = float(getattr(n, "ExDG", 1.0))
    ExAG = float(getattr(n, "ExAG", 0.0))
    gG = float(getattr(n, "gG", 1.0))
    gR = float(getattr(n, "gR", 1.0))
    cGD = float(getattr(n, "cGD", 1.0))
    cGA = float(getattr(n, "cGA", 0.0))
    cRD = float(getattr(n, "cRD", 0.0))
    cRA = float(getattr(n, "cRA", 1.0))
    QYD = float(getattr(n, "QYD", 1.0))
    QYA = float(getattr(n, "QYA", 1.0))
    S_DQ = QYD * ExDG * (1.0 - E)
    S_AQ = QYA * (ExDG * E + ExAG)
    G = gG * (cGD * S_DQ + cGA * S_AQ)
    R = gR * (cRD * S_DQ + cRA * S_AQ)
    denom = G + R
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0.0, G / denom, 0.5)


#: Named 1D PDA histogram axes. Each callback receives the tttrlib S1S2
#: counts as (ch1, ch2) = (red, green) and returns the scalar plotted on the
#: x-axis. Keeping these as named strings (not lambdas) lets the distribution
#: plot be authored declaratively in ``*.view.json`` (PRD-38).
_PDA_HISTOGRAM_AXES = {
    # Red fraction S1/(S0+S1) with S0=green, S1=red.
    "S1/(S0+S1)": lambda ch1, ch2: ch1 / max(1, ch1 + ch2),
    # Green/red ratio S0/S1.
    "S0/S1": lambda ch1, ch2: ch2 / max(1, ch1),
}


def get_pda_distribution(fit, kw_hist: dict | None = None) -> list:
    """Return data/model/residual 1D-histogram curves for a PDA fit.

    Qt-free accessor used by the data-driven (AutoForm) ``distribution`` plot.
    ``kw_hist`` carries the usual ``tttrlib`` histogram settings plus a
    ``histogram`` key naming the x-axis (see :data:`_PDA_HISTOGRAM_AXES`), so
    the whole plot stays authorable in JSON (no GUI lambdas).

    Parameters
    ----------
    fit : chisurf.core.fitting.fit.Fit
        Fit whose ``model.pda`` and ``data.pda`` provide model/experimental
        S1S2 matrices.
    kw_hist : dict, optional
        Histogram settings; ``histogram`` selects the axis
        (default ``"S1/(S0+S1)"``).

    Returns
    -------
    list
        ``[[data_y, data_x], [model_y, model_x], [wres, data_x]]`` (residual
        curve omitted if shapes disagree).
    """
    model = getattr(fit, "model", None)
    pda = getattr(model, "pda", None)
    if pda is None:
        return []

    kw = dict(kw_hist or {})
    axis = kw.pop("histogram", "S1/(S0+S1)")

    if axis in ("E", "R"):
        # Corrected axes need the model's gamma and Forster radius. ch1/ch2 are
        # (red, green); PR = red / (red + green); E = PR / (PR + gamma (1 - PR)).
        gamma = float(getattr(getattr(model, "nuisance", None), "gamma", 1.0) or 1.0)
        if not np.isfinite(gamma) or gamma <= 0.0:
            gamma = 1.0
        R0 = float(getattr(getattr(model, "fret_parameters", None), "forster_radius", 52.0) or 52.0)

        def histogram_function(ch1, ch2, _g=gamma, _r0=R0, _axis=axis):
            """Return corrected FRET efficiency (or distance) for S1S2 counts."""
            total = ch1 + ch2
            pr = ch1 / total if total > 0 else 0.0
            e = pr / (pr + _g * (1.0 - pr)) if (pr + _g * (1.0 - pr)) > 0 else 0.0
            if _axis == "E":
                return e
            e = min(max(e, 1e-6), 1.0 - 1e-6)
            return _r0 * (1.0 / e - 1.0) ** (1.0 / 6.0)
    else:
        inner = _PDA_HISTOGRAM_AXES.get(axis, _PDA_HISTOGRAM_AXES["S1/(S0+S1)"])

        def histogram_function(ch1, ch2, _cb=inner):
            """Return the selected PDA axis value for tttrlib.Pda S1S2 counts."""
            return _cb(ch1, ch2)

    pda.histogram_function = histogram_function

    pda_meta = getattr(getattr(fit, "data", None), "pda", None)
    if not isinstance(pda_meta, dict):
        return []
    s1s2_experimental = pda_meta.get("s1s2")
    if s1s2_experimental is None:
        return []

    try:
        s1s2_model = np.asarray(pda.get_S1S2_matrix(), dtype=float)
        s1s2_data = np.asarray(s1s2_experimental, dtype=float)
        shp = pda_meta.get("shape")
        if shp is not None and len(shp) == 2:
            ny, nx = int(shp[0]), int(shp[1])
            s1s2_model = s1s2_model[:ny, :nx]
            s1s2_data = s1s2_data[:ny, :nx]

        model_x, model_y = pda.get_1dhistogram(s1s2=s1s2_model.flatten(), **kw)
        data_x, data_y = pda.get_1dhistogram(s1s2=s1s2_data.flatten(), **kw)
    except Exception:
        return []

    curves = [[data_y, data_x], [model_y, model_x]]
    try:
        dy = np.asarray(data_y, dtype=float)
        my = np.asarray(model_y, dtype=float)
        if dy.shape == my.shape:
            sigma = np.sqrt(np.maximum(dy, 1.0))
            curves.append([(dy - my) / sigma, data_x])
    except Exception:
        pass
    return curves


def pda_1d_residuals_from_s1s2(
        fit,
        pda_obj,
        nuisance=None,
        kw_hist: dict | None = None,
) -> np.ndarray:
    """Compute 1D PDA histogram residuals from S1S2 data.

    This is a shared implementation used by discrete and Gaussian-distance
    PDA models to build 1D residuals from the experimental S1S2 histogram
    and the tttrlib.Pda model S1S2 matrix.
    """
    pda_meta = getattr(fit.data, "pda", None)
    if not isinstance(pda_meta, dict):
        return np.zeros(0, dtype=np.float64)
    s1s2_experimental = pda_meta.get("s1s2")
    if s1s2_experimental is None:
        return np.zeros(0, dtype=np.float64)

    if kw_hist is None:
        kw_hist = {
            "x_max": 1.0,
            "x_min": 0.0,
            "log_x": False,
            "n_bins": 81,
            "n_min": 10,
        }

    def _inner(ch1, ch2):
        """Return ch2 / max(1, ch1 + ch2)."""
        return ch2 / max(1, ch1 + ch2)

    def histogram_function(ch1, ch2, _cb=_inner):
        """Callback for tttrlib.Pda histogram, swapping (red, green) to (green, red)."""
        return _cb(ch2, ch1)

    pda_obj.histogram_function = histogram_function

    try:
        s1s2_model = np.asarray(pda_obj.get_S1S2_matrix(), dtype=float)
        s1s2_data = np.asarray(s1s2_experimental, dtype=float)

        try:
            shp = pda_meta.get("shape")
            if shp is not None and len(shp) == 2:
                ny, nx = int(shp[0]), int(shp[1])
                s1s2_model = s1s2_model[:ny, :nx]
                s1s2_data = s1s2_data[:ny, :nx]
        except Exception:
            pass

        # Effective photon-number bounds used for gating (0, 0 => disabled).
        gating_nmin = 0
        gating_nmax = 0

        # Apply photon-number gating (nPh_min/nPh_max) if a nuisance group is
        # provided. This mirrors the logic used in the PDA distance model so
        # that 1D projections respect the same N-range as the 2D residuals.
        try:
            if nuisance is not None:
                row_indices = np.asarray(pda_meta.get("row_indices"), dtype=np.int64)
                col_indices = np.asarray(pda_meta.get("col_indices"), dtype=np.int64)
                if row_indices.size and col_indices.size and row_indices.size == col_indices.size:
                    pda_nmin = int(pda_meta.get("minimum_number_of_photons", 0) or 0)
                    pda_nmax = int(pda_meta.get("maximum_number_of_photons", 0) or 0)
                    try:
                        nmin_param = int(round(float(nuisance.nPh_min)))
                    except Exception:
                        nmin_param = 0
                    try:
                        nmax_param = int(round(float(nuisance.nPh_max)))
                    except Exception:
                        nmax_param = 0
                    if nmin_param != 0 or nmax_param != 0:
                        nmin = nmin_param if nmin_param > 0 else pda_nmin
                        nmax = nmax_param if nmax_param > 0 else pda_nmax
                        if nmax >= nmin:
                            gating_nmin = nmin
                            gating_nmax = nmax
                            shp2 = getattr(s1s2_data, "shape", None)
                            if shp2 is not None and len(shp2) == 2:
                                ny2, nx2 = int(shp2[0]), int(shp2[1])
                                data_obj = getattr(fit, "data", None)
                                key = (
                                    id(pda_meta),
                                    int(ny2),
                                    int(nx2),
                                    int(nmin),
                                    int(nmax),
                                )
                                mask2d = None
                                try:
                                    cache_key = getattr(data_obj, "_pda_1d_mask2d_key", None)
                                    cache_mask = getattr(data_obj, "_pda_1d_mask2d", None)
                                except Exception:
                                    cache_key = None
                                    cache_mask = None
                                if cache_mask is not None and cache_key == key:
                                    mask2d = cache_mask
                                else:
                                    mask2d = np.zeros((ny2, nx2), dtype=bool)
                                    N = row_indices + col_indices
                                    sel = (N >= nmin) & (N <= nmax)
                                    if np.any(sel):
                                        mask2d[row_indices[sel], col_indices[sel]] = True
                                    try:
                                        data_obj._pda_1d_mask2d = mask2d
                                        data_obj._pda_1d_mask2d_key = key
                                    except Exception:
                                        pass
                                if mask2d is not None:
                                    s1s2_model = np.where(mask2d, s1s2_model, 0.0)
                                    s1s2_data = np.where(mask2d, s1s2_data, 0.0)
        except Exception:
            pass

        s1s2_model = s1s2_model.flatten()
        s1s2_data = s1s2_data.flatten()

        # Experimental 1D histogram depends only on the experimental S1S2
        # matrix, histogram settings, and photon-number gating. Cache it on the
        # data object so it is not recomputed on every model evaluation.
        data_obj = getattr(fit, "data", None)
        try:
            hist_cache_key = getattr(data_obj, "_pda_1d_hist_key", None)
            hist_cache_val = getattr(data_obj, "_pda_1d_hist", None)
        except Exception:
            hist_cache_key = None
            hist_cache_val = None

        eff_nmin = int(gating_nmin)
        eff_nmax = int(gating_nmax)
        hist_key = (
            id(pda_meta),
            int(s1s2_data.size),
            eff_nmin,
            eff_nmax,
            int(kw_hist.get("n_bins", 81)),
            float(kw_hist.get("x_min", 0.0)),
            float(kw_hist.get("x_max", 1.0)),
            bool(kw_hist.get("log_x", False)),
        )

        if hist_cache_val is not None and hist_cache_key == hist_key:
            data_x, data_y = hist_cache_val
        else:
            data_x, data_y = pda_obj.get_1dhistogram(
                s1s2=s1s2_data,
                **kw_hist,
            )
            try:
                data_obj._pda_1d_hist_key = hist_key
                data_obj._pda_1d_hist = (
                    np.asarray(data_x, dtype=float),
                    np.asarray(data_y, dtype=float),
                )
            except Exception:
                pass

        # Model histogram is recomputed for each evaluation since the PDA
        # probability spectrum changes during fitting.
        model_x, model_y = pda_obj.get_1dhistogram(
            s1s2=s1s2_model,
            **kw_hist,
        )
    except Exception:
        return np.zeros(0, dtype=np.float64)

    try:
        dy = np.asarray(data_y, dtype=float)
        my = np.asarray(model_y, dtype=float)
        if dy.shape != my.shape:
            return np.zeros(0, dtype=np.float64)
        sigma = np.sqrt(np.maximum(dy, 1.0))
        wres = (dy - my) / sigma
        return wres
    except Exception:
        return np.zeros(0, dtype=np.float64)
