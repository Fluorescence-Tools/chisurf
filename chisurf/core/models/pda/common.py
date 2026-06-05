from __future__ import annotations

import numpy as np


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
