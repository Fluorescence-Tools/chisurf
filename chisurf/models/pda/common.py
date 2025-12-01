from __future__ import annotations

import numpy as np


def mask_zero_photon_bins(fit, xmin: int, wres: np.ndarray) -> np.ndarray:
    """Return residuals with 2D PDA bins with zero experimental photons masked.

    Parameters
    ----------
    fit : chisurf.fitting.fit.Fit
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
        n_points = wres.size
        start = int(max(0, xmin))
        stop = start + n_points
        y_slice = y_arr[start:stop]
        nonzero = y_slice > 0.0
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
        return ch2 / max(1, ch1 + ch2)

    def histogram_function(ch1, ch2, _cb=_inner):
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
                            shp = getattr(s1s2_data, "shape", None)
                            if shp is not None and len(shp) == 2:
                                ny, nx = int(shp[0]), int(shp[1])
                                mask2d = np.zeros((ny, nx), dtype=bool)
                                N = row_indices + col_indices
                                sel = (N >= nmin) & (N <= nmax)
                                if np.any(sel):
                                    mask2d[row_indices[sel], col_indices[sel]] = True
                                    s1s2_model = np.where(mask2d, s1s2_model, 0.0)
                                    s1s2_data = np.where(mask2d, s1s2_data, 0.0)
        except Exception:
            pass

        s1s2_model = s1s2_model.flatten()
        s1s2_data = s1s2_data.flatten()
        model_x, model_y = pda_obj.get_1dhistogram(
            s1s2=s1s2_model,
            **kw_hist,
        )
        data_x, data_y = pda_obj.get_1dhistogram(
            s1s2=s1s2_data,
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
