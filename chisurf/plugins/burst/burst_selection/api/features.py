"""Burst feature extraction and GMM fitting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .models import GMMSettings

_FEATURE_COLUMNS = [
    "nphotons",
    "duration",
    "brightness",
    "interphoton",
    "fret",
]


def _feature_column(frame: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    """Return a feature column using API or ChiSurf display names."""
    if preferred in frame:
        return frame[preferred]
    return frame[fallback]


def extract_features(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Extract numerical burst features from burst summary DataFrames.

    Parameters
    ----------
    frames : sequence of pandas.DataFrame
        Burst summary tables.

    Returns
    -------
    pandas.DataFrame
        Feature table with one row per burst.
    """
    if not frames:
        return pd.DataFrame(columns=_FEATURE_COLUMNS)

    records: list[dict[str, Any]] = []
    for frame in frames:
        if frame.empty:
            continue
        n_photons = _feature_column(frame, "nphotons", "Number of Photons").to_numpy(dtype=float)
        duration = _feature_column(frame, "duration", "Duration (ms)").to_numpy(dtype=float)
        brightness = n_photons / np.maximum(duration, np.finfo(float).eps)
        interphoton = np.divide(
            duration,
            np.maximum(n_photons - 1.0, 1.0),
            out=np.zeros_like(duration, dtype=float),
            where=n_photons > 1.0,
        )
        if "fret" in frame:
            fret = frame["fret"].to_numpy(dtype=float)
        elif "Proximity Ratio" in frame:
            fret = frame["Proximity Ratio"].to_numpy(dtype=float)
        else:
            fret = np.zeros(len(frame))
        records.extend(
            {
                "nphotons": n,
                "duration": d,
                "brightness": b,
                "interphoton": i,
                "fret": f,
            }
            for n, d, b, i, f in zip(n_photons, duration, brightness, interphoton, fret)
        )
    return pd.DataFrame.from_records(records, columns=_FEATURE_COLUMNS)


def fit_gmm(features: pd.DataFrame, settings: GMMSettings | None = None) -> dict[str, Any]:
    """Fit a Gaussian mixture model to burst features.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature table.
    settings : GMMSettings, optional
        GMM settings.

    Returns
    -------
    dict
        Fitted model summary.
    """
    gmm_settings = settings or GMMSettings()
    if features.empty:
        return {
            "n_components": 0,
            "aic": np.nan,
            "bic": np.nan,
            "labels": [],
            "weights": [],
            "means": [],
        }

    matrix = features.to_numpy(dtype=float)
    finite_mask = np.isfinite(matrix).all(axis=1)
    if not finite_mask.any():
        return {
            "n_components": 0,
            "aic": np.nan,
            "bic": np.nan,
            "labels": [],
            "weights": [],
            "means": [],
        }

    matrix = matrix[finite_mask]
    if gmm_settings.auto_components:
        component_range = range(1, min(gmm_settings.max_components, len(matrix)) + 1)
    else:
        component_range = range(1, 2)

    best_model = None
    best_labels = None
    best_bic = np.inf
    for n_components in component_range:
        try:
            model = GaussianMixture(
                n_components=n_components,
                covariance_type=gmm_settings.covariance_type,
                random_state=gmm_settings.random_state,
                max_iter=gmm_settings.max_iter,
                n_init=gmm_settings.n_init,
                tol=gmm_settings.tol,
                reg_covar=gmm_settings.reg_covar,
            )
            labels = model.fit_predict(matrix)
            bic = float(model.bic(matrix))
        except Exception:
            continue
        if bic < best_bic:
            best_model = model
            best_labels = labels
            best_bic = bic

    if best_model is None or best_labels is None:
        return {
            "n_components": 0,
            "aic": np.nan,
            "bic": np.nan,
            "labels": [],
            "weights": [],
            "means": [],
        }

    return {
        "n_components": int(best_model.n_components),
        "aic": float(best_model.aic(matrix)),
        "bic": float(best_bic),
        "labels": best_labels.astype(int).tolist(),
        "weights": best_model.weights_.astype(float).tolist(),
        "means": best_model.means_.astype(float).tolist(),
    }
