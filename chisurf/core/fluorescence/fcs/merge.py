"""Qt-free helpers for merging / averaging FCS correlation curves.

These primitives back both the shared ``WizardFcsMerger`` widget and the
FCS-Merger plugin (its core / backend RPC). A "correlation" is a plain dict with
keys ``x`` (lag times), ``y`` (G), ``duration`` (s), ``count_rate`` (kHz),
optional ``ey`` and per-channel ``channel_a`` / ``channel_b`` count dicts.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

import numpy as np


def compute_average_correlations(correlations: List[dict]) -> dict:
    """Weighted-average a list of correlation dicts into a single curve.

    Mirrors the legacy ``WizardFcsMerger.compute_average_correlations``: the
    per-curve Suren weights are used only when more than one curve is merged,
    the acquisition time is summed, and the count rate is duration-weighted.
    """
    import chisurf.core.fluorescence.fcs as _fcs

    taus: List[np.ndarray] = []
    cors: List[np.ndarray] = []
    acquisition_time = 0.0
    weighted_count_rate_sum = 0.0
    n_curves = len(correlations)

    for correlation in correlations:
        tau = np.array(correlation["x"], dtype=float)
        cor = np.array(correlation["y"], dtype=float)
        duration = float(correlation["duration"])
        acquisition_time += duration
        counts = (
            correlation["channel_a"]["counts"] + correlation["channel_b"]["counts"]
        )
        cr = (counts / 2.0) / duration / 1000.0 if duration > 0 else 0.0
        weighted_count_rate_sum += duration * cr
        taus.append(tau)
        cors.append(cor)

    ys = np.array(cors)
    avg_count_rate = (
        weighted_count_rate_sum / acquisition_time if acquisition_time > 0 else 0.0
    )

    if n_curves == 1:
        ey = np.zeros_like(ys[0])
    else:
        ey = np.std(ys, axis=0) / np.sqrt(n_curves)

    return {
        "x": np.array(taus).mean(axis=0)[1:],
        "y": ys.mean(axis=0)[1:],
        "ey": ey[1:],
        "duration": acquisition_time,
        "count_rate": avg_count_rate,
    }


def _correlation_from_cor_array(arr: np.ndarray) -> Dict[str, Any]:
    """Build a correlation dict from a loaded ``.cor`` array (PAM/ChiSurf format)."""
    if arr.ndim == 1 and arr.size >= 2:
        arr = arr.reshape(-1, arr.size)
    x = arr[:, 0]
    y = arr[:, 1]
    duration = float(arr[0, 2]) if arr.shape[1] > 2 and arr.shape[0] >= 1 else 0.0
    count_rate = float(arr[1, 2]) if arr.shape[1] > 2 and arr.shape[0] >= 2 else 0.0
    ey = arr[:, 3] if arr.shape[1] > 3 else np.zeros_like(x)
    total_counts = count_rate * duration
    half_counts = 0.5 * total_counts
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "ey": ey.tolist(),
        "duration": duration,
        "count_rate": count_rate,
        "channel_a": {"channels": [], "microtime_range": None, "counts": half_counts},
        "channel_b": {"channels": [], "microtime_range": None, "counts": half_counts},
    }


def parse_correlation_folder(folder: pathlib.Path) -> List[Dict[str, Any]]:
    """Load all ``.cor`` (and legacy ``.json.gz``) correlation chunks in a folder."""
    folder = pathlib.Path(folder)
    out: List[Dict[str, Any]] = []
    if not folder.is_dir():
        return out

    for file in sorted(folder.glob("*.json.gz")):
        try:
            import json

            import chisurf.core.fio as io

            with io.open_maybe_zipped(file) as fp:
                out.append(json.load(fp))
        except Exception:
            continue

    for file in sorted(folder.glob("*.cor")):
        try:
            arr = np.loadtxt(str(file), delimiter="\t")
            out.append(_correlation_from_cor_array(arr))
        except Exception:
            continue
    return out


def save_mean_correlation(correlation: dict, filename: pathlib.Path) -> None:
    """Write a merged correlation to a ChiSurf ``.cor`` file (Kristine format)."""
    filename = pathlib.Path(filename)
    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    x = np.asarray(correlation["x"], dtype=float)
    y = np.asarray(correlation["y"], dtype=float)
    ey = np.asarray(correlation.get("ey", np.zeros_like(x)), dtype=float)
    suren_column = np.zeros_like(x)
    if suren_column.size > 0:
        suren_column[0] = correlation["duration"]
    if suren_column.size > 1:
        suren_column[1] = correlation["count_rate"]
    if np.any(ey != 0):
        c = np.vstack([x, y, suren_column, ey])
    else:
        c = np.vstack([x, y, suren_column])
    np.savetxt(str(filename), c.T, delimiter="\t", fmt="%.5g")


def merge_folder(folder: pathlib.Path, output: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Parse, average and (optionally) save the correlations in ``folder``.

    Returns the merged correlation as a transport-friendly dict.
    """
    correlations = parse_correlation_folder(folder)
    if not correlations:
        raise ValueError(f"No .cor/.json.gz correlation files found in {folder}")
    merged = compute_average_correlations(correlations)
    if output is not None:
        save_mean_correlation(merged, output)
    return {
        "x": np.asarray(merged["x"], dtype=float).tolist(),
        "y": np.asarray(merged["y"], dtype=float).tolist(),
        "ey": np.asarray(merged["ey"], dtype=float).tolist(),
        "duration": float(merged["duration"]),
        "count_rate": float(merged["count_rate"]),
        "n_curves": len(correlations),
    }
