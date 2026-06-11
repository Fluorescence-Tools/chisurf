"""Summary statistics for burst-selection results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def summarize_dataframes(frames: Sequence[pd.DataFrame]) -> dict[str, float]:
    """Compute summary statistics for burst summary tables.

    Parameters
    ----------
    frames : sequence of pandas.DataFrame
        Burst summary tables.

    Returns
    -------
    dict
        Summary statistics.
    """
    n_bursts = sum(len(frame) for frame in frames)
    n_photons = sum(float(frame["nphotons"].sum()) for frame in frames if "nphotons" in frame)
    durations = [frame["duration"].to_numpy(dtype=float) for frame in frames if "duration" in frame]
    brightness = [
        (frame["nphotons"] / np.maximum(frame["duration"], np.finfo(float).eps)).to_numpy(dtype=float)
        for frame in frames
        if "nphotons" in frame and "duration" in frame
    ]
    all_durations = np.concatenate(durations) if durations else np.array([], dtype=float)
    all_brightness = np.concatenate(brightness) if brightness else np.array([], dtype=float)

    return {
        "n_bursts": float(n_bursts),
        "n_photons": float(n_photons),
        "mean_duration": float(np.mean(all_durations)) if len(all_durations) else np.nan,
        "std_duration": float(np.std(all_durations)) if len(all_durations) else np.nan,
        "mean_brightness": float(np.mean(all_brightness)) if len(all_brightness) else np.nan,
        "std_brightness": float(np.std(all_brightness)) if len(all_brightness) else np.nan,
    }
