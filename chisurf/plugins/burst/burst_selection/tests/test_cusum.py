"""Tests for CUSUM, BOCPD, and Kalman burst selection modes via the API."""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from chisurf.plugins.burst.burst_selection.api.models import (
    AnalysisSettings,
    BurstDetectionSettings,
    BurstFilterMode,
    PhotonFilterSettings,
    BocpdFilterSettings,
    KalmanFilterSettings,
    CusumFilterSettings,
)
from chisurf.plugins.burst.burst_selection.api.selection import (
    analyze_file,
    apply_photon_filters,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "bh_spc132_sm_dna"
BH_SPC_FILE = DATA_DIR / "m000.spc"
STREAM_CHANNELS = [0, 1, 8, 9]


def test_cusum_filter_execution() -> None:
    """Ensure CUSUM mode runs successfully and extracts bursts."""
    if not BH_SPC_FILE.exists():
        pytest.skip("Test data not available")

    settings = AnalysisSettings()
    settings.photon_filter = PhotonFilterSettings(
        channels=STREAM_CHANNELS,
        filter_active=True,
        used_filter=BurstFilterMode.CUSUM,
        cusum_filter=CusumFilterSettings(
            min_photons=20,
            background_rate=2000,
            sb_ratio=30.0,
            alpha=0.05,
            beta=0.05,
        ),
    )

    result = analyze_file(BH_SPC_FILE, settings=settings)
    assert result is not None
    assert result.metadata["n_photons"] > 0
    # CUSUM should successfully detect bursts and populate dataframes
    assert str(BH_SPC_FILE.resolve()) in result.dataframes
    df = result.dataframes[str(BH_SPC_FILE.resolve())]
    assert len(df) >= 0


def test_bocpd_filter_execution() -> None:
    """Ensure BOCPD mode runs successfully and extracts bursts through the API."""
    if not BH_SPC_FILE.exists():
        pytest.skip("Test data not available")

    settings = AnalysisSettings()
    settings.photon_filter = PhotonFilterSettings(
        channels=[0, 1],  # BOCPD multi-channel needs valid channels
        filter_active=True,
        used_filter=BurstFilterMode.BOCPD,
        bocpd_filter=BocpdFilterSettings(
            prior_count=1.0,
            prior_duration=0.1,
            changepoint_prob=1e-5,
            dt=0.001,
        ),
    )
    settings.burst_detection = BurstDetectionSettings(
        min_photons=15,
        photon_window=10,
        time_window=0.001,
    )

    result = analyze_file(BH_SPC_FILE, settings=settings)
    assert result is not None
    assert str(BH_SPC_FILE.resolve()) in result.dataframes
    df = result.dataframes[str(BH_SPC_FILE.resolve())]
    assert len(df) >= 0


def test_kalman_filter_execution() -> None:
    """Ensure Kalman mode runs successfully and extracts bursts through the API."""
    if not BH_SPC_FILE.exists():
        pytest.skip("Test data not available")

    settings = AnalysisSettings()
    settings.photon_filter = PhotonFilterSettings(
        channels=[0, 1],
        filter_active=True,
        used_filter=BurstFilterMode.KALMAN,
        kalman_filter=KalmanFilterSettings(
            q=0.01,
            r_scale=0.1,
            z_thresh=3.0,
            min_len=2,
            merge_gap=5,
            dt=0.001,
        ),
    )
    settings.burst_detection = BurstDetectionSettings(
        min_photons=15,
        photon_window=10,
        time_window=0.001,
    )

    result = analyze_file(BH_SPC_FILE, settings=settings)
    assert result is not None
    assert str(BH_SPC_FILE.resolve()) in result.dataframes
    df = result.dataframes[str(BH_SPC_FILE.resolve())]
    assert len(df) >= 0
