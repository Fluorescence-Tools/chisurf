"""PRD-16: Burst Selection conforms to the transformer contract."""

from __future__ import annotations

import os

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.transform import check_transformer_conformance, get_transformer
from chisurf.plugins.burst.burst_selection.api.transformer import (
    BurstSelectionTransformer,
    settings_from_parameters,
)


def test_burst_selection_is_conformant(tmp_path):
    db = MFDatabase(os.path.join(tmp_path, "t.db"))
    try:
        check_transformer_conformance(BurstSelectionTransformer(), conn=db.conn)
    finally:
        db.close()


def test_burst_selection_self_registers():
    assert get_transformer("burst_selection") is not None


def test_flat_params_map_to_nested_settings():
    s = settings_from_parameters(
        {
            "min_photons": 42,
            "photon_window": 7,
            "time_window": 2e-3,
            "filter_active": False,
            "count_rate_n_ph_max": 99,
            "count_rate_time_window": 5e-3,
            "delta_macro_time_min": 1e-3,
            "delta_macro_time_max": 0.2,
            "gmm_max_components": 4,
        }
    )
    assert s.burst_detection.min_photons == 42
    assert s.burst_detection.photon_window == 7
    assert s.burst_detection.time_window == 2e-3
    assert s.photon_filter.filter_active is False
    assert s.photon_filter.count_rate_filter.n_ph_max == 99
    assert s.photon_filter.count_rate_filter.time_window == 5e-3
    assert s.photon_filter.delta_macro_time_filter.dT_min == 1e-3
    assert s.photon_filter.delta_macro_time_filter.dT_max == 0.2
    assert s.gmm.max_components == 4


def test_burst_params_roundtrip_with_extract():
    """The adapter's mapping is the inverse of extract_burst_parameters for the
    declared names — the values an operation records map back to settings."""
    from chisurf.plugins.burst.burst_selection.api.mfdb import extract_burst_parameters
    from chisurf.plugins.burst.burst_selection.api.models import AnalysisRequest

    s = settings_from_parameters(
        {
            "min_photons": 42,
            "photon_window": 7,
            "count_rate_n_ph_max": 99,
            "gmm_max_components": 4,
        }
    )
    params = extract_burst_parameters(AnalysisRequest(files=[], settings=s))
    assert params["min_photons"] == 42
    assert params["photon_window"] == 7
    assert params["count_rate_n_ph_max"] == 99
    assert params["gmm_max_components"] == 4
