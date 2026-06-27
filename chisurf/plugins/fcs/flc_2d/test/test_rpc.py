"""RPC smoke tests for the 2D-FLCS plugin."""

from __future__ import annotations

import numpy as np


def test_flc_client_contract_and_correlate():
    """Exercise the in-process RPC client and 2D-FDC service."""
    from chisurf.plugins.fcs.flc_2d.gui.client import FlcClient

    client = FlcClient()
    contract = client.describe_contract()
    assert "flc2d.correlate" in contract["methods"]
    assert "flc2d.lifetime_spectrum" in contract["methods"]

    macro = np.arange(0, 200, dtype=np.int64)
    micro = (macro % 8 + 1).astype(np.int64)
    result = client.correlate(
        macro,
        micro,
        dT=4,
        ddT=2,
        tMin=0,
        tMax=12,
        logt_imax=8,
        max_bins=16,
    )
    assert "mat_lin" in result
    assert np.asarray(result["mat_lin"]).sum() > 0


def test_flc_client_lifetime_spectrum():
    """Exercise the 1D lifetime-spectrum RPC service."""
    from chisurf.plugins.fcs.flc_2d.gui.client import FlcClient

    rng = np.random.default_rng(1)
    micro = rng.exponential(scale=12.0, size=2000).astype(np.int64)
    micro = micro[(micro >= 0) & (micro < 128)]
    result = FlcClient().lifetime_spectrum(
        micro,
        n_microtime_bins=128,
        micro_time_resolution_ns=0.05,
        tau_range=(0.2, 6.0),
        n_components=12,
        method="nnls",
    )
    assert len(result["tau_grid"]) == 12
    assert len(result["amplitudes"]) == 12
    assert np.isfinite(result["chi2"])
