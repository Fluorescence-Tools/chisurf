"""Foundation tests: Qt-free core + in-process RPC client + manifest."""

import pathlib

import numpy as np


def _synthetic_curve(td_true: float = 1.0):
    tau = np.logspace(-3, 2, 60)
    s = 3.5
    g = 0.5 * (1.0 / (1.0 + tau / td_true)) / np.sqrt(1.0 + tau / (td_true * s ** 2)) + 1.0
    return tau, g


def test_core_simple_fit_recovers_diffusion_time():
    from chisurf.plugins.burst.burst_fcs_correlator.core import BurstFcsSettings, fit_curve

    tau, g = _synthetic_curve(1.0)
    res = fit_curve(tau, g, BurstFcsSettings(fit_mode="simple"))
    assert res["fit_mode"] == "simple"
    assert 0.5 < res["td_mean"] < 2.0  # recovers ~1 ms on the coarse grid
    assert len(res["tau"]) == len(res["g"])


def test_settings_roundtrip():
    from chisurf.plugins.burst.burst_fcs_correlator.core import BurstFcsSettings

    s = BurstFcsSettings(n_bins=5, n_casc=25, make_fine=True, fit_mode="maxent")
    s2 = BurstFcsSettings.from_dict(s.to_dict())
    assert s2 == s


def test_inprocess_rpc_client():
    from chisurf.plugins.burst.burst_fcs_correlator.gui.client import BurstFcsClient

    tau, g = _synthetic_curve(1.0)
    client = BurstFcsClient()
    r = client.fit_simple(tau.tolist(), g.tolist())
    assert r.get("ok") is True
    assert 0.5 < r["result"]["td"] < 2.0

    r2 = client.fit_curve(tau.tolist(), g.tolist(), {"fit_mode": "simple"})
    assert r2.get("ok") is True
    assert r2["result"]["fit_mode"] == "simple"


def test_manifest_loads_with_rpc_methods():
    from chisurf.core.plugin import load_manifest

    here = pathlib.Path(__file__).resolve().parent.parent
    m = load_manifest(here / "manifest.json")
    assert m is not None
    assert m.id == "burst_fcs_correlator"
    names = {rpc.name for rpc in m.rpc_methods}
    assert {"burst_fcs.correlate_file", "burst_fcs.fit_simple"} <= names


def test_legacy_helpers_shim_reexports():
    from chisurf.plugins.burst.burst_fcs_correlator.helpers import (  # noqa: F401
        correlate_single_burst,
        open_tttr,
        parse_bst_file,
        parse_bur_file,
        parse_channel_list,
    )
