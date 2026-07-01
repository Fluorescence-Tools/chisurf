"""Headless tests for the anisotropy core (IRF, spectra, link plan)."""

from __future__ import annotations

import numpy as np

from chisurf.plugins.fluorescence_decay.tr_anisotropy.core import fits, irf, spectra


# ── IRF ─────────────────────────────────────────────────────────────────────
def test_initial_region():
    assert irf.initial_region(1000) == (300, 800)


def test_channel_background():
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0, 0.0])
    assert irf.channel_background(y, 2, 4) == 10.0
    # empty / inverted region → 0
    assert irf.channel_background(y, 4, 2) == 0.0


def test_correct_irfs_subtracts_and_matches_intensity():
    # flat background of 5 in the tail region, a peak up front
    y_vv = np.concatenate([[100.0, 50.0], np.full(8, 5.0)])
    y_vh = np.concatenate([[40.0, 20.0], np.full(8, 5.0)])
    vv, vh = irf.correct_irfs(y_vv, y_vh, 2, 10)
    # background removed → tail ~0
    assert vv[2:].max() < 1e-9
    assert vh[2:].max() < 1e-9
    # intensity matched → equal integrals
    assert np.isclose(vv.sum(), vh.sum())


def test_correct_irfs_empty():
    vv, vh = irf.correct_irfs([], [], 0, 1)
    assert len(vv) == 0 and len(vh) == 0


def test_correct_irfs_length_mismatch_truncates():
    vv, vh = irf.correct_irfs(np.ones(10), np.ones(6), 0, 3)
    assert len(vv) == 6 and len(vh) == 6


# ── spectra ─────────────────────────────────────────────────────────────────
def test_flatten_and_to_pairs_roundtrip():
    pairs = [[0.3, 1.8], [0.7, 4.1]]
    flat = spectra.flatten(pairs)
    assert list(flat) == [0.3, 1.8, 0.7, 4.1]
    assert spectra.to_pairs(flat) == pairs


def test_save_and_load_spectra(tmp_path):
    path = tmp_path / "s.spk.json"
    spectra.save_spectra(path, [[0.3, 1.8]], [[0.5, 2.0]], mirror_to_default=False)
    got = spectra.load_spectra(path)
    assert got["lifetime_spectrum"] == [[0.3, 1.8]]
    assert got["rotation_spectrum"] == [[0.5, 2.0]]


# ── link plan ────────────────────────────────────────────────────────────────
def test_build_link_plan_structure():
    plan = fits.build_link_plan(
        n_lifetime=2, n_rotation=1, corrections={"g_factor": 1.2, "l1": 0.1, "l2": 0.2}
    )
    ops = [p["op"] for p in plan]
    # n0 link + corrections set + component links + fixes + correction links
    assert plan[0] == {"op": "link", "name": "n0", "target": "n0"}
    # rotation component i=1 linked (b + rho)
    assert {"op": "link", "name": "rho(1)", "target": "rho(1)"} in plan
    assert {"op": "link", "name": "b(1)", "target": "b(1)"} in plan
    # lifetimes 1..2 linked
    assert {"op": "link", "name": "xL1", "target": "xL1"} in plan
    assert {"op": "link", "name": "tL2", "target": "tL2"} in plan
    # g value taken from corrections
    assert {"op": "set_value", "name": "g", "value": 1.2, "fit": "vv"} in plan
    assert "update" in ops


def test_apply_link_plan_replays_calls():
    calls = []

    class FakeClient:
        def link_parameters(self, **k):
            calls.append(("link", k))

        def set_parameter_value(self, **k):
            calls.append(("set_value", k))

        def set_parameter_fixed(self, **k):
            calls.append(("set_fixed", k))

        def update_fit(self, **k):
            calls.append(("update", k))

    plan = [
        {"op": "link", "name": "n0", "target": "n0"},
        {"op": "set_value", "name": "g", "value": 1.1, "fit": "vv"},
        {"op": "set_fixed", "name": "g", "fixed": True, "fit": "vh"},
        {"op": "update", "fit": "vv"},
    ]
    fits.apply_link_plan(FakeClient(), plan, vv_index=3, vh_index=4)
    assert calls[0] == (
        "link",
        {
            "parameter_name": "n0",
            "target_parameter_name": "n0",
            "fit_index": 4,
            "target_fit_index": 3,
        },
    )
    assert calls[1] == ("set_value", {"parameter_name": "g", "value": 1.1, "fit_index": 3})
    assert calls[2] == ("set_fixed", {"parameter_name": "g", "fixed": True, "fit_index": 4})
    assert calls[3] == ("update", {"fit_index": 3})
