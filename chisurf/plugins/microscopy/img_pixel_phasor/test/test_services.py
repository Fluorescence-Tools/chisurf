"""RPC contract tests for the ``phasor.*`` service (PRD-56 §7).

Drives the handlers through an in-process ``ServiceDispatcher`` — the same dispatch
path the ZMQ server uses — and asserts shapes and round-tripping of array payloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.microscopy.img_pixel_phasor import analysis
from chisurf.plugins.microscopy.img_pixel_phasor.backend.services import register_services
from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState


@pytest.fixture()
def dispatcher() -> ServiceDispatcher:
    d = ServiceDispatcher(SessionState())
    register_services(d)
    return d


def _semicircle_point(tau, freq):
    g, s = analysis.lifetime_to_phasor(tau, freq)
    return [float(g), float(s)]


def test_all_methods_registered(dispatcher):
    methods = set(dispatcher.list_methods())
    expected = {
        "phasor.describe",
        "phasor.apparent_lifetime",
        "phasor.filter",
        "phasor.component_fraction",
        "phasor.unmix",
        "phasor.cursor_mask",
        "phasor.pseudo_color",
        "phasor.overlays",
        "phasor.contours",
    }
    assert expected <= methods


def test_describe(dispatcher):
    res = dispatcher.dispatch("phasor.describe", {})
    assert res["ok"]
    assert res["result"]["default_frequency_mhz"] == 80.0
    assert "semicircle" in res["result"]["overlay_sets"]


def test_apparent_lifetime_roundtrip(dispatcher):
    freq = 80.0
    g, s = _semicircle_point(2.0, freq)
    res = dispatcher.dispatch(
        "phasor.apparent_lifetime", {"g": [g], "s": [s], "frequency_mhz": freq}
    )
    assert res["ok"]
    assert res["result"]["tau_phi"][0] == pytest.approx(2.0, rel=1e-6)
    assert res["result"]["tau_m"][0] == pytest.approx(2.0, rel=1e-6)


def test_filter_preserves_shape(dispatcher):
    g = np.full((4, 4), 0.4).tolist()
    s = np.full((4, 4), 0.3).tolist()
    res = dispatcher.dispatch("phasor.filter", {"g": g, "s": s, "kind": "median", "size": 3})
    assert res["ok"]
    assert np.asarray(res["result"]["g"]).shape == (4, 4)


def test_filter_unknown_kind_fails(dispatcher):
    res = dispatcher.dispatch("phasor.filter", {"g": [[0.4]], "s": [[0.3]], "kind": "bogus"})
    assert not res["ok"]
    assert "bogus" in res["error"]


def test_component_fraction(dispatcher):
    freq = 80.0
    c1 = _semicircle_point(1.0, freq)
    c2 = _semicircle_point(4.0, freq)
    f = 0.4
    g = f * c1[0] + (1 - f) * c2[0]
    s = f * c1[1] + (1 - f) * c2[1]
    res = dispatcher.dispatch(
        "phasor.component_fraction", {"g": [g], "s": [s], "c1": c1, "c2": c2}
    )
    assert res["ok"]
    assert res["result"]["fraction"][0] == pytest.approx(f, abs=1e-6)


def test_unmix(dispatcher):
    freq = 80.0
    comps = [_semicircle_point(0.5, freq), _semicircle_point(2.0, freq), _semicircle_point(6.0, freq)]
    w = [0.2, 0.5, 0.3]
    g = [sum(wi * c[0] for wi, c in zip(w, comps))]
    s = [sum(wi * c[1] for wi, c in zip(w, comps))]
    res = dispatcher.dispatch("phasor.unmix", {"g": g, "s": s, "components": comps})
    assert res["ok"]
    fractions = res["result"]["fractions"]
    assert len(fractions) == 3
    assert sum(f[0] for f in fractions) == pytest.approx(1.0, abs=1e-6)


def test_cursor_mask(dispatcher):
    g = [[0.1, 0.5], [0.51, 0.9]]
    s = [[0.1, 0.5], [0.49, 0.1]]
    res = dispatcher.dispatch(
        "phasor.cursor_mask", {"g": g, "s": s, "center": [0.5, 0.5], "radius": 0.05}
    )
    assert res["ok"]
    assert res["result"]["n_selected"] == 2
    assert np.asarray(res["result"]["mask"]).shape == (2, 2)


def test_pseudo_color(dispatcher):
    m0 = [[True, False], [False, False]]
    m1 = [[False, True], [True, False]]
    res = dispatcher.dispatch(
        "phasor.pseudo_color", {"masks": [m0, m1], "colors": [[1, 0, 0], [0, 0, 1]]}
    )
    assert res["ok"]
    assert res["result"]["shape"] == [2, 2, 3]


def test_overlays_default_sets(dispatcher):
    res = dispatcher.dispatch("phasor.overlays", {"frequency_mhz": 80.0})
    assert res["ok"]
    names = {o["name"] for o in res["result"]["overlays"]}
    assert "universal semicircle" in names
    assert "lifetime ticks" in names


def test_overlays_fret_and_component_line(dispatcher):
    res = dispatcher.dispatch(
        "phasor.overlays",
        {
            "frequency_mhz": 80.0,
            "sets": ["fret", "component_line"],
            "tau_d0": 4.0,
            "c1": [0.7, 0.3],
            "c2": [0.3, 0.4],
        },
    )
    assert res["ok"]
    names = {o["name"] for o in res["result"]["overlays"]}
    assert "FRET trajectory" in names
    assert "component line" in names


def test_harmonic_scales_frequency(dispatcher):
    res = dispatcher.dispatch("phasor.overlays", {"frequency_mhz": 40.0, "harmonic": 2})
    assert res["ok"]
    assert res["result"]["frequency_mhz"] == pytest.approx(80.0)


def test_overlays_new_sets(dispatcher):
    res = dispatcher.dispatch(
        "phasor.overlays",
        {
            "frequency_mhz": 80.0,
            "sets": ["polar_grid", "components", "cursor"],
            "components": [[0.2, 0.3], [0.6, 0.4]],
            "fractions": [0.5, 0.5],
            "cursors": [{"center": [0.5, 0.3], "radius": 0.05, "name": "gate"}],
        },
    )
    assert res["ok"]
    names = {o["name"] for o in res["result"]["overlays"]}
    assert {"unit circle", "components", "gate"} <= names


def test_contours_of_gaussian_blob(dispatcher):
    g = np.linspace(0.0, 1.0, 50)[:, None]
    s = np.linspace(0.0, 1.0, 50)[None, :]
    density = np.exp(-((g - 0.5) ** 2 + (s - 0.3) ** 2) / 0.01)
    res = dispatcher.dispatch(
        "phasor.contours",
        {"density": density.tolist(), "g_range": [0.0, 1.0], "s_range": [0.0, 1.0], "levels": 3},
    )
    assert res["ok"]
    assert len(res["result"]["overlays"]) >= 1
    assert res["result"]["overlays"][0]["kind"] == "curve"
