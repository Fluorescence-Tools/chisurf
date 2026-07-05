"""Contract tests for the shared FRET-line overlay interface (PRD-56).

``fret_line.overlays`` must return the same LineSet shape as ``phasor.overlays`` so
ndXplorer can draw FRET lines and phasor lines through one uniform path.
"""

from __future__ import annotations

import numpy as np
import pytest

from chisurf.plugins.fret_line.backend.services import register_services
from chisurf.plugins.fret_line.core.algorithms import list_fret_line_projections
from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState


@pytest.fixture()
def dispatcher() -> ServiceDispatcher:
    d = ServiceDispatcher(SessionState())
    register_services(d)
    return d


@pytest.fixture(autouse=True)
def _rda_axis():
    # Pin the FRET distance axis so the sweep does not depend on runtime settings.
    import chisurf.core.models.tcspc.fret as fret_mod

    fret_mod.rda_axis = np.logspace(np.log10(1), np.log10(500))


_COMPONENTS = [{"model_name": "FRET: FD (Gaussian)", "n_components": 1, "params": {}}]
_SWEEP = {"kind": "param", "component": 0, "name": "R(G,1)"}


def test_overlays_method_registered(dispatcher):
    methods = set(dispatcher.list_methods())
    assert {"fret_line.overlays", "fret_line.list_projections"} <= methods


def test_overlays_returns_lineset_shape(dispatcher):
    res = dispatcher.dispatch(
        "fret_line.overlays",
        {"components": _COMPONENTS, "sweep": _SWEEP, "param_min": 20.0,
         "param_max": 100.0, "n_points": 8},
    )
    assert res["ok"], res
    overlays = res["result"]["overlays"]
    assert len(overlays) == 1
    line = overlays[0]
    # Same keys as a phasor LineSet: name, kind, x, y, style (+ axes hint).
    assert {"name", "kind", "x", "y", "style"} <= set(line)
    assert line["kind"] == "curve"
    assert len(line["x"]) == len(line["y"]) == 8
    assert line["axes"]["x"] == "tau_f" and line["axes"]["y"] == "e_fret"


def test_projection_choice(dispatcher):
    res = dispatcher.dispatch(
        "fret_line.overlays",
        {"components": _COMPONENTS, "sweep": _SWEEP, "param_min": 20.0,
         "param_max": 100.0, "n_points": 6, "line": "tau_x_vs_tau_f"},
    )
    assert res["ok"], res
    assert res["result"]["overlays"][0]["axes"]["y"] == "tau_x"


def test_unknown_projection_fails(dispatcher):
    res = dispatcher.dispatch(
        "fret_line.overlays",
        {"components": _COMPONENTS, "sweep": _SWEEP, "param_min": 20.0,
         "param_max": 100.0, "n_points": 4, "line": "bogus"},
    )
    assert not res["ok"]
    assert "bogus" in res["error"]


def test_list_projections(dispatcher):
    res = dispatcher.dispatch("fret_line.list_projections", {})
    assert res["ok"]
    assert "static" in res["result"]["projections"]
    assert set(res["result"]["projections"]) == set(list_fret_line_projections())
