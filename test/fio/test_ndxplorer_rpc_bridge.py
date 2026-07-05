"""End-to-end headless test of the ndXplorer ↔ ChiSurf in-process RPC bridge (PRD-56 §7).

Builds the in-process ChiSurf client (phasor + FRET-line services), then drives it
through ndXplorer's own chisurf-free ``PhasorService`` / ``LinesService`` facades — the
exact path the GUI uses, minus the socket.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from chisurf.plugins.ndxplorer.rpc_bridge import make_inprocess_chisurf_client

# ndXplorer is an optional submodule under modules/ndxplorer; put it on the path (as the
# plugin does at load time) so this cross-module test can import its RPC facades.
_NDX_MODULE = pathlib.Path(__file__).resolve().parents[2] / "modules" / "ndxplorer"
if _NDX_MODULE.is_dir() and str(_NDX_MODULE) not in sys.path:
    sys.path.insert(0, str(_NDX_MODULE))

pytest.importorskip("ndxplorer", reason="ndXplorer submodule not available")


@pytest.fixture()
def client():
    c = make_inprocess_chisurf_client()
    assert c is not None
    return c


def test_phasor_service_over_inprocess_client(client):
    from ndxplorer.rpc import PhasorService

    svc = PhasorService(client)
    # A single-exponential phasor point round-trips to its true lifetime.
    from chisurf.plugins.microscopy.img_pixel_phasor import analysis

    gp, sp = analysis.lifetime_to_phasor(2.0, 80.0)
    tau_phi, tau_m = svc.apparent_lifetime([float(gp)], [float(sp)], 80.0)
    assert tau_phi[0] == pytest.approx(2.0, rel=1e-6)
    assert tau_m[0] == pytest.approx(2.0, rel=1e-6)

    overlays = svc.overlays(80.0, sets=["semicircle", "lifetime_ticks"])
    names = {o["name"] for o in overlays}
    assert "universal semicircle" in names


def test_lines_service_phasor_and_fret_over_inprocess_client(client):
    from ndxplorer.rpc import LinesService

    import chisurf.core.models.tcspc.fret as fret_mod

    fret_mod.rda_axis = np.logspace(np.log10(1), np.log10(500))
    svc = LinesService(client)

    phasor_lines = svc.phasor.overlays(frequency_mhz=80.0, sets=["semicircle"])
    assert phasor_lines[0]["name"] == "universal semicircle"

    fret_lines = svc.fret_line.overlays(
        components=[{"model_name": "FRET: FD (Gaussian)", "n_components": 1, "params": {}}],
        sweep={"kind": "param", "component": 0, "name": "R(G,1)"},
        param_min=20.0,
        param_max=100.0,
        n_points=6,
    )
    assert fret_lines[0]["kind"] == "curve"
    assert len(fret_lines[0]["x"]) == 6


def test_gui_window_installs_phasor_toolbar(client, qtbot):
    """A real NDXplorer given the in-process client wires services + installs the toolbar."""
    import ndxplorer
    from qtpy import QtWidgets

    win = ndxplorer.NDXplorer(chisurf_rpc=client)
    qtbot.addWidget(win)
    # Run the deferred init (scheduled via singleShot(0)), which wires the RPC services.
    qtbot.waitUntil(lambda: getattr(win, "phasor_service", None) is not None, timeout=5000)
    assert win.lines_service is not None
    toolbar = win.findChild(QtWidgets.QToolBar, "chisurfPhasorToolbar")
    assert toolbar is not None


def test_toolbar_surfaces_control_panel(client, qtbot):
    """The toolbar button opens the control panel, populated with FRET models over RPC."""
    import ndxplorer
    from qtpy import QtWidgets

    win = ndxplorer.NDXplorer(chisurf_rpc=client)
    qtbot.addWidget(win)
    qtbot.waitUntil(lambda: getattr(win, "phasor_service", None) is not None, timeout=5000)
    toolbar = win.findChild(QtWidgets.QToolBar, "chisurfPhasorToolbar")

    # The panel exists (hidden) and the toolbar action reveals it.
    panel = win.findChild(QtWidgets.QWidget, "chisurfPhasorPanel")
    assert panel is not None
    assert panel.isHidden()  # starts hidden
    toolbar._on_open_panel()
    assert not panel.isHidden()  # toolbar button surfaced it

    # Overlay-set checkboxes and the FRET model list are populated from ChiSurf.
    assert set(panel._set_checks) >= {"semicircle", "polar_grid", "fret"}
    models = [panel._fret_model.itemText(i) for i in range(panel._fret_model.count())]
    assert any("FRET" in m for m in models)
