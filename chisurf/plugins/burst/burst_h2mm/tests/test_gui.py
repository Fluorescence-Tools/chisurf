"""Head-less GUI smoke tests for the H2MM tool (collected by the gui suite)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_h2mm_gui_instantiates(qapp):
    from chisurf.plugins.burst.burst_h2mm.gui.tool import H2mmTool

    w = H2mmTool(embedded=True)
    tabs = [w.dock_area.tabText(i) for i in range(3)]
    assert "H2MM Settings" in tabs
    assert "Results" in tabs
    w.close()


def test_h2mm_gui_plots_render(qapp):
    import types

    import pandas as pd

    from chisurf.plugins.burst.burst_h2mm.api.models import H2mmSettings
    from chisurf.plugins.burst.burst_h2mm.backend.services import (
        H2mmAnalysisBundle,
        _result_from_analysis,
    )
    from chisurf.plugins.burst.burst_h2mm.core import analysis, h2mm
    from chisurf.plugins.burst.burst_h2mm.core.photons import (
        StreamDef,
        bursts_from_dataframe,
    )
    from chisurf.plugins.burst.burst_h2mm.gui.tool import H2mmTool

    gt = h2mm.H2mmModel(
        np.array([0.5, 0.5]),
        np.array([[0.99, 0.01], [0.02, 0.98]]),
        np.array([[0.85, 0.15], [0.20, 0.80]]),
    )
    rng = np.random.default_rng(1)
    times = [
        np.concatenate([[0], np.cumsum(rng.poisson(4, size=59) + 1)]).astype(np.int64)
        for _ in range(120)
    ]
    streams = h2mm.simulate_bursts(gt, times, seed=2)
    macro, chan, micro, rows = [], [], [], []
    off = base = 0
    for t, s in zip(times, streams):
        macro.append(t + base)
        chan.append(s.astype(np.int64))
        micro.append(np.zeros_like(s))
        rows.append(("f.spc", off, off + len(t)))
        off += len(t)
        base += int(t[-1]) + 1000
    hdr = types.SimpleNamespace(tag=lambda k: {"value": 1e-6}, macro_time_resolution=1e-6)
    tttr = types.SimpleNamespace(
        macro_times=np.concatenate(macro),
        routing_channels=np.concatenate(chan),
        micro_times=np.concatenate(micro),
        header=hdr,
    )
    df = pd.DataFrame(rows, columns=["First File", "First Photon", "Last Photon"])
    data = bursts_from_dataframe(
        df, {"f.spc": tttr}, [StreamDef("green", [0]), StreamDef("red", [1])], min_photons=5
    )
    ana = analysis.analyze(data, state_counts=(1, 2), base_time_s=1e-6, n_restarts=1, max_iter=150)

    w = H2mmTool(embedded=True)
    w._result = _result_from_analysis(ana, H2mmSettings())
    w._bundle = H2mmAnalysisBundle(ana, data, H2mmSettings())
    w._update_plots()  # must not raise
    assert w._result.n_states == 2
    w.close()
