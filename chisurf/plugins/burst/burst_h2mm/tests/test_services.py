"""Backend/service and photon-extraction tests for the H2MM plugin."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

from chisurf.plugins.burst.burst_h2mm.core import analysis, h2mm
from chisurf.plugins.burst.burst_h2mm.core.photons import (
    StreamDef,
    bursts_from_dataframe,
)


def _fake_tttr(macro, chan, micro, resolution=1e-6):
    """Build a duck-typed TTTR object for tests (no disk IO)."""
    header = types.SimpleNamespace()
    header.tag = lambda key: {"value": resolution}
    header.macro_time_resolution = resolution
    return types.SimpleNamespace(
        macro_times=np.asarray(macro),
        routing_channels=np.asarray(chan),
        micro_times=np.asarray(micro),
        header=header,
    )


def _synthetic_dataset(n_bursts=250, burst_len=80, seed=7):
    """Simulate a 2-state dataset laid out as one TTTR file + burst table."""
    gt = h2mm.H2mmModel(
        np.array([0.5, 0.5]),
        np.array([[0.99, 0.01], [0.02, 0.98]]),
        np.array([[0.85, 0.15], [0.20, 0.80]]),
    )
    rng = np.random.default_rng(seed)
    times = [
        np.concatenate([[0], np.cumsum(rng.poisson(4, size=burst_len - 1) + 1)]).astype(np.int64)
        for _ in range(n_bursts)
    ]
    streams = h2mm.simulate_bursts(gt, times, seed=seed + 1)

    macro, chan, micro, rows = [], [], [], []
    offset = 0
    base = 0
    for t, s in zip(times, streams):
        macro.append(t + base)
        chan.append(s.astype(np.int64))  # stream 0 -> ch 0, stream 1 -> ch 1
        micro.append(np.zeros_like(s))
        rows.append(("f.spc", offset, offset + t.shape[0]))
        offset += t.shape[0]
        base += int(t[-1]) + 1000
    tttr = _fake_tttr(np.concatenate(macro), np.concatenate(chan), np.concatenate(micro))
    df = pd.DataFrame(rows, columns=["First File", "First Photon", "Last Photon"])
    return df, {"f.spc": tttr}


def test_bursts_from_dataframe_and_analyze():
    df, tttrs = _synthetic_dataset()
    streams = [StreamDef("green", [0], []), StreamDef("red", [1], [])]
    data = bursts_from_dataframe(df, tttrs, streams, min_photons=5)
    assert data.n_bursts == 250
    assert data.n_streams == 2

    ana = analysis.analyze(
        data, state_counts=(1, 2, 3), criterion="bic",
        base_time_s=1e-6, n_restarts=1, max_iter=200,
    )
    assert ana.best.n_states == 2
    order = np.argsort(-ana.best.model.obs[:, 0])
    fret = ana.fret[order]
    assert fret[0] < 0.35 and fret[1] > 0.65
    assert len(ana.transitions) > 0


def test_stream_microtime_gating():
    macro = np.array([0, 1, 2, 3])
    chan = np.array([0, 0, 1, 1])
    micro = np.array([100, 5000, 100, 5000])
    tttr = _fake_tttr(macro, chan, micro)
    df = pd.DataFrame([("f", 0, 4)], columns=["First File", "First Photon", "Last Photon"])
    # Green accepts ch0 with micro<=1000; red accepts ch1 any micro.
    streams = [StreamDef("green", [0], [(0, 1000)]), StreamDef("red", [1], [])]
    from chisurf.plugins.burst.burst_h2mm.core.photons import extract_burst_photons

    times, sidx = extract_burst_photons(df, {"f": tttr}, streams, min_photons=1)
    # Photon 1 (ch0, micro 5000) is dropped by the micro-time gate.
    assert len(times) == 1
    assert list(sidx[0]) == [0, 1, 1]


def test_contract_describe_rpc():
    from chisurf.core.plugin.client import InProcessClient
    from chisurf.plugins.burst.burst_h2mm.backend.services import register_services
    from chisurf.server.dispatcher import ServiceDispatcher
    from chisurf.server.session import SessionState

    state = SessionState()
    dispatcher = ServiceDispatcher(state)
    dispatcher._build_default_registry()
    register_services(dispatcher)
    client = InProcessClient(dispatcher)

    res = client.call("burst_h2mm.contract.describe", {})
    assert res.get("ok") is True
    assert res["result"]["plugin_id"] == "burst_h2mm"
