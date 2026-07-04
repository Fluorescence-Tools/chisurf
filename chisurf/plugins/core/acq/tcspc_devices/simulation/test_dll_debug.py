"""Replacement tests for the acquisition photon simulator.

The historical file name is kept so existing targeted commands still work, but
the active simulator is now tttrlib-backed rather than Burbulator/DLL-backed.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest


def _requires_tttrlib_simulator():
    """Import tttrlib and skip when the simulator API is unavailable.

    Returns
    -------
    module
        Imported ``tttrlib`` module.
    """
    tttrlib = pytest.importorskip("tttrlib")
    if not hasattr(tttrlib, "SimEngine"):
        pytest.skip("tttrlib photon simulator is unavailable")
    return tttrlib


def _small_params(output_path: str = "") -> dict:
    """Return fast deterministic simulation parameters.

    Parameters
    ----------
    output_path : str, optional
        SPC output folder.

    Returns
    -------
    dict
        Parameter dictionary accepted by the tttrlib simulator backend.
    """
    return {
        "N_species": 1,
        "M": [1.0],
        "D": [1.0],
        "N_channels": 2,
        "q": [5.0, 5.0],
        "q_bg": [0.0, 0.0],
        "k_rad": [0.0],
        "k_nrad": [0.0],
        "box_xy": 2.0,
        "box_z": 4.0,
        "focus_param": [0.3, 2.0],
        "dt": 0.01,
        "N_ph_max": 16,
        "N_ph_per_file": 8,
        "spc_output_path": output_path,
        "rmt1seed": 12345,
        "rmt2seed": 54321,
    }


def test_tttrlib_core_generates_spc_words():
    """Core generator should produce uint32 SPC records from tttrlib."""
    _requires_tttrlib_simulator()
    from chisurf.plugins.core.acq.tcspc_devices.simulation.core.algorithms import (
        generate_spc132_uint32,
    )

    words = generate_spc132_uint32(_small_params())

    assert isinstance(words, np.ndarray)
    assert words.dtype == np.uint32
    assert len(words) > 0


def test_tttrlib_stream_writes_readable_spc_file(tmp_path):
    """Streaming backend should write SPC files readable by tttrlib."""
    tttrlib = _requires_tttrlib_simulator()
    from chisurf.plugins.core.acq.tcspc_devices.simulation.core.streaming import (
        TttrlibSimulator,
    )

    simulator = TttrlibSimulator()
    data_queue = queue.Queue()
    stop_event = threading.Event()

    assert simulator.simulate_photons_streaming(
        _small_params(str(tmp_path)),
        data_queue,
        stop_event,
    )
    simulator.generation_thread.join(timeout=30)

    assert not simulator.generation_thread.is_alive()
    assert data_queue.get_nowait() is not None

    files = sorted(tmp_path.glob("m*.spc"))
    assert files
    assert tttrlib.TTTR(str(files[0]), "SPC-130").get_n_valid_events() > 0


def test_simulation_device_reads_fifo(tmp_path, qapp):
    """SimulationDevice should start tttrlib generation and expose FIFO chunks."""
    _requires_tttrlib_simulator()
    from chisurf.plugins.core.acq.tcspc_devices.simulation.wrapper import SimulationDevice

    device = SimulationDevice()
    device.simulation_params.update(_small_params(str(tmp_path)))

    assert device.start_measurement()
    device.generation_thread.join(timeout=30)

    chunks = []
    while device.measurement_running:
        chunk = device.read_fifo(1024)
        if len(chunk):
            chunks.append(chunk)

    device.close()

    assert chunks
    assert all(chunk.dtype == np.uint32 for chunk in chunks)
