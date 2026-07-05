"""Tests for the AutoForm-backed acquisition photon simulator."""

from __future__ import annotations

import queue
import threading

import pytest


def test_simulation_settings_model_serializes_enabled_channels():
    """The AutoForm model should emit tttrlib-compatible channel vectors."""
    from chisurf.plugins.core.acq.tcspc_devices.simulation.setup_dialog import (
        SimulationSettingsModel,
    )

    model = SimulationSettingsModel(
        n_species=2,
        green_enabled=True,
        red_enabled=True,
        yellow_enabled=False,
        q_green_p=11.0,
        q_green_s=12.0,
        q_red_p=21.0,
        q_red_s=22.0,
        n_ph_max=123,
        n_ph_per_file=10,
    )

    params = model.to_parameters()

    assert params["N_species"] == 2
    assert params["N_channels"] == 4
    assert params["q"] == [11.0, 12.0, 21.0, 22.0, 11.0, 12.0, 21.0, 22.0]
    assert params["ch_conversion"] == [8, 0, 9, 1, 10, 2, 11, 3]
    assert params["N_ph_max"] == 123
    assert params["N_ph_per_file"] == 10


def test_simulation_setup_dialog_renders_autoform(qapp):
    """The setup dialog should render through AutoForm and expose parameters."""
    from chisurf.gui.autoform import AutoForm
    from chisurf.plugins.core.acq.tcspc_devices.simulation.setup_dialog import (
        EnhancedSimulationSetupDialog,
    )

    dialog = EnhancedSimulationSetupDialog()
    assert dialog.findChildren(AutoForm)

    params = dialog.get_parameters()
    assert params["N_species"] == 1
    assert params["N_channels"] == 2
    assert params["pulsed_exc"] == 0


def test_tttrlib_streaming_writes_split_spc_files(tmp_path):
    """The tttrlib backend should stream words and split SPC output files."""
    tttrlib = pytest.importorskip("tttrlib")
    if not hasattr(tttrlib, "SimEngine"):
        pytest.skip("tttrlib photon simulator is unavailable")

    from chisurf.plugins.core.acq.tcspc_devices.simulation.core.streaming import (
        TttrlibSimulator,
    )

    params = {
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
        "N_ph_max": 12,
        "N_ph_per_file": 3,
        "spc_output_path": str(tmp_path),
        "stream_words_per_batch": 4,
    }
    data_queue = queue.Queue()
    stop_event = threading.Event()
    simulator = TttrlibSimulator()

    assert simulator.simulate_photons_streaming(params, data_queue, stop_event)
    simulator.generation_thread.join(timeout=30)
    assert not simulator.generation_thread.is_alive()

    chunks = []
    while True:
        item = data_queue.get_nowait()
        if item is None:
            break
        chunks.append(item)

    assert chunks
    files = sorted(tmp_path.glob("m*.spc"))
    assert files
    assert tttrlib.TTTR(str(files[0]), "SPC-130").get_n_valid_events() > 0
    assert (tmp_path / "simulation_config.json").exists()
