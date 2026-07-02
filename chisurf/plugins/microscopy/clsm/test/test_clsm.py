"""Headless tests for the CLSM plugin (core / api / services / cli / gui).

The image tests use the real CLSM fixtures under ``<repo>/test/data/clsm``; they
skip cleanly when ``tttrlib`` or the data files are unavailable.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

# <repo>/chisurf/plugins/microscopy/clsm/test/test_clsm.py -> parents[5] == <repo>
REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
DATA_DIR = REPO_ROOT / "test" / "data" / "clsm"
SP5 = DATA_DIR / "Leica_SP5.ptu"


def _require_data():
    pytest.importorskip("tttrlib")
    if not SP5.exists():
        pytest.skip(f"CLSM test data not available: {SP5}")


# ── pure helpers (no data needed) ───────────────────────────────────────────


def test_frc_and_noise_pure():
    from chisurf.plugins.microscopy.clsm.core import frc

    rng = np.random.RandomState(0)
    a, b = rng.rand(64, 64), rng.rand(64, 64)
    density, bins = frc.compute_frc(a, b)
    assert density.shape == bins.shape
    assert np.isclose(frc.gaussian_kernel(7, 3).sum(), 1.0)
    assert np.all(frc.counting_noise(np.array([0.0, 4.0, 9.0])) == np.array([1.0, 2.0, 3.0]))


def test_brush_kernel_deselect_is_negative():
    from chisurf.plugins.microscopy.clsm.core import imaging

    assert imaging.brush_kernel(7, 3, select=False).min() < 0
    assert imaging.brush_kernel(7, 3, select=True).min() >= 0


def test_setups_presets():
    from chisurf.plugins.microscopy.clsm.core import setups

    presets = setups.builtin_setups()
    assert "Leica SP5" in presets
    assert presets["Leica SP5"]["frame_marker"] == [4, 6]


def test_contract_descriptor():
    from chisurf.plugins.microscopy.clsm.api import contract

    desc = contract.contract_descriptor()
    assert desc["plugin_id"] == "clsm"
    assert contract.METHOD_DECAY in desc["methods"]


# ── core / api with real data ───────────────────────────────────────────────


def test_core_image_representation_decay_frc():
    _require_data()
    import tttrlib

    from chisurf.plugins.microscopy.clsm.api.models import ClsmSetup
    from chisurf.plugins.microscopy.clsm.core import frc, imaging, setups

    tttr = tttrlib.TTTR(str(SP5), "PTU")
    preset = setups.builtin_setups()["Leica SP5"]
    detected = setups.read_clsm_markers(tttr)
    setup = ClsmSetup.from_preset(preset, channels=[0, 1])
    if detected.get("pixel_per_line"):
        setup.pixel_per_line = detected["pixel_per_line"]

    clsm = imaging.build_clsm_image(tttr, setup)
    assert clsm.n_frames > 0 and clsm.n_lines > 0 and clsm.n_pixel > 0

    image = imaging.representation(clsm, tttr, "Intensity", 1)
    assert image.ndim == 3
    current, s1, s2 = imaging.reduce_frames(image, "sum")
    assert current.shape == s1.shape == s2.shape

    mask = (current > current.mean()).astype(np.uint8)
    t, y, ey = imaging.decay_of_selection(clsm, tttr, mask, tac_coarsening=4, stack_frames=True)
    assert t.shape == y.shape == ey.shape
    assert y.sum() > 0

    density, bins = frc.compute_frc(s1, s2)
    assert density.shape == bins.shape


def test_api_orchestration_and_save(tmp_path):
    _require_data()
    from chisurf.plugins.microscopy.clsm import api

    info = api.image_info(str(SP5), setup_name="Leica SP5", channels=[0, 1])
    assert info["n_frames"] > 0
    assert info["micro_time_resolution_ns"] > 0

    out = tmp_path / "decay.txt"
    dec = api.extract_decay(
        str(SP5),
        setup_name="Leica SP5",
        channels=[0, 1],
        threshold=0.5,
        tac_coarsening=4,
        output_path=str(out),
    )
    assert dec["n_photons"] > 0
    assert out.exists()
    assert len(dec["counts"]) == len(dec["time_ns"])

    frc = api.compute_frc(str(SP5), setup_name="Leica SP5", channels=[0, 1])
    assert len(frc["correlation"]) == len(frc["frequency"])


# ── services / client ───────────────────────────────────────────────────────


def test_register_services():
    from chisurf.plugins.microscopy.clsm.api import contract
    from chisurf.plugins.microscopy.clsm.backend.services import register_services
    from chisurf.server.dispatcher import ServiceDispatcher
    from chisurf.server.session import SessionState

    dispatcher = ServiceDispatcher(SessionState())
    register_services(dispatcher)
    for method in contract.ALL_METHODS:
        assert dispatcher.has_method(method)


def test_client_inprocess_roundtrip():
    from chisurf.plugins.microscopy.clsm.client import ClsmClient

    client = ClsmClient()
    assert client.contract()["plugin_id"] == "clsm"
    assert "Leica SP5" in client.setups()


def test_client_decay_real_data():
    _require_data()
    from chisurf.plugins.microscopy.clsm.client import ClsmClient

    dec = ClsmClient().decay(str(SP5), setup_name="Leica SP5", channels=[0, 1], threshold=0.5)
    assert dec["n_photons"] > 0


# ── cli ─────────────────────────────────────────────────────────────────────


def test_cli_contract_and_setups():
    from click.testing import CliRunner

    from chisurf.plugins.microscopy.clsm.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["contract"])
    assert result.exit_code == 0
    assert "clsm" in result.output

    result = runner.invoke(cli, ["setups", "--json"])
    assert result.exit_code == 0
    assert "Leica SP5" in result.output


# ── view-model (no Qt) ──────────────────────────────────────────────────────


def test_view_model_workflow():
    _require_data()
    from chisurf.plugins.microscopy.clsm.gui.view_model import ClsmViewModel

    vm = ClsmViewModel()
    events: list[str] = []
    vm.add_observer(events.append)

    vm.apply_preset("Leica SP5")
    vm.setup.channels_text = "0,1"
    vm.load_tttr(str(SP5))
    assert vm.setup.pixel_per_line > 0  # auto-detected

    vm.add_clsm()
    vm.add_representation()
    assert vm.current_image is not None

    vm.selection_mask = (vm.current_image > vm.current_image.mean()).astype(np.float64)
    decay = vm.recompute_decay()
    assert decay is not None and np.sum(decay["counts"]) > 0
    vm.add_decay_curve()
    assert len(vm.decay_series()) == 2  # saved curve + current selection
    assert len(vm.frc_series()) == 1
    assert {"setup", "clsm", "image", "decay"} <= set(events)


# ── gui smoke (Qt) ──────────────────────────────────────────────────────────


def test_tool_widget_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    from qtpy import QtWidgets

    from chisurf.plugins.microscopy.clsm.gui.tool import CLSMPixelSelect

    widget = CLSMPixelSelect()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert widget.model is not None
    assert widget.auto_form is not None
