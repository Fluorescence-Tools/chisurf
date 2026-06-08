"""GUI fit-creation workflow tests.

Each test loads data for an experiment type and creates a fit.  Slow
reader-heavy tests (TTTR-based experiments) are marked ``slow``.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import utils

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import pytest

import chisurf as cs
import chisurf.gui
import chisurf.macros

cs_app = cs.gui.get_app()

# Run fit creation locally, not via server RPC, to avoid hangs
if hasattr(cs, "api") and cs.core.api is not None:
    cs.core.api.mode = "local"


def _clear():
    cs.core.actions.dispatch(name="fit.close_all", payload={})
    cs.imported_datasets.clear()


def _set_exp(exp_name: str, setup_name: str, **kw):
    gui = cs.cs
    idx = gui.comboBox_experimentSelect.findText(exp_name)
    if idx >= 0:
        gui.comboBox_experimentSelect.setCurrentIndex(idx)
        gui._refresh_experiment_ui()
    idx = gui.comboBox_setupSelect.findText(setup_name)
    if idx >= 0:
        gui.comboBox_setupSelect.setCurrentIndex(idx)
        gui._refresh_setup_ui()
    for k, v in kw.items():
        try:
            setattr(gui.current_setup, k, v)
        except Exception:
            pass


def test_tcspc_lifetime_fit():
    _clear()
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=11, reading_routine='csv', is_jordi=False,
             use_header=True, matrix_columns=[], polarization='vm',
             rep_rate=10.0, dt=0.0141)
    cs.macros.add_dataset(
        filename="./test/data/tcspc/ibh_sample/Decay_577D.txt"
    )
    before = len(cs.fits)
    cs.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [0], "model_name": "Lifetime "},
    )
    assert len(cs.fits) == before + 1


@pytest.mark.slow
@pytest.mark.parametrize("model_name", [
    "Lifetime ",
    "FRET: FD (Discrete)",
    "FRET: FD (Gaussian)",
    "FRET: PDDEM",
    "FRET: FD (Worm-like chain)",
    "Parse-Model",
    "Lifetime mixer",
    "Et-Model free",
])
def test_tcspc_each_model(model_name):
    """Create a fit for each registered TCSPC model."""
    _clear()
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=11, reading_routine='csv', is_jordi=False,
             use_header=True, matrix_columns=[], polarization='vm',
             rep_rate=10.0, dt=0.0141)
    cs.macros.add_dataset(
        filename="./test/data/tcspc/ibh_sample/Decay_577D.txt"
    )
    before = len(cs.fits)
    cs.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [0], "model_name": model_name},
    )
    assert len(cs.fits) == before + 1


def test_fcs_parse_model_fit():
    _clear()
    _set_exp("FCS", "Seidel Kristine")
    cs.macros.add_dataset(
        filename="./test/data/fcs/kristine/Kristine_with_error.cor"
    )
    before = len(cs.fits)
    cs.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [0], "model_name": "Parse-Model"},
    )
    assert len(cs.fits) == before + 1


def test_global_fit():
    """Create a Global fit using restore_global_fit_dataset first."""
    _clear()
    cs.macros.restore_global_fit_dataset(_from_controller=True)
    global_idx = None
    for i, ds in enumerate(cs.imported_datasets):
        name = str(getattr(ds, "name", "") or "")
        if "Global" in name or "global" in name.lower():
            global_idx = i
            break
    if global_idx is None:
        pytest.skip("No Global-Fit dataset available")
    before = len(cs.fits)
    cs.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [global_idx], "model_name": "Global fit"},
    )
    assert len(cs.fits) == before + 1
