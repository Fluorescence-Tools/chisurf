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

import chisurf
import chisurf.gui
import chisurf.macros

cs_app = chisurf.gui.get_app()

# Run fit creation locally, not via server RPC, to avoid hangs
if hasattr(chisurf, "api") and chisurf.core.api is not None:
    chisurf.core.api.mode = "local"


def _clear():
    chisurf.core.actions.dispatch(name="fit.close_all", payload={})
    chisurf.imported_datasets.clear()


def _set_exp(exp_name: str, setup_name: str, **kw):
    cs = chisurf.cs
    idx = cs.comboBox_experimentSelect.findText(exp_name)
    if idx >= 0:
        cs.comboBox_experimentSelect.setCurrentIndex(idx)
        cs._refresh_experiment_ui()
    idx = cs.comboBox_setupSelect.findText(setup_name)
    if idx >= 0:
        cs.comboBox_setupSelect.setCurrentIndex(idx)
        cs._refresh_setup_ui()
    for k, v in kw.items():
        try:
            setattr(cs.current_setup, k, v)
        except Exception:
            pass


def test_tcspc_lifetime_fit():
    _clear()
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=11, reading_routine='csv', is_jordi=False,
             use_header=True, matrix_columns=[], polarization='vm',
             rep_rate=10.0, dt=0.0141)
    chisurf.macros.add_dataset(
        filename="./test/data/tcspc/ibh_sample/Decay_577D.txt"
    )
    before = len(chisurf.fits)
    chisurf.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [0], "model_name": "Lifetime "},
    )
    assert len(chisurf.fits) == before + 1


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
    chisurf.macros.add_dataset(
        filename="./test/data/tcspc/ibh_sample/Decay_577D.txt"
    )
    before = len(chisurf.fits)
    chisurf.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [0], "model_name": model_name},
    )
    assert len(chisurf.fits) == before + 1


def test_fcs_parse_model_fit():
    _clear()
    _set_exp("FCS", "Seidel Kristine")
    chisurf.macros.add_dataset(
        filename="./test/data/fcs/kristine/Kristine_with_error.cor"
    )
    before = len(chisurf.fits)
    chisurf.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [0], "model_name": "Parse-Model"},
    )
    assert len(chisurf.fits) == before + 1


def test_global_fit():
    """Create a Global fit using restore_global_fit_dataset first."""
    _clear()
    chisurf.macros.restore_global_fit_dataset(_from_controller=True)
    global_idx = None
    for i, ds in enumerate(chisurf.imported_datasets):
        name = str(getattr(ds, "name", "") or "")
        if "Global" in name or "global" in name.lower():
            global_idx = i
            break
    if global_idx is None:
        pytest.skip("No Global-Fit dataset available")
    before = len(chisurf.fits)
    chisurf.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [global_idx], "model_name": "Global fit"},
    )
    assert len(chisurf.fits) == before + 1
