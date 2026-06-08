"""GUI file-open tests.

These tests monkey-patch QFileDialog to simulate file-open dialogs
without user interaction.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import utils

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import pytest

from unittest.mock import patch

from qtpy import QtWidgets

import chisurf as cs
import chisurf.gui

cs_app = cs.gui.get_app()

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


def test_open_tcspc_via_action():
    """Dispatch dataset.add with a file path (as the open dialog would)."""
    _clear()
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=11, reading_routine='csv', is_jordi=False,
             use_header=True, matrix_columns=[], polarization='vm',
             rep_rate=10.0, dt=0.0141)
    before = len(cs.imported_datasets)
    cs.core.actions.dispatch(
        name="dataset.add",
        payload={
            "filename": os.path.abspath(
                "./test/data/tcspc/ibh_sample/Decay_577D.txt"
            ),
            "experiment_reader": None,
        },
    )
    assert len(cs.imported_datasets) == before + 1


def test_open_fcs_via_action():
    _clear()
    _set_exp("FCS", "Seidel Kristine")
    before = len(cs.imported_datasets)
    cs.core.actions.dispatch(
        name="dataset.add",
        payload={
            "filename": os.path.abspath(
                "./test/data/fcs/kristine/Kristine_with_error.cor"
            ),
            "experiment_reader": None,
        },
    )
    assert len(cs.imported_datasets) == before + 1
