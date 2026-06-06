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

import chisurf
import chisurf.gui

cs_app = chisurf.gui.get_app()

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


def test_open_tcspc_via_action():
    """Dispatch dataset.add with a file path (as the open dialog would)."""
    _clear()
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=11, reading_routine='csv', is_jordi=False,
             use_header=True, matrix_columns=[], polarization='vm',
             rep_rate=10.0, dt=0.0141)
    before = len(chisurf.imported_datasets)
    chisurf.core.actions.dispatch(
        name="dataset.add",
        payload={
            "filename": os.path.abspath(
                "./test/data/tcspc/ibh_sample/Decay_577D.txt"
            ),
            "experiment_reader": None,
        },
    )
    assert len(chisurf.imported_datasets) == before + 1


def test_open_fcs_via_action():
    _clear()
    _set_exp("FCS", "Seidel Kristine")
    before = len(chisurf.imported_datasets)
    chisurf.core.actions.dispatch(
        name="dataset.add",
        payload={
            "filename": os.path.abspath(
                "./test/data/fcs/kristine/Kristine_with_error.cor"
            ),
            "experiment_reader": None,
        },
    )
    assert len(chisurf.imported_datasets) == before + 1
