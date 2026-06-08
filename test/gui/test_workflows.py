"""GUI workflow tests: group/ungroup datasets, switch experiments, clear."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import utils

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import pytest
from unittest.mock import patch

from qtpy.QtWidgets import QMessageBox

import chisurf as cs
import chisurf.gui

cs_app = cs.gui.get_app()

if hasattr(cs, "api") and cs.core.api is not None:
    cs.core.api.mode = "local"

# Auto-confirm dataset removals (avoids QMessageBox blocking in offscreen mode)
_patcher = patch.object(QMessageBox, "question", return_value=QMessageBox.Yes)
_patcher.start()


def _clear():
    cs.core.actions.dispatch(name="fit.close_all", payload={})
    cs.imported_datasets.clear()


def _setup_two_ds():
    _clear()
    gui = cs.cs
    exp_idx = gui.comboBox_experimentSelect.findText("TCSPC")
    gui.comboBox_experimentSelect.setCurrentIndex(exp_idx)
    gui._refresh_experiment_ui()
    setup_idx = gui.comboBox_setupSelect.findText("TXT/CSV")
    gui.comboBox_setupSelect.setCurrentIndex(setup_idx)
    gui._refresh_setup_ui()
    gui.current_setup.skiprows = 11
    gui.current_setup.reading_routine = 'csv'
    gui.current_setup.is_jordi = False
    gui.current_setup.use_header = True
    gui.current_setup.matrix_columns = []
    gui.current_setup.polarization = 'vm'
    gui.current_setup.rep_rate = 10.0
    gui.current_setup.dt = 0.0141
    cs.core.actions.dispatch(
        name="dataset.add",
        payload={
            "filename": "./test/data/tcspc/ibh_sample/Decay_577D.txt",
            "experiment_reader": None,
        },
    )
    cs.core.actions.dispatch(
        name="dataset.add",
        payload={
            "filename": "./test/data/tcspc/ibh_sample/Prompt.txt",
            "experiment_reader": None,
        },
    )


def test_dataset_group_ungroup():
    _setup_two_ds()
    before = len(cs.imported_datasets)

    cs.core.actions.dispatch(
        name="dataset.group",
        payload={"dataset_indices": [0, 1]},
    )
    # Grouping replaces two entries with one group, so count drops by 1
    assert len(cs.imported_datasets) == before - 1

    cs.core.actions.dispatch(
        name="dataset.ungroup",
        payload={"dataset_indices": [0]},
    )
    assert len(cs.imported_datasets) == before


def test_remove_dataset():
    _setup_two_ds()
    before = len(cs.imported_datasets)
    cs.core.actions.dispatch(
        name="dataset.remove",
        payload={"dataset_indices": [0]},
    )
    assert len(cs.imported_datasets) == before - 1


def test_clear_datasets():
    _setup_two_ds()
    _clear()
    assert len(cs.imported_datasets) == 0


def test_switch_experiment():
    gui = cs.cs
    names = [gui.comboBox_experimentSelect.itemText(i)
             for i in range(gui.comboBox_experimentSelect.count())]
    assert "TCSPC" in names
    assert "FCS" in names
    assert "Modelling" in names

    gui.comboBox_experimentSelect.setCurrentIndex(
        gui.comboBox_experimentSelect.findText("FCS")
    )
    gui._refresh_experiment_ui()
    assert gui.current_experiment.name == "FCS"
    assert gui.comboBox_setupSelect.count() > 0

    gui.comboBox_experimentSelect.setCurrentIndex(
        gui.comboBox_experimentSelect.findText("TCSPC")
    )
    gui._refresh_experiment_ui()
    assert gui.current_experiment.name == "TCSPC"
