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

import chisurf
import chisurf.gui

cs_app = chisurf.gui.get_app()

if hasattr(chisurf, "api") and chisurf.core.api is not None:
    chisurf.core.api.mode = "local"

# Auto-confirm dataset removals (avoids QMessageBox blocking in offscreen mode)
_patcher = patch.object(QMessageBox, "question", return_value=QMessageBox.Yes)
_patcher.start()


def _clear():
    chisurf.core.actions.dispatch(name="fit.close_all", payload={})
    chisurf.imported_datasets.clear()


def _setup_two_ds():
    _clear()
    cs = chisurf.cs
    exp_idx = cs.comboBox_experimentSelect.findText("TCSPC")
    cs.comboBox_experimentSelect.setCurrentIndex(exp_idx)
    cs._refresh_experiment_ui()
    setup_idx = cs.comboBox_setupSelect.findText("TXT/CSV")
    cs.comboBox_setupSelect.setCurrentIndex(setup_idx)
    cs._refresh_setup_ui()
    cs.current_setup.skiprows = 11
    cs.current_setup.reading_routine = 'csv'
    cs.current_setup.is_jordi = False
    cs.current_setup.use_header = True
    cs.current_setup.matrix_columns = []
    cs.current_setup.polarization = 'vm'
    cs.current_setup.rep_rate = 10.0
    cs.current_setup.dt = 0.0141
    chisurf.core.actions.dispatch(
        name="dataset.add",
        payload={
            "filename": "./test/data/tcspc/ibh_sample/Decay_577D.txt",
            "experiment_reader": None,
        },
    )
    chisurf.core.actions.dispatch(
        name="dataset.add",
        payload={
            "filename": "./test/data/tcspc/ibh_sample/Prompt.txt",
            "experiment_reader": None,
        },
    )


def test_dataset_group_ungroup():
    _setup_two_ds()
    before = len(chisurf.imported_datasets)

    chisurf.core.actions.dispatch(
        name="dataset.group",
        payload={"dataset_indices": [0, 1]},
    )
    # Grouping replaces two entries with one group, so count drops by 1
    assert len(chisurf.imported_datasets) == before - 1

    chisurf.core.actions.dispatch(
        name="dataset.ungroup",
        payload={"dataset_indices": [0]},
    )
    assert len(chisurf.imported_datasets) == before


def test_remove_dataset():
    _setup_two_ds()
    before = len(chisurf.imported_datasets)
    chisurf.core.actions.dispatch(
        name="dataset.remove",
        payload={"dataset_indices": [0]},
    )
    assert len(chisurf.imported_datasets) == before - 1


def test_clear_datasets():
    _setup_two_ds()
    _clear()
    assert len(chisurf.imported_datasets) == 0


def test_switch_experiment():
    cs = chisurf.cs
    names = [cs.comboBox_experimentSelect.itemText(i)
             for i in range(cs.comboBox_experimentSelect.count())]
    assert "TCSPC" in names
    assert "FCS" in names
    assert "Modelling" in names

    cs.comboBox_experimentSelect.setCurrentIndex(
        cs.comboBox_experimentSelect.findText("FCS")
    )
    cs._refresh_experiment_ui()
    assert cs.current_experiment.name == "FCS"
    assert cs.comboBox_setupSelect.count() > 0

    cs.comboBox_experimentSelect.setCurrentIndex(
        cs.comboBox_experimentSelect.findText("TCSPC")
    )
    cs._refresh_experiment_ui()
    assert cs.current_experiment.name == "TCSPC"
