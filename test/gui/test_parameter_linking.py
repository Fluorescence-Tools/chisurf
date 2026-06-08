"""GUI parameter-linking tests.

Creates two fits and exercises link/unlink actions.
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

if hasattr(cs, "api") and cs.core.api is not None:
    cs.core.api.mode = "local"


def _clear():
    cs.core.actions.dispatch(name="fit.close_all", payload={})
    cs.imported_datasets.clear()


def _setup():
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

    cs.macros.add_dataset(
        filename="./test/data/tcspc/ibh_sample/Decay_577D.txt"
    )
    cs.macros.add_dataset(
        filename="./test/data/tcspc/ibh_sample/Prompt.txt"
    )

    cs.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [0], "model_name": "Lifetime "},
    )
    cs.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [1], "model_name": "Lifetime "},
    )


def test_two_fits_created():
    _setup()
    assert len(cs.fits) == 2


def test_fits_have_parameters():
    _setup()
    for fit in cs.fits:
        model = getattr(fit, "model", None)
        assert model is not None
        params = list(getattr(model, "parameters_all", []) or [])
        assert len(params) > 0


def test_within_fit_parameter_linking():
    _setup()
    fit = cs.fits[0]
    model = fit.model
    params = list(getattr(model, "parameters_all", []) or [])
    assert len(params) >= 2

    p1 = params[0]
    p2 = params[1]
    p1_name = str(getattr(p1, "name", ""))
    p2_name = str(getattr(p2, "name", ""))

    cs.core.actions.dispatch(
        name="parameter.link",
        payload={
            "source_fit_index": 0,
            "source_parameter": p1_name,
            "target_fit_index": 0,
            "target_parameter": p2_name,
        },
    )


def test_cross_fit_parameter_linking():
    _setup()
    fit0 = cs.fits[0]
    fit1 = cs.fits[1]
    model0 = fit0.model
    model1 = fit1.model
    params0 = list(getattr(model0, "parameters_all", []) or [])
    params1 = list(getattr(model1, "parameters_all", []) or [])
    assert len(params0) >= 1
    assert len(params1) >= 1

    p1 = params0[0]
    p2 = params1[0]
    p1_name = str(getattr(p1, "name", ""))
    p2_name = str(getattr(p2, "name", ""))

    cs.core.actions.dispatch(
        name="parameter.link",
        payload={
            "source_fit_index": 0,
            "source_parameter": p1_name,
            "target_fit_index": 1,
            "target_parameter": p2_name,
        },
    )


def test_parameter_unlink():
    _setup()
    fit = cs.fits[0]
    model = fit.model
    params = list(getattr(model, "parameters_all", []) or [])
    assert len(params) >= 2

    p1 = params[0]
    p2 = params[1]
    p1_name = str(getattr(p1, "name", ""))
    p2_name = str(getattr(p2, "name", ""))

    cs.core.actions.dispatch(
        name="parameter.link",
        payload={
            "source_fit_index": 0,
            "source_parameter": p1_name,
            "target_fit_index": 0,
            "target_parameter": p2_name,
        },
    )

    cs.core.actions.dispatch(
        name="parameter.unlink",
        payload={"fit_index": 0, "source_parameter": p1_name},
    )
