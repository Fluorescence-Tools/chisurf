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

import chisurf
import chisurf.gui
import chisurf.macros

cs_app = chisurf.gui.get_app()

if hasattr(chisurf, "api") and chisurf.core.api is not None:
    chisurf.core.api.mode = "local"


def _clear():
    chisurf.core.actions.dispatch(name="fit.close_all", payload={})
    chisurf.imported_datasets.clear()


def _setup():
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

    chisurf.macros.add_dataset(
        filename="./test/data/tcspc/ibh_sample/Decay_577D.txt"
    )
    chisurf.macros.add_dataset(
        filename="./test/data/tcspc/ibh_sample/Prompt.txt"
    )

    chisurf.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [0], "model_name": "Lifetime "},
    )
    chisurf.core.actions.dispatch(
        name="fit.add",
        payload={"dataset_indices": [1], "model_name": "Lifetime "},
    )


def test_two_fits_created():
    _setup()
    assert len(chisurf.fits) == 2


def test_fits_have_parameters():
    _setup()
    for fit in chisurf.fits:
        model = getattr(fit, "model", None)
        assert model is not None
        params = list(getattr(model, "parameters_all", []) or [])
        assert len(params) > 0


def test_within_fit_parameter_linking():
    _setup()
    fit = chisurf.fits[0]
    model = fit.model
    params = list(getattr(model, "parameters_all", []) or [])
    assert len(params) >= 2

    p1 = params[0]
    p2 = params[1]
    p1_name = str(getattr(p1, "name", ""))
    p2_name = str(getattr(p2, "name", ""))

    chisurf.core.actions.dispatch(
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
    fit0 = chisurf.fits[0]
    fit1 = chisurf.fits[1]
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

    chisurf.core.actions.dispatch(
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
    fit = chisurf.fits[0]
    model = fit.model
    params = list(getattr(model, "parameters_all", []) or [])
    assert len(params) >= 2

    p1 = params[0]
    p2 = params[1]
    p1_name = str(getattr(p1, "name", ""))
    p2_name = str(getattr(p2, "name", ""))

    chisurf.core.actions.dispatch(
        name="parameter.link",
        payload={
            "source_fit_index": 0,
            "source_parameter": p1_name,
            "target_fit_index": 0,
            "target_parameter": p2_name,
        },
    )

    chisurf.core.actions.dispatch(
        name="parameter.unlink",
        payload={"fit_index": 0, "source_parameter": p1_name},
    )
