import os
import sys
import glob
import pytest
from qtpy.QtCore import Qt

import pathlib
TOPDIR = pathlib.Path(__file__).parent.parent
import utils
utils.set_search_paths(TOPDIR)

from chisurf.plugins.kappa2_dist.k2dgui import Kappa2Dist
from chisurf.plugins.tttr.tttr_histogram.gui import HistogramTTTR


@pytest.fixture
def kappa2_form(qtbot):
    form = Kappa2Dist()
    qtbot.addWidget(form)
    return form


@pytest.fixture
def histogram_form(qtbot):
    form = HistogramTTTR()
    qtbot.addWidget(form)
    return form


# --- Kappa2Dist Tests ---

def test_kappa2_defaults(kappa2_form):
    assert kappa2_form._model.r_0 == 0.380
    assert kappa2_form._model.r_Dinf == 0.050
    assert kappa2_form._model.r_Ainf == 0.100
    assert kappa2_form._model.step == 1.5
    assert kappa2_form._model.n_bins == 131
    assert kappa2_form._model.r_ADinf == 0.005


def test_kappa2_calculation_1(kappa2_form, qtbot):
    ok_button = kappa2_form.pushButton
    qtbot.mouseClick(ok_button, Qt.LeftButton)

    assert round(kappa2_form._model.k2_mean, 1) == 0.7
    assert round(kappa2_form._model.k2_sd, 1) == 0.2
    assert round(kappa2_form._model.Rapp_mean, 1) == 1.0
    assert round(kappa2_form._model.RappSD, 1) == 0.0


def test_kappa2_calculation_2(kappa2_form, qtbot):
    kappa2_form._model.rAD_known = True

    ok_button = kappa2_form.pushButton
    qtbot.mouseClick(ok_button, Qt.LeftButton)

    assert round(kappa2_form._model.k2_mean, 1) == 0.7
    assert round(kappa2_form._model.k2_sd, 1) == 0.2
    assert round(kappa2_form._model.Rapp_mean, 1) == 1.0
    assert round(kappa2_form._model.RappSD, 1) == 0.0


# --- HistogramTTTR Tests ---

def test_histogram_load_data(histogram_form, qtbot):
    make_decay_button = histogram_form.tcspc_setup_widget.pushButton

    assert len(histogram_form.curve_selector.get_data_sets()) == 0

    spcFileWidget = histogram_form.tcspc_setup_widget.spcFileWidget
    filenames = glob.glob("./test/data/tttr/BH/132/*.spc")
    file_type = "bh132"

    spcFileWidget.onLoadSample(
        event=None,
        filenames=filenames,
        file_type=file_type
    )

    qtbot.mouseClick(make_decay_button, Qt.LeftButton)

    assert len(histogram_form.curve_selector.get_data_sets()) == 1
