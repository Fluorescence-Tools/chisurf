import os
import sys
import glob
import pytest
from qtpy.QtCore import Qt

# Ensure search paths are set up
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
    assert kappa2_form.doubleSpinBox_2.value() == 0.380
    assert kappa2_form.doubleSpinBox.value() == 0.050
    assert kappa2_form.doubleSpinBox_5.value() == 0.100
    assert kappa2_form.doubleSpinBox_3.value() == 1.500000
    assert kappa2_form.spinBox.value() == 131
    assert kappa2_form.doubleSpinBox_7.value() == 0.005

def test_kappa2_calculation_1(kappa2_form, qtbot):
    ok_button = kappa2_form.pushButton
    qtbot.mouseClick(ok_button, Qt.LeftButton)

    assert round(kappa2_form.doubleSpinBox_10.value(), 1) == 0.8
    assert round(kappa2_form.doubleSpinBox_9.value(), 1) == 0.2
    assert round(kappa2_form.doubleSpinBox_6.value(), 1) == 1.0
    assert round(kappa2_form.doubleSpinBox_8.value(), 1) == 0.0

def test_kappa2_calculation_2(kappa2_form, qtbot):
    check_box = kappa2_form.checkBox
    check_box.setCheckState(Qt.Checked)

    ok_button = kappa2_form.pushButton
    qtbot.mouseClick(ok_button, Qt.LeftButton)

    assert round(kappa2_form.doubleSpinBox_10.value(), 1) == 0.7
    assert round(kappa2_form.doubleSpinBox_9.value(), 1) == 0.2
    assert round(kappa2_form.doubleSpinBox_6.value(), 1) == 1.0
    assert round(kappa2_form.doubleSpinBox_8.value(), 1) == 0.0


# --- HistogramTTTR Tests ---

def test_histogram_load_data(histogram_form, qtbot):
    make_decay_button = histogram_form.tcspc_setup_widget.pushButton

    assert len(histogram_form.curve_selector.get_data_sets()) == 0

    spcFileWidget = histogram_form.tcspc_setup_widget.spcFileWidget
    filenames = glob.glob("./test/data/tttr/BH/132/*.spc")
    file_type = "bh132"
    
    # Simulate loading files
    spcFileWidget.onLoadSample(
        event=None,
        filenames=filenames,
        file_type=file_type
    )
    
    qtbot.mouseClick(make_decay_button, Qt.LeftButton)

    assert len(histogram_form.curve_selector.get_data_sets()) == 1
