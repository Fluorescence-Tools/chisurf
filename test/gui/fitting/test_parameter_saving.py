from qtpy import QtWidgets

from chisurf.gui.widgets.wizard.tttr_photonfilter import WizardTTTRPhotonFilter


def test_parameter_collection(qtbot):
    windows = {}
    detectors = {}

    widget = WizardTTTRPhotonFilter(windows=windows, detectors=detectors)
    qtbot.addWidget(widget)

    widget.spinBox.setValue(100)
    widget.doubleSpinBox.setValue(2.0)
    widget.checkBox.setChecked(True)
    widget.checkBox_4.setChecked(True)
    widget.checkBox_5.setChecked(True)
    widget.spinBox_7.setValue(5)
    widget.doubleSpinBox_4.setValue(1.5)
    widget.spinBox_6.setValue(50)
    widget.lineEdit_4.setText("1,2,3")
    widget.spinBox_5.setValue(8)

    params = widget.get_burst_selection_parameters()

    assert params["photon_threshold"] == 100
    assert params["count_rate_window_ms"] == 2.0
    assert params["invert_filter"] == True
    assert params["filter_active"] == True
    assert params["use_gap_fill"] == True
    assert params["max_gap"] == 5
    assert params["trace_bin_width"] == 1.5
    assert params["number_of_burst_bins"] == 50
    assert params["channels"] == [1, 2, 3]
    assert params["decay_coarse"] == 8
