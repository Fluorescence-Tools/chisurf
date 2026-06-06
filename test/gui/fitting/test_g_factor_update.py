from qtpy.QtWidgets import QLineEdit

from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizard


def test_g_factor_update(qtbot):
    wizard = DetectorWizard()
    qtbot.addWidget(wizard)

    page = wizard.page(0)

    page.new_detector_le.setText("test_detector_1")
    page._add_detector()
    page.new_detector_le.setText("test_detector_2")
    page._add_detector()

    page.detectors_form.selectRow(0)

    row = 0
    g_factor_value = "1.234"

    page.detectors_form.removeCellWidget(row, 3)
    new_cell_widget = QLineEdit(g_factor_value)
    page.detectors_form.setCellWidget(row, 3, new_cell_widget)

    cell_widget = page.detectors_form.cellWidget(row, 3)
    assert cell_widget is not None
    assert cell_widget.text() == g_factor_value

    row = 1
    g_factor_value = "2.345"

    page.detectors_form.removeCellWidget(row, 3)
    new_cell_widget = QLineEdit(g_factor_value)
    page.detectors_form.setCellWidget(row, 3, new_cell_widget)

    cell_widget = page.detectors_form.cellWidget(row, 3)
    assert cell_widget is not None
    assert cell_widget.text() == g_factor_value

    row = 0
    expected_value = "1.234"
    cell_widget = page.detectors_form.cellWidget(row, 3)
    assert cell_widget is not None
    assert cell_widget.text() == expected_value
