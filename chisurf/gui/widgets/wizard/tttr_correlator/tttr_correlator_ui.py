import chisurf.gui.decorators

import pyqtgraph as pg

from chisurf.gui import QtWidgets


def setup_ui(page):
    page.textEdit.setVisible(False)
    chisurf.gui.decorators.lineEdit_dragFile_injector(page.lineEdit_3, call=page.open_analysis_folder)

    # Preset combobox is defined in the .ui (comboBox_fcs_preset) in a row
    # directly above the "Correlation channels" group box.
    cb = getattr(page, 'comboBox_fcs_preset', None)
    if isinstance(cb, QtWidgets.QComboBox):
        page.comboBox_fcs_preset = cb
        page.comboBox_fcs_preset.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        page.comboBox_fcs_preset.clear()
        page.comboBox_fcs_preset.addItem("")
        page.comboBox_fcs_preset.currentIndexChanged.connect(page._on_fcs_preset_changed)
    else:
        page.comboBox_fcs_preset = None

    page.pw_fcs = pg.PlotWidget()
    page.pw_fcs.resize(150, 150)

    page.plot_item_fcs = page.pw_fcs.getPlotItem()
    page.plot_item_fcs.setLogMode(True, False)
    page.plot_item_fcs.setLabel('bottom', 'Correlation time, t_c (ms)')
    page.plot_item_fcs.setLabel('left', 'Correlation amplitude, G')
    page.verticalLayout_2.addWidget(page.pw_fcs)

    page.actionUpdate_ouput_path.triggered.connect(page.update_output_path)
    page.toolButton_3.clicked.connect(page.correlate_data)
    page.toolButton_4.clicked.connect(page.onClearFiles)

    if hasattr(page, 'comboBox_micro_binning'):
        page.comboBox_micro_binning.addItems(['1', '2', '4', '8', '16'])
        page.comboBox_micro_binning.setEnabled(False)
        page.checkBox_2.toggled.connect(page.comboBox_micro_binning.setEnabled)

    page._channel_defs = {}
    try:
        if hasattr(page, 'comboBox'):
            page.comboBox.currentTextChanged.connect(lambda _=None: page._on_combo_changed('A'))
        if hasattr(page, 'comboBox_2'):
            page.comboBox_2.currentTextChanged.connect(lambda _=None: page._on_combo_changed('B'))
    except Exception:
        pass

    try:
        page.lineEdit.textChanged.connect(page._on_channel_text_changed)
        page.lineEdit_2.textChanged.connect(page._on_channel_text_changed)
    except Exception:
        pass
