def setup_connections(page):
    """
    Set up all signal-slot connections for UI elements.
    This centralizes all connections in one place for better maintainability.
    """
    import tttrlib
    import chisurf as cs
    import chisurf.gui.decorators
    from chisurf.gui import QtCore, QtWidgets
    from .tttr_photon_filter_support import ProgressWindow

    # Action connections
    page.actionUpdate_Values.triggered.connect(page.update_parameter)
    page.actionUpdateUI.triggered.connect(page.updateUI)
    page.actionFile_changed.triggered.connect(page.read_tttr)
    page.actionRegionUpdate.triggered.connect(page.onRegionUpdate)

    # Tool button connections
    page.toolButton_2.toggled.connect(page.pw_mcs.setVisible)
    page.toolButton_3.toggled.connect(page.pw_decay.setVisible)
    page.toolButton_4.toggled.connect(page.pw_filter.setVisible)
    page.toolButton_7.toggled.connect(page.pw_burst_histogram.setVisible)
    page.toolButton_5.clicked.connect(page.save_selection)
    page.toolButton_6.clicked.connect(page.onClearFiles)

    # Combo box connections
    page.comboBox_2.currentTextChanged.connect(page.update_detectors)
    page.comboBox_3.currentTextChanged.connect(page.update_pie_windows)
    page.comboBox.currentTextChanged.connect(page.update_micro_time_binning)
    page.comboBox.currentTextChanged.connect(page.update_burst_selection_parameters)
    page.comboBox.currentTextChanged.connect(page.update_channel_routing)
    page.comboBox.currentTextChanged.connect(page.update_pie_windows_from_setup)

    # BOCPD element connections
    page.doubleSpinBox_5.valueChanged.connect(page.update_parameter)  # Alpha
    page.doubleSpinBox_6.valueChanged.connect(page.update_parameter)  # Beta
    page.doubleSpinBox_7.valueChanged.connect(page.update_parameter)  # Hazard

    # Gap fill checkbox connection
    page.checkBox_5.stateChanged.connect(page.update_spinbox_7_state)

    # Save parameters button connection is already connected in the UI file
    # Removing duplicate connection to prevent the save_burst_selection_parameters method from being called twice
