def install_filter_mode_visibility(page, default_filter_mode: str):
    # Find the layout containing the burst filter combobox
    import chisurf as cs
    from chisurf.gui import QtWidgets
    layout = page.comboBox_burst_filter.parentWidget().layout()

    # Connect the burst filter combobox to the actionUpdate_Values action
    page.comboBox_burst_filter.currentIndexChanged.connect(page.actionUpdate_Values.trigger)

    # Function to update parameter visibility based on selected filter mode
    def update_parameter_visibility(filter_mode):
        is_kalman = filter_mode == "Kalman Burst"
        is_bocpd = filter_mode == "BOCPD Burst"
        is_count_rate_or_burst = filter_mode in ["Count rate", "Burst"]

        # Show/hide Kalman filter parameters (label_15, label_16, label_17, doubleSpinBox_8, doubleSpinBox_9, doubleSpinBox_10)
        page.label_15.setVisible(is_kalman)  # Q parameter
        page.doubleSpinBox_8.setVisible(is_kalman)  # Q parameter
        page.label_16.setVisible(is_kalman)  # R scale parameter
        page.doubleSpinBox_9.setVisible(is_kalman)  # R scale parameter
        page.label_17.setVisible(is_kalman)  # Z threshold parameter
        page.doubleSpinBox_10.setVisible(is_kalman)  # Z threshold parameter
        page.label_18.setVisible(is_kalman)  # Min length parameter
        page.spinBox_9.setVisible(is_kalman)  # Min length parameter

        # Show/hide BOCPD parameters (label_12, label_13, label_14, doubleSpinBox_5, doubleSpinBox_6, doubleSpinBox_7)
        page.label_12.setVisible(is_bocpd)
        page.label_13.setVisible(is_bocpd)
        page.label_14.setVisible(is_bocpd)
        page.doubleSpinBox_5.setVisible(is_bocpd)
        page.doubleSpinBox_6.setVisible(is_bocpd)
        page.doubleSpinBox_7.setVisible(is_bocpd)

        # Show/hide count rate & burstwise parameters (label, label_11, spinBox, spinBox_8)
        # label_2 and doubleSpinBox should be shown only for "Count rate"
        is_count_rate = filter_mode == "Count rate"
        page.label_2.setVisible(is_count_rate)
        page.label.setVisible(is_count_rate_or_burst)
        page.label_11.setVisible(is_count_rate_or_burst)
        page.doubleSpinBox.setVisible(is_count_rate)
        page.spinBox.setVisible(is_count_rate_or_burst)
        page.spinBox_8.setVisible(is_count_rate_or_burst)

    # Assign the function to the instance
    page.update_parameter_visibility = update_parameter_visibility

    # Set up connections to show/hide parameters based on the selected filter mode
    page.comboBox_burst_filter.currentTextChanged.connect(page.update_parameter_visibility)

    # Initialize parameter visibility based on current selection
    page.update_parameter_visibility(page.comboBox_burst_filter.currentText())

    # Set the default filter mode
    if default_filter_mode == 'count_rate':
        page.comboBox_burst_filter.setCurrentText("Count rate")
    elif default_filter_mode == 'burst':
        page.comboBox_burst_filter.setCurrentText("Burst")
    elif default_filter_mode == 'bocpd':
        page.comboBox_burst_filter.setCurrentText("BOCPD Burst")
    elif default_filter_mode == 'kalman':
        page.comboBox_burst_filter.setCurrentText("Kalman Burst")
