def install_filter_mode_visibility(page, default_filter_mode: str):
    # Find the layout containing the burst filter combobox
    import chisurf as cs
    from chisurf.gui import QtWidgets
    layout = page.comboBox_burst_filter.parentWidget().layout()

    # Block signals to add CUSUM Burst programmatically
    page.comboBox_burst_filter.blockSignals(True)
    if page.comboBox_burst_filter.findText("CUSUM Burst") == -1:
        page.comboBox_burst_filter.addItem("CUSUM Burst")
    page.comboBox_burst_filter.blockSignals(False)

    # Connect the burst filter combobox to the actionUpdate_Values action
    page.comboBox_burst_filter.currentIndexChanged.connect(page.actionUpdate_Values.trigger)

    # Function to update parameter visibility and tooltips based on selected filter mode
    def update_parameter_visibility(filter_mode):
        is_kalman = filter_mode == "Kalman Burst"
        is_bocpd = filter_mode == "BOCPD Burst"
        is_cusum = filter_mode == "CUSUM Burst"
        is_count_rate_or_burst_or_cusum = filter_mode in ["Count rate", "Burst", "CUSUM Burst"]

        # Restore original label text
        page.label.setText("Min photons")
        page.label_12.setText("Alpha")
        page.label_13.setText("Beta")
        page.label_14.setText("Hazard")
        page.label_15.setText("Q")

        # Set specific settings for CUSUM
        if is_cusum:
            page.label_12.setText("BG Rate (cps)")
            page.label_13.setText("S/B Ratio")
            page.label_14.setText("Alpha")
            page.label_15.setText("Beta")

            page.doubleSpinBox_5.setMinimum(0.0)
            page.doubleSpinBox_5.setMaximum(1000000.0)
            page.doubleSpinBox_5.setSingleStep(100.0)
            page.doubleSpinBox_5.setDecimals(1)
            page.doubleSpinBox_5.setValue(2000.0)

            page.doubleSpinBox_6.setMinimum(0.0)
            page.doubleSpinBox_6.setMaximum(1000.0)
            page.doubleSpinBox_6.setSingleStep(1.0)
            page.doubleSpinBox_6.setDecimals(1)
            page.doubleSpinBox_6.setValue(30.0)

            page.doubleSpinBox_7.setMinimum(0.0001)
            page.doubleSpinBox_7.setMaximum(1.0)
            page.doubleSpinBox_7.setSingleStep(0.001)
            page.doubleSpinBox_7.setDecimals(4)
            page.doubleSpinBox_7.setValue(0.05)

            page.doubleSpinBox_8.setMinimum(0.0001)
            page.doubleSpinBox_8.setMaximum(1.0)
            page.doubleSpinBox_8.setSingleStep(0.001)
            page.doubleSpinBox_8.setDecimals(4)
            page.doubleSpinBox_8.setValue(0.05)

            page.spinBox.setToolTip("Minimum number of photons for a burst (L parameter).")
            page.doubleSpinBox_5.setToolTip("Expected background count rate in counts per second (cps) (m parameter).")
            page.doubleSpinBox_6.setToolTip("Signal-to-background ratio (T parameter). Set to 0 to auto-estimate.")
            page.doubleSpinBox_7.setToolTip("False alarm probability (probability of detecting a false changepoint).")
            page.doubleSpinBox_8.setToolTip("Missed detection probability (probability of failing to detect a true changepoint).")

        elif is_bocpd:
            page.doubleSpinBox_5.setMinimum(0.01)
            page.doubleSpinBox_5.setMaximum(100.0)
            page.doubleSpinBox_5.setSingleStep(0.1)
            page.doubleSpinBox_5.setDecimals(2)
            page.doubleSpinBox_5.setValue(1.0)

            page.doubleSpinBox_6.setMinimum(0.01)
            page.doubleSpinBox_6.setMaximum(100.0)
            page.doubleSpinBox_6.setSingleStep(0.1)
            page.doubleSpinBox_6.setDecimals(2)
            page.doubleSpinBox_6.setValue(0.1)

            page.doubleSpinBox_7.setMinimum(1e-10)
            page.doubleSpinBox_7.setMaximum(1.0)
            page.doubleSpinBox_7.setSingleStep(1e-5)
            page.doubleSpinBox_7.setDecimals(6)
            page.doubleSpinBox_7.setValue(1e-5)

            page.spinBox.setToolTip("Minimum number of photons for a burst.")
            page.doubleSpinBox_5.setToolTip("Alpha parameter for Gamma prior in BOCPD (prior count of photons).")
            page.doubleSpinBox_6.setToolTip("Beta parameter for Gamma prior in BOCPD (prior duration in seconds).")
            page.doubleSpinBox_7.setToolTip("Hazard rate / probability of a changepoint occurring at any time step.")

        elif is_kalman:
            page.doubleSpinBox_8.setMinimum(0.0)
            page.doubleSpinBox_8.setMaximum(100.0)
            page.doubleSpinBox_8.setSingleStep(0.01)
            page.doubleSpinBox_8.setDecimals(4)
            page.doubleSpinBox_8.setValue(0.01)

            page.spinBox.setToolTip("Minimum number of photons for a burst.")
            page.doubleSpinBox_8.setToolTip("Process noise covariance Q for state transition.")
            page.doubleSpinBox_9.setToolTip("Measurement noise scaling parameter R.")
            page.doubleSpinBox_10.setToolTip("Z-score threshold for burst detection.")
            page.spinBox_9.setToolTip("Minimum burst length in bins.")
            page.spinBox_7.setToolTip("Maximum gap in bins between bursts to merge them.")
            
        elif filter_mode == "Count rate":
            # Set Tooltips dynamically
            page.spinBox.setToolTip("Minimum number of photons in the time window to define a burst.")
            page.spinBox_8.setToolTip("Number of photons to compute a local count rate.")
            page.doubleSpinBox.setToolTip("Time window size in milliseconds for count rate computation.")
            
        elif filter_mode == "Burst":
            # Set Tooltips dynamically
            page.spinBox.setToolTip("Minimum number of photons for a burst (L parameter).")
            page.spinBox_8.setToolTip("Number of consecutive photons for rate calculation (m parameter).")
            page.doubleSpinBox.setToolTip("Maximum time window in milliseconds for rate calculation (T parameter).")

        # Show/hide Kalman filter parameters (label_15, label_16, label_17, doubleSpinBox_8, doubleSpinBox_9, doubleSpinBox_10)
        # Note: doubleSpinBox_8 is shared between Kalman (Q) and CUSUM (Beta)
        page.label_15.setVisible(is_kalman or is_cusum)
        page.doubleSpinBox_8.setVisible(is_kalman or is_cusum)
        
        page.label_16.setVisible(is_kalman)
        page.doubleSpinBox_9.setVisible(is_kalman)
        
        page.label_17.setVisible(is_kalman)
        page.doubleSpinBox_10.setVisible(is_kalman)
        
        page.label_18.setVisible(is_kalman)
        page.spinBox_9.setVisible(is_kalman)

        # Show/hide BOCPD parameters (label_12, label_13, label_14, doubleSpinBox_5, doubleSpinBox_6, doubleSpinBox_7)
        # Shared with CUSUM (BG Rate, S/B Ratio, Alpha)
        page.label_12.setVisible(is_bocpd or is_cusum)
        page.doubleSpinBox_5.setVisible(is_bocpd or is_cusum)
        
        page.label_13.setVisible(is_bocpd or is_cusum)
        page.doubleSpinBox_6.setVisible(is_bocpd or is_cusum)
        
        page.label_14.setVisible(is_bocpd or is_cusum)
        page.doubleSpinBox_7.setVisible(is_bocpd or is_cusum)

        # Show/hide count rate & burstwise parameters (label, label_11, spinBox, spinBox_8)
        # label_2 and doubleSpinBox should be shown only for "Count rate"
        is_count_rate = filter_mode == "Count rate"
        page.label_2.setVisible(is_count_rate)
        page.label.setVisible(is_count_rate_or_burst_or_cusum)
        page.label_11.setVisible(filter_mode in ["Count rate", "Burst"])
        page.doubleSpinBox.setVisible(is_count_rate)
        page.spinBox.setVisible(is_count_rate_or_burst_or_cusum)
        page.spinBox_8.setVisible(filter_mode in ["Count rate", "Burst"])

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
    elif default_filter_mode == 'cusum':
        page.comboBox_burst_filter.setCurrentText("CUSUM Burst")
