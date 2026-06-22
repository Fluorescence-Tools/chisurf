"""Plot controller classes for SM Acquisition plugin."""

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QToolButton,
    QComboBox,
)
import chisurf
import numpy as np


class DecayPlotController(QWidget):
    """Plot controller for fluorescence decay settings."""

    def __init__(self, parent=None):
        """Initialize the decay plot controller."""
        super().__init__(parent)
        self.setWindowTitle("Decay Plot Settings")

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Reduced from 2 to 0
        layout.setSpacing(0)  # Reduced from 2 to 0

        # Channel selection group
        channel_group = QGroupBox("Channel Selection")
        channel_group.setFlat(True)
        channel_layout = QVBoxLayout(channel_group)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.setSpacing(0)

        self.channel_widgets = []
        for i in range(4):
            channel_widget = QWidget()
            channel_layout_inner = QHBoxLayout(channel_widget)
            channel_layout_inner.setContentsMargins(0, 0, 0, 0)

            spinbox = QSpinBox()
            spinbox.setRange(-1, 255)
            spinbox.setValue([8, 9, 10, -1][i])  # Default routing channels
            spinbox.valueChanged.connect(self.on_channel_changed)
            channel_layout_inner.addWidget(spinbox)

            checkbox = QCheckBox(f"Curve {i}")
            checkbox.setChecked(i < 3)  # Enable first 3 by default
            checkbox.stateChanged.connect(self.on_channel_changed)
            channel_layout_inner.addWidget(checkbox)

            self.channel_widgets.append((spinbox, checkbox))
            channel_layout.addWidget(channel_widget)

        layout.addWidget(channel_group)

        # Display options group
        display_group = QGroupBox("Display Options")
        display_group.setFlat(True)
        display_layout = QVBoxLayout(display_group)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(0)

        self.log_y_checkbox = QCheckBox("Log Y Scale")
        self.log_y_checkbox.setChecked(False)  # Default to linear scale
        self.log_y_checkbox.stateChanged.connect(self.on_display_changed)
        display_layout.addWidget(self.log_y_checkbox)

        layout.addWidget(display_group)

        # Update frequency control
        frequency_group = QGroupBox("Update Frequency")
        frequency_group.setFlat(True)
        frequency_layout = QHBoxLayout(frequency_group)
        frequency_layout.setContentsMargins(0, 0, 0, 0)
        frequency_layout.setSpacing(0)
        frequency_layout.addWidget(QLabel("Update every"))
        self.update_frequency_spinbox = QSpinBox()
        self.update_frequency_spinbox.setRange(1, 1000)
        self.update_frequency_spinbox.setValue(1)
        self.update_frequency_spinbox.setSuffix(" data chunks")
        frequency_layout.addWidget(self.update_frequency_spinbox)
        layout.addWidget(frequency_group)

        layout.addStretch()

    def on_channel_changed(self):
        """Handle channel checkbox changes."""
        self.update_plot()

    def on_display_changed(self):
        """Handle display option changes."""
        self.update_plot()

    def update_plot(self):
        """Update the decay plot based on current settings."""
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            manager.update_decay_plot()


class CorrelationPlotController(QWidget):
    """Plot controller for correlation curve settings."""

    def __init__(self, parent=None):
        """Initialize the correlation plot controller."""
        super().__init__(parent)
        self.setWindowTitle("Correlation Plot Settings")

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Reduced from 2 to 0
        layout.setSpacing(0)  # Reduced from 2 to 0

        # Correlation curve controls group
        correlation_group = QGroupBox("Correlation Curves")
        correlation_layout = QVBoxLayout(correlation_group)
        correlation_layout.setContentsMargins(0, 0, 0, 0)
        correlation_layout.setSpacing(0)

        self.correlation_widgets = []
        for i in range(4):
            curve_widget = QWidget()
            curve_layout_inner = QHBoxLayout(curve_widget)
            curve_layout_inner.setContentsMargins(0, 0, 0, 0)

            # Channel A spinbox
            label_a = QLabel("Ch A:")
            curve_layout_inner.addWidget(label_a)
            spinbox_a = QSpinBox()
            spinbox_a.setRange(-1, 255)
            spinbox_a.setValue(-1)  # Default to all photons
            spinbox_a.valueChanged.connect(self.on_channel_changed)
            curve_layout_inner.addWidget(spinbox_a)

            # Channel B spinbox
            label_b = QLabel("Ch B:")
            curve_layout_inner.addWidget(label_b)
            spinbox_b = QSpinBox()
            spinbox_b.setRange(-1, 255)
            spinbox_b.setValue(-1)  # Default to all photons
            spinbox_b.valueChanged.connect(self.on_channel_changed)
            curve_layout_inner.addWidget(spinbox_b)

            # Enable checkbox
            checkbox = QCheckBox(f"Curve {i}")
            checkbox.setChecked(i == 0)  # Enable first by default
            checkbox.stateChanged.connect(self.on_channel_changed)
            curve_layout_inner.addWidget(checkbox)

            self.correlation_widgets.append((spinbox_a, spinbox_b, checkbox))
            correlation_layout.addWidget(curve_widget)

        # Info label for channel -1 meaning
        info_label = QLabel("Channel -1 = All Channels")
        info_label.setStyleSheet("font-style: italic; color: #666;")
        correlation_layout.addWidget(info_label)

        layout.addWidget(correlation_group)

        # Update frequency control
        frequency_group = QGroupBox("Correlation Update")
        frequency_layout = QHBoxLayout(frequency_group)
        frequency_layout.setContentsMargins(0, 0, 0, 0)
        frequency_layout.setSpacing(0)
        frequency_layout.addWidget(QLabel("Correlate every"))
        self.update_frequency_spinbox = QSpinBox()
        self.update_frequency_spinbox.setRange(1, 1000)
        self.update_frequency_spinbox.setValue(5)  # Default to every 5 chunks
        self.update_frequency_spinbox.setSuffix(" chunks")
        self.update_frequency_spinbox.setToolTip(
            "Accumulate photons from N data chunks before computing and displaying correlation.\n"
            "Higher values = smoother updates but less frequent."
        )
        frequency_layout.addWidget(self.update_frequency_spinbox)
        layout.addWidget(frequency_group)

        layout.addStretch()

    def on_channel_changed(self):
        """Handle channel selection changes."""
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            manager._compute_correlation(None, None, None)
            manager.update_correlation_plot()

    def update_plot(self):
        """Update the correlation plot based on current settings."""
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            manager.update_correlation_plot()


class CountRatePlotController(QWidget):
    """Plot controller for count rate trace settings."""

    def __init__(self, parent=None):
        """Initialize the count rate plot controller."""
        super().__init__(parent)
        self.setWindowTitle("Count Rate Plot Settings")

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Reduced from 2 to 0
        layout.setSpacing(0)  # Reduced from 2 to 0

        # Channel selection group
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout(channel_group)

        # Individual channel spinboxes with checkboxes and "Set to mean" buttons
        self.channel_widgets = []
        for i in range(5):
            channel_widget = QWidget()
            channel_layout_inner = QHBoxLayout(channel_widget)
            channel_layout_inner.setContentsMargins(0, 0, 0, 0)
            channel_layout_inner.setSpacing(0)

            if i < 4:
                spinbox = QSpinBox()
                spinbox.setRange(-1, 255)
                # Default: first curve plots sum (-1), others keep routing defaults
                default_channels = [-1, 9, 10, 8]
                spinbox.setValue(default_channels[i])
                spinbox.valueChanged.connect(self.on_channel_changed)
                channel_layout_inner.addWidget(spinbox)

                checkbox = QCheckBox(f"Curve {i}")
                # Default: only first curve enabled
                checkbox.setChecked(i == 0)
                checkbox.stateChanged.connect(self.on_channel_changed)
                channel_layout_inner.addWidget(checkbox)

                set_mean_btn = QToolButton()  # Changed from QPushButton to QToolButton
                set_mean_btn.setText("Set to mean")
                set_mean_btn.clicked.connect(lambda checked, ch=i: self.set_mean_line(ch))
                channel_layout_inner.addWidget(set_mean_btn)

                self.channel_widgets.append((spinbox, checkbox, set_mean_btn))
            else:
                # Label explaining channel -1 means all channels, with "Set to mean" button
                label = QLabel("Channel -1 = All Channels")
                label.setStyleSheet("font-style: italic; color: #666;")
                channel_layout_inner.addWidget(label)

                set_mean_btn = QToolButton()  # Changed from QPushButton to QToolButton
                set_mean_btn.setText("Set to mean")
                set_mean_btn.clicked.connect(lambda: self.set_mean_line(4))  # 4 = All channels
                channel_layout_inner.addWidget(set_mean_btn)

                self.channel_widgets.append((None, None, set_mean_btn))  # For consistency

            channel_layout.addWidget(channel_widget)

        layout.addWidget(channel_group)

        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)

        self.log_y_checkbox = QCheckBox("Log Y Scale")
        self.log_y_checkbox.setChecked(False)  # Default to linear scale
        self.log_y_checkbox.stateChanged.connect(self.on_display_changed)
        display_layout.addWidget(self.log_y_checkbox)

        layout.addWidget(display_group)

        # Rolling window group
        rolling_group = QGroupBox("Rolling Window")
        rolling_group.setFlat(True)
        rolling_layout = QVBoxLayout(rolling_group)
        rolling_layout.setContentsMargins(0, 0, 0, 0)
        rolling_layout.setSpacing(0)

        # Window size control
        window_layout = QHBoxLayout()
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.setSpacing(0)
        window_layout.addWidget(QLabel("Window Size:"))
        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(10, 10000)  # 10 to 10,000 data points
        self.window_size_spinbox.setValue(500)  # Default 500 points
        self.window_size_spinbox.setSuffix(" points")
        window_layout.addWidget(self.window_size_spinbox)
        rolling_layout.addLayout(window_layout)

        # Binning control
        binning_layout = QHBoxLayout()
        binning_layout.setContentsMargins(0, 0, 0, 0)
        binning_layout.setSpacing(0)
        binning_layout.addWidget(QLabel("Binning:"))
        self.binning_spinbox = QSpinBox()
        self.binning_spinbox.setRange(1, 100)  # 1 to 100 points per bin
        self.binning_spinbox.setValue(1)  # Default no binning
        self.binning_spinbox.setSuffix(" points/bin")
        binning_layout.addWidget(self.binning_spinbox)
        rolling_layout.addLayout(binning_layout)

        # Enable/disable rolling window
        self.rolling_window_checkbox = QCheckBox("Enable Rolling Window")
        self.rolling_window_checkbox.setChecked(True)  # Default enabled
        self.rolling_window_checkbox.stateChanged.connect(self.on_rolling_changed)
        rolling_layout.addWidget(self.rolling_window_checkbox)

        layout.addWidget(rolling_group)
        frequency_group = QGroupBox("Update Frequency")
        frequency_layout = QHBoxLayout(frequency_group)
        frequency_layout.addWidget(QLabel("Update every"))
        self.update_frequency_spinbox = QSpinBox()
        self.update_frequency_spinbox.setRange(1, 1000)
        self.update_frequency_spinbox.setValue(1)  # Default to every chunk
        self.update_frequency_spinbox.setSuffix(" data chunks")
        frequency_layout.addWidget(self.update_frequency_spinbox)
        layout.addWidget(frequency_group)

        layout.addStretch()

    def on_channel_changed(self):
        """Handle channel checkbox changes."""
        self.update_plot()

    def on_display_changed(self):
        """Handle display option changes."""
        self.update_plot()

    def on_rolling_changed(self):
        """Handle rolling window option changes."""
        self.update_plot()

    def set_mean_line(self, curve_index):
        """Add a horizontal mean line for the specified curve."""
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            # Calculate the current mean value for the curve
            if curve_index < 4:  # Individual channels
                spinbox, _, _ = self.channel_widgets[curve_index]
                channel = spinbox.value()
                if channel in {8, 9, 10}:
                    data_idx = {8: 0, 9: 1, 10: 2}[channel]
                    if len(manager.count_rate_data[data_idx]) > 0:
                        mean_value = np.mean(manager.count_rate_data[data_idx])
                        manager.add_count_rate_mean_line(curve_index, mean_value)
                elif curve_index == 3:  # 4th curve
                    if len(manager.count_rate_data[3]) > 0:
                        mean_value = np.mean(manager.count_rate_data[3])
                        manager.add_count_rate_mean_line(curve_index, mean_value)
            elif curve_index == 4:  # "All" channels
                if len(manager.count_rate_data[4]) > 0:
                    mean_value = np.mean(manager.count_rate_data[4])
                    manager.add_count_rate_mean_line(curve_index, mean_value)

    def update_plot(self):
        """Update the count rate plot based on current settings."""
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            manager.update_count_rate_plot()


class MCSPlotController(QWidget):
    """Plot controller for MCS trace settings."""

    def __init__(self, parent=None):
        """Initialize the MCS plot controller."""
        super().__init__(parent)
        self.setWindowTitle("MCS Trace Settings")

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # MCS trace controls
        mcs_group = QGroupBox("MCS Trace")
        mcs_layout = QVBoxLayout(mcs_group)

        # Bin width control (milliseconds)
        bin_width_layout = QHBoxLayout()
        bin_width_layout.addWidget(QLabel("Bin Width (ms):"))
        self.bin_width_spinbox = QDoubleSpinBox()
        self.bin_width_spinbox.setRange(0.1, 100.0)
        self.bin_width_spinbox.setValue(1.0)
        self.bin_width_spinbox.setSingleStep(0.1)
        self.bin_width_spinbox.setSuffix(" ms")
        self.bin_width_spinbox.valueChanged.connect(self.update_plot)
        bin_width_layout.addWidget(self.bin_width_spinbox)
        mcs_layout.addLayout(bin_width_layout)

        # Rollaround time control (seconds)
        rollaround_layout = QHBoxLayout()
        rollaround_layout.addWidget(QLabel("Rollaround Time (s):"))
        self.rollaround_spinbox = QDoubleSpinBox()
        self.rollaround_spinbox.setRange(0.01, 100.0)
        self.rollaround_spinbox.setValue(1.0)
        self.rollaround_spinbox.setSingleStep(0.1)
        self.rollaround_spinbox.setSuffix(" s")
        self.rollaround_spinbox.valueChanged.connect(self.update_plot)
        rollaround_layout.addWidget(self.rollaround_spinbox)
        mcs_layout.addLayout(rollaround_layout)

        # Update frequency control
        update_layout = QHBoxLayout()
        update_layout.addWidget(QLabel("Update Frequency:"))
        self.update_frequency_spinbox = QSpinBox()
        self.update_frequency_spinbox.setRange(1, 100)
        self.update_frequency_spinbox.setValue(5)  # Default every 5 chunks
        self.update_frequency_spinbox.setSingleStep(1)
        self.update_frequency_spinbox.setSuffix(" chunks")
        update_layout.addWidget(self.update_frequency_spinbox)
        mcs_layout.addLayout(update_layout)
        layout.addLayout(mcs_layout)

        # Y range controls
        y_range_group = QGroupBox("Y Range")
        y_range_layout = QVBoxLayout(y_range_group)

        # Manual/auto range toggle
        self.manual_y_range_checkbox = QCheckBox("Manual Y Range")
        self.manual_y_range_checkbox.setChecked(False)  # Default to auto range
        self.manual_y_range_checkbox.stateChanged.connect(self.update_plot)
        y_range_layout.addWidget(self.manual_y_range_checkbox)

        # Y min control
        y_min_layout = QHBoxLayout()
        y_min_layout.addWidget(QLabel("Y Min:"))
        self.y_min_spinbox = QDoubleSpinBox()
        self.y_min_spinbox.setRange(-1000000.0, 1000000.0)
        self.y_min_spinbox.setValue(0.0)
        self.y_min_spinbox.setSingleStep(100.0)
        self.y_min_spinbox.valueChanged.connect(self.update_plot)
        y_min_layout.addWidget(self.y_min_spinbox)
        y_range_layout.addLayout(y_min_layout)

        # Y max control
        y_max_layout = QHBoxLayout()
        y_max_layout.addWidget(QLabel("Y Max:"))
        self.y_max_spinbox = QDoubleSpinBox()
        self.y_max_spinbox.setRange(-1000000.0, 1000000.0)
        self.y_max_spinbox.setValue(1000.0)
        self.y_max_spinbox.setSingleStep(100.0)
        self.y_max_spinbox.valueChanged.connect(self.update_plot)
        y_max_layout.addWidget(self.y_max_spinbox)
        y_range_layout.addLayout(y_max_layout)

        # Add y_range_group to layout
        layout.addWidget(y_range_group)

        layout.addStretch()

    def update_plot(self):
        """Update the MCS plot based on current settings."""
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            manager.update_mcs_plot()


class MacrotimePlotController(QWidget):
    """Plot controller for macrotime time series settings."""

    def __init__(self, parent=None):
        """Initialize the macrotime plot controller."""
        super().__init__(parent)
        self.setWindowTitle("Macrotime Plot Settings")

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Reduced from 2 to 0
        layout.setSpacing(0)  # Reduced from 2 to 0

        # Display options
        options_group = QGroupBox("Display Options")
        options_layout = QVBoxLayout(options_group)

        self.show_macrotimes_checkbox = QCheckBox("Show Macrotime Plot")
        self.show_macrotimes_checkbox.setChecked(True)
        options_layout.addWidget(self.show_macrotimes_checkbox)

        # Plot type selection
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItem("Time Differences (dt)")
        self.plot_type_combo.addItem("Absolute Times")
        self.plot_type_combo.setCurrentText("Time Differences (dt)")  # Default to accumulated DTs
        self.plot_type_combo.currentTextChanged.connect(self.update_plot)
        options_layout.addWidget(self.plot_type_combo)

        layout.addWidget(options_group)

        # Update frequency control
        frequency_group = QGroupBox("Update Frequency")
        frequency_layout = QHBoxLayout(frequency_group)
        frequency_layout.addWidget(QLabel("Update every"))
        self.update_frequency_spinbox = QSpinBox()
        self.update_frequency_spinbox.setRange(1, 1000)
        self.update_frequency_spinbox.setValue(1)
        self.update_frequency_spinbox.setSuffix(" data chunks")
        frequency_layout.addWidget(self.update_frequency_spinbox)
        layout.addWidget(frequency_group)

        layout.addStretch()

    def update_plot(self):
        """Update the macrotime plot based on current settings."""
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            manager.update_macrotime_plot()
