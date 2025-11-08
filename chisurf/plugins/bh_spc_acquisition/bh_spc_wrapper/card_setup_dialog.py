"""
BH SPC Card Setup Dialog

This module provides a dialog for configuring the BH SPC card parameters.
It allows users to set various parameters of the card, organized into logical groups.
"""

import os
import json
from PyQt5.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QGroupBox, QDialogButtonBox, QFileDialog, QMessageBox,
    QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal

from .wrapper import BHSPC, ParID, SPCMError, InitStatus

class ParamWidget:
    """Base class for parameter widgets."""

    def __init__(self, label, param_id, device, mod_no=0):
        """Initialize the parameter widget.

        Args:
            label (str): The label for the parameter.
            param_id (ParID): The parameter ID.
            device (BHSPC): The device to get/set the parameter.
            mod_no (int): The module number.
        """
        self.label = label
        self.param_id = param_id
        self.device = device
        self.mod_no = mod_no
        self.widget = None

    def create_widget(self):
        """Create the widget for the parameter."""
        raise NotImplementedError("Subclasses must implement create_widget")

    def get_value(self):
        """Get the value from the widget."""
        raise NotImplementedError("Subclasses must implement get_value")

    def set_value(self, value):
        """Set the value in the widget."""
        raise NotImplementedError("Subclasses must implement set_value")

    def read_from_device(self):
        """Read the parameter value from the device."""
        try:
            value = self.device.get_parameter(self.mod_no, self.param_id)
            self.set_value(value)
            return True
        except SPCMError as e:
            print(f"Error reading parameter {self.param_id}: {e}")
            return False

    def write_to_device(self):
        """Write the parameter value to the device."""
        try:
            value = self.get_value()
            self.device.set_parameter(self.mod_no, self.param_id, value)
            return True
        except SPCMError as e:
            print(f"Error writing parameter {self.param_id}: {e}")
            return False


class IntParamWidget(ParamWidget):
    """Widget for integer parameters."""

    def __init__(self, label, param_id, device, mod_no=0, min_val=0, max_val=100, step=1):
        """Initialize the integer parameter widget.

        Args:
            label (str): The label for the parameter.
            param_id (ParID): The parameter ID.
            device (BHSPC): The device to get/set the parameter.
            mod_no (int): The module number.
            min_val (int): The minimum value.
            max_val (int): The maximum value.
            step (int): The step value.
        """
        super().__init__(label, param_id, device, mod_no)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

    def create_widget(self):
        """Create the widget for the parameter."""
        self.widget = QSpinBox()
        self.widget.setRange(self.min_val, self.max_val)
        self.widget.setSingleStep(self.step)
        return QLabel(self.label), self.widget

    def get_value(self):
        """Get the value from the widget."""
        return self.widget.value()

    def set_value(self, value):
        """Set the value in the widget."""
        self.widget.setValue(int(value))


class FloatParamWidget(ParamWidget):
    """Widget for float parameters."""

    def __init__(self, label, param_id, device, mod_no=0, min_val=0.0, max_val=100.0, step=0.1, decimals=2):
        """Initialize the float parameter widget.

        Args:
            label (str): The label for the parameter.
            param_id (ParID): The parameter ID.
            device (BHSPC): The device to get/set the parameter.
            mod_no (int): The module number.
            min_val (float): The minimum value.
            max_val (float): The maximum value.
            step (float): The step value.
            decimals (int): The number of decimal places.
        """
        super().__init__(label, param_id, device, mod_no)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.decimals = decimals

    def create_widget(self):
        """Create the widget for the parameter."""
        self.widget = QDoubleSpinBox()
        self.widget.setRange(self.min_val, self.max_val)
        self.widget.setSingleStep(self.step)
        self.widget.setDecimals(self.decimals)
        return QLabel(self.label), self.widget

    def get_value(self):
        """Get the value from the widget."""
        return self.widget.value()

    def set_value(self, value):
        """Set the value in the widget."""
        self.widget.setValue(float(value))


class ComboParamWidget(ParamWidget):
    """Widget for parameters with predefined options."""

    def __init__(self, label, param_id, device, mod_no=0, options=None):
        """Initialize the combo parameter widget.

        Args:
            label (str): The label for the parameter.
            param_id (ParID): The parameter ID.
            device (BHSPC): The device to get/set the parameter.
            mod_no (int): The module number.
            options (dict): A dictionary mapping option values to display names.
        """
        super().__init__(label, param_id, device, mod_no)
        self.options = options or {}

    def create_widget(self):
        """Create the widget for the parameter."""
        self.widget = QComboBox()
        for value, name in self.options.items():
            self.widget.addItem(name, value)
        return QLabel(self.label), self.widget

    def get_value(self):
        """Get the value from the widget."""
        return self.widget.currentData()

    def set_value(self, value):
        """Set the value in the widget."""
        index = self.widget.findData(value)
        if index >= 0:
            self.widget.setCurrentIndex(index)


class BHSPCCardSetupDialog(QDialog):
    """Dialog for setting up the BH SPC card."""

    def __init__(self, device, parent=None):
        """Initialize the dialog.

        Args:
            device (BHSPC): The device to configure.
            parent (QWidget): The parent widget.
        """
        super().__init__(parent)
        self.device = device
        self.mod_no = 0  # Default module number
        self.param_widgets = []
        self.available_cards = []
        self.active_cards = []

        self.setWindowTitle("BH SPC Card Setup")
        self.resize(800, 300)

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create hardware detection tab (first tab)
        self.create_hardware_tab()

        # Create tabs for different parameter groups
        self.create_cfd_tab()
        self.create_sync_tab()
        self.create_tac_tab()
        self.create_timing_tab()
        self.create_mode_tab()

        # Create buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)

        # Add save/load buttons
        save_load_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Settings...")
        self.load_button = QPushButton("Load Settings...")
        self.save_button.clicked.connect(self.save_settings)
        self.load_button.clicked.connect(self.load_settings)
        save_load_layout.addWidget(self.save_button)
        save_load_layout.addWidget(self.load_button)
        save_load_layout.addStretch()
        save_load_layout.addWidget(button_box)

        main_layout.addLayout(save_load_layout)

    def create_cfd_tab(self):
        """Create the CFD (Constant Fraction Discriminator) tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        # Add CFD parameters
        self.add_param_widget(layout, 0, 0, FloatParamWidget("CFD Limit Low (mV)", ParID.CFD_LIMIT_LOW, self.device, self.mod_no, -1000.0, 1000.0, 10.0))
        self.add_param_widget(layout, 1, 0, FloatParamWidget("CFD Limit High (mV)", ParID.CFD_LIMIT_HIGH, self.device, self.mod_no, -1000.0, 1000.0, 10.0))
        self.add_param_widget(layout, 2, 0, FloatParamWidget("CFD Zero Cross Level (mV)", ParID.CFD_ZC_LEVEL, self.device, self.mod_no, -100.0, 100.0, 1.0))
        self.add_param_widget(layout, 3, 0, FloatParamWidget("CFD Holdoff (ns)", ParID.CFD_HOLDOFF, self.device, self.mod_no, 0.0, 100.0, 1.0))

        self.tab_widget.addTab(tab, "CFD")

    def create_sync_tab(self):
        """Create the Sync tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        # Add Sync parameters
        self.add_param_widget(layout, 0, 0, FloatParamWidget("Sync Zero Cross Level (mV)", ParID.SYNC_ZC_LEVEL, self.device, self.mod_no, -100.0, 100.0, 1.0))
        self.add_param_widget(layout, 1, 0, IntParamWidget("Sync Frequency Divider", ParID.SYNC_FREQ_DIV, self.device, self.mod_no, 1, 16, 1))
        self.add_param_widget(layout, 2, 0, FloatParamWidget("Sync Holdoff (ns)", ParID.SYNC_HOLDOFF, self.device, self.mod_no, 0.0, 100.0, 1.0))
        self.add_param_widget(layout, 3, 0, FloatParamWidget("Sync Threshold (mV)", ParID.SYNC_THRESHOLD, self.device, self.mod_no, -1000.0, 1000.0, 10.0))

        self.tab_widget.addTab(tab, "Sync")

    def create_tac_tab(self):
        """Create the TAC (Time-to-Amplitude Converter) tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        # Add TAC parameters
        self.add_param_widget(layout, 0, 0, FloatParamWidget("TAC Range (ns)", ParID.TAC_RANGE, self.device, self.mod_no, 0.0, 1000.0, 10.0))

        tac_gain_options = {1: "1", 2: "2", 4: "4", 8: "8"}
        self.add_param_widget(layout, 1, 0, ComboParamWidget("TAC Gain", ParID.TAC_GAIN, self.device, self.mod_no, tac_gain_options))

        self.add_param_widget(layout, 2, 0, FloatParamWidget("TAC Offset (%)", ParID.TAC_OFFSET, self.device, self.mod_no, 0.0, 100.0, 1.0))
        self.add_param_widget(layout, 3, 0, FloatParamWidget("TAC Limit Low (%)", ParID.TAC_LIMIT_LOW, self.device, self.mod_no, 0.0, 100.0, 1.0))
        self.add_param_widget(layout, 4, 0, FloatParamWidget("TAC Limit High (%)", ParID.TAC_LIMIT_HIGH, self.device, self.mod_no, 0.0, 100.0, 1.0))
        self.add_param_widget(layout, 5, 0, FloatParamWidget("TAC Enable Hold (ns)", ParID.TAC_ENABLE_HOLD, self.device, self.mod_no, 0.0, 100.0, 1.0))

        self.tab_widget.addTab(tab, "TAC")

    def create_timing_tab(self):
        """Create the Timing tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        # Add Timing parameters
        self.add_param_widget(layout, 0, 0, FloatParamWidget("Collection Time (s)", ParID.COLLECT_TIME, self.device, self.mod_no, 0.001, 1000.0, 1.0))
        self.add_param_widget(layout, 1, 0, FloatParamWidget("Display Time (s)", ParID.DISPLAY_TIME, self.device, self.mod_no, 0.001, 1000.0, 1.0))
        self.add_param_widget(layout, 2, 0, FloatParamWidget("Repeat Time (s)", ParID.REPEAT_TIME, self.device, self.mod_no, 0.001, 1000.0, 1.0))

        stop_options = {0: "No", 1: "Yes"}
        self.add_param_widget(layout, 3, 0, ComboParamWidget("Stop on Time", ParID.STOP_ON_TIME, self.device, self.mod_no, stop_options))
        self.add_param_widget(layout, 4, 0, ComboParamWidget("Stop on Overflow", ParID.STOP_ON_OVFL, self.device, self.mod_no, stop_options))

        self.add_param_widget(layout, 5, 0, FloatParamWidget("Rate Count Time (s)", ParID.RATE_COUNT_TIME, self.device, self.mod_no, 0.001, 10.0, 0.1))

        macro_time_options = {0: "25 ns", 1: "50 ns", 2: "100 ns", 3: "200 ns", 4: "400 ns", 5: "800 ns", 6: "1.6 µs", 7: "3.2 µs"}
        self.add_param_widget(layout, 6, 0, ComboParamWidget("Macro Time Clock", ParID.MACRO_TIME_CLK, self.device, self.mod_no, macro_time_options))

        self.tab_widget.addTab(tab, "Timing")

    def create_hardware_tab(self):
        """Create the Hardware tab for card detection and selection."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Add a button to detect cards
        detect_button = QPushButton("Detect Cards")
        detect_button.clicked.connect(self.detect_cards)
        layout.addWidget(detect_button)

        # Add a group box for card selection
        self.cards_group = QGroupBox("Available Cards")
        cards_layout = QVBoxLayout(self.cards_group)

        # Add a label for instructions
        instructions = QLabel("Select the cards you want to use for acquisition:")
        cards_layout.addWidget(instructions)

        # Add a widget to display detected cards
        self.cards_widget = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_widget)
        cards_layout.addWidget(self.cards_widget)

        # Add the group box to the main layout
        layout.addWidget(self.cards_group)

        # Add a stretch to push everything to the top
        layout.addStretch()

        self.tab_widget.addTab(tab, "Hardware")

    def detect_cards(self):
        """Detect available cards and update the UI."""
        # Clear the current cards layout
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Detect available cards
        try:
            self.available_cards = self.device.detect_cards()

            if not self.available_cards:
                # No cards detected
                label = QLabel("No cards detected.")
                self.cards_layout.addWidget(label)
                return

            # Create checkboxes for each card
            self.card_checkboxes = []
            for card in self.available_cards:
                mod_no = card['module_number']
                status = card['status']
                active = card['active']

                checkbox = QCheckBox(f"Module {mod_no}: {status.message()}")
                checkbox.setChecked(active)
                checkbox.setProperty("module_number", mod_no)
                checkbox.setEnabled(status == InitStatus.INIT_OK)

                self.card_checkboxes.append(checkbox)
                self.cards_layout.addWidget(checkbox)

            # Set the active cards based on the checkboxes
            self.update_active_cards()
        except Exception as e:
            # Error detecting cards
            label = QLabel(f"Error detecting cards: {e}")
            self.cards_layout.addWidget(label)

    def update_active_cards(self):
        """Update the active cards based on the selected checkboxes."""
        self.active_cards = []
        for checkbox in self.card_checkboxes:
            if checkbox.isChecked():
                mod_no = checkbox.property("module_number")
                self.active_cards.append(mod_no)

    def create_mode_tab(self):
        """Create the Mode tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        # Add Mode parameters
        mode_options = {0: "Histogramming", 1: "FIFO", 2: "Scan", 3: "Imaging"}
        self.add_param_widget(layout, 0, 0, ComboParamWidget("Mode", ParID.MODE, self.device, self.mod_no, mode_options))

        routing_options = {0: "Off", 1: "4 Bits", 2: "16 Bits"}
        self.add_param_widget(layout, 1, 0, ComboParamWidget("Routing Mode", ParID.ROUTING_MODE, self.device, self.mod_no, routing_options))

        adc_resolution_options = {6: "6 Bits", 8: "8 Bits", 10: "10 Bits", 12: "12 Bits", 14: "14 Bits", 16: "16 Bits"}
        self.add_param_widget(layout, 2, 0, ComboParamWidget("ADC Resolution", ParID.ADC_RESOLUTION, self.device, self.mod_no, adc_resolution_options))

        self.tab_widget.addTab(tab, "Mode")

    def add_param_widget(self, layout, row, col, param_widget):
        """Add a parameter widget to the layout.

        Args:
            layout (QGridLayout): The layout to add the widget to.
            row (int): The row in the layout.
            col (int): The column in the layout.
            param_widget (ParamWidget): The parameter widget to add.
        """
        label, widget = param_widget.create_widget()
        layout.addWidget(label, row, col * 2)
        layout.addWidget(widget, row, col * 2 + 1)
        self.param_widgets.append(param_widget)

    def read_all_params(self):
        """Read all parameters from the device."""
        for widget in self.param_widgets:
            widget.read_from_device()

    def write_all_params(self):
        """Write all parameters to the device."""
        success = True
        for widget in self.param_widgets:
            if not widget.write_to_device():
                success = False
        return success

    def apply_settings(self):
        """Apply the settings to the device."""
        # Update active cards based on the checkboxes
        self.update_active_cards()

        # Set the active cards in the device
        try:
            self.device.set_active_cards(self.active_cards)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error setting active cards: {e}")

        # Write all parameters to the device
        if self.write_all_params():
            QMessageBox.information(self, "Success", "Settings applied successfully.")
        else:
            QMessageBox.warning(self, "Warning", "Some settings could not be applied.")

    def save_settings(self):
        """Save the settings to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "", "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        # Create a dictionary of parameter values
        settings = {}
        for widget in self.param_widgets:
            # Use the parameter ID name as the key
            param_name = widget.param_id.name
            param_value = widget.get_value()
            settings[param_name] = param_value

        try:
            # Write the settings to the file
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)

            QMessageBox.information(self, "Success", f"Settings saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving settings: {e}")

    def load_settings(self):
        """Load settings from a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Read the settings from the file
            with open(file_path, 'r') as f:
                settings = json.load(f)

            # Apply the settings to the widgets
            success_count = 0
            for widget in self.param_widgets:
                param_name = widget.param_id.name
                if param_name in settings:
                    widget.set_value(settings[param_name])
                    success_count += 1

            # Apply the settings to the device
            if self.write_all_params():
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Settings loaded from {file_path}\n{success_count} parameters updated"
                )
            else:
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    f"Settings loaded from {file_path}, but some could not be applied to the device"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading settings: {e}")

    def showEvent(self, event):
        """Handle the show event."""
        super().showEvent(event)

        # Detect cards
        self.detect_cards()

        # Read all parameters
        self.read_all_params()

    def accept(self):
        """Handle the accept event."""
        # Update active cards based on the checkboxes
        self.update_active_cards()

        # Set the active cards in the device
        try:
            self.device.set_active_cards(self.active_cards)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error setting active cards: {e}")

        # Write all parameters to the device
        if self.write_all_params():
            super().accept()
        else:
            QMessageBox.warning(self, "Warning", "Some settings could not be applied.")
