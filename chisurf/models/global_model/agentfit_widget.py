from __future__ import annotations

import chisurf.fitting
import chisurf.plots
from chisurf.gui import QtCore, QtWidgets, QtGui
from chisurf import logging

from .agentfit import AgentFitModel
from chisurf.models import model


class AgentFitModelWidget(AgentFitModel, model.ModelWidget):
    """
    Widget for the agent-based global fit model.
    Provides UI for managing the agent-based fitting process.
    """

    # Register the AgentFitPlot in the plot_classes
    plot_classes = [
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.agent_fit.AgentFitPlot, {})
    ]

    def __init__(self, fit: chisurf.fitting.fit.Fit, settings_file: str = None):
        """
        Initialize the agent-based fit model widget.

        Args:
            fit: The fit object to use
            settings_file: Path to the YAML settings file
        """
        logging.info("AgentFitModelWidget.__init__ called")

        # Initialize both parent classes
        AgentFitModel.__init__(self, fit, settings_file=settings_file)
        model.ModelWidget.__init__(self, fit)

        # Create main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Add run button (moved up and renamed)
        self.run_agent_button = QtWidgets.QPushButton("Run")
        self.run_agent_button.clicked.connect(self.run_agent)
        main_layout.addWidget(self.run_agent_button)

        # Create fits section
        fits_group = QtWidgets.QGroupBox("Fits")
        fits_layout = QtWidgets.QVBoxLayout(fits_group)

        # Table for displaying fits
        self.fits_table = QtWidgets.QTableWidget()
        self.fits_table.setColumnCount(1)
        self.fits_table.setHorizontalHeaderLabels(["Fit Name"])
        self.fits_table.horizontalHeader().setStretchLastSection(True)
        fits_layout.addWidget(self.fits_table)

        # Buttons for managing fits
        fits_buttons_layout = QtWidgets.QHBoxLayout()
        self.add_fit_button = QtWidgets.QPushButton("Add Fit")
        self.add_fit_button.clicked.connect(self.add_fit)
        self.remove_fit_button = QtWidgets.QPushButton("Remove Fit")
        self.remove_fit_button.clicked.connect(self.remove_fit)
        self.clear_fits_button = QtWidgets.QPushButton("Clear Fits")
        self.clear_fits_button.clicked.connect(self.clear_fits)
        fits_buttons_layout.addWidget(self.add_fit_button)
        fits_buttons_layout.addWidget(self.remove_fit_button)
        fits_buttons_layout.addWidget(self.clear_fits_button)
        fits_layout.addLayout(fits_buttons_layout)

        main_layout.addWidget(fits_group)

        # Create agent settings section
        agent_group = QtWidgets.QGroupBox("Agent Settings")
        agent_layout = QtWidgets.QVBoxLayout(agent_group)

        # Add button to open settings editor
        settings_button_layout = QtWidgets.QHBoxLayout()
        self.edit_settings_button = QtWidgets.QPushButton("Edit Settings with Config Editor")
        self.edit_settings_button.clicked.connect(self.open_settings_editor)
        settings_button_layout.addWidget(self.edit_settings_button)
        agent_layout.addLayout(settings_button_layout)

        # Create form layout for settings
        form_layout = QtWidgets.QFormLayout()

        # Add max iterations setting
        self.max_iterations_spin = QtWidgets.QSpinBox()
        self.max_iterations_spin.setRange(10, 10000)
        self.max_iterations_spin.setValue(self.max_iterations)
        self.max_iterations_spin.valueChanged.connect(self.update_agent_settings)
        form_layout.addRow("Max Iterations:", self.max_iterations_spin)

        # Add step size setting
        self.step_size_spin = QtWidgets.QDoubleSpinBox()
        self.step_size_spin.setRange(0.001, 1.0)
        self.step_size_spin.setValue(self.step_size)
        self.step_size_spin.setSingleStep(0.01)
        self.step_size_spin.valueChanged.connect(self.update_agent_settings)
        form_layout.addRow("Step Size:", self.step_size_spin)

        # Add cooling rate setting
        self.cooling_spin = QtWidgets.QDoubleSpinBox()
        self.cooling_spin.setRange(0.5, 0.999)
        self.cooling_spin.setValue(self.cooling_rate)
        self.cooling_spin.setSingleStep(0.01)
        self.cooling_spin.valueChanged.connect(self.update_agent_settings)
        form_layout.addRow("Cooling Rate:", self.cooling_spin)

        # Add tolerance setting
        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-10, 1e-2)
        self.tolerance_spin.setValue(self.tolerance)
        self.tolerance_spin.setDecimals(10)
        self.tolerance_spin.valueChanged.connect(self.update_agent_settings)
        form_layout.addRow("Tolerance:", self.tolerance_spin)

        # Add explore/exploit ratio setting
        self.explore_exploit_spin = QtWidgets.QDoubleSpinBox()
        self.explore_exploit_spin.setRange(0.1, 0.9)
        self.explore_exploit_spin.setValue(self.explore_exploit_ratio)
        self.explore_exploit_spin.setSingleStep(0.05)
        self.explore_exploit_spin.valueChanged.connect(self.update_agent_settings)
        form_layout.addRow("Explore/Exploit Ratio:", self.explore_exploit_spin)

        # Add sensitivity memory setting
        self.sensitivity_memory_spin = QtWidgets.QSpinBox()
        self.sensitivity_memory_spin.setRange(1, 50)
        self.sensitivity_memory_spin.setValue(self.sensitivity_memory)
        self.sensitivity_memory_spin.valueChanged.connect(self.update_agent_settings)
        form_layout.addRow("Sensitivity Memory:", self.sensitivity_memory_spin)

        # Add fit frequency setting
        self.fit_frequency_spin = QtWidgets.QSpinBox()
        self.fit_frequency_spin.setRange(0, 100)
        self.fit_frequency_spin.setValue(self.fit_frequency)
        self.fit_frequency_spin.setSpecialValueText("Never")  # 0 means never perform scheduled fits
        self.fit_frequency_spin.valueChanged.connect(self.update_agent_settings)
        form_layout.addRow("Fit Frequency (iterations):", self.fit_frequency_spin)

        # Add state machine info section
        state_group = QtWidgets.QGroupBox("State Machine Info")
        state_layout = QtWidgets.QVBoxLayout(state_group)

        # Current state display
        self.state_label = QtWidgets.QLabel("Current State: EXPLORE")
        state_layout.addWidget(self.state_label)

        # Parameter sensitivities display
        self.sensitivities_table = QtWidgets.QTableWidget()
        self.sensitivities_table.setColumnCount(2)
        self.sensitivities_table.setHorizontalHeaderLabels(["Parameter", "Sensitivity"])
        self.sensitivities_table.horizontalHeader().setStretchLastSection(True)
        state_layout.addWidget(self.sensitivities_table)

        agent_layout.addWidget(state_group)

        # Create a groupbox for agent parameters
        params_group = QtWidgets.QGroupBox("Agent Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        params_layout.addLayout(form_layout)
        agent_layout.addWidget(params_group)

        main_layout.addWidget(agent_group)

        # Update the UI with current fits
        self.update_widgets()

    def open_settings_editor(self):
        """Open the settings editor for the agent settings."""
        from chisurf.gui.widgets.settings_editor import SettingsEditor

        # Documentation for settings
        documentation = {
            'max_iterations': 'Maximum number of iterations for the agent',
            'step_size': 'Initial step size for parameter changes',
            'cooling_rate': 'Rate at which step size decreases',
            'tolerance': 'Convergence tolerance for chi-squared improvement',
            'explore_exploit_ratio': 'Ratio of exploration vs exploitation (0-1)',
            'sensitivity_memory': 'Number of parameter changes to remember for sensitivity analysis',
            'fit_frequency': 'How often to perform a fit (every N iterations, 0 = never)'
        }

        # Create and show the settings editor
        editor = SettingsEditor(
            filename=self.settings_file,
            documentation_dict=documentation,
            window_title="Agent Fit Settings Editor"
        )
        editor.show()

        # Connect the editor's save signal to update our settings
        editor.save_button.clicked.connect(self.reload_settings_from_file)

    def _load_settings(self) -> dict:
        """
        Load settings from the YAML file.

        Returns:
            dict: The settings dictionary loaded from the YAML file, or an empty dict if the file doesn't exist.
        """
        import os
        import yaml

        settings = {}
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    settings = yaml.safe_load(f)
                logging.info(f"Loaded agent settings from {self.settings_file}")
            except Exception as e:
                logging.error(f"Error loading agent settings from {self.settings_file}: {str(e)}")
        else:
            logging.warning(f"Settings file {self.settings_file} not found, using default settings")

        return settings or {}

    def reload_settings_from_file(self):
        """Reload settings from the YAML file and update the UI."""
        # Load settings from file
        settings = self._load_settings()

        # Update instance variables
        self.max_iterations = settings.get('max_iterations', self.max_iterations)
        self.step_size = settings.get('step_size', self.step_size)
        self.cooling_rate = settings.get('cooling_rate', self.cooling_rate)
        self.tolerance = settings.get('tolerance', self.tolerance)
        self.explore_exploit_ratio = settings.get('explore_exploit_ratio', self.explore_exploit_ratio)
        self.sensitivity_memory = settings.get('sensitivity_memory', self.sensitivity_memory)
        self.fit_frequency = settings.get('fit_frequency', self.fit_frequency)

        # Update UI controls
        self.update_widgets()

        logging.info("Reloaded agent settings from file")

    def save_settings(self) -> None:
        """
        Save the current settings to the YAML file.
        """
        import yaml

        settings = {
            'max_iterations': self.max_iterations,
            'step_size': self.step_size,
            'cooling_rate': self.cooling_rate,
            'tolerance': self.tolerance,
            'explore_exploit_ratio': self.explore_exploit_ratio,
            'sensitivity_memory': self.sensitivity_memory,
            'fit_frequency': self.fit_frequency
        }

        try:
            with open(self.settings_file, 'w') as f:
                yaml.dump(settings, f, default_flow_style=False)
            logging.info(f"Saved agent settings to {self.settings_file}")
        except Exception as e:
            logging.error(f"Error saving agent settings to {self.settings_file}: {str(e)}")

    def update_agent_settings(self):
        """Update the agent settings from the UI controls and save to file."""
        self.max_iterations = self.max_iterations_spin.value()
        self.step_size = self.step_size_spin.value()
        self.cooling_rate = self.cooling_spin.value()
        self.tolerance = self.tolerance_spin.value()
        self.explore_exploit_ratio = self.explore_exploit_spin.value()
        self.sensitivity_memory = self.sensitivity_memory_spin.value()
        self.fit_frequency = self.fit_frequency_spin.value()

        # Save settings to file
        self.save_settings()

        logging.info("Updated and saved agent settings")

    def update_state_machine_info(self):
        """Update the state machine info display."""
        # Update current state
        if hasattr(self, 'current_state'):
            self.state_label.setText(f"Current State: {self.current_state.name}")

        # Update parameter sensitivities
        if hasattr(self, 'parameter_sensitivities') and self.parameter_sensitivities:
            # Sort parameters by sensitivity (highest first)
            sorted_sensitivities = sorted(
                self.parameter_sensitivities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )

            # Update table
            self.sensitivities_table.setRowCount(len(sorted_sensitivities))
            for i, (param_name, sensitivity) in enumerate(sorted_sensitivities):
                # Parameter name
                name_item = QtWidgets.QTableWidgetItem(param_name)
                self.sensitivities_table.setItem(i, 0, name_item)

                # Sensitivity value
                sensitivity_item = QtWidgets.QTableWidgetItem(f"{sensitivity:.6g}")
                # Color code: green for positive sensitivity (good), red for negative (bad)
                if sensitivity > 0:
                    sensitivity_item.setForeground(QtGui.QBrush(QtGui.QColor('green')))
                else:
                    sensitivity_item.setForeground(QtGui.QBrush(QtGui.QColor('red')))
                self.sensitivities_table.setItem(i, 1, sensitivity_item)

            self.sensitivities_table.resizeColumnsToContents()

    def update_widgets(self):
        """Update the UI with current data."""
        logging.info("AgentFitModelWidget.update_widgets called")

        # Update fits table
        self.fits_table.setRowCount(0)
        for i, fit in enumerate(self.fits):
            self.fits_table.insertRow(i)
            name_item = QtWidgets.QTableWidgetItem(fit.name)
            name_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.fits_table.setItem(i, 0, name_item)

        self.fits_table.resizeRowsToContents()

        # Update agent settings
        self.max_iterations_spin.setValue(self.max_iterations)
        self.step_size_spin.setValue(self.step_size)
        self.cooling_spin.setValue(self.cooling_rate)
        self.tolerance_spin.setValue(self.tolerance)
        self.explore_exploit_spin.setValue(self.explore_exploit_ratio)
        self.sensitivity_memory_spin.setValue(self.sensitivity_memory)
        self.fit_frequency_spin.setValue(self.fit_frequency)

        # Update state machine info
        self.update_state_machine_info()

    def add_fit(self):
        """Add a fit to the model."""
        logging.info("AgentFitModelWidget.add_fit called")

        # Get available fits from chisurf.fits
        import chisurf
        available_fits = [
            fit for fit in chisurf.fits
            if isinstance(fit, chisurf.fitting.fit.Fit) and fit.model is not self
        ]

        if not available_fits:
            logging.warning("No available fits to add")
            return

        # Create a dialog to select a fit
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Fit")
        layout = QtWidgets.QVBoxLayout(dialog)

        combo = QtWidgets.QComboBox()
        combo.addItems([fit.name for fit in available_fits])
        layout.addWidget(combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_index = combo.currentIndex()
            if 0 <= selected_index < len(available_fits):
                selected_fit = available_fits[selected_index]
                self.append_fit(selected_fit)
                self.update_widgets()
                logging.info(f"Added fit: {selected_fit.name}")

    def remove_fit(self):
        """Remove the selected fit from the model."""
        logging.info("AgentFitModelWidget.remove_fit called")

        selected_row = self.fits_table.currentRow()
        if 0 <= selected_row < len(self.fits):
            fit = self.fits[selected_row]
            self.remove_local_fit(selected_row)
            self.update_widgets()
            logging.info(f"Removed fit at index {selected_row}: {fit.name}")
        else:
            logging.warning("No fit selected to remove")

    def clear_fits(self):
        """Clear all fits from the model."""
        logging.info("AgentFitModelWidget.clear_fits called")

        self.clear_local_fits()
        self.update_widgets()
        logging.info("Cleared all fits")

    def append_fit(self, fit: chisurf.fitting.fit.Fit):
        """Append a fit to the model."""
        logging.info(f"AgentFitModelWidget.append_fit called with fit: {fit.name}")

        if fit not in self.fits:
            AgentFitModel.append_fit(self, fit)
            self.update_widgets()
            logging.info(f"Appended fit: {fit.name}")

    def run_agent(self):
        """Run the agent-based optimization."""
        logging.info("AgentFitModelWidget.run_agent called")
        self.update_agent_settings()
        logging.info(f"Agent settings: max_iterations={self.max_iterations}, step_size={self.step_size}, cooling_rate={self.cooling_rate}, tolerance={self.tolerance}, explore_exploit_ratio={self.explore_exploit_ratio}, sensitivity_memory={self.sensitivity_memory}, fit_frequency={self.fit_frequency}")

        # Disable the button while running
        self.run_agent_button.setEnabled(False)
        self.run_agent_button.setText("Running...")

        # Create a timer to update the state machine info during optimization
        update_timer = QtCore.QTimer()
        update_timer.timeout.connect(self.update_state_machine_info)
        update_timer.start(500)  # Update every 500ms

        try:
            # Run the agent directly in the main thread
            logging.info("Running agent directly in main thread")
            # Call the parent class's run_agent method
            logging.info("Calling AgentFitModel.run_agent")
            AgentFitModel.run_agent(self)
            logging.info("AgentFitModel.run_agent completed")
        except Exception as e:
            logging.error(f"Error running agent: {str(e)}")
        finally:
            # Stop the update timer
            update_timer.stop()

            # Update UI when done
            self.run_agent_button.setEnabled(True)
            self.run_agent_button.setText("Run")

            # Update the state machine info and plots
            self.update_state_machine_info()
            self.update_plots()

            # Show a message with optimization results
            if hasattr(self, 'best_chi2') and hasattr(self, 'state_transitions'):
                QtWidgets.QMessageBox.information(
                    self,
                    "Optimization Complete",
                    f"Optimization completed with final chi² = {self.best_chi2:.6g}\n"
                    f"State transitions: {self.state_transitions}\n"
                    f"Total iterations: {self.current_iteration}"
                )
