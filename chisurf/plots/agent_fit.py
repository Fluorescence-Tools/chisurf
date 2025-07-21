from __future__ import annotations

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

import chisurf.fitting
import chisurf.plots
from chisurf import logging
from chisurf.plots.plotbase import Plot

__all__ = ['AgentFitPlot']


class AgentFitPlot(Plot):
    """
    Plot window for displaying agent-based fitting progress and results.
    Shows the agent's actions and weighted residuals of the fits.
    """
    name = "Agent-Fit Plot"

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            logy: bool = False,
            logx: bool = False
    ):
        super(AgentFitPlot, self).__init__(fit)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.fit = fit

        # Create main splitter
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.layout.addWidget(self.main_splitter)

        # Create chi2 history plot
        self.chi2_widget = pg.PlotWidget(title="Chi² History")
        self.chi2_widget.setLabel('left', 'Chi²')
        self.chi2_widget.setLabel('bottom', 'Iteration')
        self.chi2_plot = self.chi2_widget.plot(pen='r')
        self.main_splitter.addWidget(self.chi2_widget)

        # Create action history widget
        self.action_widget = QtWidgets.QTableWidget()
        self.action_widget.setColumnCount(6)
        self.action_widget.setHorizontalHeaderLabels([
            "Iteration", "Action", "Details", "Details", "Chi² Change", "Accepted"
        ])
        self.action_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.main_splitter.addWidget(self.action_widget)

        # Create weighted residuals plot
        self.wres_widget = pg.PlotWidget(title="Weighted Residuals")
        self.wres_widget.setLabel('left', 'Weighted Residuals')
        self.wres_widget.setLabel('bottom', 'Point Index')
        self.wres_plots = {}  # Will hold plot items for each fit
        self.main_splitter.addWidget(self.wres_widget)

        # Set initial splitter sizes
        self.main_splitter.setSizes([200, 200, 200])

        # Create control panel
        self.control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout(self.control_panel)

        # Add run button
        self.run_button = QtWidgets.QPushButton("Run Agent")
        self.run_button.clicked.connect(self.run_agent)
        control_layout.addWidget(self.run_button)

        # Add settings
        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(10, 10000)
        self.iterations_spin.setValue(100)
        self.iterations_spin.setPrefix("Iterations: ")
        control_layout.addWidget(self.iterations_spin)

        self.step_size_spin = QtWidgets.QDoubleSpinBox()
        self.step_size_spin.setRange(0.001, 1.0)
        self.step_size_spin.setValue(0.1)
        self.step_size_spin.setSingleStep(0.01)
        self.step_size_spin.setPrefix("Step Size: ")
        control_layout.addWidget(self.step_size_spin)

        self.cooling_spin = QtWidgets.QDoubleSpinBox()
        self.cooling_spin.setRange(0.5, 0.999)
        self.cooling_spin.setValue(0.95)
        self.cooling_spin.setSingleStep(0.01)
        self.cooling_spin.setPrefix("Cooling Rate: ")
        control_layout.addWidget(self.cooling_spin)

        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-10, 1e-2)
        self.tolerance_spin.setValue(1e-6)
        self.tolerance_spin.setDecimals(10)
        self.tolerance_spin.setPrefix("Tolerance: ")
        control_layout.addWidget(self.tolerance_spin)

        self.layout.addWidget(self.control_panel)

    def run_agent(self):
        """Run the agent-based optimization when the button is clicked."""
        # Get settings from UI
        max_iterations = self.iterations_spin.value()
        step_size = self.step_size_spin.value()
        cooling_rate = self.cooling_spin.value()
        tolerance = self.tolerance_spin.value()

        # Update model settings
        model = self.fit.model
        if hasattr(model, 'max_iterations'):
            model.max_iterations = max_iterations
            model.step_size = step_size
            model.cooling_rate = cooling_rate
            model.tolerance = tolerance

            # Run the agent
            if hasattr(model, 'run_agent'):
                self.run_button.setEnabled(False)
                self.run_button.setText("Running...")

                try:
                    # Run the agent directly in the main thread
                    model.run_agent()
                except Exception as e:
                    logging.error(f"Error running agent: {str(e)}")
                finally:
                    # Update UI when done
                    self.run_button.setEnabled(True)
                    self.run_button.setText("Run Agent")
                    # Update the plot one final time
                    self.update()

    def update(self, *args, **kwargs) -> None:
        """Update the plot with current agent state."""
        super().update(*args, **kwargs)

        model = self.fit.model

        # Update chi2 history plot if available
        if hasattr(model, 'chi2_history') and model.chi2_history:
            iterations = list(range(len(model.chi2_history)))
            self.chi2_plot.setData(iterations, model.chi2_history)

        # Update action history table if available
        if hasattr(model, 'action_history') and model.action_history:
            self.action_widget.setRowCount(len(model.action_history))

            for i, action in enumerate(model.action_history):
                # Common fields for all action types
                self.action_widget.setItem(i, 0, QtWidgets.QTableWidgetItem(str(action['iteration'])))

                # Handle different action types
                action_type = action.get('action_type', 'parameter_change')  # Default to parameter_change

                if action_type == 'individual_fit':
                    # Individual fit action
                    self.action_widget.setItem(i, 1, QtWidgets.QTableWidgetItem("Individual Fit"))
                    self.action_widget.setItem(i, 2, QtWidgets.QTableWidgetItem(f"Fit: {action['fit_name']}"))
                    self.action_widget.setItem(i, 3, QtWidgets.QTableWidgetItem(f"Fit χ²: {action['old_chi2']:.6g} → {action['new_chi2']:.6g}"))

                    # Use global chi2 for the change
                    chi2_change = action['new_global_chi2'] - action['old_global_chi2']
                    chi2_item = QtWidgets.QTableWidgetItem(f"{chi2_change:.6g}")

                elif action_type == 'global_fit':
                    # Global fit action
                    self.action_widget.setItem(i, 1, QtWidgets.QTableWidgetItem("Global Fit"))
                    self.action_widget.setItem(i, 2, QtWidgets.QTableWidgetItem("All fits"))
                    self.action_widget.setItem(i, 3, QtWidgets.QTableWidgetItem(f"Global χ²: {action['old_chi2']:.6g} → {action['new_chi2']:.6g}"))

                    # Use global chi2 for the change
                    chi2_change = action['new_chi2'] - action['old_chi2']
                    chi2_item = QtWidgets.QTableWidgetItem(f"{chi2_change:.6g}")

                else:
                    # Parameter change action (default)
                    self.action_widget.setItem(i, 1, QtWidgets.QTableWidgetItem(f"Parameter: {action['parameter']}"))
                    self.action_widget.setItem(i, 2, QtWidgets.QTableWidgetItem(f"Old: {action['old_value']:.6g}"))
                    self.action_widget.setItem(i, 3, QtWidgets.QTableWidgetItem(f"New: {action['new_value']:.6g}"))

                    # Use regular chi2 for the change
                    chi2_change = action['new_chi2'] - action['old_chi2']
                    chi2_item = QtWidgets.QTableWidgetItem(f"{chi2_change:.6g}")

                # Set color based on whether chi2 improved (decreased)
                if chi2_change < 0:
                    chi2_item.setForeground(QtGui.QBrush(QtGui.QColor('green')))
                else:
                    chi2_item.setForeground(QtGui.QBrush(QtGui.QColor('red')))
                self.action_widget.setItem(i, 4, chi2_item)

                # Accepted column is common for all action types
                accepted_item = QtWidgets.QTableWidgetItem("Yes" if action['accepted'] else "No")
                accepted_item.setForeground(
                    QtGui.QBrush(QtGui.QColor('green' if action['accepted'] else 'red'))
                )
                self.action_widget.setItem(i, 5, accepted_item)

            # Scroll to the bottom to show the latest actions
            self.action_widget.scrollToBottom()

        # Update weighted residuals plot if available
        if hasattr(model, 'wres_history') and model.wres_history and model.wres_history[-1]:
            # Clear previous plots
            self.wres_widget.clear()
            self.wres_plots = {}

            # Get the latest weighted residuals
            latest_wres = model.wres_history[-1]

            # Create a plot for each fit's weighted residuals
            colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Cycle through these colors
            legend = self.wres_widget.addLegend()

            for i, wres in enumerate(latest_wres):
                color = colors[i % len(colors)]
                fit_name = f"Fit {i+1}"
                if i < len(model.fits) and hasattr(model.fits[i], 'name'):
                    fit_name = model.fits[i].name

                # Convert to numpy array if needed
                if not isinstance(wres, np.ndarray):
                    wres = np.array(wres)

                # Create x values (point indices)
                x = np.arange(len(wres))

                # Plot the weighted residuals
                self.wres_plots[i] = self.wres_widget.plot(
                    x, wres, pen=color, name=fit_name
                )
