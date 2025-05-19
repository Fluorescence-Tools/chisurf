import os
import sys
import pathlib
import subprocess
import tempfile
import yaml
import threading
import json
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

import chisurf.decorators
import chisurf.gui.decorators
import chisurf.gui.widgets.settings_editor
import chisurf.fluorescence.tcspc.convolve
import chisurf.fluorescence.general
import chisurf.fio as io

# Import matplotlib for plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import lltf module
import lltf

class QTextLogger(QtCore.QObject):
    """
    A QObject that you can assign to sys.stdout. It emits newText(str)
    whenever someone .write()s to it—and in the slot we update the QTextEdit.
    """
    newText = QtCore.pyqtSignal(str)

    def __init__(self, text_edit: QtWidgets.QPlainTextEdit):
        super().__init__()
        self.text_edit = text_edit
        self.newText.connect(self._append_text)

    def write(self, text: str):
        # called from worker thread: emit a signal to update UI
        self.newText.emit(text)

    def flush(self):
        pass  # no‐op

    @QtCore.pyqtSlot(str)
    def _append_text(self, text: str):
        """
        Append text to the widget, handling carriage returns so the
        progress bar overwrites the last line instead of spamming new lines.
        """
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.text_edit.setTextCursor(cursor)

        if '\r' in text:
            # strip trailing newline, split off the overwritten line
            new_part = text.strip('\r\n')
            # remove the last line entirely:
            #   move cursor to end → select last block → remove
            cursor.select(QtGui.QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            # insert the new bar text
            self.text_edit.insertPlainText(new_part)
        else:
            self.text_edit.insertPlainText(text)

        # autoscroll
        self.text_edit.verticalScrollBar().setValue(
            self.text_edit.verticalScrollBar().maximum()
        )


class LLTFSettingsEditor(chisurf.gui.widgets.settings_editor.SettingsEditor):
    """
    A settings editor for LLTF settings files.

    This is a specialized version of the SettingsEditor that is tailored for
    editing LLTF settings files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, window_title="LLTF Settings Editor", **kwargs)


class ProcessOutputWidget(QtWidgets.QWidget):
    """
    A widget for displaying the output of a process.

    This widget displays the output of a process in a text area and provides
    controls for stopping the process.
    """

    # Signal emitted when the process completes successfully
    process_completed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.process = None
        self.output_thread = None
        self.running = False
        self.logger = None

    def setup_ui(self):
        """Set up the user interface."""
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Create output text area
        self.output_text = QtWidgets.QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QtGui.QFont("Courier New", 9))
        layout.addWidget(self.output_text)

        # Create logger for the output text
        self.logger = QTextLogger(self.output_text)

        # Create control buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.stop_button = QtWidgets.QPushButton("Stop Process")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_process)

        self.clear_button = QtWidgets.QPushButton("Clear Output")
        self.clear_button.clicked.connect(self.clear_output)

        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

    def run_process(self, cmd: List[str], cwd: Optional[str] = None):
        """
        Run a process and display its output.

        Parameters
        ----------
        cmd : List[str]
            The command to run as a list of strings.
        cwd : Optional[str]
            The working directory for the process.
        """
        # Ensure logger is initialized
        if self.logger is None:
            self.logger = QTextLogger(self.output_text)

        if self.running:
            self.logger.write("Error: A process is already running.\n")
            return

        self.running = True
        self.stop_button.setEnabled(True)
        self.output_text.clear()
        self.logger.write(f"Running command: {' '.join(cmd)}\n\n")

        try:
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                bufsize=1,
                universal_newlines=True
            )

            # Start a thread to read the output
            self.output_thread = threading.Thread(
                target=self._read_output,
                daemon=True
            )
            self.output_thread.start()

        except Exception as e:
            self.logger.write(f"Error starting process: {str(e)}\n")
            self.running = False
            self.stop_button.setEnabled(False)

    def _read_output(self):
        """Read the output of the process and display it."""
        try:
            # Ensure logger is initialized
            if self.logger is None:
                self.logger = QTextLogger(self.output_text)

            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                # Write directly to the logger
                if self.logger:  # Check if logger still exists
                    self.logger.write(line)

            # Wait for the process to finish
            self.process.wait()

            # Update UI when process is done
            QtCore.QMetaObject.invokeMethod(
                self,
                "_process_finished",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, self.process.returncode)
            )

        except Exception as e:
            # Write error to the logger if it exists
            if self.logger:
                self.logger.write(f"Error reading process output: {str(e)}\n")
            QtCore.QMetaObject.invokeMethod(
                self,
                "_process_finished",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, -1)
            )

    @QtCore.pyqtSlot(int)
    def _process_finished(self, return_code: int):
        """Handle process completion."""
        self.running = False
        self.stop_button.setEnabled(False)

        # Ensure logger is initialized
        if self.logger is None:
            self.logger = QTextLogger(self.output_text)

        if return_code == 0:
            self.logger.write("\nProcess completed successfully.\n")
            # Emit signal for successful completion
            self.process_completed.emit(return_code)
        else:
            self.logger.write(f"\nProcess failed with return code {return_code}.\n")

    def stop_process(self):
        """Stop the running process."""
        if self.process and self.running:
            self.process.terminate()

            # Ensure logger is initialized
            if self.logger is None:
                self.logger = QTextLogger(self.output_text)

            self.logger.write("\nProcess terminated by user.\n")
            self.running = False
            self.stop_button.setEnabled(False)

    def clear_output(self):
        """Clear the output text area."""
        self.output_text.clear()


class LLTFGUIWizard(QtWidgets.QMainWindow):
    """
    Wizard for LLTF (Lazy Lifetime Fitter) analysis.

    This wizard provides a GUI for the lltf module, which implements advanced fitting
    procedures for extracting fluorescence lifetimes from time-correlated single photon
    counting (TCSPC) measurements.
    """

    name = "LLTFGUIWizard"

    def __init__(self, verbose: bool = True, *args, **kwargs):
        """
        Initialize the LLTFGUIWizard.

        Parameters
        ----------
        verbose : bool
            Whether to print verbose output.
        """
        super().__init__(*args, **kwargs)
        self.verbose = verbose

        # Set up the UI
        self.setup_ui()

        # Initialize variables
        self.decay_file = None
        self.irf_file = None
        self.output_file = None
        self.config_file = None
        self.output_dir = None
        self.last_fit_result = None

        # Set default config file
        self.set_default_config_file()

    def setup_ui(self):
        """Set up the user interface."""
        # Set window title
        self.setWindowTitle("LLTF: Lazy Lifetime Analysis")

        # Create central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Create input files group
        input_group = QtWidgets.QGroupBox("Input Files")
        input_layout = QtWidgets.QGridLayout()
        input_group.setLayout(input_layout)

        # Decay file
        input_layout.addWidget(QtWidgets.QLabel("Decay File:"), 0, 0)
        self.decay_file_edit = QtWidgets.QLineEdit()
        self.decay_file_edit.setReadOnly(True)
        input_layout.addWidget(self.decay_file_edit, 0, 1)
        self.load_decay_button = QtWidgets.QPushButton("Load...")
        self.load_decay_button.clicked.connect(self.on_load_decay)
        input_layout.addWidget(self.load_decay_button, 0, 2)

        # IRF file
        input_layout.addWidget(QtWidgets.QLabel("IRF File:"), 1, 0)
        self.irf_file_edit = QtWidgets.QLineEdit()
        self.irf_file_edit.setReadOnly(True)
        input_layout.addWidget(self.irf_file_edit, 1, 1)
        self.load_irf_button = QtWidgets.QPushButton("Load...")
        self.load_irf_button.clicked.connect(self.on_load_irf)
        input_layout.addWidget(self.load_irf_button, 1, 2)

        # Config file
        input_layout.addWidget(QtWidgets.QLabel("Config File:"), 2, 0)
        self.config_file_edit = QtWidgets.QLineEdit()
        input_layout.addWidget(self.config_file_edit, 2, 1)
        self.edit_config_button = QtWidgets.QPushButton("Edit...")
        self.edit_config_button.clicked.connect(self.on_edit_config)
        input_layout.addWidget(self.edit_config_button, 2, 2)

        # Output file
        input_layout.addWidget(QtWidgets.QLabel("Output Directory:"), 3, 0)
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        input_layout.addWidget(self.output_dir_edit, 3, 1)
        self.select_output_button = QtWidgets.QPushButton("Select...")
        self.select_output_button.clicked.connect(self.on_select_output)
        input_layout.addWidget(self.select_output_button, 3, 2)

        # Add input group to main layout
        main_layout.addWidget(input_group)

        # Create fitting options group
        options_group = QtWidgets.QGroupBox("Fitting Options")
        options_layout = QtWidgets.QGridLayout()
        options_group.setLayout(options_layout)

        # Number of lifetimes
        options_layout.addWidget(QtWidgets.QLabel("Number of Lifetimes:"), 0, 0)
        self.n_lifetimes_spin = QtWidgets.QSpinBox()
        self.n_lifetimes_spin.setMinimum(1)
        self.n_lifetimes_spin.setMaximum(6)
        self.n_lifetimes_spin.setValue(1)
        options_layout.addWidget(self.n_lifetimes_spin, 0, 1)

        # Find optimal checkbox
        self.find_optimal_check = QtWidgets.QCheckBox("Find Optimal Number of Lifetimes")
        self.find_optimal_check.stateChanged.connect(self.on_find_optimal_changed)
        options_layout.addWidget(self.find_optimal_check, 1, 0, 1, 2)

        # Max lifetimes for optimal search
        options_layout.addWidget(QtWidgets.QLabel("Max Lifetimes to Try:"), 2, 0)
        self.max_lifetimes_spin = QtWidgets.QSpinBox()
        self.max_lifetimes_spin.setMinimum(2)
        self.max_lifetimes_spin.setMaximum(6)
        self.max_lifetimes_spin.setValue(4)
        self.max_lifetimes_spin.setEnabled(False)
        options_layout.addWidget(self.max_lifetimes_spin, 2, 1)

        # Probability threshold
        options_layout.addWidget(QtWidgets.QLabel("Probability Threshold:"), 3, 0)
        self.prob_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.prob_threshold_spin.setMinimum(0.0)
        self.prob_threshold_spin.setMaximum(1.0)
        self.prob_threshold_spin.setValue(0.68)
        self.prob_threshold_spin.setSingleStep(0.01)
        self.prob_threshold_spin.setEnabled(False)
        options_layout.addWidget(self.prob_threshold_spin, 3, 1)

        # Verbose checkbox
        self.verbose_check = QtWidgets.QCheckBox("Verbose Output")
        self.verbose_check.setChecked(True)
        options_layout.addWidget(self.verbose_check, 4, 0, 1, 2)

        # Add options group to main layout
        main_layout.addWidget(options_group)

        # Create tab widget for output
        self.tab_widget = QtWidgets.QTabWidget()

        # Create info tab
        self.info_tab = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout()
        self.info_tab.setLayout(info_layout)

        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)

        # Set welcome message
        self.info_text.setHtml("""
        <h3>Welcome to LLTF Lazy Lifetime Analysis</h3>
        <p>This wizard guides you through the process of analyzing fluorescence lifetime data using the LLTF module.</p>
        <ol>
            <li>Load decay data using the "Load..." button</li>
            <li>Load IRF data using the "Load..." button</li>
            <li>Edit configuration if needed using the "Edit..." button</li>
            <li>Set fitting options</li>
            <li>Run the analysis using the "Fit" button</li>
        </ol>
        <p>The analysis will be run in a separate process and the output will be displayed in the "Analysis Output" tab.</p>
        <p>After fitting, the results will be displayed in the "Results" tab.</p>
        """)

        # Create analysis tab
        self.analysis_tab = ProcessOutputWidget()
        # Connect the process_completed signal to the display_fit_results method
        self.analysis_tab.process_completed.connect(self.display_fit_results)

        # Create results tab
        self.results_tab = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout()
        self.results_tab.setLayout(results_layout)

        # Create a splitter for the results tab
        results_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        results_layout.addWidget(results_splitter)

        # Create results text area
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        results_splitter.addWidget(self.results_text)

        # Create a widget for the matplotlib figure
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout()
        plot_widget.setLayout(plot_layout)

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        results_splitter.addWidget(plot_widget)

        # Set initial sizes for the splitter
        results_splitter.setSizes([200, 400])

        # Add tabs to tab widget
        self.tab_widget.addTab(self.info_tab, "Information")
        self.tab_widget.addTab(self.analysis_tab, "Analysis Output")
        self.tab_widget.addTab(self.results_tab, "Results")

        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)

        # Create button layout
        button_layout = QtWidgets.QHBoxLayout()

        # Create buttons
        self.fit_button = QtWidgets.QPushButton("Fit")
        self.fit_button.clicked.connect(self.on_fit)
        self.fit_button.setEnabled(False)

        # Add buttons to layout
        button_layout.addStretch()
        button_layout.addWidget(self.fit_button)

        # Add button layout to main layout
        main_layout.addLayout(button_layout)

        # Create menu bar
        menu_bar = self.menuBar()

        # Create file menu
        file_menu = menu_bar.addMenu("File")

        # Create load decay action
        load_decay_action = QtWidgets.QAction("Load Decay Data...", self)
        load_decay_action.triggered.connect(self.on_load_decay)
        file_menu.addAction(load_decay_action)

        # Create load IRF action
        load_irf_action = QtWidgets.QAction("Load IRF Data...", self)
        load_irf_action.triggered.connect(self.on_load_irf)
        file_menu.addAction(load_irf_action)

        # Add separator
        file_menu.addSeparator()

        # Create select output action
        select_output_action = QtWidgets.QAction("Select Output Directory...", self)
        select_output_action.triggered.connect(self.on_select_output)
        file_menu.addAction(select_output_action)

        # Add separator
        file_menu.addSeparator()

        # Create exit action
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Create settings menu
        settings_menu = menu_bar.addMenu("Settings")

        # Create edit config action
        edit_config_action = QtWidgets.QAction("Edit Configuration...", self)
        edit_config_action.triggered.connect(self.on_edit_config)
        settings_menu.addAction(edit_config_action)

        # Create analysis menu
        analysis_menu = menu_bar.addMenu("Analysis")

        # Create fit action
        fit_action = QtWidgets.QAction("Fit...", self)
        fit_action.triggered.connect(self.on_fit)
        analysis_menu.addAction(fit_action)

        # Set window size
        self.resize(800, 600)

    def set_default_config_file(self):
        """
        Set default configuration file from the lltf module.
        """
        # Create a temporary config file if none exists
        temp_dir = tempfile.gettempdir()
        config_file = os.path.join(temp_dir, "lltf_config.yml")

        # Default configuration
        default_config = {
            "verbose": True,
            "estimate_background_parameter": {
                "enabled": True,
                "initial_irf_background": 0.0,
                "fit_irf_background": True,
                "average_window": 10
            },
            "analysis_range_parameter": {
                "count_threshold": 10.0,
                "area": 0.999,
                "start_at_peak": False,
                "start_fraction": 0.1,
                "skip_first_channels": 0,
                "skip_last_channels": 0
            },
            "estimate_irf_shift_parameters": {
                "enabled": True,
                "apply_shift": True,
                "irf_time_shift_scan_range": [-8.0, 8.0],
                "irf_time_shift_scan_n_steps": 20
            },
            "lifetime_fit_parameter": {
                "find_optimal": False,
                "maximum_number_of_lifetimes": 6,
                "prob_threshold": 0.68,
                "plot_probabilities": True,
                "plot_weighted_residuals": True,
                "randomize_initial_values": {
                    "enabled": False,
                    "min_lifetime": 0.5,
                    "max_lifetime": 5.0,
                    "amplitude_variation": 0.5
                }
            },
            "pile_up_correction": {
                "enabled": False,
                "rep_rate": 80.0,
                "dead_time": 85.0,
                "measurement_time": 60.0
            },
            "plot_resulting_fit": False
        }

        # Write default config to file if it doesn't exist
        if not os.path.exists(config_file):
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)

        # Set config file
        self.config_file = config_file
        self.config_file_edit.setText(config_file)

    def on_load_decay(self):
        """
        Load decay data file.
        """
        # Open file dialog to select decay file
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            'Open Decay Data File',
            str(chisurf.working_path),
            'Data Files (*.dat *.txt *.csv);;All Files (*.*)'
        )
        decay_file = filenames[0] if filenames else None

        # Update working path if a file was selected
        if filenames:
            chisurf.working_path = pathlib.Path(filenames[0]).parent

        if not decay_file:
            return

        self.decay_file = decay_file
        self.decay_file_edit.setText(decay_file)

        # Update fit button state
        self._update_fit_button_state()

    def on_load_irf(self):
        """
        Load IRF data file.
        """
        # Open file dialog to select IRF file
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            'Open IRF Data File',
            str(chisurf.working_path),
            'Data Files (*.dat *.txt *.csv);;All Files (*.*)'
        )
        irf_file = filenames[0] if filenames else None

        # Update working path if a file was selected
        if filenames:
            chisurf.working_path = pathlib.Path(filenames[0]).parent

        if not irf_file:
            return

        self.irf_file = irf_file
        self.irf_file_edit.setText(irf_file)

        # Update fit button state
        self._update_fit_button_state()

    def on_edit_config(self):
        """
        Edit the configuration file.
        """
        # Get the current config file
        config_file = self.config_file_edit.text()

        # Create a settings editor
        editor = LTFSettingsEditor(filename=config_file)
        editor.show()

    def on_select_output(self):
        """
        Select output directory.
        """
        # Open directory dialog to select output directory
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            'Select Output Directory',
            str(chisurf.working_path)
        )

        # Update working path if a directory was selected
        if output_dir:
            chisurf.working_path = pathlib.Path(output_dir)
            self.output_dir = output_dir
            self.output_dir_edit.setText(output_dir)

    def on_find_optimal_changed(self, state):
        """
        Handle changes to the find optimal checkbox.
        """
        # Enable/disable related controls
        self.max_lifetimes_spin.setEnabled(state)
        self.prob_threshold_spin.setEnabled(state)
        self.n_lifetimes_spin.setEnabled(not state)

    def _update_fit_button_state(self):
        """
        Update the state of the fit button based on input files.
        """
        self.fit_button.setEnabled(self.decay_file is not None and self.irf_file is not None)

    def on_fit(self):
        """
        Run the LTF analysis.
        """
        if not self.decay_file or not self.irf_file:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please load decay and IRF data first."
            )
            return

        # Get output directory
        if not hasattr(self, 'output_dir') or not self.output_dir:
            # Use decay file directory as default
            self.output_dir = str(pathlib.Path(self.decay_file).parent)
            self.output_dir_edit.setText(self.output_dir)

        # Get config file
        config_file = self.config_file_edit.text()

        # Get fitting options
        find_optimal = self.find_optimal_check.isChecked()
        n_lifetimes = self.n_lifetimes_spin.value()
        max_lifetimes = self.max_lifetimes_spin.value()
        prob_threshold = self.prob_threshold_spin.value()
        verbose = self.verbose_check.isChecked()

        # Switch to analysis tab
        self.tab_widget.setCurrentIndex(1)

        # Generate output filename
        decay_name = pathlib.Path(self.decay_file).stem
        self.output_file = os.path.join(self.output_dir, f"{decay_name}_fit.json")

        # Build command
        cmd = [
            sys.executable,
            "-m", "lltf",
            "fit",
            self.decay_file,
            self.irf_file,
            "-sp", self.output_dir,
            "-o", self.output_file
        ]

        # Add options
        if config_file:
            cmd.extend(["-c", config_file])

        if find_optimal:
            cmd.append("-f")
            cmd.extend(["-m", str(max_lifetimes)])
            cmd.extend(["-pt", str(prob_threshold)])
        else:
            cmd.extend(["-n", str(n_lifetimes)])

        if verbose:
            cmd.append("-v")

        # Run the process
        self.analysis_tab.run_process(cmd)

    def display_fit_results(self, return_code):
        """
        Display the fit results after the process completes.

        Parameters
        ----------
        return_code : int
            The return code of the process
        """
        if return_code != 0 or not self.output_file or not os.path.exists(self.output_file):
            return

        try:
            # Load the fit results from the JSON file
            with open(self.output_file, 'r') as f:
                self.last_fit_result = json.load(f)

            # Display the fit results
            self.results_text.clear()

            # Format the results as HTML
            html = "<h3>Fit Results</h3>"

            # Add lifetimes
            if 'lifetimes' in self.last_fit_result:
                html += "<h4>Lifetimes</h4><table border='1' cellpadding='4'>"
                html += "<tr><th>Component</th><th>Amplitude</th><th>Lifetime (ns)</th></tr>"

                for i, lifetime in enumerate(self.last_fit_result['lifetimes']):
                    html += f"<tr><td>{i+1}</td><td>{lifetime['amplitude']:.3f}</td><td>{lifetime['lifetime']:.3f}</td></tr>"

                html += "</table>"

            # Add chi-square
            if 'chi_square' in self.last_fit_result:
                html += f"<p><b>Chi-square:</b> {self.last_fit_result['chi_square']:.3f}</p>"

            # Add reduced chi-square
            if 'reduced_chi_square' in self.last_fit_result:
                html += f"<p><b>Reduced chi-square:</b> {self.last_fit_result['reduced_chi_square']:.3f}</p>"

            # Add time range
            if 'time_range' in self.last_fit_result:
                time_range = self.last_fit_result['time_range']
                html += f"<p><b>Time range:</b> {time_range['start']:.3f} - {time_range['stop']:.3f} ns</p>"

            # Add number of lifetimes
            if 'n_lifetimes' in self.last_fit_result:
                html += f"<p><b>Number of lifetimes:</b> {self.last_fit_result['n_lifetimes']}</p>"

            # Set the HTML content
            self.results_text.setHtml(html)

            # Create the plot
            self.create_fit_plot()

            # Switch to the results tab
            self.tab_widget.setCurrentWidget(self.results_tab)

        except Exception as e:
            print(f"Error displaying fit results: {str(e)}")

    def create_fit_plot(self):
        """
        Create a plot of the fit results using matplotlib.
        """
        if not hasattr(self, 'last_fit_result') or not self.last_fit_result:
            return

        # Check if model data is available
        if 'model' not in self.last_fit_result:
            print("No model data available for plotting")
            return

        # Clear the figure
        self.figure.clear()

        # Create a figure with two subplots
        ax1 = self.figure.add_subplot(211)  # Top subplot for decay and fit
        ax2 = self.figure.add_subplot(212, sharex=ax1)  # Bottom subplot for residuals

        # Get model data
        model_time = self.last_fit_result['model']['time']
        model_decay = self.last_fit_result['model']['decay']

        # Load the original data files to get the full decay and IRF
        try:
            # Load decay data
            decay_data = np.genfromtxt(
                self.decay_file,
                delimiter=None,
                skip_header=0,
                usecols=[0, 1]
            )
            time_axis = decay_data[:, 0]
            decay = decay_data[:, 1]

            # Load IRF data
            irf_data = np.genfromtxt(
                self.irf_file,
                delimiter=None,
                skip_header=0,
                usecols=[0, 1]
            )
            irf = irf_data[:, 1]

            # Get time range
            time_range = self.last_fit_result['time_range']
            start_idx = time_range['start_idx']
            stop_idx = time_range['stop_idx']

            # Plot decay data
            ax1.semilogy(time_axis, decay, 'b-', label='Data', alpha=0.7)

            # Plot model
            ax1.semilogy(model_time, model_decay, 'r-', label='Fit', linewidth=2)

            # Plot IRF (scaled to the maximum of the decay data in the fit range)
            max_decay_in_range = np.max(decay[start_idx:stop_idx])
            max_irf = np.max(irf)
            ax1.semilogy(time_axis, irf * max_decay_in_range / max_irf, 'g-', label='IRF', alpha=0.5)

            # Add vertical lines for analysis range
            ax1.axvline(x=time_axis[start_idx], color='k', linestyle='--', alpha=0.5)
            ax1.axvline(x=time_axis[stop_idx-1], color='k', linestyle='--', alpha=0.5)

            # Set labels and legend
            ax1.set_ylabel('Counts')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Calculate residuals
            residuals = decay[start_idx:stop_idx] - model_decay
            weights = 1.0 / np.sqrt(np.maximum(decay[start_idx:stop_idx], 1.0))
            weighted_residuals = residuals * weights

            # Plot residuals
            ax2.plot(model_time, weighted_residuals, 'b-')
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)

            # Add vertical lines for analysis range
            ax2.axvline(x=time_axis[start_idx], color='k', linestyle='--', alpha=0.5)
            ax2.axvline(x=time_axis[stop_idx-1], color='k', linestyle='--', alpha=0.5)

            # Set labels
            ax2.set_xlabel('Time (ns)')
            ax2.set_ylabel('Weighted Residuals')
            ax2.grid(True, alpha=0.3)

            # Add fit information
            n_lifetimes = self.last_fit_result['n_lifetimes']
            chi_square = self.last_fit_result['reduced_chi_square']

            info_text = f"χ² = {chi_square:.3f}\n"
            for i, lifetime in enumerate(self.last_fit_result['lifetimes']):
                info_text += f"τ{i+1} = {lifetime['lifetime']:.3f} ns, A{i+1} = {lifetime['amplitude']:.3f}\n"

            # Add text box with fit information
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=props, fontsize=9)

            # Adjust layout
            self.figure.tight_layout()

            # Draw the canvas
            self.canvas.draw()

        except Exception as e:
            print(f"Error creating plot: {str(e)}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = LTFGUIWizard()
    w.show()
    sys.exit(app.exec_())
