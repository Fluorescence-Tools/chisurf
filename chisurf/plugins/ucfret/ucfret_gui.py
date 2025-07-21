import os
import sys
import pathlib
import subprocess
import tempfile
import yaml
import threading
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

import chisurf.decorators
import chisurf.gui.decorators
import chisurf.gui.widgets.settings_editor
import chisurf.fluorescence.tcspc.convolve
import chisurf.fluorescence.general
import chisurf.fio as io

# Import scikit_fluorescence for TCSPC data handling
import ucfret.skf as skf

# Import ucfret module
import ucfret
import ucfret.sampling
import ucfret.analyze

# Import custom UCFRET editors
from chisurf.plugins.ucfret.ucfret_sampling_editor import UCFRETSamplingSettingsEditor
from chisurf.plugins.ucfret.ucfret_experiment_editor import UCFRETExperimentEditor

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


class UCFRETSettingsEditor(chisurf.gui.widgets.settings_editor.SettingsEditor):
    """
    A settings editor for UCFRET settings files.

    This is a specialized version of the SettingsEditor that is tailored for
    editing UCFRET settings files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, window_title="UCFRET Settings Editor", **kwargs)


class ProcessOutputWidget(QtWidgets.QWidget):
    """
    A widget for displaying the output of a process.

    This widget displays the output of a process in a text area and provides
    controls for stopping the process.
    """

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


class UCFRETGUIWizard(QtWidgets.QMainWindow):
    """
    Enhanced wizard for UCFRET analysis.

    This wizard provides a GUI for the ucfret module, which implements Bayesian analysis
    of time-resolved FRET data. It enhances the original UCFRETWizard with:

    1. A settings editor for editing UCFRET settings files
    2. Running the ucfret CLI in a separate process
    3. Displaying the sampling output in a window
    """

    name = "UCFRETGUIWizard"

    def __init__(self, verbose: bool = True, *args, **kwargs):
        """
        Initialize the UCFRETGUIWizard.

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
        self.data_file = None
        self.output_file = None
        self.analysis_settings_file = None
        self.sample_settings_file = None

        # Set default settings files
        self.set_default_settings_files()

    def setup_ui(self):
        """Set up the user interface."""
        # Set window title
        self.setWindowTitle("UCFRET: Bayesian FRET Analysis")

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

        # Data file
        input_layout.addWidget(QtWidgets.QLabel("Data File:"), 0, 0)
        self.data_file_edit = QtWidgets.QLineEdit()
        self.data_file_edit.setReadOnly(True)
        input_layout.addWidget(self.data_file_edit, 0, 1)
        self.load_data_button = QtWidgets.QPushButton("Load...")
        self.load_data_button.clicked.connect(self.on_load_data)
        input_layout.addWidget(self.load_data_button, 0, 2)

        # Analysis settings
        input_layout.addWidget(QtWidgets.QLabel("Analysis Settings:"), 1, 0)
        self.analysis_settings_edit = QtWidgets.QLineEdit()
        input_layout.addWidget(self.analysis_settings_edit, 1, 1)
        self.edit_analysis_settings_button = QtWidgets.QPushButton("Edit...")
        self.edit_analysis_settings_button.clicked.connect(self.on_edit_analysis_settings)
        input_layout.addWidget(self.edit_analysis_settings_button, 1, 2)

        # Sample settings
        input_layout.addWidget(QtWidgets.QLabel("Sample Settings:"), 2, 0)
        self.sample_settings_edit = QtWidgets.QLineEdit()
        input_layout.addWidget(self.sample_settings_edit, 2, 1)
        self.edit_sample_settings_button = QtWidgets.QPushButton("Edit...")
        self.edit_sample_settings_button.clicked.connect(self.on_edit_sample_settings)
        input_layout.addWidget(self.edit_sample_settings_button, 2, 2)

        # Output file
        input_layout.addWidget(QtWidgets.QLabel("Output File:"), 3, 0)
        self.output_file_edit = QtWidgets.QLineEdit()
        self.output_file_edit.setReadOnly(True)
        input_layout.addWidget(self.output_file_edit, 3, 1)

        # Add input group to main layout
        main_layout.addWidget(input_group)

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
        <h3>Welcome to UCFRET Bayesian FRET Analysis</h3>
        <p>This wizard guides you through the process of analyzing time-resolved FRET data using Bayesian inference.</p>
        <ol>
            <li>Load data using the "Load..." button</li>
            <li>Edit settings if needed using the "Edit..." buttons</li>
            <li>Run sampling using the "Sample" button</li>
            <li>Analyze results using the "Analyze" button</li>
        </ol>
        <p>The sampling and analysis will be run in separate processes and their output will be displayed in the respective tabs.</p>
        """)

        # Create sampling tab
        self.sampling_tab = ProcessOutputWidget()

        # Create analysis tab
        self.analysis_tab = ProcessOutputWidget()

        # Add tabs to tab widget
        self.tab_widget.addTab(self.info_tab, "Information")
        self.tab_widget.addTab(self.sampling_tab, "Sampling Output")
        self.tab_widget.addTab(self.analysis_tab, "Analysis Output")

        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)

        # Create button layout
        button_layout = QtWidgets.QHBoxLayout()

        # Create buttons
        self.sample_button = QtWidgets.QPushButton("Sample")
        self.sample_button.clicked.connect(self.on_sample)

        self.analyze_button = QtWidgets.QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.on_analyze)

        # Add buttons to layout
        button_layout.addStretch()
        button_layout.addWidget(self.sample_button)
        button_layout.addWidget(self.analyze_button)

        # Add button layout to main layout
        main_layout.addLayout(button_layout)

        # Create menu bar
        menu_bar = self.menuBar()

        # Create file menu
        file_menu = menu_bar.addMenu("File")

        # Create load data action
        load_data_action = QtWidgets.QAction("Load Data...", self)
        load_data_action.triggered.connect(self.on_load_data)
        file_menu.addAction(load_data_action)

        # Create load experiment action
        load_experiment_action = QtWidgets.QAction("Load Experiment...", self)
        load_experiment_action.triggered.connect(self.on_load_experiment)
        file_menu.addAction(load_experiment_action)

        # Add separator
        file_menu.addSeparator()

        # Create exit action
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Create settings menu
        settings_menu = menu_bar.addMenu("Settings")

        # Create edit analysis settings action
        edit_analysis_settings_action = QtWidgets.QAction("Edit Analysis Settings...", self)
        edit_analysis_settings_action.triggered.connect(self.on_edit_analysis_settings)
        settings_menu.addAction(edit_analysis_settings_action)

        # Create edit sample settings action
        edit_sample_settings_action = QtWidgets.QAction("Edit Sample Settings...", self)
        edit_sample_settings_action.triggered.connect(self.on_edit_sample_settings)
        settings_menu.addAction(edit_sample_settings_action)

        # Create create experiment action
        create_experiment_action = QtWidgets.QAction("Create Experiment...", self)
        create_experiment_action.triggered.connect(self.on_create_experiment)
        settings_menu.addAction(create_experiment_action)

        # Create analysis menu
        analysis_menu = menu_bar.addMenu("Analysis")

        # Create sample action
        sample_action = QtWidgets.QAction("Sample...", self)
        sample_action.triggered.connect(self.on_sample)
        analysis_menu.addAction(sample_action)

        # Create analyze action
        analyze_action = QtWidgets.QAction("Analyze...", self)
        analyze_action.triggered.connect(self.on_analyze)
        analysis_menu.addAction(analyze_action)

        # Set window size
        self.resize(800, 600)

    def set_default_settings_files(self):
        """
        Set default settings files from the ucfret module.
        """
        # Get the path to the ucfret module
        ucfret_path = pathlib.Path(ucfret.__file__).parent

        # Set default settings files
        self.analysis_settings_file = str(ucfret_path / 'settings/lifetime_analysis_settings.yml')
        self.sample_settings_file = str(ucfret_path / 'settings/ucfret_settings.yml')

        # Update UI
        self.analysis_settings_edit.setText(self.analysis_settings_file)
        self.sample_settings_edit.setText(self.sample_settings_file)

    def on_load_data(self):
        """
        Load TCSPC data from a YAML file.
        """
        # Open file dialog to select data file
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            'Open TCSPC Data File',
            str(chisurf.working_path),
            'YAML Files (*.yml *.yaml)'
        )
        data_file = filenames[0] if filenames else None

        # Update working path if a file was selected
        if filenames:
            chisurf.working_path = pathlib.Path(filenames[0]).parent

        if not data_file:
            return

        self.data_file = data_file
        self.data_file_edit.setText(data_file)

        # Load data and display information
        try:
            # Convert string path to Path object
            data_file_path = pathlib.Path(data_file)
            data = skf.io.tcspc.read_tcspc_yaml(data_file_path)

            # Display detailed experiment information
            self._display_experiment_info(data, data_file)

        except Exception as e:
            self.info_text.setText(f"<h3>Error loading data</h3><p>{str(e)}</p>")

    def on_edit_analysis_settings(self):
        """
        Edit the analysis settings file.
        """
        # Get the current settings file
        settings_file = self.analysis_settings_edit.text()

        # Create a settings editor
        editor = UCFRETSettingsEditor(filename=settings_file)
        editor.show()

    def on_edit_sample_settings(self):
        """
        Edit the sample settings file.

        Uses the specialized UCFRETSamplingSettingsEditor which provides tooltips
        based on the documentation in the YAML files.
        """
        # Get the current settings file
        settings_file = self.sample_settings_edit.text()

        # Create a specialized sampling settings editor
        editor = UCFRETSamplingSettingsEditor(filename=settings_file)
        editor.show()

    def _display_experiment_info(self, data, filename):
        """
        Display detailed information about an experiment file.

        Parameters
        ----------
        data : dict
            The loaded experiment data
        filename : str
            The path to the experiment file
        """
        self.info_text.setText(f"<h3>Loaded experiment from {filename}</h3>")

        # Display sample information
        if 'Sample' in data:
            sample = data['Sample']
            self.info_text.append("<h4>Sample Information:</h4><ul>")

            # Sample name
            if 'Name' in sample:
                self.info_text.append(f"<li><b>Name:</b> {sample['Name']}</li>")

            # Measurement ID
            if 'Measurement ID' in sample:
                self.info_text.append(f"<li><b>Measurement ID:</b> {sample['Measurement ID']}</li>")

            # Reference information
            if 'Reference' in sample and isinstance(sample['Reference'], dict):
                self.info_text.append("<li><b>Reference:</b><ul>")
                for ref_key, ref_value in sample['Reference'].items():
                    self.info_text.append(f"<li>{ref_key}: {ref_value}</li>")
                self.info_text.append("</ul></li>")

            self.info_text.append("</ul>")

        # Display measurement datasets
        if 'Measurement datasets' in data:
            measurements = data['Measurement datasets']
            self.info_text.append("<h4>Measurement Datasets:</h4><ul>")

            for meas_key, meas_data in measurements.items():
                self.info_text.append(f"<li><b>{meas_key}</b><ul>")

                # Setup settings
                if 'Setup settings' in meas_data:
                    self.info_text.append(f"<li><b>Type:</b> {meas_data['Setup settings']}</li>")

                # Description
                if 'Description' in meas_data:
                    desc = meas_data['Description']
                    # Truncate long descriptions
                    if len(desc) > 100:
                        desc = desc[:97] + "..."
                    self.info_text.append(f"<li><b>Description:</b> {desc}</li>")

                # Data filename
                if 'Data' in meas_data and 'Filename' in meas_data['Data']:
                    self.info_text.append(f"<li><b>Data file:</b> {meas_data['Data']['Filename']}</li>")

                # IRF information
                if 'Instrument response function' in meas_data:
                    irf = meas_data['Instrument response function']
                    if 'Measurement ID' in irf:
                        self.info_text.append(f"<li><b>IRF:</b> {irf['Measurement ID']}</li>")

                self.info_text.append("</ul></li>")

            self.info_text.append("</ul>")

        # Switch to info tab
        self.tab_widget.setCurrentIndex(0)

    def on_load_experiment(self):
        """
        Load an experiment file.

        Opens a file dialog to select an experiment YAML file and loads it.
        The file is then opened in the UCFRETExperimentEditor for viewing or editing.
        """
        # Open file dialog to select experiment file
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            'Load Experiment File',
            str(chisurf.working_path),
            'YAML Files (*.yml *.yaml)'
        )

        # If no file was selected, return
        if not filenames:
            return

        filename = filenames[0]

        # Update working path
        chisurf.working_path = pathlib.Path(filename).parent

        # Update data file if it's not already set
        if not self.data_file:
            self.data_file = filename
            self.data_file_edit.setText(filename)

            # Load data and display information
            try:
                # Convert string path to Path object
                data_file_path = pathlib.Path(filename)
                data = skf.io.tcspc.read_tcspc_yaml(data_file_path)

                # Display detailed experiment information
                self._display_experiment_info(data, filename)

            except Exception as e:
                self.info_text.setText(f"<h3>Error loading experiment</h3><p>{str(e)}</p>")

        # Create the editor and show it
        editor = UCFRETExperimentEditor(filename=filename)
        editor.show()

    def on_create_experiment(self):
        """
        Create a new experiment file.

        Uses the UCFRETExperimentEditor.create_experiment_from_template static method
        to create a new experiment file from a template, then opens the file in the editor.
        """
        # Create a new experiment file from a template
        filename = UCFRETExperimentEditor.create_experiment_from_template(self)

        # If a file was created, open it in the editor
        if filename:
            editor = UCFRETExperimentEditor(filename=filename)
            editor.show()

    def on_sample(self):
        """
        Run the sampling step of the UCFRET analysis.
        """
        if not self.data_file:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please load data first."
            )
            return

        # Get output file
        output_file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save Sampling Results',
            str(chisurf.working_path),
            'HDF5 Files (*.hdf *.h5)'
        )

        # Update working path if a file was selected
        if output_file:
            chisurf.working_path = pathlib.Path(output_file).parent

        if not output_file:
            return

        self.output_file = output_file
        self.output_file_edit.setText(output_file)

        # Get settings files
        analysis_settings_file = self.analysis_settings_edit.text()
        sample_settings_file = self.sample_settings_edit.text()

        # Get experiment name from data file
        name = pathlib.Path(self.data_file).stem

        # Switch to sampling tab
        self.tab_widget.setCurrentIndex(1)

        # Run ucfret CLI in a separate process
        cmd = [
            sys.executable,
            "-m", "ucfret.ucfret",
            "sample",
            self.data_file,
            "-o", output_file,
            "-l", analysis_settings_file,
            "-s", sample_settings_file,
            "-v", "1"  # Verbose output
        ]

        # Run the process
        self.sampling_tab.run_process(cmd)

    def on_analyze(self):
        """
        Run the analysis step of the UCFRET analysis.
        """
        if not self.output_file:
            # Check if we can use the output file from the lineEdit
            output_file = self.output_file_edit.text()
            if not output_file or not os.path.exists(output_file):
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Please run sampling first or specify a valid sampling output file."
                )
                return
            self.output_file = output_file

        # Get output directory
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            'Select Output Directory',
            str(chisurf.working_path)
        )

        # Update working path if a directory was selected
        if output_dir:
            chisurf.working_path = pathlib.Path(output_dir)

        if not output_dir:
            return

        # Get settings file
        sample_settings_file = self.sample_settings_edit.text()

        # Get experiment name from output file
        name = pathlib.Path(self.output_file).stem

        # Switch to analysis tab
        self.tab_widget.setCurrentIndex(2)

        # Run ucfret CLI in a separate process
        cmd = [
            sys.executable,
            "-m", "ucfret.ucfret",
            "analyze",
            self.output_file,
            "-o", output_dir,
            "-s", sample_settings_file,
            "-v", "1"  # Verbose output
        ]

        # Run the process
        self.analysis_tab.run_process(cmd)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = UCFRETGUIWizard()
    w.show()
    sys.exit(app.exec_())
