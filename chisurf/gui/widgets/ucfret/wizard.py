from __future__ import annotations

import os
import pathlib
import tempfile
import yaml

import numpy as np
from PyQt5 import QtWidgets, QtCore

import chisurf.decorators
import chisurf.gui.decorators
import chisurf.gui.widgets
import chisurf.fluorescence.tcspc.convolve
import chisurf.fluorescence.general
import chisurf.fio as io

# Import scikit_fluorescence for TCSPC data handling
import scikit_fluorescence as skf

# Import ucfret module
import ucfret
import ucfret.sampling
import ucfret.analyze


class UCFRETWizard(QtWidgets.QMainWindow):
    """
    Wizard for UCFRET analysis.

    This wizard provides a GUI for the ucfret module, which implements Bayesian analysis
    of time-resolved FRET data.
    """

    name = "UCFRETWizard"

    @chisurf.gui.decorators.init_with_ui(ui_filename="ucfret_wizard.ui")
    def __init__(self, verbose: bool = True, *args, **kwargs):
        """
        Initialize the UCFRETWizard.

        Parameters
        ----------
        verbose : bool
            Whether to print verbose output.
        """
        self.verbose = verbose

        # Connect signals to slots
        self.actionLoad_Data.triggered.connect(self.on_load_data)
        self.actionSample.triggered.connect(self.on_sample)
        self.actionAnalyze.triggered.connect(self.on_analyze)

        # Initialize variables
        self.data_file = None
        self.output_file = None
        self.analysis_settings_file = None
        self.sample_settings_file = None

        # Set default settings files
        self.set_default_settings_files()

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
        self.lineEdit_analysis_settings.setText(self.analysis_settings_file)
        self.lineEdit_sample_settings.setText(self.sample_settings_file)

    def on_load_data(self):
        """
        Load TCSPC data from a YAML file.
        """
        # Open file dialog to select data file
        filenames = chisurf.gui.widgets.open_files(
            description='Open TCSPC Data File',
            file_type='YAML Files (*.yml *.yaml)'
        )
        data_file = filenames[0] if filenames else None

        if not data_file:
            return

        self.data_file = data_file
        self.lineEdit_data_file.setText(data_file)

        # Load data and display information
        try:
            # Convert string path to Path object
            data_file_path = pathlib.Path(data_file)
            data = skf.io.tcspc.read_tcspc_yaml(data_file_path)
            self.textEdit_info.setText(f"Loaded data from {data_file}")

            # Display data structure
            self.textEdit_info.append("\nData structure:")
            for key in data:
                self.textEdit_info.append(f"- {key}")
                if isinstance(data[key], dict):
                    for subkey in data[key]:
                        self.textEdit_info.append(f"  - {subkey}")

        except Exception as e:
            self.textEdit_info.setText(f"Error loading data: {str(e)}")

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
        output_file = chisurf.gui.widgets.save_file(
            description='Save Sampling Results',
            file_type='HDF5 Files (*.hdf *.h5)'
        )

        if not output_file:
            return

        self.output_file = output_file
        self.lineEdit_output_file.setText(output_file)

        # Get settings files
        analysis_settings_file = self.lineEdit_analysis_settings.text()
        sample_settings_file = self.lineEdit_sample_settings.text()

        # Run sampling
        try:
            # Get experiment name from data file
            name = pathlib.Path(self.data_file).stem

            # Run sampling
            self.textEdit_info.setText(f"Running sampling for {name}...")

            # Use ChiSurf's progress dialog
            with chisurf.gui.widgets.progress.ProgressDialog(
                title="UCFRET Sampling",
                label="Running UCFRET sampling...",
                parent=self
            ) as progress:
                # Run sampling in a separate thread to avoid freezing the UI
                def run_sampling():
                    ucfret.sample(
                        experiment_file=self.data_file,
                        output=output_file,
                        analysis_settings_file=analysis_settings_file,
                        sample_settings_file=sample_settings_file,
                        name=name,
                        verbose=self.verbose
                    )

                # Run in thread
                worker = chisurf.gui.widgets.progress.Worker(run_sampling)
                worker.signals.finished.connect(
                    lambda: self.textEdit_info.append("Sampling completed.")
                )
                worker.signals.error.connect(
                    lambda e: self.textEdit_info.append(f"Error during sampling: {str(e)}")
                )

                # Start the worker
                progress.start_worker(worker)

        except Exception as e:
            self.textEdit_info.setText(f"Error during sampling: {str(e)}")

    def on_analyze(self):
        """
        Run the analysis step of the UCFRET analysis.
        """
        if not self.output_file:
            # Check if we can use the output file from the lineEdit
            output_file = self.lineEdit_output_file.text()
            if not output_file or not os.path.exists(output_file):
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Please run sampling first or specify a valid sampling output file."
                )
                return
            self.output_file = output_file

        # Get output directory
        output_dir, _ = chisurf.gui.widgets.get_directory(
            directory=None
        )

        if not output_dir:
            return

        # Get settings file
        sample_settings_file = self.lineEdit_sample_settings.text()

        # Run analysis
        try:
            # Get experiment name from output file
            name = pathlib.Path(self.output_file).stem

            # Run analysis
            self.textEdit_info.setText(f"Running analysis for {name}...")

            # Use ChiSurf's progress dialog
            with chisurf.gui.widgets.progress.ProgressDialog(
                title="UCFRET Analysis",
                label="Running UCFRET analysis...",
                parent=self
            ) as progress:
                # Run analysis in a separate thread to avoid freezing the UI
                def run_analysis():
                    # Load settings
                    with open(sample_settings_file, 'r') as fp:
                        settings = yaml.safe_load(fp)

                    # Create output directory if it doesn't exist
                    analysis_output_path = pathlib.Path(output_dir)
                    if not analysis_output_path.is_dir():
                        os.mkdir(analysis_output_path)

                    # Run analysis
                    ucfret.analyze.analyze(
                        filename=self.output_file,
                        name=name,
                        output_path=analysis_output_path,
                        **settings['analyze']
                    )

                # Run in thread
                worker = chisurf.gui.widgets.progress.Worker(run_analysis)
                worker.signals.finished.connect(
                    lambda: self.textEdit_info.append(f"Analysis completed. Results saved to {output_dir}")
                )
                worker.signals.error.connect(
                    lambda e: self.textEdit_info.append(f"Error during analysis: {str(e)}")
                )

                # Start the worker
                progress.start_worker(worker)

        except Exception as e:
            self.textEdit_info.setText(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = UCFRETWizard()
    w.show()
    sys.exit(app.exec_())
