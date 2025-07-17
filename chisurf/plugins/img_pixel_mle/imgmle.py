"""
Lifetime MLE Analysis Wizard

This module provides a GUI wizard for analyzing fluorescence lifetime data
from imaging experiments using Maximum Likelihood Estimation (MLE).

Features:
- Batch processing of multiple TTTR files with a single IRF
- Interactive file selection for viewing results
- Export options for single or multiple files
- Pixel-by-pixel lifetime analysis
"""

import typing
import faulthandler
faulthandler.enable(all_threads=True)

from typing import Union, Dict, List, Tuple, Optional

from qtpy import QtWidgets, QtCore, QtGui, uic
from qtpy.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QDialog, QProgressBar, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QApplication
import pyqtgraph as pg
import numpy as np
import pandas as pd
import os
import time

import chisurf
import chisurf.gui.decorators
import chisurf.settings
import chisurf.gui.widgets.wizard

import tttrlib


class CombinedProgressDialog(QDialog):
    """
    A custom dialog that shows progress for files, frames, and lines in a single window.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Progress")
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.resize(400, 200)

        # Create layout
        layout = QVBoxLayout()

        # File progress
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Files:")
        file_layout.addWidget(self.file_label)
        self.file_progress = QProgressBar()
        file_layout.addWidget(self.file_progress)
        self.file_value_label = QLabel("0/0")
        file_layout.addWidget(self.file_value_label)
        layout.addLayout(file_layout)

        # Frame progress
        frame_layout = QHBoxLayout()
        self.frame_label = QLabel("Frames:")
        frame_layout.addWidget(self.frame_label)
        self.frame_progress = QProgressBar()
        frame_layout.addWidget(self.frame_progress)
        self.frame_value_label = QLabel("0/0")
        frame_layout.addWidget(self.frame_value_label)
        layout.addLayout(frame_layout)

        # Line progress
        line_layout = QHBoxLayout()
        self.line_label = QLabel("Lines:")
        line_layout.addWidget(self.line_label)
        self.line_progress = QProgressBar()
        line_layout.addWidget(self.line_progress)
        self.line_value_label = QLabel("0/0")
        line_layout.addWidget(self.line_value_label)
        layout.addLayout(line_layout)

        # Status label
        self.status_label = QLabel("Initializing...")
        layout.addWidget(self.status_label)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

        # Initialize progress values
        self.total_files = 0
        self.current_file = 0
        self.total_frames = 0
        self.current_frame = 0
        self.total_lines = 0
        self.current_line = 0


    def set_file_progress(self, current, total):
        """Set the progress for files."""
        self.current_file = current
        self.total_files = total
        self.file_progress.setMaximum(total)
        self.file_progress.setValue(current)
        self.file_value_label.setText(f"{current}/{total}")
        self.update_status()

    def set_frame_progress(self, current, total):
        """Set the progress for frames."""
        self.current_frame = current
        self.total_frames = total
        self.frame_progress.setMaximum(total)
        self.frame_progress.setValue(current)
        self.frame_value_label.setText(f"{current}/{total}")
        self.update_status()

    def set_line_progress(self, current, total):
        """Set the progress for lines."""
        self.current_line = current
        self.total_lines = total
        self.line_progress.setMaximum(total)
        self.line_progress.setValue(current)
        self.line_value_label.setText(f"{current}/{total}")
        self.update_status()

    def update_status(self):
        """Update the status label with current progress information."""
        self.status_label.setText(
            f"Processing file {self.current_file}/{self.total_files}, "
            f"frame {self.current_frame}/{self.total_frames}, "
            f"line {self.current_line}/{self.total_lines}..."
        )


class FileListWidget(QtWidgets.QListWidget):
    """
    QListWidget subclass that accepts drag-and-drop of files.
    Each dropped file becomes an item with a checkbox, checked by default.
    """
    def __init__(self, parent=None, file_added_callback=None, process_on_drop=False):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)
        self.file_added_callback = file_added_callback
        self.process_on_drop = process_on_drop

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if local_path:
                    self.add_file(local_path)
            event.acceptProposedAction()
            if self.process_on_drop and self.file_added_callback:
                self.file_added_callback()
        else:
            super().dropEvent(event)

    def add_file(self, file_path: str):
        """Add a file to the list if it doesn't already exist."""
        for i in range(self.count()):
            if self.item(i).text() == file_path:
                return  # File already in list

        item = QtWidgets.QListWidgetItem(file_path)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        self.addItem(item)

        if self.file_added_callback:
            self.file_added_callback()

    def get_selected_files(self):
        """Return a list of file paths for items whose checkboxes are checked."""
        files = []
        for index in range(self.count()):
            item = self.item(index)
            if item.checkState() == QtCore.Qt.Checked:
                files.append(item.text())
        return files


class LifetimeMleAnalysisWizard(QtWidgets.QMainWindow):
    """
    Main wizard for Lifetime MLE Analysis.
    """
    def _interpolate_shift(self, arr: np.ndarray, shift: Union[int, float]) -> np.ndarray:
        """
        Shift a 1D array by a given number of bins, supporting fractional shifts.

        Parameters
        ----------
        arr : np.ndarray
            Input array to shift.
        shift : int or float
            Number of bins to shift (positive rightwards, negative leftwards).

        Returns
        -------
        np.ndarray
            Shifted array with zeros filled.
        """
        result = arr.astype(np.float64).copy()
        if shift == 0:
            return result
        int_shift = int(np.trunc(shift))
        if int_shift != 0:
            result = np.roll(result, int_shift)
            if int_shift > 0:
                result[:int_shift] = 0.0
            else:
                result[int_shift:] = 0.0
        frac_shift = shift - int_shift
        if frac_shift != 0:
            x = np.arange(result.size)
            result = np.interp(x - frac_shift, x, result, left=0.0, right=0.0)
        return result

    def prepare_irf(self, irf_p: np.ndarray, irf_s: np.ndarray, 
                   threshold: float = -1, 
                   shift: int = 0, 
                   shift_sp: float = 0, 
                   shift_ss: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare IRF by applying threshold, normalization, and shifts.

        Parameters
        ----------
        irf_p : np.ndarray
            Parallel channel IRF.
        irf_s : np.ndarray
            Perpendicular channel IRF.
        threshold : float
            Threshold value as a fraction of the maximum. Values below this threshold will be set to zero.
        shift : int
            Integer shift of the second (ss) decay relative to the first (sp).
        shift_sp : float
            Sub-channel (fractional) shift to apply to the sp IRF.
        shift_ss : float
            Sub-channel (fractional) shift to apply to the ss IRF.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Processed parallel and perpendicular IRFs.
        """
        # Make copies to avoid modifying the original arrays
        irf_p = irf_p.astype(np.float64).copy()
        irf_s = irf_s.astype(np.float64).copy()

        # Apply threshold to IRF histograms
        if threshold > 0:
            irf_p[irf_p < threshold * irf_p.max()] = 0
            irf_s[irf_s < threshold * irf_s.max()] = 0

        # Normalize after thresholding
        if np.sum(irf_p) > 0:
            irf_p = irf_p / np.sum(irf_p)
        if np.sum(irf_s) > 0:
            irf_s = irf_s / np.sum(irf_s)

        # Apply sub-bin shifts
        irf_p = self._interpolate_shift(irf_p, shift_sp)
        irf_s = self._interpolate_shift(irf_s, shift_ss)

        # Apply integer relative shift to the second decay
        if shift != 0:
            irf_s = np.roll(irf_s, shift)
            if shift > 0:
                irf_s[:shift] = 0.0
            else:
                irf_s[shift:] = 0.0

        return irf_p, irf_s

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize variables
        self.tttr_data = None
        self.irf_tttr = None
        self.clsm_p = None
        self.clsm_s = None
        self.irf_p = None
        self.irf_s = None
        self.tau = None
        self.rho = None
        self._fit = None
        self.decay_all_photons = None
        self.micro_time_range = [0, 0]

        # Background pattern variables
        self.bg_tttr = None
        self.bg_p = None
        self.bg_s = None

        # Load UI from file
        ui_file = os.path.join(os.path.dirname(__file__), 'imgmle.ui')
        uic.loadUi(ui_file, self)

        # Add p2s_twoIstar_flag checkbox
        self.checkBox_2IStar = QtWidgets.QCheckBox("Enable 2I* calculation")
        self.checkBox_2IStar.setChecked(True)
        self.checkBox_2IStar.setToolTip("Enable calculation of 2I* and 2I*: P+2S? values")

        # Add BIFL scatter fit checkbox
        self.checkBox_BIFL_scatter = QtWidgets.QCheckBox("Enable BIFL scatter fit")
        self.checkBox_BIFL_scatter.setChecked(False)
        self.checkBox_BIFL_scatter.setToolTip("Enable BIFL scatter fit calculation")

        # Find a suitable layout to add the checkboxes to
        # Add them to the same layout as the other fit parameter checkboxes
        # Assuming the fix_tau_checkbox is in a layout
        if hasattr(self, 'fix_tau_checkbox') and self.fix_tau_checkbox is not None:
            layout = self.fix_tau_checkbox.parent().layout()
            if layout is not None:
                layout.addWidget(self.checkBox_2IStar)
                layout.addWidget(self.checkBox_BIFL_scatter)

        # Make file format radio buttons mutually exclusive
        self.file_format_group = QtWidgets.QButtonGroup(self)
        self.file_format_group.addButton(self.radioButton_FileHDF)
        self.file_format_group.addButton(self.radioButton_FileCsv)

        # Set up FileListWidget callbacks
        self.tttr_list.file_added_callback = self.update_tttr_files
        self.irf_list.file_added_callback = self.update_irf_files
        self.bg_list.file_added_callback = self.update_bg_files

        # Connect button signals
        self.browse_tttr_button.clicked.connect(lambda: self.browse_files(self.tttr_list))
        self.clear_tttr_button.clicked.connect(lambda: self.clear_files(self.tttr_list))
        self.browse_irf_button.clicked.connect(lambda: self.browse_files(self.irf_list))
        self.clear_irf_button.clicked.connect(lambda: self.clear_files(self.irf_list))
        self.browse_bg_button.clicked.connect(lambda: self.browse_files(self.bg_list, "Background Files (*.ht3 *.ptu *.pt3);;All Files (*.*)"))
        self.clear_bg_button.clicked.connect(lambda: self.clear_files(self.bg_list))
        self.process_button.clicked.connect(self.process_data)
        self.export_button.clicked.connect(self.export_results)
        self.update_fit_button.clicked.connect(self.update_fit)

        # Connect background radio buttons
        self.bg_fixed_radio.toggled.connect(self.toggle_background_source)
        self.bg_file_radio.toggled.connect(self.toggle_background_source)

        # Connect file selector combo box
        self.file_selector_combo.currentIndexChanged.connect(self.on_file_selection_changed)

        # Initialize plots
        self.residual_plot.setLabel('left', 'Residuals')
        self.combined_plot.setLabel('bottom', 'Time (ch.)')
        self.combined_plot.setLabel('left', 'Intensity')
        self.combined_plot.setLogMode(y=True)
        self.combined_plot.setYRange(-1, 5)

        # Link x-axis of residual and combined plots
        self.residual_plot.setXLink(self.combined_plot)

        # Connect signals
        self.connect_signals()

        # Initialize UI values
        self.initialize_ui_values()

        # Set window size
        self.resize(700, 500)

    def connect_signals(self):
        """Connect signals to slots."""
        # Connect tab changed signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Connect parameter change signals
        self.ch_p_spinbox.valueChanged.connect(self.update_parameters)
        self.ch_s_spinbox.valueChanged.connect(self.update_parameters)
        self.binning_factor_spinbox.valueChanged.connect(self.update_parameters)
        self.min_photons_spinbox.valueChanged.connect(self.update_parameters)
        self.g_factor_spinbox.valueChanged.connect(self.update_parameters)
        self.l1_spinbox.valueChanged.connect(self.update_parameters)
        self.l2_spinbox.valueChanged.connect(self.update_parameters)
        self.period_spinbox.valueChanged.connect(self.update_parameters)
        self.tau_spinbox.valueChanged.connect(self.update_parameters)
        self.gamma_spinbox.valueChanged.connect(self.update_parameters)
        self.r0_spinbox.valueChanged.connect(self.update_parameters)
        self.rho_spinbox.valueChanged.connect(self.update_parameters)
        self.fix_tau_checkbox.stateChanged.connect(self.update_parameters)
        self.fix_gamma_checkbox.stateChanged.connect(self.update_parameters)
        self.fix_r0_checkbox.stateChanged.connect(self.update_parameters)
        self.fix_rho_checkbox.stateChanged.connect(self.update_parameters)

        # Connect new checkboxes
        self.checkBox_2IStar.stateChanged.connect(self.update_parameters)
        self.checkBox_BIFL_scatter.stateChanged.connect(self.update_parameters)

        # Connect micro time range parameter change signals
        self.micro_time_start_spinbox.valueChanged.connect(self.update_parameters)
        self.micro_time_stop_spinbox.valueChanged.connect(self.update_parameters)
        self.adjust_stop_checkbox.stateChanged.connect(self.update_parameters)
        self.read_period_checkbox.stateChanged.connect(self.update_parameters)

        # Connect time shift parameter change signals
        self.shift_spinbox.valueChanged.connect(self.update_parameters)
        self.shift_sp_spinbox.valueChanged.connect(self.update_parameters)
        self.shift_ss_spinbox.valueChanged.connect(self.update_parameters)

        # Connect irf_threshold parameter change signal
        self.doubleSpinBox_irf_threshold.valueChanged.connect(self.update_parameters)

        # Connect background correction parameter change signals
        self.bg_p_spinbox.valueChanged.connect(self.update_parameters)
        self.bg_s_spinbox.valueChanged.connect(self.update_parameters)
        self.use_bg_checkbox.stateChanged.connect(self.update_parameters)
        self.bg_fixed_radio.toggled.connect(self.update_parameters)
        self.bg_file_radio.toggled.connect(self.update_parameters)

    def initialize_ui_values(self):
        """Initialize UI values."""
        # Initialize micro time range
        start, stop = self.micro_time_range
        self.micro_time_start_spinbox.setValue(start)
        self.micro_time_stop_spinbox.setValue(stop)

        # Initialize IRF threshold
        self.doubleSpinBox_irf_threshold.setValue(0.02)

        # Initialize background source visibility
        #self.bg_fixed_widget.setVisible(self.bg_fixed_radio.isChecked())
        #self.bg_file_widget.setVisible(self.bg_file_radio.isChecked())

    def on_tab_changed(self, index: int):
        """Handle tab changed event."""
        pass

    def update_parameters(self):
        """Update parameters from UI."""
        # If we have data loaded, update the fit
        if hasattr(self, 'tttr_data') and self.tttr_data is not None:
            # Get the sender (which parameter was changed)
            sender = self.sender()

            # If channel parameters, binning factor, micro time range, or time shift parameters changed, regenerate decay curves
            if sender in [self.ch_p_spinbox, self.ch_s_spinbox, self.binning_factor_spinbox,
                         self.micro_time_start_spinbox, self.micro_time_stop_spinbox,
                         self.adjust_stop_checkbox, self.read_period_checkbox,
                         self.shift_spinbox, self.shift_sp_spinbox, self.shift_ss_spinbox]:
                # Regenerate decay curves with new parameters
                self.load_data_and_compute_decays()
            else:
                # For other parameters, just update the fit
                self.update_fit()

        # Update micro time range from UI
        self.micro_time_range = [self.micro_time_start_spinbox.value(), self.micro_time_stop_spinbox.value()]

    def toggle_background_source(self):
        """Toggle the visibility of background source widgets based on the selected radio button."""
        #self.bg_fixed_widget.setVisible(self.bg_fixed_radio.isChecked())
        #self.bg_file_widget.setVisible(self.bg_file_radio.isChecked())

        # If switching to file-based background, load the background pattern
        if self.bg_file_radio.isChecked():
            self.load_background_pattern()

        # Update the fit if the background source changes
        self.update_fit()

    def browse_files(self, list_widget, name_filter="TTTR Files (*.ht3 *.ptu *.pt3);;All Files (*.*)"):
        """Browse for files and add them to the list widget."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter(name_filter)

        if file_dialog.exec_():
            file_names = file_dialog.selectedFiles()
            for file_name in file_names:
                list_widget.add_file(file_name)

    def clear_files(self, list_widget):
        """Clear all files from the list widget."""
        list_widget.clear()

        # If clearing TTTR or IRF files, also clear decay curves and fit
        if list_widget in [self.tttr_list, self.irf_list]:
            # Clear data
            if list_widget == self.tttr_list:
                self.tttr_data = None
                self.clsm_p = None
                self.clsm_s = None
                self.decay_all_photons = None
            elif list_widget == self.irf_list:
                self.irf_tttr = None
                self.irf_p = None
                self.irf_s = None

            # Clear fit
            self._fit = None

            # Clear plots
            if hasattr(self, 'combined_plot') and self.combined_plot is not None:
                self.combined_plot.clear()
            if hasattr(self, 'residual_plot') and self.residual_plot is not None:
                self.residual_plot.clear()

            # Reset fit parameter labels
            if hasattr(self, 'tau_label'):
                self.tau_label.setText("Tau: -")
            if hasattr(self, 'gamma_label'):
                self.gamma_label.setText("Gamma: -")
            if hasattr(self, 'r0_label'):
                self.r0_label.setText("R0: -")
            if hasattr(self, 'rho_label'):
                self.rho_label.setText("Rho: -")
            if hasattr(self, 'chi2_label'):
                self.chi2_label.setText("Chi²: -")

        # If clearing background files, clear background data
        elif list_widget == self.bg_list:
            self.bg_tttr = None
            self.bg_p = None
            self.bg_s = None

            # Update the fit
            self.update_fit()

    def update_tttr_files(self):
        """Update TTTR files and compute decays if both TTTR and IRF files are available."""
        self.load_data_and_compute_decays()

    @property
    def irf_threshold(self) -> float:
        """
        Threshold for IRF processing, as a fraction of the maximum value.
        Values below this threshold will be set to zero.
        """
        return float(self.doubleSpinBox_irf_threshold.value())

    @property
    def p2s_twoIstar(self) -> bool:
        """Get the p2s_twoIstar flag value."""
        return self.checkBox_2IStar.isChecked()

    @p2s_twoIstar.setter
    def p2s_twoIstar(self, value: bool):
        """Set the p2s_twoIstar flag value."""
        self.checkBox_2IStar.setChecked(value)

    @property
    def BIFL_scatter(self) -> bool:
        """Get the BIFL_scatter flag value."""
        return self.checkBox_BIFL_scatter.isChecked()

    @BIFL_scatter.setter
    def BIFL_scatter(self, value: bool):
        """Set the BIFL_scatter flag value."""
        self.checkBox_BIFL_scatter.setChecked(value)

    @irf_threshold.setter
    def irf_threshold(self, v: float):
        self.doubleSpinBox_irf_threshold.setValue(v)

    def update_irf_files(self):
        """Update IRF files and compute decays if both TTTR and IRF files are available."""
        self.load_data_and_compute_decays()

    def update_bg_files(self):
        """Update background files and load the background pattern."""
        self.load_background_pattern()
        self.update_fit()

    def load_background_pattern(self):
        """Load background pattern from the selected file."""
        # If not using file-based background, return
        if not self.bg_file_radio.isChecked():
            return

        # Get selected files
        bg_files = self.bg_list.get_selected_files()

        if not bg_files:
            # No files selected, clear background data
            self.bg_tttr = None
            self.bg_p = None
            self.bg_s = None
            return

        try:
            # Get parameters from UI
            ch_p = [self.ch_p_spinbox.value()]
            ch_s = [self.ch_s_spinbox.value()]
            binning_factor = self.binning_factor_spinbox.value()

            # Load background data
            fn_bg = bg_files[0]
            self.bg_tttr = tttrlib.TTTR(fn_bg)

            # Get micro time histograms for the background
            bg_data_p = self.bg_tttr[self.bg_tttr.get_selection_by_channel(ch_p)]
            bg_data_s = self.bg_tttr[self.bg_tttr.get_selection_by_channel(ch_s)]
            self.bg_p, _ = bg_data_p.get_microtime_histogram(binning_factor)
            self.bg_s, _ = bg_data_s.get_microtime_histogram(binning_factor)

            # Apply micro time range
            start, stop = self.micro_time_range
            if stop > len(self.bg_p):
                stop = len(self.bg_p)

            self.bg_p = self.bg_p[:stop]
            self.bg_s = self.bg_s[:stop]

            if start > 0:
                self.bg_p = self.bg_p[start:]
                self.bg_s = self.bg_s[start:]

            chisurf.logging.info(f"Loaded background pattern from {fn_bg}")

        except Exception as e:
            # If there's an error, show a message but don't crash
            QMessageBox.warning(self, "Error", f"Error loading background pattern: {str(e)}")
            self.bg_tttr = None
            self.bg_p = None
            self.bg_s = None

    def load_data_and_compute_decays(self):
        """Load data and compute decays for selected channels."""
        # Get parameters from UI
        ch_p = [self.ch_p_spinbox.value()]
        ch_s = [self.ch_s_spinbox.value()]
        binning_factor = self.binning_factor_spinbox.value()
        # Get micro time range from UI
        start = self.micro_time_start_spinbox.value()
        stop = self.micro_time_stop_spinbox.value()

        # Get selected files
        tttr_files = self.tttr_list.get_selected_files()
        irf_files = self.irf_list.get_selected_files()

        if not tttr_files or not irf_files:
            # Not enough files selected, can't compute decays yet
            return

        try:
            # Store all TTTR data objects
            self.tttr_data_list = []

            # Process the first TTTR file to set parameters
            fn_clsm = tttr_files[0]
            self.tttr_data = tttrlib.TTTR(fn_clsm)
            self.tttr_data_list.append(self.tttr_data)

            # Add all other TTTR files to the list
            for i in range(1, len(tttr_files)):
                tttr_data = tttrlib.TTTR(tttr_files[i])
                self.tttr_data_list.append(tttr_data)

            # Read period from PTU file if enabled
            if self.read_period_checkbox.isChecked() and fn_clsm.lower().endswith('.ptu'):
                try:
                    # Try to read period from PTU file
                    header = self.tttr_data.header
                    if hasattr(header, 'laser_period'):
                        # Period is in seconds, convert to nanoseconds
                        period_ns = header.laser_period * 1e9
                        self.period_spinbox.setValue(period_ns)
                        chisurf.logging.info(f"Read period from PTU: {period_ns} ns")
                    elif hasattr(header, 'tttr_info') and 'SyncRate' in header.tttr_info:
                        # SyncRate is in Hz, convert to period in nanoseconds
                        sync_rate = float(header.tttr_info['SyncRate'])
                        if sync_rate > 0:
                            period_ns = 1e9 / sync_rate
                            self.period_spinbox.setValue(period_ns)
                            chisurf.logging.info(f"Read period from PTU SyncRate: {period_ns} ns")
                except Exception as e:
                    chisurf.logging.warning(f"Failed to read period from PTU: {str(e)}")

            # Load IRF data
            fn_irf = irf_files[0]
            self.irf_tttr = tttrlib.TTTR(fn_irf)

            # Get micro time histograms for the IRF
            irf_data_p = self.irf_tttr[self.irf_tttr.get_selection_by_channel(ch_p)]
            irf_data_s = self.irf_tttr[self.irf_tttr.get_selection_by_channel(ch_s)]
            self.irf_p, t = irf_data_p.get_microtime_histogram(binning_factor)
            self.irf_s, _ = irf_data_s.get_microtime_histogram(binning_factor)

            # Get time shift parameters from UI
            shift = self.shift_spinbox.value()
            shift_sp = self.shift_sp_spinbox.value()
            shift_ss = self.shift_ss_spinbox.value()

            # Prepare IRF (threshold, normalize, shift)
            self.irf_p, self.irf_s = self.prepare_irf(
                self.irf_p, self.irf_s,
                threshold=self.irf_threshold,
                shift=shift,
                shift_sp=shift_sp,
                shift_ss=shift_ss
            )

            # Calculate micro time range
            n_channels = self.tttr_data.header.number_of_micro_time_channels // binning_factor

            # Update UI with actual number of channels if needed
            if self.micro_time_stop_spinbox.maximum() < n_channels:
                self.micro_time_stop_spinbox.setMaximum(n_channels)

            # Adjust stop based on period if enabled
            if self.adjust_stop_checkbox.isChecked():
                # Get period in nanoseconds
                period_ns = self.period_spinbox.value()

                # Calculate time resolution in nanoseconds
                time_resolution_ns = self.tttr_data.header.micro_time_resolution * 1e9 * binning_factor

                # Calculate number of channels corresponding to one period
                period_channels = int(period_ns / time_resolution_ns)

                # Set stop to period_channels or n_channels, whichever is smaller
                stop = min(period_channels, n_channels)

                # Update UI
                self.micro_time_stop_spinbox.setValue(stop)
                chisurf.logging.info(f"Adjusted stop to {stop} based on period {period_ns} ns")

            # Set micro time range
            self.micro_time_range = [start, stop]

            # Generate decay curve for all photons in the image
            micro_times = self.tttr_data.micro_times // binning_factor

            # Get indices for parallel and perpendicular channels
            idx_p = self.tttr_data.get_selection_by_channel(ch_p)
            idx_s = self.tttr_data.get_selection_by_channel(ch_s)

            # Create histograms with the specified range
            hist_p = np.bincount(micro_times[idx_p], minlength=n_channels)[:stop]
            hist_s = np.bincount(micro_times[idx_s], minlength=n_channels)[:stop]

            # Apply start index
            if start > 0:
                hist_p = hist_p[start:]
                hist_s = hist_s[start:]

            # Combine histograms
            self.decay_all_photons = np.hstack([hist_p, hist_s])

            # Update the fit and plot the result
            self.update_fit()

            # Switch to analysis tab
            self.tab_widget.setCurrentIndex(3)

        except Exception as e:
            # If there's an error, show a message but don't crash
            QMessageBox.warning(self, "Error", f"Error loading data: {str(e)}")
            return

    def process_data(self):
        """Process the data using the current parameters."""
        # First, load data and compute decays for all photons
        self.load_data_and_compute_decays()

        # Get all settings using the get_settings method
        all_settings = self.get_settings()

        # Extract needed parameters
        ch_p = [all_settings['ch_p']]
        ch_s = [all_settings['ch_s']]
        binning_factor = all_settings['binning_factor']
        minimum_n_photons = all_settings['min_photons']
        auto_export = all_settings['auto_export']

        # Check if data was loaded successfully
        if not hasattr(self, 'tttr_data_list') or not self.tttr_data_list:
            return

        # Get IRF from settings
        irf = all_settings['irf']

        # Get initial values and fixed parameters
        x0 = np.array([all_settings['tau'], all_settings['gamma'], all_settings['r0'], all_settings['rho']])

        fixed = np.array([
            1 if all_settings['fix_tau'] else 0,
            1 if all_settings['fix_gamma'] else 0,
            1 if all_settings['fix_r0'] else 0,
            1 if all_settings['fix_rho'] else 0
        ])

        # Create results list for export (one list per file)
        self.results_list = []

        # Store tau and rho arrays for each file
        self.tau_list = []
        self.rho_list = []

        # For backward compatibility
        self.results = []

        # Get selected files for naming
        tttr_files = self.tttr_list.get_selected_files()

        # Create combined progress dialog
        total_files = len(self.tttr_data_list)
        progress_dialog = CombinedProgressDialog(self)
        progress_dialog.set_file_progress(1, total_files)
        progress_dialog.show()
        QApplication.processEvents()

        # Process the data
        time_start = time.time()

        # Get micro time range once
        start, stop = self.micro_time_range

        # Process each TTTR file
        for file_idx, tttr_data in enumerate(self.tttr_data_list):
            # Update file progress
            progress_dialog.set_file_progress(file_idx, total_files)
            QApplication.processEvents()

            # Set current TTTR data
            self.tttr_data = tttr_data

            # Get file name for results
            file_name = os.path.basename(tttr_files[file_idx]) if file_idx < len(tttr_files) else f"File_{file_idx+1}"

            # Create a new list for this file's results
            file_results = []
            self.results_list.append(file_results)

            # Create CLSM containers for pixel-by-pixel analysis
            self.clsm_p = tttrlib.CLSMImage(self.tttr_data, channels=ch_p, fill=True)
            self.clsm_s = tttrlib.CLSMImage(self.tttr_data, channels=ch_s, fill=True)

            # Stack frames if checkbox is checked
            if all_settings['stack_frames']:
                self.clsm_p.stack_frames()
                self.clsm_s.stack_frames()

            # Calculate micro time range
            n_channels = self.tttr_data.header.number_of_micro_time_channels // binning_factor

            # Settings for MLE - use the settings from get_settings but update dt for current file
            settings = {
                'dt': self.tttr_data.header.micro_time_resolution * 1e9 * binning_factor,
                'g_factor': all_settings['g_factor'],
                'l1': all_settings['l1'],
                'l2': all_settings['l2'],
                'convolution_stop': -1,
                'irf': irf,
                'period': all_settings['period'],
                'background': all_settings['background'],
                'p2s_twoIstar_flag': all_settings['p2s_twoIstar'],  # Enable 2I* and 2I*: P+2S? calculation
                'soft_bifl_scatter_flag': all_settings['BIFL_scatter']  # Enable BIFL scatter fit
            }

            # Create Fit23 instance
            fit23 = tttrlib.Fit23(**settings)

            # Get image dimensions
            intensity = self.clsm_p.intensity
            micro_times = self.tttr_data.micro_times // binning_factor
            n_channels = self.tttr_data.header.number_of_micro_time_channels // binning_factor

            # Create arrays for tau and rho
            tau_array = np.zeros_like(intensity, dtype=np.float32)
            rho_array = np.zeros_like(intensity, dtype=np.float32)
            n_frames, n_lines, n_pixel = self.clsm_p.shape

            # Update progress dialog with frame and line information
            progress_dialog.set_frame_progress(0, n_frames)
            progress_dialog.set_line_progress(0, n_lines)
            QApplication.processEvents()

            # Pre-compute line durations for all lines to avoid repeated calculations
            line_durations = np.zeros((n_frames, n_lines))
            for i in range(n_frames):
                for j in range(n_lines):
                    line_durations[i, j] = self.clsm_p.get_line_duration(i, j)  # in seconds

            # Pre-allocate arrays for histograms to avoid repeated memory allocations
            hist_p_template = np.zeros(stop - start, dtype=np.int64)
            hist_s_template = np.zeros(stop - start, dtype=np.int64)

            # Create a list to store results for batch processing
            batch_results = []

            # Process each frame
            for i in range(n_frames):
                # Update frame progress
                progress_dialog.set_frame_progress(i + 1, n_frames)
                QApplication.processEvents()

                # Process each line in the frame
                for j in range(n_lines):
                    # Update line progress
                    progress_dialog.set_line_progress(j + 1, n_lines)
                    QApplication.processEvents()

                    # Get line duration once per line
                    line_duration = line_durations[i, j]
                    pixel_duration = line_duration / n_pixel  # in seconds

                    # Process all pixels in the line in batches
                    # First collect all pixel data for the line
                    line_data = []
                    for k in range(n_pixel):
                        idx_p = self.clsm_p[i][j][k].tttr_indices
                        idx_s = self.clsm_s[i][j][k].tttr_indices
                        n_p = len(idx_p)
                        n_s = len(idx_s)
                        total_photons = n_p + n_s

                        # Store pixel data for processing
                        # Make copies of the indices to prevent garbage collection issues
                        line_data.append({
                            'idx_p': np.array(idx_p, copy=True) if len(idx_p) > 0 else idx_p,
                            'idx_s': np.array(idx_s, copy=True) if len(idx_s) > 0 else idx_s,
                            'n_p': n_p,
                            'n_s': n_s,
                            'total_photons': total_photons,
                            'pixel_idx': k,
                            'pixel_duration': pixel_duration
                        })

                    # Process each pixel in the line
                    for pixel_data in line_data:
                        k = pixel_data['pixel_idx']
                        n_p = pixel_data['n_p']
                        n_s = pixel_data['n_s']
                        total_photons = pixel_data['total_photons']
                        idx_p = pixel_data['idx_p']
                        idx_s = pixel_data['idx_s']
                        pixel_duration = pixel_data['pixel_duration']

                        # Skip processing if not enough photons
                        if total_photons < minimum_n_photons:
                            # Add result with actual photon counts but no fit data
                            result_dict = {
                                'Y pixel': j,
                                'X pixel': k,
                                'Green Count Rate (KHz)': total_photons / (pixel_duration * 1000.0),  # Convert to KHz
                                'Number of Photons (green)': total_photons,
                                'Pixel Number': j * n_pixel + k,
                                'Number of Photons (fit window) (green)': 0,  # We don't calculate this for skipped pixels
                                'tau (green)': 0.0,
                                'gamma (green)': 0.0,
                                'r0 (green)': 0.0,
                                'rho (green)': 0.0,
                                'BIFL scatter fit? (green)': 0.0,  # No fit performed
                                '2I*: P+2S? (green)': 0.0,  # No fit performed
                                'rS (green)': 0.0,  # Not implemented
                                'rE (green)': 0.0,  # Not implemented
                                '2I* (green)': 0.0,  # No fit performed
                                'Ng-p-all': n_p,
                                'Ng-s-all': n_s,
                                'Ng-all': total_photons
                            }
                            # Only add Z pixel if there's more than one frame
                            if n_frames > 1:
                                result_dict['Z pixel'] = i
                            batch_results.append(result_dict)
                            continue

                        # Create histograms efficiently
                        # Use pre-allocated arrays for better memory efficiency
                        if n_p > 0:
                            # Use bincount directly with slicing for efficiency
                            hist_p = np.bincount(micro_times[idx_p], minlength=n_channels)[start:stop] if start < n_channels else hist_p_template.copy()
                            fit_window_photons_p = np.sum(hist_p)
                        else:
                            hist_p = hist_p_template.copy()
                            fit_window_photons_p = 0

                        if n_s > 0:
                            # Use bincount directly with slicing for efficiency
                            hist_s = np.bincount(micro_times[idx_s], minlength=n_channels)[start:stop] if start < n_channels else hist_s_template.copy()
                            fit_window_photons_s = np.sum(hist_s)
                        else:
                            hist_s = hist_s_template.copy()
                            fit_window_photons_s = 0

                        fit_window_photons = fit_window_photons_p + fit_window_photons_s

                        # Combine histograms - use np.concatenate for better performance
                        hist = np.concatenate([hist_p, hist_s])

                        # Perform the fit
                        r = fit23(hist, x0, fixed)

                        # Store the results
                        tau_array[i, j, k] = r['x'][0]
                        rho_array[i, j, k] = r['x'][3]

                        # Add result for export
                        result_dict = {
                            'Y pixel': j,
                            'X pixel': k,
                            'Green Count Rate (KHz)': total_photons / (pixel_duration * 1000.0),  # Convert to KHz
                            'Number of Photons (green)': total_photons,
                            'Pixel Number': j * n_pixel + k,
                            'Number of Photons (fit window) (green)': fit_window_photons,
                            'tau (green)': r['x'][0],
                            'gamma (green)': r['x'][1],
                            'r0 (green)': r['x'][2],
                            'rho (green)': r['x'][3],
                            'BIFL scatter fit? (green)': all_settings['BIFL_scatter'],
                            '2I*: P+2S? (green)': all_settings['p2s_twoIstar'],
                            'rS (green)': 0.0,  # Not implemented
                            'rE (green)': 0.0,  # Not implemented
                            '2I* (green)': r.get('twoIstar', -1),
                            'Ng-p-all': n_p,
                            'Ng-s-all': n_s,
                            'Ng-all': total_photons
                        }
                        # Only add Z pixel if there's more than one frame
                        if n_frames > 1:
                            result_dict['Z pixel'] = i
                        batch_results.append(result_dict)

                    # Add batch results to this file's results list
                    file_results.extend(batch_results)
                    # Also add to main results list for backward compatibility
                    self.results.extend(batch_results)
                    batch_results = []  # Clear batch results for next line

            # Store the tau and rho arrays for this file
            self.tau_list.append(tau_array)
            self.rho_list.append(rho_array)

            # Set the current tau and rho for display (last processed file)
            self.tau = tau_array
            self.rho = rho_array

            # Reset frame and line progress for next file
            progress_dialog.set_frame_progress(0, 1)
            progress_dialog.set_line_progress(0, 1)
            QApplication.processEvents()

        time_stop = time.time()
        # Update progress dialog to show completion
        progress_dialog.set_file_progress(total_files, total_files)
        QApplication.processEvents()

        # Hide the progress dialog once processing is complete
        progress_dialog.hide()

        # Display the results
        self.display_results()

        # Update the fit and plot the result
        self.update_fit()

        # Enable export button
        self.export_button.setEnabled(True)

        # Show processing time
        QMessageBox.information(self, "Processing Complete", 
                               f"Processing completed in {time_stop - time_start:.2f} seconds.")

        # Auto export if enabled
        if auto_export:
            self.auto_export_results()

    def on_file_selection_changed(self, index):
        """Handle file selection changes in the combo box."""
        if index < 0 or not hasattr(self, 'tau_list') or not self.tau_list:
            return

        # Display the selected file's results
        if index < len(self.tau_list):
            self.display_file_results(index)

    def display_file_results(self, file_index):
        """Display results for a specific file."""
        if file_index < 0 or not hasattr(self, 'tau_list') or file_index >= len(self.tau_list):
            return

        # Get the tau array for the selected file
        tau = self.tau_list[file_index]

        # Display the lifetime image
        self.image_view.setImage(tau[0], levels=(0, 5))

        # Display the lifetime histogram
        self.hist_plot.clear()
        y, x = np.histogram(tau[0].flatten(), bins=131, range=(0.01, 5))
        self.hist_plot.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        self.hist_plot.setLabel('left', 'Count')
        self.hist_plot.setLabel('bottom', 'Lifetime (ns)')

    def display_results(self):
        """Display the results in the results tab."""
        if not hasattr(self, 'tau_list') or not self.tau_list:
            return

        # Update the file selector combo box
        self.file_selector_combo.blockSignals(True)
        self.file_selector_combo.clear()

        # Get selected files for naming
        tttr_files = self.tttr_list.get_selected_files()

        # Add file names to the combo box
        for i, _ in enumerate(self.tau_list):
            if i < len(tttr_files):
                file_name = os.path.basename(tttr_files[i])
            else:
                file_name = f"File {i+1}"
            self.file_selector_combo.addItem(file_name)

        self.file_selector_combo.blockSignals(False)

        # Display the first file's results
        if self.tau_list:
            self.display_file_results(0)

        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)

    def auto_export_results(self):
        """Automatically export the results to a file without user interaction."""
        if not hasattr(self, 'results_list') or not self.results_list:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return

        # Get all settings
        all_settings = self.get_settings()

        # Always export all results (flatten the results list for all files)
        filtered_results = [item for sublist in self.results_list for item in sublist]

        # Get selected files for naming
        tttr_files = self.tttr_list.get_selected_files()
        if not tttr_files:
            return

        # Determine base name for the filename
        if len(tttr_files) == 1:
            base_name = os.path.splitext(os.path.basename(tttr_files[0]))[0]
        else:
            base_name = "MultipleFiles"

        # Get parameters for filename from settings
        min_photons = all_settings['min_photons']
        binning_factor = all_settings['binning_factor']
        micro_time_start, micro_time_stop = all_settings['micro_time_range']

        # Get number of frames if available
        n_frames = 0
        if hasattr(self, 'clsm_p') and self.clsm_p is not None:
            n_frames = self.clsm_p.shape[0]

        # Create DataFrame from filtered results
        df = pd.DataFrame(filtered_results)

        # Check which file format is selected
        if all_settings['file_format_hdf']:
            # Save as HDF5
            file_name = f"{base_name}_Frames_{n_frames}_Green_Fit#23_BinFactor_{binning_factor}_MinPh#{min_photons}_MicroTime_{micro_time_start}-{micro_time_stop}.h5"
            file_dir = os.path.dirname(tttr_files[0])
            file_path = os.path.join(file_dir, file_name)

            # Save DataFrame to HDF5 file with compression
            # complevel: Compression level (0-9, 9 is highest compression)
            # complib: Compression library ('blosc' is fast and efficient)
            # format: 'table' allows for partial reading and querying (slower but more flexible)
            df.to_hdf(file_path, key='results', mode='w', complevel=9, complib='blosc', format='table')
        else:
            # Save as CSV (tab-separated)
            file_name = f"{base_name}_Frames_{n_frames}_Green_Fit#23_BinFactor_{binning_factor}_MinPh#{min_photons}_MicroTime_{micro_time_start}-{micro_time_stop}.pg4"
            file_dir = os.path.dirname(tttr_files[0])
            file_path = os.path.join(file_dir, file_name)

            # Export to file
            with open(file_path, 'w') as f:
                # Write header
                f.write('\t'.join(df.columns) + '\n')

                # Write data
                for _, row in df.iterrows():
                    f.write('\t'.join([f"{val:.8f}" if isinstance(val, float) else f"{val}" for val in row]) + '\n')

        QMessageBox.information(self, "Auto Export Complete", f"Results automatically exported to {file_path}")

    def export_results(self):
        """Export the results to a file."""
        if not hasattr(self, 'results_list') or not self.results_list:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return

        # Get all settings
        all_settings = self.get_settings()

        # Initialize export_option with a default value
        export_option = None

        # Ask if user wants to export all files or just the current file
        if len(self.tau_list) > 1:
            export_option = QtWidgets.QMessageBox.question(
                self, 
                "Export Options", 
                "Do you want to export results for all files or just the currently selected file?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.YesToAll | QtWidgets.QMessageBox.Cancel
            )

            if export_option == QtWidgets.QMessageBox.Cancel:
                return

            # Filter results based on selection
            if export_option == QtWidgets.QMessageBox.Yes:
                current_file_index = self.file_selector_combo.currentIndex()
                if current_file_index >= 0 and current_file_index < len(self.tau_list):
                    # Get results for the current file
                    filtered_results = self.results_list[current_file_index]
                else:
                    QMessageBox.warning(self, "Error", "No file currently selected.")
                    return
            else:  # YesToAll (all files)
                # Flatten the results list for all files
                filtered_results = [item for sublist in self.results_list for item in sublist]
        else:
            # Only one file, export all results
            filtered_results = self.results_list[0] if self.results_list else []

        # Get file name for export
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)

        # Set file filter based on selected format
        if all_settings['file_format_hdf']:
            file_dialog.setNameFilter("HDF5 Files (*.h5);;All Files (*.*)")
            file_dialog.setDefaultSuffix("h5")
        else:
            file_dialog.setNameFilter("PG4 Files (*.pg4);;All Files (*.*)")
            file_dialog.setDefaultSuffix("pg4")

        tttr_files = self.tttr_list.get_selected_files()
        if tttr_files:
            # Get base name for the filename
            if len(tttr_files) == 1 or (export_option == QtWidgets.QMessageBox.Yes and len(self.tau_list) > 1):
                # Single file export
                if export_option == QtWidgets.QMessageBox.Yes and len(self.tau_list) > 1:
                    current_file_index = self.file_selector_combo.currentIndex()
                    if current_file_index < len(tttr_files):
                        base_name = os.path.splitext(os.path.basename(tttr_files[current_file_index]))[0]
                    else:
                        base_name = f"File_{current_file_index+1}"
                else:
                    base_name = os.path.splitext(os.path.basename(tttr_files[0]))[0]
            else:
                # Multiple files export
                base_name = "MultipleFiles"

            # Get parameters for filename from settings
            min_photons = all_settings['min_photons']
            binning_factor = all_settings['binning_factor']
            micro_time_start, micro_time_stop = all_settings['micro_time_range']

            # Get number of frames if available
            n_frames = 0
            if hasattr(self, 'clsm_p') and self.clsm_p is not None:
                n_frames = self.clsm_p.shape[0]

            # Generate filename based on parameters and selected format
            if all_settings['file_format_hdf']:
                file_ext = ".h5"
            else:
                file_ext = ".pg4"

            file_name = f"{base_name}_Frames_{n_frames}_Green_Fit#23_BinFactor_{binning_factor}_MinPh#{min_photons}_MicroTime_{micro_time_start}-{micro_time_stop}{file_ext}"
            file_dialog.selectFile(file_name)

        if file_dialog.exec_():
            file_names = file_dialog.selectedFiles()
            if file_names:
                file_path = file_names[0]

                # Create DataFrame from filtered results
                df = pd.DataFrame(filtered_results)

                # Export to file based on selected format
                if all_settings['file_format_hdf']:
                    # Save as HDF5 with compression
                    # complevel: Compression level (0-9, 9 is highest compression)
                    # complib: Compression library ('blosc' is fast and efficient)
                    # fletcher32: Adds a checksum for data integrity
                    # format: 'table' allows for partial reading and querying (slower but more flexible)
                    df.to_hdf(file_path, key='results', mode='w', complevel=9, complib='blosc', fletcher32=True, format='table')
                else:
                    # Save as CSV (tab-separated)
                    with open(file_path, 'w') as f:
                        # Write header
                        f.write('\t'.join(df.columns) + '\n')

                        # Write data
                        for _, row in df.iterrows():
                            f.write('\t'.join([f"{val:.8f}" if isinstance(val, float) else f"{val}" for val in row]) + '\n')

                QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")

    def fit_parameters(self):
        """Get the fit parameters."""
        tau = self.tau_spinbox.value()
        gamma = self.gamma_spinbox.value()
        r0 = self.r0_spinbox.value()
        rho = self.rho_spinbox.value()
        x0 = np.array([tau, gamma, r0, rho])

        fixed = np.array([
            1 if self.fix_tau_checkbox.isChecked() else 0,
            1 if self.fix_gamma_checkbox.isChecked() else 0,
            1 if self.fix_r0_checkbox.isChecked() else 0,
            1 if self.fix_rho_checkbox.isChecked() else 0
        ])

        return x0, fixed

    def create_background(self, irf, use_bg=None, bg_fixed=None, bg_file=None, bg_p=None, bg_s=None):
        """Create background array based on UI controls or loaded background pattern.

        Parameters
        ----------
        irf : np.ndarray
            IRF array to match the shape of the background array.
        use_bg : bool, optional
            Whether to use background correction. If None, uses the value from the UI.
        bg_fixed : bool, optional
            Whether to use fixed background values. If None, uses the value from the UI.
        bg_file : bool, optional
            Whether to use background pattern from file. If None, uses the value from the UI.
        bg_p : float, optional
            Fixed background value for parallel channel. If None, uses the value from the UI.
        bg_s : float, optional
            Fixed background value for perpendicular channel. If None, uses the value from the UI.

        Returns
        -------
        np.ndarray
            Background array with the same shape as the IRF.
        """
        # Use values from parameters if provided, otherwise use values from UI
        use_bg = use_bg if use_bg is not None else self.use_bg_checkbox.isChecked()
        bg_fixed = bg_fixed if bg_fixed is not None else self.bg_fixed_radio.isChecked()
        bg_file = bg_file if bg_file is not None else self.bg_file_radio.isChecked()
        bg_p = bg_p if bg_p is not None else self.bg_p_spinbox.value()
        bg_s = bg_s if bg_s is not None else self.bg_s_spinbox.value()

        # If background correction is disabled, return zeros
        if not use_bg:
            return np.zeros_like(irf)

        # The first half is for parallel channel, the second half is for perpendicular channel
        n_half = len(irf) // 2
        background = np.zeros_like(irf)

        # Check if using file-based background
        if bg_file and self.bg_p is not None and self.bg_s is not None:
            # Use loaded background pattern
            # Make sure the background arrays match the expected size
            bg_p_array = self.bg_p
            bg_s_array = self.bg_s

            # Resize if necessary
            if len(bg_p_array) > n_half:
                bg_p_array = bg_p_array[:n_half]
            elif len(bg_p_array) < n_half:
                # Pad with zeros
                bg_p_array = np.pad(bg_p_array, (0, n_half - len(bg_p_array)), 'constant')

            if len(bg_s_array) > n_half:
                bg_s_array = bg_s_array[:n_half]
            elif len(bg_s_array) < n_half:
                # Pad with zeros
                bg_s_array = np.pad(bg_s_array, (0, n_half - len(bg_s_array)), 'constant')

            # Set background values
            background[:n_half] = bg_p_array
            background[n_half:] = bg_s_array

            chisurf.logging.info("Using background pattern from file")
        else:
            # Use fixed background values
            # Set background values
            background[:n_half] = bg_p
            background[n_half:] = bg_s

            chisurf.logging.info(f"Using fixed background values: P={bg_p}, S={bg_s}")

        return background

    def update_fit(self):
        """Update the fit based on current parameters."""
        if not hasattr(self, 'tttr_data') or self.tttr_data is None:
            return

        if not hasattr(self, 'irf_p') or self.irf_p is None:
            return

        if not hasattr(self, 'decay_all_photons') or self.decay_all_photons is None:
            return

        # Get all settings using the get_settings method
        all_settings = self.get_settings()

        # Get parameters for the fit
        x0 = np.array([all_settings['tau'], all_settings['gamma'], all_settings['r0'], all_settings['rho']])
        fixed = np.array([
            1 if all_settings['fix_tau'] else 0,
            1 if all_settings['fix_gamma'] else 0,
            1 if all_settings['fix_r0'] else 0,
            1 if all_settings['fix_rho'] else 0
        ])

        # Extract only the settings needed for Fit23
        settings = {
            'dt': all_settings['dt'],
            'g_factor': all_settings['g_factor'],
            'l1': all_settings['l1'],
            'l2': all_settings['l2'],
            'convolution_stop': -1,
            'irf': all_settings['irf'],
            'period': all_settings['period'],
            'background': all_settings['background'],
            'p2s_twoIstar_flag': all_settings['p2s_twoIstar'],  # Enable 2I* and 2I*: P+2S? calculation
            'soft_bifl_scatter_flag': all_settings['BIFL_scatter']  # Enable BIFL scatter fit
        }

        # Create Fit23 instance
        self._fit = tttrlib.Fit23(**settings)

        # Perform the fit
        res = self._fit(data=self.decay_all_photons, initial_values=x0, fixed=fixed)

        # Plot the fit result
        self.plot_fit_result(res)

    def plot_fit_result(self, fit_result):
        """Plot the fit result."""
        # Clear both plots
        self.combined_plot.clear()
        self.residual_plot.clear()

        if self._fit is None:
            return

        # Plot data and model in the combined plot
        self.combined_plot.plot(self._fit.data,
                               pen=None,
                               symbol='o',
                               symbolSize=3)
        self.combined_plot.plot(self._fit.model, pen='g')

        # Plot IRF with the same micro time range
        start, stop = self.micro_time_range
        irf_p_range = self.irf_p[:stop]
        irf_s_range = self.irf_s[:stop]
        if start > 0:
            irf_p_range = irf_p_range[start:]
            irf_s_range = irf_s_range[start:]

        # Prepare IRF (threshold, normalize, shift)
        irf_p_range, irf_s_range = self.prepare_irf(
            irf_p_range, irf_s_range,
            threshold=self.irf_threshold,
            shift=self.shift_spinbox.value(),
            shift_sp=self.shift_sp_spinbox.value(),
            shift_ss=self.shift_ss_spinbox.value()
        )

        irf = np.hstack([irf_p_range, irf_s_range])

        # Scale IRF for display purposes
        max_irf = np.max(irf) if np.max(irf) > 0 else 1
        max_data = max(self._fit.data) if max(self._fit.data) > 0 else 1
        irf = irf * (max_data / max_irf)
        self.combined_plot.plot(irf, pen='r', name='IRF')

        # Compute and plot weighted residuals
        data = self._fit.data
        model = self._fit.model
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            resid = (data - model) / np.sqrt(model)
        # Fall-back zeros where model==0
        resid = np.nan_to_num(resid)

        # Draw residuals in the top panel
        pen = pg.mkPen(color=(200, 20, 20), width=1)
        self.residual_plot.plot(resid,
                               pen=pen,
                               symbol='o',
                               symbolSize=3)

        # Update fit parameters display
        self.update_fit_ui(fit_result)

    def update_fit_ui(self, fit_result):
        """Update the fit parameters display."""
        if fit_result is None:
            return

        # Update labels with fit results
        self.tau_label.setText(f"Tau: {fit_result['x'][0]:.3f}")
        self.gamma_label.setText(f"Gamma: {fit_result['x'][1]:.3f}")
        self.r0_label.setText(f"R0: {fit_result['x'][2]:.3f}")
        self.rho_label.setText(f"Rho: {fit_result['x'][3]:.3f}")

        # Calculate chi-square
        if hasattr(self._fit, 'chi_square'):
            chi2 = self._fit.chi_square
        else:
            # Approximate chi-square calculation
            data = self._fit.data
            model = self._fit.model
            with np.errstate(divide='ignore', invalid='ignore'):
                chi2 = np.sum(((data - model) ** 2) / np.maximum(model, 1))

        self.chi2_label.setText(f"Chi²: {chi2:.3f}")

    def get_settings(self) -> Dict:
        """
        Gather all settings into a dictionary.

        Returns
        -------
        Dict
            Dictionary containing all settings.
        """
        # Get binning factor
        binning_factor = self.binning_factor_spinbox.value()

        # Get micro time range
        start, stop = self.micro_time_range

        # Get IRF with the applied micro time range
        if hasattr(self, 'irf_p') and self.irf_p is not None and hasattr(self, 'irf_s') and self.irf_s is not None:
            irf_p_range = self.irf_p[:stop]
            irf_s_range = self.irf_s[:stop]
            if start > 0:
                irf_p_range = irf_p_range[start:]
                irf_s_range = irf_s_range[start:]

            # Prepare IRF (threshold, normalize, shift)
            irf_p_range, irf_s_range = self.prepare_irf(
                irf_p_range, irf_s_range,
                threshold=self.irf_threshold,
                shift=self.shift_spinbox.value(),
                shift_sp=self.shift_sp_spinbox.value(),
                shift_ss=self.shift_ss_spinbox.value()
            )

            irf = np.hstack([irf_p_range, irf_s_range])
        else:
            irf = None

        # Get dt (time resolution)
        dt = None
        if hasattr(self, 'tttr_data') and self.tttr_data is not None:
            dt = self.tttr_data.header.micro_time_resolution * 1e9 * binning_factor

        # Get fit parameters
        x0, fixed = self.fit_parameters()

        # Create settings dictionary
        settings = {
            'dt': dt,
            'g_factor': self.g_factor_spinbox.value(),
            'l1': self.l1_spinbox.value(),
            'l2': self.l2_spinbox.value(),
            'convolution_stop': -1,
            'irf': irf,
            'period': self.period_spinbox.value(),
            'background': self.create_background(
                irf,
                use_bg=self.use_bg_checkbox.isChecked(),
                bg_fixed=self.bg_fixed_radio.isChecked(),
                bg_file=self.bg_file_radio.isChecked(),
                bg_p=self.bg_p_spinbox.value(),
                bg_s=self.bg_s_spinbox.value()
            ) if irf is not None else None,
            'binning_factor': binning_factor,
            'micro_time_range': self.micro_time_range,
            'ch_p': self.ch_p_spinbox.value(),
            'ch_s': self.ch_s_spinbox.value(),
            'min_photons': self.min_photons_spinbox.value(),
            'tau': x0[0],
            'gamma': x0[1],
            'r0': x0[2],
            'rho': x0[3],
            'fix_tau': self.fix_tau_checkbox.isChecked(),
            'fix_gamma': self.fix_gamma_checkbox.isChecked(),
            'fix_r0': self.fix_r0_checkbox.isChecked(),
            'fix_rho': self.fix_rho_checkbox.isChecked(),
            'irf_threshold': self.irf_threshold,
            'shift': self.shift_spinbox.value(),
            'shift_sp': self.shift_sp_spinbox.value(),
            'shift_ss': self.shift_ss_spinbox.value(),
            'use_bg': self.use_bg_checkbox.isChecked(),
            'bg_fixed': self.bg_fixed_radio.isChecked(),
            'bg_file': self.bg_file_radio.isChecked(),
            'bg_p': self.bg_p_spinbox.value(),
            'bg_s': self.bg_s_spinbox.value(),
            'adjust_stop': self.adjust_stop_checkbox.isChecked(),
            'read_period': self.read_period_checkbox.isChecked(),
            # Additional GUI elements
            'auto_export': self.checkBoxAutoExport.isChecked(),
            'stack_frames': self.checkBoxStackFrames.isChecked(),
            'file_format_hdf': self.radioButton_FileHDF.isChecked(),
            'file_format_csv': self.radioButton_FileCsv.isChecked(),
            'p2s_twoIstar': self.p2s_twoIstar,
            'BIFL_scatter': self.BIFL_scatter
        }

        return settings

    def load_settings_from_dict(self, settings: Dict):
        """
        Load settings from a dictionary.

        Parameters
        ----------
        settings : Dict
            Dictionary containing settings to load.
        """
        # Update UI elements with values from settings
        if 'g_factor' in settings:
            self.g_factor_spinbox.setValue(settings['g_factor'])
        if 'l1' in settings:
            self.l1_spinbox.setValue(settings['l1'])
        if 'l2' in settings:
            self.l2_spinbox.setValue(settings['l2'])
        if 'period' in settings:
            self.period_spinbox.setValue(settings['period'])
        if 'binning_factor' in settings:
            self.binning_factor_spinbox.setValue(settings['binning_factor'])
        if 'micro_time_range' in settings:
            self.micro_time_start_spinbox.setValue(settings['micro_time_range'][0])
            self.micro_time_stop_spinbox.setValue(settings['micro_time_range'][1])
            self.micro_time_range = settings['micro_time_range']
        if 'ch_p' in settings:
            self.ch_p_spinbox.setValue(settings['ch_p'])
        if 'ch_s' in settings:
            self.ch_s_spinbox.setValue(settings['ch_s'])
        if 'min_photons' in settings:
            self.min_photons_spinbox.setValue(settings['min_photons'])
        if 'tau' in settings:
            self.tau_spinbox.setValue(settings['tau'])
        if 'gamma' in settings:
            self.gamma_spinbox.setValue(settings['gamma'])
        if 'r0' in settings:
            self.r0_spinbox.setValue(settings['r0'])
        if 'rho' in settings:
            self.rho_spinbox.setValue(settings['rho'])
        if 'fix_tau' in settings:
            self.fix_tau_checkbox.setChecked(settings['fix_tau'])
        if 'fix_gamma' in settings:
            self.fix_gamma_checkbox.setChecked(settings['fix_gamma'])
        if 'fix_r0' in settings:
            self.fix_r0_checkbox.setChecked(settings['fix_r0'])
        if 'fix_rho' in settings:
            self.fix_rho_checkbox.setChecked(settings['fix_rho'])
        if 'irf_threshold' in settings:
            self.doubleSpinBox_irf_threshold.setValue(settings['irf_threshold'])
        if 'shift' in settings:
            self.shift_spinbox.setValue(settings['shift'])
        if 'shift_sp' in settings:
            self.shift_sp_spinbox.setValue(settings['shift_sp'])
        if 'shift_ss' in settings:
            self.shift_ss_spinbox.setValue(settings['shift_ss'])
        if 'use_bg' in settings:
            self.use_bg_checkbox.setChecked(settings['use_bg'])
        if 'bg_fixed' in settings and 'bg_file' in settings:
            self.bg_fixed_radio.setChecked(settings['bg_fixed'])
            self.bg_file_radio.setChecked(settings['bg_file'])
        if 'bg_p' in settings:
            self.bg_p_spinbox.setValue(settings['bg_p'])
        if 'bg_s' in settings:
            self.bg_s_spinbox.setValue(settings['bg_s'])
        if 'adjust_stop' in settings:
            self.adjust_stop_checkbox.setChecked(settings['adjust_stop'])
        if 'read_period' in settings:
            self.read_period_checkbox.setChecked(settings['read_period'])
        # Additional GUI elements
        if 'auto_export' in settings:
            self.checkBoxAutoExport.setChecked(settings['auto_export'])
        if 'stack_frames' in settings:
            self.checkBoxStackFrames.setChecked(settings['stack_frames'])
        if 'file_format_hdf' in settings:
            self.radioButton_FileHDF.setChecked(settings['file_format_hdf'])
        if 'file_format_csv' in settings:
            self.radioButton_FileCsv.setChecked(settings['file_format_csv'])
        if 'p2s_twoIstar' in settings:
            self.p2s_twoIstar = settings['p2s_twoIstar']
        if 'BIFL_scatter' in settings:
            self.BIFL_scatter = settings['BIFL_scatter']

        # Update parameters
        self.update_parameters()
