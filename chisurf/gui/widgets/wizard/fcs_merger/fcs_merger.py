import os
import pathlib
import typing
import json
import re
import numpy as np
import pyqtgraph as pg

import chisurf
import chisurf.core.fio as io
import chisurf.core.data
import chisurf.core.fluorescence.fcs
import chisurf.gui.decorators
from chisurf.gui import QtGui, QtWidgets, QtCore, uic
from .fcs_merger_ui import setup_ui as _setup_ui

colors = chisurf.core.settings.gui['plot']['colors']


class WizardFcsMerger(QtWidgets.QWizardPage):

    @property
    def correlation_folder(self):
        return pathlib.Path(self.lineEdit.text())

    @property
    def current_curve_idx(self) -> int:
        table = self.tableWidget
        idx = int(table.currentIndex().row())
        return idx

    @staticmethod
    def compute_average_correlations(correlations: typing.List[dict]) -> dict:
        taus = []
        cors = []
        ws = []
        acquisition_time = 0.0
        weighted_count_rate_sum = 0.0
        n_curves = len(correlations)
        
        for correlation in correlations:
            tau = np.array(correlation['x'])
            cor = np.array(correlation['y'])
            duration = correlation['duration']
            acquisition_time += duration
            counts = correlation['channel_a']['counts'] + correlation['channel_b']['counts']
            # Average count rate per channel in kHz: (total_counts / 2) / duration / 1000
            cr = (counts / 2.0) / duration / 1000.0
            # Weighted sum for proper averaging: sum(duration * count_rate)
            weighted_count_rate_sum += duration * cr
            taus.append(tau)
            cors.append(cor)
            # Only compute weights if merging multiple curves
            if n_curves > 1:
                w = chisurf.core.fluorescence.fcs.noise(tau, cor, duration, cr, weight_type='suren')
                ws.append(w)
        
        ys = np.array(cors)
        # Weighted average count rate: sum(duration * count_rate) / sum(duration)
        avg_count_rate = weighted_count_rate_sum / acquisition_time if acquisition_time > 0 else 0.0
        
        # For single curve, use zeros for error; for multiple curves, compute standard error
        if n_curves == 1:
            ey = np.zeros_like(ys[0])
        else:
            ey = np.std(ys, axis=0) / np.sqrt(n_curves)
        
        correlation = {
            'x': np.array(taus).mean(axis=0)[1:],
            'y': ys.mean(axis=0)[1:],
            'ey': ey[1:],
            'duration': acquisition_time,
            'count_rate': avg_count_rate
        }
        return correlation

    @property
    def mean_correlation(self) -> dict:
        # Use only curves with the checkbox checked
        selected_correlations = []
        for row in range(self.tableWidget.rowCount()):
            item = self.tableWidget.item(row, 0)
            if item is not None and item.checkState() == QtCore.Qt.Checked:
                selected_correlations.append(self.correlations[row])
        if not selected_correlations:
            # Fallback: if none are selected, use all curves.
            selected_correlations = self.correlations
        data = self.compute_average_correlations(selected_correlations)
        return data

    @property
    def merge_folder(self):
        return pathlib.Path('..')

    def update_plots(self, *args, **kwargs):
        chisurf.logging.info('WizardTTTRCorrelator::Updating plots')
        self.pw_fcs.clear()
        idx = self.current_curve_idx
        for i, cor in enumerate(self.correlations):
            # Check if the curve is selected for merging
            checkbox_item = self.tableWidget.item(i, 0)
            if checkbox_item is not None and checkbox_item.checkState() == QtCore.Qt.Unchecked:
                # Draw not-used curves with a dashed grey pen
                pen = pg.mkPen('grey', width=1.0, style=QtCore.Qt.DashLine)
            else:
                width = 3.0 if i == idx else 1.0
                pen = pg.mkPen(chisurf.core.settings.colors[i % len(chisurf.core.settings.colors)]['hex'], width=width)
            self.plot_item_fcs.plot(x=cor['x'], y=cor['y'], pen=pen)

        self.pw_fcs_mean.clear()
        corr_mean = self.mean_correlation
        self.plot_item_fcs_mean.plot(x=corr_mean['x'], y=corr_mean['y'])

    def onClearFiles(self):
        chisurf.logging.info("WizardTTTRCorrelator::onClearFiles")
        self.settings['tttr_filenames'].clear()
        self.comboBox.setEnabled(True)
        self.lineEdit.clear()

    def writeTableValues(self, table: QtWidgets.QTableWidget, arr):
        table.setRowCount(0)
        for row in arr:
            rc = table.rowCount()
            table.insertRow(rc)
            for i, col in enumerate(row):
                table.setItem(rc, i, QtWidgets.QTableWidgetItem(f"{col: 0.2f}"))
                table.resizeRowsToContents()
            table.resizeRowsToContents()

    def append_correlation(self, filename: pathlib.Path, correlation_dict):
        self.correlations.append(correlation_dict)
        table = self.tableWidget
        rc = table.rowCount()
        table.insertRow(rc)
        duration = float(correlation_dict.get('duration', 0.0))
        # Compute count rates in kHz
        try:
            cr_a = float(correlation_dict['channel_a']['counts']) / duration / 1000.0 if duration > 0 else 0.0
            cr_b = float(correlation_dict['channel_b']['counts']) / duration / 1000.0 if duration > 0 else 0.0
        except Exception:
            # Fallback: if only total count_rate present (already in kHz)
            total_cr = float(correlation_dict.get('count_rate', 0.0))
            cr_a = total_cr / 2.0
            cr_b = total_cr / 2.0

        fnw = QtWidgets.QTableWidgetItem(f"{filename.stem}")
        fnw.setToolTip(f'{filename.as_posix()}')
        table.setItem(rc, 1, fnw)
        table.setItem(rc, 2, QtWidgets.QTableWidgetItem(f"{cr_a: 0.2f}"))
        table.setItem(rc, 3, QtWidgets.QTableWidgetItem(f"{cr_b: 0.2f}"))
        table.setItem(rc, 4, QtWidgets.QTableWidgetItem(f"{duration: 0.2f}"))
        # Add a checkable item for using the curve in merging
        checkbox_item = QtWidgets.QTableWidgetItem()
        checkbox_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        checkbox_item.setCheckState(QtCore.Qt.Checked)
        table.setItem(rc, 0, checkbox_item)
        # Update the correlation dictionary accordingly
        correlation_dict['use_curve'] = True

    def open_correlation_folder(self, folder: pathlib.Path = None):
        chisurf.logging.info( "WizardFcsMerger::open_correlation_folder")
        self.tableWidget.setRowCount(0)
        if folder is None:
            folder = self.correlation_folder
        # Support both legacy JSON chunks and new .cor files
        json_files = sorted(list(folder.glob('*.json.gz')))
        cor_files = sorted(list(folder.glob('*.cor')))
        self.correlations.clear()
        # Load JSON chunks if present
        for file in json_files:
            try:
                with io.open_maybe_zipped(file) as fp:
                    d = json.load(fp)
                    self.append_correlation(file, d)
            except Exception:
                continue
        # Load .cor chunk files
        import numpy as _np
        for file in cor_files:
            try:
                arr = _np.loadtxt(str(file), delimiter='\t')
                if arr.ndim == 1 and arr.size >= 2:
                    arr = arr.reshape(-1, arr.size)
                x = arr[:, 0]
                y = arr[:, 1]
                # Third column encodes duration (row 0) and count_rate (row 1)
                duration = float(arr[0, 2]) if arr.shape[1] > 2 and arr.shape[0] >= 1 else 0.0
                count_rate = float(arr[1, 2]) if arr.shape[1] > 2 and arr.shape[0] >= 2 else 0.0
                ey = arr[:, 3] if arr.shape[1] > 3 else _np.zeros_like(x)
                # Derive per-channel counts by splitting total equally (best effort)
                total_counts = count_rate * duration
                half_counts = 0.5 * total_counts
                d = {
                    'x': x.tolist(),
                    'y': y.tolist(),
                    'ey': ey.tolist(),
                    'duration': duration,
                    'count_rate': count_rate,
                    'channel_a': {'channels': [], 'microtime_range': None, 'counts': half_counts},
                    'channel_b': {'channels': [], 'microtime_range': None, 'counts': half_counts}
                }
                self.append_correlation(file, d)
            except Exception:
                continue
        chisurf.logging.info('Opening analysis folder...')
        self.lineEdit_2.setText(self.target_filepath.as_posix())
        self.update_plots()

    def set_correlations(self, correlations: typing.List[dict], source_folder: pathlib.Path = None):
        """
        Populate the table and plots from already computed correlations (in-memory),
        without requiring chnk-*.json.gz files on disk.
        """
        self.tableWidget.setRowCount(0)
        self.correlations.clear()
        # Optionally bind the correlation folder for saving .cor output later
        if source_folder is not None:
            try:
                self.lineEdit.setText(source_folder.as_posix())
            except Exception:
                pass
        for i, cor in enumerate(correlations):
            fake_file = (source_folder / f'chnk-{i:04}.json.gz') if source_folder is not None else pathlib.Path(f'chnk-{i:04}.json.gz')
            self.append_correlation(fake_file, cor)
        try:
            self.lineEdit_2.setText(self.target_filepath.as_posix())
        except Exception:
            pass
        self.update_plots()

    @property
    def target_filepath(self) -> pathlib.Path:
        stem = self.correlation_folder.stem
        # Sanitize filename: allow letters, numbers, dot, dash, underscore
        safe_stem = re.sub(r'[^A-Za-z0-9._-]+', '_', stem)
        if not safe_stem or safe_stem in {'.', '..', '_'}:
            safe_stem = 'correlation'
        correlation_filename = safe_stem + '.cor'
        filename = self.correlation_folder.parent / correlation_filename
        return filename

    def save_mean_correlation(self, evt=None, filename: pathlib.Path = None):
        chisurf.logging.info("WizardFcsMerger::save_mean_correlation")
        correlation = self.mean_correlation
        if filename is None:
            filename = self.target_filepath
        # Ensure parent directory exists (e.g., .../cr5)
        try:
            filename.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        chisurf.logging.info(f"Saving: {filename}")
        suren_column = np.zeros_like(correlation['x'])
        suren_column[0] = correlation['duration']
        suren_column[1] = correlation['count_rate']
        
        # Only include error column if it contains non-zero values (multiple curves merged)
        if np.any(correlation['ey'] != 0):
            c = np.vstack([correlation['x'], correlation['y'], suren_column, correlation['ey']])
        else:
            # For single curve, save only 3 columns (x, y, suren)
            c = np.vstack([correlation['x'], correlation['y'], suren_column])
        
        # Use native path string to avoid UNC/as_posix issues on Windows
        # Format with 5 significant digits, suppress scientific notation for small numbers
        np.savetxt(str(filename), c.T, delimiter='\t', fmt='%.5g')

    def onRemoveRow(self):
        table = self.tableWidget
        rc = table.rowCount()
        idx = int(table.currentIndex().row())
        if rc >= 0:
            if idx < 0:
                idx = 0
            table.removeRow(idx)
            self.correlations.pop(idx)
            self.update_plots()

    def onRowDoubleClicked(self, item: QtWidgets.QTableWidgetItem):
        # Instead of removing the row, toggle the checkbox state
        row = item.row()
        checkbox_item = self.tableWidget.item(row, 0)
        if checkbox_item is not None:
            current_state = checkbox_item.checkState()
            new_state = QtCore.Qt.Unchecked if current_state == QtCore.Qt.Checked else QtCore.Qt.Checked
            checkbox_item.setCheckState(new_state)
            # Update the correlation dictionary if needed
            self.correlations[row]['use_curve'] = (new_state == QtCore.Qt.Checked)
            self.update_plots()

    def add_to_chisurf(self):
        """
        Add the generated correlation curve to chisurf as a dataset using FCS Kristine correlation.
        This method uses the already generated .cor file instead of creating a new one.
        """
        print("Adding correlation to ChiSurf...")
        chisurf.logging.info("WizardFcsMerger::adding correlation to chisurf")

        # Ensure the correlation file exists
        cor_file = self.target_filepath
        if not cor_file.exists():
            # Save the correlation file if it doesn't exist
            print("Saving correlation file...")
            print("Filename: ", cor_file.as_posix(), "\n")
            self.save_mean_correlation(filename=cor_file)

        if not cor_file.exists():
            # Display a message box to the user if file still doesn't exist
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setWindowTitle("No Correlation File")
            msg_box.setText("No correlation file available. Please save correlation data before adding to ChiSurf.")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg_box.exec_()
            return

        # Use the standard approach as specified in the issue description
        # Set the current experiment and setup using the global cs instance
        chisurf.core.actions.dispatch(
            name="experiment.set",
            payload={"name": "FCS"},
        )
        chisurf.core.actions.dispatch(
            name="setup.select",
            payload={"name": "Seidel Kristine"},
        )

        # Add dataset to chisurf using the standard approach
        chisurf.core.actions.dispatch(
            name="dataset.add",
            payload={"filename": cor_file.as_posix(), "experiment_reader": None},
        )

        # Show success message
        chisurf.logging.info(f"Added correlation to ChiSurf: {cor_file.name}")

    @chisurf.gui.decorators.init_with_ui("fcs_merger.ui")
    def __init__(self, *args, **kwargs):
        _setup_ui(self)
