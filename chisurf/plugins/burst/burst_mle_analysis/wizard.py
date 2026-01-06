import typing
import faulthandler

from chisurf.plugins.burst.burst_mle_analysis.utils import \
    LazyTTTRDict, NumpyEncoder, FileListWidget, random_search_hpo
from chisurf.plugins.burst.burst_mle_analysis.interpolate import interpolate_shift

faulthandler.enable(all_threads=True)

from typing import Union

from qtpy import QtWidgets, QtCore
from qtpy.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
import pyqtgraph as pg
import numpy as np
import pandas as pd

import json

import chisurf
import chisurf.gui.decorators
import chisurf.settings
import chisurf.gui.widgets.wizard
from chisurf.gui.widgets.wizard.tttr_channeldefinition import \
    load_detector_setups, save_detector_setups

from pathlib import Path
import tttrlib
from typing import Dict
from chisurf.fio import write_jordi


class MLELifetimeAnalysisWizard(QtWidgets.QMainWindow):
    """
    Note on legacy burst processors:
    Older implementations (process_bursts_old, process_bursts_new, process_bursts_new2, process_bursts_new3)
    were moved to chisurf.plugins.burst_mle_analysis.wizard_old for documentation/archiving.
    Only process_bursts_new4 is kept here and exposed as `process_bursts`.
    """

    def _save_burst_results_fast(self, result_df: pd.DataFrame) -> None:
        """
        Save burst-fit results grouped by (file stem, detector) with a fast, vectorized path.
        - Builds zero-interleaved rows (zero, data, zero, data, ..., zero) via NumPy.
        - Writes each output once with np.savetxt.
        - Writes channel_settings.json once per output folder.
        - Shows a modal QProgressDialog and supports cancelation.

        Expects columns:
          'First File', 'Detector', and the per-detector numeric columns produced above.
        Uses:
          self.burst_files_list, self.channel_definer, self.channel_settings
        """
        # Collect selected files and map by stem
        files = self.burst_files_list.get_selected_files()
        files_by_stem = {Path(p).stem: Path(p) for p in files}
        selected_stems = set(files_by_stem.keys())

        if result_df is None or result_df.empty:
            QtWidgets.QMessageBox.information(self, "Done", "No burst-fit results to save.")
            return
        if not selected_stems:
            QtWidgets.QMessageBox.information(self, "Done", "No files selected to save.")
            return

        # Compute stems once; filter to selected stems; attach "First Stem" without double-mapping
        stems_series = result_df['First File'].map(lambda fn: Path(fn).stem)
        mask = stems_series.isin(selected_stems)
        if not mask.any():
            QtWidgets.QMessageBox.information(self, "Done", "No burst-fit rows matched the selected files.")
            return
        res = result_df.loc[mask].copy()
        res['First Stem'] = stems_series.loc[mask].values

        # Prepare detector metadata (columns per detector)
        dets = list(self.channel_definer.detectors.keys())
        det_meta = {}
        for det in dets:
            color = det.lower()
            letter = color[0]
            cols = [
                'Ng-p-all', 'Ng-s-all',
                f'Number of Photons (fit window) ({color})',
                f'2I*  ({color})', f'Tau ({color})', f'gamma ({color})',
                f'r0 ({color})', f'rho ({color})', f'BIFL scatter? ({color})',
                f'2I*: P+2S? ({color})', f'r Scatter ({color})', f'r Experimental ({color})'
            ]
            det_meta[det] = (color, letter, cols)

        # Group once by (First Stem, Detector)
        groups = res.groupby(['First Stem', 'Detector'], sort=False)
        total_tasks = len(groups)

        progress = QProgressDialog("Saving burst-fit results...", "Cancel", 0, total_tasks, self)
        progress.setWindowTitle("Saving burst-fit results")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        current_task = 0
        written_dirs = set()
        wrote_settings_for = set()

        def maybe_pump_ui(k: int) -> None:
            # Throttle UI event processing
            if (k % 25) == 0:
                QtWidgets.QApplication.processEvents()

        for (stem, det), df_g in groups:
            meta = det_meta.get(det)
            if meta is None:
                # Unknown detector label in results; skip gracefully
                current_task += 1
                progress.setValue(current_task)
                maybe_pump_ui(current_task)
                if progress.wasCanceled():
                    progress.close()
                    QtWidgets.QMessageBox.information(self, "Canceled", "Save operation was canceled.")
                    return
                continue

            color, letter, cols = meta
            file_path = files_by_stem.get(stem)
            if file_path is None:
                # Group doesn't map to a selected file (filtered above, but guard anyway)
                current_task += 1
                progress.setValue(current_task)
                maybe_pump_ui(current_task)
                if progress.wasCanceled():
                    progress.close()
                    QtWidgets.QMessageBox.information(self, "Canceled", "Save operation was canceled.")
                    return
                continue

            out_dir = file_path.parent.parent / f"b{letter}4"
            out_dir.mkdir(parents=True, exist_ok=True)
            written_dirs.add(out_dir.name)

            # Build zero-interleaved matrix efficiently
            arr = df_g[cols].to_numpy(dtype=float, copy=False)
            out = np.zeros((arr.shape[0] * 2 + 1, arr.shape[1]), dtype=float)
            out[1::2] = arr  # fill odd rows with data

            out_file = out_dir / f"{stem}.b{letter}4"
            with open(out_file, 'w', newline='') as f:
                f.write('\t'.join(cols) + '\t\n')  # keep trailing tab + newline
                np.savetxt(f, out, delimiter='\t', fmt='%.6f')

            # Write channel settings once per folder
            if out_dir not in wrote_settings_for:
                settings_file = out_dir / 'channel_settings.json'
                with open(settings_file, 'w') as sf:
                    json.dump(self.channel_settings, sf, indent=4, cls=NumpyEncoder)
                wrote_settings_for.add(out_dir)

            current_task += 1
            progress.setValue(current_task)
            maybe_pump_ui(current_task)
            if progress.wasCanceled():
                progress.close()
                QtWidgets.QMessageBox.information(self, "Canceled", "Save operation was canceled.")
                return

        progress.close()
        folder_list = ", ".join(sorted(written_dirs)) if written_dirs else "(no data)"
        QtWidgets.QMessageBox.information(self, "Done", f"Burst-fit results saved in folders: {folder_list}")

    @property
    def scatter_count_rate(self) -> float:
        """
        Total background count‐rate (in counts per second)
        for the currently selected detector.
        """
        # self.bg is an array of counts/sec per channel bin
        return float(np.sum(self.bg))

    @property
    def irf_threshold_vv(self) -> float:
        return float(self.doubleSpinBox_irf_threshold_vv.value())

    @irf_threshold_vv.setter
    def irf_threshold_vv(
            self,
            v: float
    ):
        self.doubleSpinBox_irf_threshold_vv.setValue(v)
        try:
            if not self.doubleSpinBox_irf_threshold_vv.signalsBlocked():
                self.on_irf_parameters_changed()
        except Exception:
            pass

    @property
    def irf_threshold_vh(self) -> float:
        return float(self.doubleSpinBox_irf_threshold_vh.value())

    @irf_threshold_vh.setter
    def irf_threshold_vh(
            self,
            v: float
    ):
        self.doubleSpinBox_irf_threshold_vh.setValue(v)
        try:
            if not self.doubleSpinBox_irf_threshold_vh.signalsBlocked():
                self.on_irf_parameters_changed()
        except Exception:
            pass

    @property
    def min_photons(self) -> float:
        return float(self.spinBox_min_photons.value())

    @min_photons.setter
    def min_photons(
            self,
            v: int
    ):
        self.spinBox_min_photons.setValue(int(v))

    @property
    def one_for_all_irf(self) -> bool:
        return self.checkBox_irf_one_for_all.isChecked()

    @one_for_all_irf.setter
    def one_for_all_irf(
            self,
            v: bool
    ):
        self.checkBox_irf_one_for_all.setChecked(v)

    @property
    def one_for_all_bg(self) -> bool:
        return self.checkBox_bg_one_for_all.isChecked()

    @one_for_all_bg.setter
    def one_for_all_bg(
            self,
            v: bool
    ):
        self.checkBox_bg_one_for_all.setChecked(v)

    @property
    def p2s_twoIstar(self) -> bool:
        return self.checkBox_2IStar.isChecked()

    @p2s_twoIstar.setter
    def p2s_twoIstar(self, v: bool):
        self.checkBox_2IStar.setChecked(v)

    @property
    def shift(self) -> int:
        """Global shift of the second (ss) decay relative to the first (sp).
        Stored as double in settings JSON and restored as float.
        """
        return int(self.doubleSpinBox_shift.value())

    @shift.setter
    def shift(self, v: int):
        # accept int or float, store/display as float
        self.doubleSpinBox_shift.setValue(int(v))
        try:
            if not self.doubleSpinBox_shift.signalsBlocked():
                self.on_irf_parameters_changed()
        except Exception:
            pass

    @property
    def shift_sp(self) -> float:
        """Sub-channel (fractional) shift to apply to the sp IRF."""
        return float(self.doubleSpinBox_shift_sp.value())

    @shift_sp.setter
    def shift_sp(self, v: float):
        self.doubleSpinBox_shift_sp.setValue(float(v))
        try:
            if not self.doubleSpinBox_shift_sp.signalsBlocked():
                self.on_irf_parameters_changed()
        except Exception:
            pass

    @property
    def shift_ss(self) -> float:
        """Sub-channel (fractional) shift to apply to the ss IRF."""
        return float(self.doubleSpinBox_shift_ss.value())

    @shift_ss.setter
    def shift_ss(self, v: float):
        self.doubleSpinBox_shift_ss.setValue(float(v))
        try:
            if not self.doubleSpinBox_shift_ss.signalsBlocked():
                self.on_irf_parameters_changed()
        except Exception:
            pass

    @property
    def irf_start(self) -> int:
        """Start index for IRF windowing. If < 0, do not zero the beginning."""
        return int(self.spinBox_irf_start.value())

    @irf_start.setter
    def irf_start(self, v: int):
        self.spinBox_irf_start.setValue(int(v))
        try:
            if not self.spinBox_irf_start.signalsBlocked():
                self.on_irf_parameters_changed()
        except Exception:
            pass

    @property
    def irf_stop(self) -> int:
        """Stop index for IRF windowing. If < 0, do not zero the end."""
        return int(self.spinBox_irf_stop.value())

    @irf_stop.setter
    def irf_stop(self, v: int):
        self.spinBox_irf_stop.setValue(int(v))
        try:
            if not self.spinBox_irf_stop.signalsBlocked():
                self.on_irf_parameters_changed()
        except Exception:
            pass

    @property
    def irf(self) -> np.ndarray:
        det = self.current_detector
        arr = self.irf_np.get(det)
        if arr is None:
            # fallback default IRF
            length = max(2, (self.micro_time_range[1] // self.micro_time_binning) * 2)
            arr = np.zeros(length, dtype=np.float64)
            arr[0] = 1.0
            arr[length // 2] = 1.0

        # split into sp / ss halves
        half = len(arr) // 2
        sp = arr[:half].astype(np.float64)
        ss = arr[half:].astype(np.float64)

        # 1) Global integer shift of VH (second half)
        # shift is stored as float in settings; np.roll requires int steps
        if float(self.shift) != 0.0:
            ss = np.roll(ss, int(round(self.shift)))

        # 2) Individual sub-bin IRF shifts
        sp = interpolate_shift(sp, self.shift_sp)
        ss = interpolate_shift(ss, self.shift_ss)

        # 3) IRF range/windowing (zero outside of [start, stop])
        try:
            start = int(self.irf_start)
            stop = int(self.irf_stop)

            if start >= 0:
                sp[:max(0, start)] = 0
                ss[:max(0, start)] = 0
            if stop >= 0 and stop + 1 < sp.size:
                sp[stop + 1:] = 0
            if stop >= 0 and stop + 1 < ss.size:
                ss[stop + 1:] = 0
        except Exception:
            # be permissive if widgets not yet constructed
            pass

        # 4) IRF thresholding (background correction)
        try:
            th_vv = float(self.irf_threshold_vv)
            th_vh = float(self.irf_threshold_vh)
            if th_vv > 0 and sp.size and sp.max() > 0:
                sp[sp < th_vv * sp.max()] = 0
            if th_vh > 0 and ss.size and ss.max() > 0:
                ss[ss < th_vh * ss.max()] = 0
        except Exception:
            pass

        # reassemble after per-channel processing
        irf = np.hstack([sp, ss])
        return irf

    @property
    def bg(self) -> np.ndarray:
        """
        Return the background array for the currently selected detector.
        Apply VH integer shift (doubleSpinBox_shift) BEFORE any other operation.
        If none was loaded, return a zeros default matching the IRF length.
        """
        det = self.current_detector
        arr = self.bg_np.get(det)
        if arr is None:
            # match IRF length
            return np.zeros_like(self.irf)
        # ensure float64 copy
        arr = np.asarray(arr, dtype=np.float64)
        half = len(arr) // 2
        vv = arr[:half].copy()
        vh = arr[half:].copy()
        # IMPORTANT: apply integer shift to VH before anything else
        if self.shift != 0:
            vh = np.roll(vh, self.shift)
        return np.hstack([vv, vh])

    @property
    def BIFL_scatter(self) -> bool:
        return self.checkBox_BIFL_scatter.isChecked()

    @BIFL_scatter.setter
    def BIFL_scatter(self, v: bool):
        self.checkBox_BIFL_scatter.setChecked(v)

    @property
    def fit(self):
        if self._fit is None:
            self._fit = self.create_fit_instance()
        return self._fit

    @property
    def save_jordis(self):
        return self.checkBox_save_jordis.isChecked()

    @property
    def fix_tau(self) -> bool:
        """Whether the tau parameter is fixed during fitting."""
        return self.checkBox_fix_tau.isChecked()

    @fix_tau.setter
    def fix_tau(self, v: bool):
        self.checkBox_fix_tau.setChecked(v)

    @property
    def fix_gamma(self) -> bool:
        """Whether the gamma parameter is fixed during fitting."""
        return self.checkBox_fix_gamma.isChecked()

    @fix_gamma.setter
    def fix_gamma(self, v: bool):
        self.checkBox_fix_gamma.setChecked(v)

    @property
    def fix_r0(self) -> bool:
        """Whether the r0 parameter is fixed during fitting."""
        return self.checkBox_fix_r0.isChecked()

    @fix_r0.setter
    def fix_r0(self, v: bool):
        self.checkBox_fix_r0.setChecked(v)

    @property
    def fix_rho(self) -> bool:
        """Whether the rho parameter is fixed during fitting."""
        return self.checkBox_fix_rho.isChecked()

    @fix_rho.setter
    def fix_rho(self, v: bool):
        self.checkBox_fix_rho.setChecked(v)

    @property
    def current_file_idx(self) -> int:
        """Current file index in the burst files list."""
        return self.spinBox_current_file_idx.value()

    @current_file_idx.setter
    def current_file_idx(self, v: int):
        self.spinBox_current_file_idx.setValue(v)

    @property
    def tau(self) -> float:
        """Fluorescence lifetime (tau) in nanoseconds."""
        return self.doubleSpinBox_tau.value()

    @tau.setter
    def tau(self, v: float):
        self.doubleSpinBox_tau.setValue(v)
        try:
            if not self.doubleSpinBox_tau.signalsBlocked():
                self.update_variable_fit_parameters()
        except Exception:
            pass

    @property
    def gamma(self) -> float:
        """Gamma parameter for fitting."""
        return self.doubleSpinBox_gamma.value()

    @gamma.setter
    def gamma(self, v: float):
        self.doubleSpinBox_gamma.setValue(v)
        try:
            if not self.doubleSpinBox_gamma.signalsBlocked():
                self.update_variable_fit_parameters()
        except Exception:
            pass

    @property
    def r0(self) -> float:
        """Fundamental anisotropy (r0) parameter."""
        return self.doubleSpinBox_r0.value()

    @r0.setter
    def r0(self, v: float):
        self.doubleSpinBox_r0.setValue(v)
        try:
            if not self.doubleSpinBox_r0.signalsBlocked():
                self.update_variable_fit_parameters()
        except Exception:
            pass

    @property
    def rho(self) -> float:
        """Rotational correlation time (rho) in nanoseconds."""
        return self.doubleSpinBox_rho.value()

    @rho.setter
    def rho(self, v: float):
        self.doubleSpinBox_rho.setValue(v)
        try:
            if not self.doubleSpinBox_rho.signalsBlocked():
                self.update_variable_fit_parameters()
        except Exception:
            pass

    @property
    def scatter_countrate(self) -> float:
        """Scatter count rate in Hz."""
        return self.doubleSpinBox_scatter_Countrate.value()

    @scatter_countrate.setter
    def scatter_countrate(self, v: float):
        self.doubleSpinBox_scatter_Countrate.setValue(v)

    @property
    def tau_result(self) -> float:
        """Fitted fluorescence lifetime (tau) result in nanoseconds."""
        return self.doubleSpinBox_tau_result.value()

    @tau_result.setter
    def tau_result(self, v: float):
        self.doubleSpinBox_tau_result.setValue(v)

    @property
    def gamma_result(self) -> float:
        """Fitted gamma parameter result."""
        return self.doubleSpinBox_gamma_result.value()

    @gamma_result.setter
    def gamma_result(self, v: float):
        self.doubleSpinBox_gamma_result.setValue(v)

    @property
    def r0_result(self) -> float:
        """Fitted fundamental anisotropy (r0) result."""
        return self.doubleSpinBox_r0_result.value()

    @r0_result.setter
    def r0_result(self, v: float):
        self.doubleSpinBox_r0_result.setValue(v)

    @property
    def rho_result(self) -> float:
        """Fitted rotational correlation time (rho) result in nanoseconds."""
        return self.doubleSpinBox_rho_result.value()

    @rho_result.setter
    def rho_result(self, v: float):
        self.doubleSpinBox_rho_result.setValue(v)

    @property
    def twoIstar_result(self) -> float:
        """Fitted 2I* result."""
        return self.doubleSpinBox_twoIstar_result.value()

    @twoIstar_result.setter
    def twoIstar_result(self, v: float):
        self.doubleSpinBox_twoIstar_result.setValue(v)

    @property
    def r_scatter_result(self) -> float:
        """Fitted r scatter result."""
        return self.doubleSpinBox_r_scatter_result.value()

    @r_scatter_result.setter
    def r_scatter_result(self, v: float):
        self.doubleSpinBox_r_scatter_result.setValue(v)

    @property
    def r_exp_result(self) -> float:
        """Fitted r experimental result."""
        return self.doubleSpinBox_r_exp_result.value()

    @r_exp_result.setter
    def r_exp_result(self, v: float):
        self.doubleSpinBox_r_exp_result.setValue(v)

    @property
    def irf_select(self) -> str:
        """Selected detector for IRF."""
        return self.comboBox_irf_select.currentText()

    @irf_select.setter
    def irf_select(self, v: str):
        index = self.comboBox_irf_select.findText(v)
        if index >= 0:
            self.comboBox_irf_select.setCurrentIndex(index)

    @property
    def background_select(self) -> str:
        """Selected detector for background."""
        return self.comboBox_background_select.currentText()

    @background_select.setter
    def background_select(self, v: str):
        index = self.comboBox_background_select.findText(v)
        if index >= 0:
            self.comboBox_background_select.setCurrentIndex(index)

    @property
    def total_burst_time_seconds(self) -> float:
        """
        Total integrated burst duration, in seconds.
        Returns 0.0 if no burst DataFrame is loaded or if the column is missing.
        """
        if self.df_bursts is None or 'Duration (ms)' not in self.df_bursts:
            return 0.0
        # sum durations (ms) and convert to seconds
        total_ms = self.df_bursts['Duration (ms)'].sum()
        return total_ms / 1000.0

    @property
    def dt_effective(self):
        return self.channel_definer.effective_micro_time_resolution

    @property
    def n_bursts(self) -> int:
        """Number of bursts currently loaded."""
        return len(self.df_bursts) if self.df_bursts is not None else 0

    @property
    def detector_channels(self):
        parallel = self.channel_definer.detectors[self.current_detector]["chs"][::2]
        perp = self.channel_definer.detectors[self.current_detector]["chs"][1::2]
        return parallel, perp

    @property
    def micro_time_range(self):
        """
        Returns [start_bin, stop_bin] as set by the spin boxes.
        """
        return [
            self.spinBox_micro_time_start.value(),
            self.spinBox_micro_time_stop.value()
        ]

    @micro_time_range.setter
    def micro_time_range(self, value):
        start, stop = value
        self.spinBox_micro_time_start.setValue(start)
        self.spinBox_micro_time_stop.setValue(stop)
        # self.spinBox_micro_time_start.setMinimum(start)
        # self.spinBox_micro_time_start.setMaximum(stop)
        # self.spinBox_micro_time_stop.setMinimum(start)
        # self.spinBox_micro_time_stop.setMaximum(stop)

    @property
    def micro_time_start(self) -> int:
        """Start bin of the micro-time window."""
        return int(self.spinBox_micro_time_start.value())

    @micro_time_start.setter
    def micro_time_start(self, v: int):
        self.spinBox_micro_time_start.setValue(int(v))

    @property
    def micro_time_stop(self) -> int:
        """Stop bin (exclusive) of the micro-time window."""
        return int(self.spinBox_micro_time_stop.value())

    @micro_time_stop.setter
    def micro_time_stop(self, v: int):
        self.spinBox_micro_time_stop.setValue(int(v))

    @property
    def micro_time_binning(self):
        return self.channel_definer.tttr_reading['micro_time_binning']

    @property
    def tttr_file_type(self):
        # Use the filetype property from DetectorWizardPage
        txt = self.channel_definer.filetype
        if txt is None:  # This means "Auto" was selected in DetectorWizardPage
            # Try to use the first tttr path from the LazyTTTRDict
            if self._tttr_paths:
                # Get the first tttr path from the LazyTTTRDict
                first_path = next(iter(self._tttr_paths.values()))
                if first_path:
                    file_type_int = tttrlib.inferTTTRFileType(str(first_path))
                    return file_type_int

            # Fall back to current_filename if no tttr paths are available
            filename = self.current_filename
            if filename:
                file_type_int = tttrlib.inferTTTRFileType(filename)
                return file_type_int
            return None
        return txt

    @property
    def fit_parameters(self):
        tau = self.tau
        gamma = self.gamma
        r0 = self.r0
        rho = self.rho
        fixed = [
            int(self.fix_tau),
            int(self.fix_gamma),
            int(self.fix_r0),
            int(self.fix_rho),
        ]
        return np.array([tau, gamma, r0, rho]), np.array(fixed)

    @property
    def current_filename(self) -> str:
        return self.lineEdit_current_filename.text()

    @current_filename.setter
    def current_filename(self, value: str):
        self.lineEdit_current_filename.setText(value)

    @property
    def current_detector(self):
        return self.comboBox_window.currentText()

    @property
    def excitation_period(self) -> float:
        return float(self.channel_definer.excitation_period)

    @property
    def g_factor(self) -> float:
        """
        Returns the g-factor value for the current detector.
        If the current detector doesn't have a g_factor value, returns the default value (1).
        """
        try:
            return float(self.channel_definer.detectors[self.current_detector].get("g_factor", 1.0))
        except (KeyError, AttributeError):
            return 1.0

    @g_factor.setter
    def g_factor(
            self,
            v: float
    ):
        """
        Sets the g-factor value for the current detector.
        Updates the detector data structure directly.
        """
        try:
            # Get the current detector data
            detector = self.current_detector
            if detector in self.channel_definer.detectors:
                # Update the g_factor value in the data structure
                self.channel_definer.detectors[detector]["g_factor"] = float(v)
                # Update the UI
                row = self._find_detector_row(detector)
                if row >= 0:
                    self.channel_definer.detectors_form.cellWidget(row, 3).setText(str(v))
        except (KeyError, AttributeError):
            pass

    @property
    def l1(self) -> float:
        """
        Returns the l1 value for the current detector.
        If the current detector doesn't have an l1 value, returns the default value (0).
        """
        try:
            return float(self.channel_definer.detectors[self.current_detector].get("l1", 0.0))
        except (KeyError, AttributeError):
            return 0.0

    @l1.setter
    def l1(
            self,
            v: float
    ):
        """
        Sets the l1 value for the current detector.
        Updates the detector data structure directly.
        """
        try:
            # Get the current detector data
            detector = self.current_detector
            if detector in self.channel_definer.detectors:
                # Update the l1 value in the data structure
                self.channel_definer.detectors[detector]["l1"] = float(v)
                # Update the UI
                row = self._find_detector_row(detector)
                if row >= 0:
                    self.channel_definer.detectors_form.cellWidget(row, 4).setText(str(v))
        except (KeyError, AttributeError):
            pass

    @property
    def l2(self) -> float:
        """
        Returns the l2 value for the current detector.
        If the current detector doesn't have an l2 value, returns the default value (0).
        """
        try:
            return float(self.channel_definer.detectors[self.current_detector].get("l2", 0.0))
        except (KeyError, AttributeError):
            return 0.0

    @l2.setter
    def l2(
            self,
            v: float
    ):
        """
        Sets the l2 value for the current detector.
        Updates the detector data structure directly.
        """
        try:
            # Get the current detector data
            detector = self.current_detector
            if detector in self.channel_definer.detectors:
                # Update the l2 value in the data structure
                self.channel_definer.detectors[detector]["l2"] = float(v)
                # Update the UI
                row = self._find_detector_row(detector)
                if row >= 0:
                    self.channel_definer.detectors_form.cellWidget(row, 5).setText(str(v))
        except (KeyError, AttributeError):
            pass

    def _ensure_channel_state(self, det: str):
        """
        Ensure self.channel_settings[det] exists and contains all required
        per-detector settings. If a key is missing, initialize it from the
        current UI/properties at the time of the call. This guarantees that
        later code can safely use st['key'] without fallbacks.
        """
        st = self.channel_settings.get(det, {})
        # micro-time
        if 'micro_time_start' not in st or 'micro_time_stop' not in st:
            sb, eb = self.micro_time_range
            st['micro_time_start'] = int(sb)
            st['micro_time_stop'] = int(eb)
        if 'micro_time_binning' not in st:
            st['micro_time_binning'] = int(self.micro_time_binning)
        # thresholds and shifts
        st.setdefault('irf_threshold_vv', float(getattr(self, 'irf_threshold_vv', 0.0)))
        st.setdefault('irf_threshold_vh', float(getattr(self, 'irf_threshold_vh', 0.0)))
        st.setdefault('shift', int(getattr(self, 'shift', 0)))
        st.setdefault('shift_sp', float(getattr(self, 'shift_sp', 0.0)))
        st.setdefault('shift_ss', float(getattr(self, 'shift_ss', 0.0)))
        # timing
        st.setdefault('dt', float(self.dt_effective))
        st.setdefault('excitation_period', float(self.excitation_period))
        # model parameters
        st.setdefault('g_factor', float(self.g_factor))
        st.setdefault('l1', float(self.l1))
        st.setdefault('l2', float(self.l2))
        # initial guesses and fixed flags
        x0, fixed = self.fit_parameters
        st.setdefault('initial_x0', np.array(x0))
        st.setdefault('fixed_flags', np.array(fixed))
        # options and counts
        st.setdefault('p2s_twoIstar', bool(getattr(self, 'p2s_twoIstar', False)))
        st.setdefault('BIFL_scatter', bool(getattr(self, 'BIFL_scatter', False)))
        st.setdefault('min_photons', int(getattr(self, 'min_photons', 0)))
        # IRF/BG arrays: ensure keys exist even if empty
        st.setdefault('irf', np.array(self.irf_np.get(det, np.array([]))))
        st.setdefault('bg', np.array(self.bg_np.get(det, np.array([]))))
        self.channel_settings[det] = st
        return st

    def _init_channels_from_wizard(self):
        """
        Called whenever the DetectorWizardPage has a new set of detectors;
        populates IRF/BG selectors, window combobox, and per-channel state.
        """
        dets = list(self.channel_definer.detectors.keys())
        chisurf.logging.info('_init_channels_from_wizard')
        # reset our per-channel state cache and pre-initialize per-detector dicts
        self.channel_settings.clear()
        for d in dets:
            self.channel_settings.setdefault(d, {})
            self.irf_np.setdefault(d, np.array([]))
            self.bg_np.setdefault(d, np.array([]))
            self._ensure_channel_state(d)

        # refill all of the "window" and IRF/BG dropdowns
        self.comboBox_window.clear()
        self.comboBox_window.addItems(dets)
        self.comboBox_window.setCurrentIndex(0)
        self._switch_filewidget(self.irf_file_widgets, dets[0])
        self._switch_filewidget(self.bg_file_widgets, dets[0])

        for cb in (self.comboBox_irf_select, self.comboBox_background_select):
            cb.clear()
            cb.addItems(dets)
            cb.setCurrentIndex(0)

        # finally, apply settings for the initially selected detector including any saved MLE settings
        if dets:
            self._on_channel_changed(dets[0])

    def _on_tab_changed(self, index: int):
        if self.tabWidget.widget(index) is self.tab_parameters:
            current_idx = self.spinBox_current_file_idx.value()
            self.spinBox_current_file_idx.blockSignals(True)
            self.spinBox_current_file_idx.setValue(current_idx)
            self.spinBox_current_file_idx.blockSignals(False)
            self.update_current_file(current_idx)
            self.update_bg_files()

    def _update_max_bins_from_tttr(self):
        try:
            tttr = next(iter(self.tttrs.values()))
            total_channels = tttr.header.number_of_micro_time_channels
        except Exception:
            return

        max_bins: int = int(total_channels // self.micro_time_binning)
        # Only update the allowed maximums; do not overwrite current values or minimums
        try:
            self.spinBox_micro_time_start.blockSignals(True)
            self.spinBox_micro_time_stop.blockSignals(True)
            # Only set maxima based on TTTR info
            self.spinBox_micro_time_start.setMaximum(max(0, max_bins - 1))
            self.spinBox_micro_time_stop.setMaximum(max_bins)
        finally:
            self.spinBox_micro_time_start.blockSignals(False)
            self.spinBox_micro_time_stop.blockSignals(False)

        # Keep a separate "full" range for internal histogram building
        self.full_range = (0, max_bins)

    def _capture_current_ui_state(self):
        chisurf.logging.info("_capture_current_ui_state")
        x0, fixed = self.fit_parameters
        start_bin, stop_bin = self.micro_time_range
        d = {
            'micro_time_start': start_bin,
            'micro_time_stop': stop_bin,
            'micro_time_binning': self.micro_time_binning,
            'irf_threshold_vv': self.irf_threshold_vv,
            'irf_threshold_vh': self.irf_threshold_vh,
            'shift': self.shift,
            'shift_sp': self.shift_sp,
            'shift_ss': self.shift_ss,
            'dt': self.dt_effective,
            'excitation_period': self.excitation_period,
            'g_factor': self.g_factor,
            'l1': self.l1,
            'l2': self.l2,
            'irf': self.irf,
            'bg': self.bg,
            'initial_x0': np.array(x0),
            'fixed_flags': fixed.astype(int),
            'p2s_twoIstar': self.p2s_twoIstar,
            'BIFL_scatter': self.BIFL_scatter,
            'min_photons': self.min_photons
        }
        return d

    def _apply_ui_state(self, state):
        """Push a saved state back into the widgets."""
        # — micro-time controls —
        self.micro_time_range = (state['micro_time_start'], state['micro_time_stop'])

        # Update micro_time_binning in DetectorWizardPage instead of spinBox
        micro_time_binning = state['micro_time_binning']
        self.channel_definer.micro_binning_combo.setCurrentText(str(micro_time_binning))

        # — IRF threshold & shifts —
        self.irf_threshold_vv = state['irf_threshold_vv']
        self.irf_threshold_vh = state['irf_threshold_vh']
        self.shift = state['shift']
        self.shift_sp = state['shift_sp']
        self.shift_ss = state['shift_ss']

        self.min_photons = state['min_photons']
        self.p2s_twoIstar = state['p2s_twoIstar']
        self.BIFL_scatter = state['BIFL_scatter']

        # — “internal” fit parameters —
        # use the property setters so the UI stays in sync
        self.g_factor = state['g_factor']
        self.l1 = state['l1']
        self.l2 = state['l2']

        # — initial‐guess & fixed flags —
        x0 = state['initial_x0']
        fixed = state['fixed_flags']
        self.tau = x0[0]
        self.gamma = x0[1]
        self.r0 = x0[2]
        self.rho = x0[3]

        self.fix_tau = bool(fixed[0])
        self.fix_gamma = bool(fixed[1])
        self.fix_r0 = bool(fixed[2])
        self.fix_rho = bool(fixed[3])

        # — restore cached IRF/BG arrays so your `.irf` & `.bg` props pick them up —
        det = self.current_detector
        self.irf_np[det] = np.array(state['irf'])
        self.bg_np [det] = np.array(state['bg'])

    def _on_channel_changed(self, new_detector):
        old = getattr(self, '_last_detector', None)
        if old is not None:
            # save old‐channel UI state
            self.channel_settings[old] = self._capture_current_ui_state()

        # pull in the channel‐definer info for the new detector
        info = self.channel_definer.detectors.get(new_detector, {})

        # set the spinboxes to any channel‐specific micro‐time defaults
        ranges = info.get('micro_time_ranges', [])
        if ranges:
            raw_start, raw_stop = ranges[0]
            bin_start = raw_start // self.micro_time_binning
            bin_stop = raw_stop // self.micro_time_binning
            # block signals so we don't trigger micro-time‐range callbacks
            widgets = (self.spinBox_micro_time_start, self.spinBox_micro_time_stop)
            self.block_widget_signals(widgets)
            self.micro_time_range = (bin_start, bin_stop)
            self.unblock_widget_signals(widgets)

        # ensure per-detector dict exists and complete to avoid KeyError later when saving arrays
        self._ensure_channel_state(new_detector)

        # restore any previously‐saved UI state for this detector, but only if it's complete
        state = self.channel_settings.get(new_detector)
        required_keys = (
            'micro_time_start', 'micro_time_stop', 'initial_x0', 'fixed_flags',
            'g_factor', 'l1', 'l2', 'irf', 'bg'
        )
        if isinstance(state, dict) and all(k in state for k in required_keys):
            widgets = (
                self.spinBox_micro_time_start,
                self.spinBox_micro_time_stop,
                self.doubleSpinBox_irf_threshold_vv,
                self.doubleSpinBox_irf_threshold_vh,
                self.doubleSpinBox_shift,
                self.doubleSpinBox_shift_sp,
                self.doubleSpinBox_shift_ss,
            )
            self.block_widget_signals(widgets)
            self._apply_ui_state(state)
            self.unblock_widget_signals(widgets)

        # Check if we have saved MLE settings for this detector in the current setup
        try:
            # Get the current setup name
            setup_name = self.channel_definer.setup_combo.currentText()
            if setup_name:
                # Get the detector_setups.json file path
                setups_file = self.channel_definer.current_setups_file

                # Load existing setups
                setups = load_detector_setups(setups_file)

                # Check if the setup exists and has the detector with MLE settings
                if (setup_name in setups.get("setups", {}) and
                    "detectors" in setups["setups"][setup_name] and
                    new_detector in setups["setups"][setup_name]["detectors"] and
                    "mle_settings" in setups["setups"][setup_name]["detectors"][new_detector]):

                    # Get the MLE settings for the detector
                    detector_params = setups["setups"][setup_name]["detectors"][new_detector]["mle_settings"]

                    # Block signals to prevent multiple updates
                    widgets = (
                        self.spinBox_micro_time_start,
                        self.spinBox_micro_time_stop,
                        self.doubleSpinBox_irf_threshold_vv,
                        self.doubleSpinBox_irf_threshold_vh,
                        self.doubleSpinBox_shift,
                        self.doubleSpinBox_shift_sp,
                        self.doubleSpinBox_shift_ss,
                    )
                    self.block_widget_signals(widgets)

                    # Update the UI with the loaded parameters (require new per-channel keys)
                    if "micro_time_start" in detector_params and "micro_time_stop" in detector_params:
                        self.micro_time_range = [detector_params["micro_time_start"], detector_params["micro_time_stop"]]
                    if "irf_threshold_vv" in detector_params:
                        self.irf_threshold_vv = detector_params["irf_threshold_vv"]
                    if "irf_threshold_vh" in detector_params:
                        self.irf_threshold_vh = detector_params["irf_threshold_vh"]
                    if "shift" in detector_params:
                        self.shift = detector_params["shift"]
                    if "shift_sp" in detector_params:
                        self.shift_sp = detector_params["shift_sp"]
                    if "shift_ss" in detector_params:
                        self.shift_ss = detector_params["shift_ss"]
                    if "min_photons" in detector_params:
                        self.min_photons = detector_params["min_photons"]

                    # IRF start/stop window
                    if "irf_start" in detector_params:
                        self.irf_start = detector_params["irf_start"]
                    if "irf_stop" in detector_params:
                        self.irf_stop = detector_params["irf_stop"]

                    # Update checkbox states
                    if "p2s_twoIstar" in detector_params:
                        self.p2s_twoIstar = detector_params["p2s_twoIstar"]
                    if "BIFL_scatter" in detector_params:
                        self.BIFL_scatter = detector_params["BIFL_scatter"]
                    if "fix_tau" in detector_params:
                        self.fix_tau = detector_params["fix_tau"]
                    if "fix_gamma" in detector_params:
                        self.fix_gamma = detector_params["fix_gamma"]
                    if "fix_r0" in detector_params:
                        self.fix_r0 = detector_params["fix_r0"]
                    if "fix_rho" in detector_params:
                        self.fix_rho = detector_params["fix_rho"]

                    # Unblock signals
                    self.unblock_widget_signals(widgets)
        except Exception as e:
            # Silently ignore errors when loading MLE settings
            pass

        # remember where we are now
        self._last_detector = new_detector

        # this covers everything update_selected_window used to do:
        self.update_irf_files()
        self.update_bg_files()
        self.update_decay_of_detector()
        self.update_scatter_count_rate_ui()
        self._fit = None
        self.update_fit()

    def block_widget_signals(self, widgets):
        """
        Block signals for a collection of widgets to prevent multiple updates.

        Parameters
        ----------
        widgets : tuple or list
            Collection of widgets whose signals should be blocked.
        """
        for w in widgets:
            w.blockSignals(True)

    def unblock_widget_signals(self, widgets):
        """
        Unblock signals for a collection of widgets after updates are complete.

        Parameters
        ----------
        widgets : tuple or list
            Collection of widgets whose signals should be unblocked.
        """
        for w in widgets:
            w.blockSignals(False)


    def _switch_filewidget(self, widgets_dict: dict, active: str):
        """
        Show only the FileListWidget corresponding to the active detector.
        """
        for det, fw in widgets_dict.items():
            fw.setVisible(det == active)

    def _set_irf_bg_widgets_enabled(self, enabled: bool):
        """
        Enable or disable file drops for all IRF and BG file widgets.

        Parameters
        ----------
        enabled : bool
            Whether to enable (True) or disable (False) file drops.
        """
        # Enable/disable all IRF file widgets
        for fw in self.irf_file_widgets.values():
            fw.setAcceptDrops(enabled)
            if enabled:
                fw.setToolTip("Drop IRF files here")
            else:
                fw.setToolTip("Drop burst files first before dropping IRF files")

        # Enable/disable all BG file widgets
        for fw in self.bg_file_widgets.values():
            fw.setAcceptDrops(enabled)
            if enabled:
                fw.setToolTip("Drop background files here")
            else:
                fw.setToolTip("Drop burst files first before dropping background files")

    def _prepare_irf_bg_widgets(self):
        """
        Initialize IRF and background file selection widgets, populate selectors,
        and synchronize widget visibility across detectors.

        Note: IRF and BG widgets are initially disabled and will be enabled
        only after burst files are loaded.
        """
        dets = list(self.channel_definer.detectors.keys())

        # Create IRF and background FileListWidgets for each detector
        for det in dets:
            irf_fw = FileListWidget(parent=self, file_added_callback=self.update_irf_files)
            irf_fw.hide()
            irf_fw.setAcceptDrops(False)  # Initially disable file drops
            self.verticalLayout_irf_files.addWidget(irf_fw)
            self.irf_file_widgets[det] = irf_fw

            bg_fw = FileListWidget(parent=self, file_added_callback=self.update_bg_files)
            bg_fw.hide()
            bg_fw.setAcceptDrops(False)  # Initially disable file drops
            self.verticalLayout_bg_files.addWidget(bg_fw)
            self.bg_file_widgets[det] = bg_fw

        # Populate combo boxes and connect signals to switch visible widget
        for combo, widgets_dict in (
                (self.comboBox_irf_select, self.irf_file_widgets),
                (self.comboBox_background_select, self.bg_file_widgets),
        ):
            combo.clear()
            combo.addItems(dets)
            combo.setCurrentIndex(0)
            combo.currentTextChanged.connect(
                lambda name, wd=widgets_dict: self._switch_filewidget(wd, name)
            )

        # Ensure the correct widgets are shown once the UI is laid out
        QtCore.QTimer.singleShot(0, lambda: (
            self._switch_filewidget(self.irf_file_widgets, self.comboBox_irf_select.currentText()),
            self._switch_filewidget(self.bg_file_widgets, self.comboBox_background_select.currentText())
        ))

    def _update_hist_files(
        self,
        widgets_dict: dict,
        np_dict: dict,
        one_for_all: bool,
        normalize: int,
        threshold: typing.Union[float, typing.Tuple[float, float]] = -1,
        state_key: str = None,  # should be 'irf' or 'bg' when called
        detector: Union[str, list[str]] = None
    ):
        """
        Populate np_dict[det] with the summed histogram for the current detector,
        and—if state_key is given—save the resulting array into
        self.channel_settings[det][state_key].
        """
        # figure out which detectors to update
        if one_for_all:
            dets = list(widgets_dict.keys())
        else:
            if detector is None:
                dets = [self.current_detector]
            else:
                dets = detector if isinstance(detector, (list, tuple)) else [detector]

        for det in dets:
            fw = widgets_dict.get(det)
            files: typing.List[Path] = fw.get_selected_files() if fw else []

            # register every file so that LazyTTTRDict knows where to find it,
            # then grab the TTTR object (loading on first access).
            tttr_inputs = list()
            for fp in files:
                # Extract just the filename part (without folder information) before getting the stem
                key = Path(fp.name).stem
                self._tttr_paths[key] = fp  # tell the lazy dict where the file lives
                tttr = self.tttrs.get(key)  # loads/caches on first use
                if tttr is not None:
                    tttr_inputs.append(tttr)

            # keep other file widgets in sync if one-for-all is checked
            if one_for_all and det == self.current_detector:
                for other_fw in widgets_dict.values():
                    if other_fw is not fw:
                        other_fw.blockSignals(True)
                        other_fw.clear()
                        for fp in files:
                            other_fw.add_file(str(fp))
                        other_fw.blockSignals(False)

            if not files:
                np_dict.pop(det, None)
                self.combined_plot.clear()
                return

            det_chs = self.channel_definer.detectors[det]["chs"]

            if tttr_inputs:
                jordi = self.make_jordi(
                    tttr_list=tttr_inputs,
                    detector_chs=det_chs,
                    micro_time_range=self.full_range,
                    micro_time_binning=self.micro_time_binning,
                    save_files=self.save_jordis,
                    normalize_counts=normalize,
                    threshold=threshold,
                    apply_vh_shift=False if state_key in ('irf', 'bg') else True
                )
                arr = np.sum(jordi, axis=0)
                np_dict[det] = arr

                if state_key is not None:
                    # Ensure nested dict exists before assignment to avoid KeyError
                    self.channel_settings.setdefault(det, {})
                    self.channel_settings[det][state_key] = arr

        # redraw decay without fitting
        self.update_decay_of_detector()

    def inspect_bursts(self, idx: int, embed: bool = False):
        if self.df_bursts is None or not self.tttrs:
            chisurf.logging.info("No burst data loaded.")
            return

        row = self.df_bursts.iloc[idx]
        key = Path(row['First File']).stem
        tttr = self.tttrs.get(key)
        if tttr is None:
            chisurf.logging.info(f"TTTR with key {key} not found.")
            return
        burst = tttr[int(row['First Photon']):int(row['Last Photon'])]

        # clear any existing plots
        self.burst_layout.clear()

        dets = list(self.channel_definer.detectors.keys())
        for det in dets:
            # create a new subplot
            p = self.burst_layout.addPlot(title=f"Detector: {det}")
            p.setLabel('bottom', 'Micro‐time channel')
            p.setLabel('left', 'Counts')
            p.setLogMode(x=False, y=True)
            p.setYRange(-1, 2)
            p.showGrid(x=True, y=True)

            info = self.channel_definer.detectors[det]
            chs = info['chs']
            pchs = chs[::2]
            schs = chs[1::2] if len(chs) > 1 else chs

            sb, eb = self.micro_time_range
            tp = self.filter_tttr(burst, self.micro_time_range, pchs)
            ts = self.filter_tttr(burst, self.micro_time_range, schs)
            cp = tp.get_microtime_histogram(self.micro_time_binning)[0].astype(np.float64, copy=False)
            cs = ts.get_microtime_histogram(self.micro_time_binning)[0].astype(np.float64, copy=False)
            # zero outside window for visualization
            if sb > 0:
                cp[:sb] = 0
                cs[:sb] = 0
            if eb < cp.size:
                cp[eb:] = 0
                cs[eb:] = 0
            data = np.hstack([cp, cs])

            # plot it
            p.plot(data, pen=None, symbol='o', symbolSize=4)

            # move to next row in the grid
            self.burst_layout.nextRow()

    @chisurf.gui.decorators.init_with_ui(
        "burst/burst_mle_analysis/wizard.ui",
        path=chisurf.settings.plugin_path
    )
    def __init__(self, *args, **kwargs):
        # Core attributes
        self.df_bursts = None
        self._fit = None
        self.stop_processing = False

        self._tttr_paths: Dict[str, Path] = {}
        self.tttrs = LazyTTTRDict(self._tttr_paths, lambda: self.tttr_file_type)

        self.irf_np = {}
        self.bg_np = {}
        self.channel_settings = {}
        self.irf_file_widgets = {}
        self.bg_file_widgets = {}

        self.channel_definer = None
        self.decay_of_current_file = None
        self.current_file_idx = 0

        # File lists
        self.burst_files_list = FileListWidget(
            parent=self,
            file_added_callback=self.load_burst_data,
            process_on_drop=True
        )
        self.verticalLayout_burst_files.addWidget(self.burst_files_list)

        # Detector definition tab - moved to first page
        self.tab_detector = QtWidgets.QWidget()
        self.tabWidget.insertTab(0, self.tab_detector, "Detector Definition")
        self.tabWidget.setCurrentIndex(0)  # Start on Detector Wizard page
        self.verticalLayout_detector_tab = QtWidgets.QVBoxLayout(self.tab_detector)
        self.channel_definer = chisurf.gui.widgets.wizard.DetectorWizardPage(parent=self)
        self.groupBox_detector = QtWidgets.QGroupBox("Detector Configuration")
        self.verticalLayout_detector = QtWidgets.QVBoxLayout(self.groupBox_detector)
        self.verticalLayout_detector.addWidget(self.channel_definer)
        self.verticalLayout_detector_tab.addWidget(self.groupBox_detector)
        # ← as soon as the user finishes defining detectors, re-build all our channel UIs

        # Prepare irf, bg widgets
        self._prepare_irf_bg_widgets()

        # Ensure IRF and BG file widgets are disabled by default
        self._set_irf_bg_widgets_enabled(False)

        # 1) draw the plots
        self.setup_plots()

        # 2) set *all* of the spin-boxes to their default values
        self.initialize_ui_values()
        self._init_channels_from_wizard()

        self.connect_signals()

    def setup_plots(self):
        # Decay plot all bursts
        self.groupBox_combined_plot = QtWidgets.QGroupBox("All selections")
        self.verticalLayout_combined_plot = QtWidgets.QVBoxLayout(self.groupBox_combined_plot)
        self.verticalLayout_plots.addWidget(self.groupBox_combined_plot)

        # Plot for IRF, Model, and Data
        self.combined_plot = pg.PlotWidget()

        # Weighted‐residuals plot
        self.residual_plot = pg.PlotWidget()
        # insert it *above* the decay plot, give it stretch=1 (residual)
        self.verticalLayout_combined_plot.insertWidget(
            0, self.residual_plot, 1
        )
        self.residual_plot.setLabel('left', 'Weighted residuals')
        # link the x‐axes so they pan/zoom together
        self.residual_plot.setXLink(self.combined_plot)
        # optional: show grid
        self.residual_plot.showGrid(x=True, y=True)

        # Data plot, give it stretch=3 (combined)
        self.verticalLayout_combined_plot.addWidget(self.combined_plot, 3)
        self.combined_plot.setLabel('bottom', 'Time (ch.)')
        self.combined_plot.setLabel('left', 'Intensity')
        self.combined_plot.setLogMode(y=True)
        self.combined_plot.setYRange(-1, 5)

    def update_variable_fit_parameters(self):
        chisurf.logging.info("update initial parameters. BLANK")
        self.update_fit()

    def update_internal_fit_parameters(self):
        chisurf.logging.info("update internal fit parameters")
        # Set fit to none to force recreation of fit
        self._fit = None
        self.update_fit()

    def on_irf_parameters_changed(self, _=None):
        """
        Called whenever any IRF parameter (threshold,
        'one‐for‐all' toggle, detector selector, or your
        shift / shift_sp / shift_ss) changes.
        """
        # 2) recompute IRFs from the files (this will use your new shift/shift_sp/shift_ss
        self.update_irf_files()

        # 3) toss out the old Fit23 so we’ll build a fresh one with the new IRF
        self._fit = None

        # 4) refresh the decay and re‐run the fit
        self.update_decay_of_detector()
        self.update_fit()

    def connect_signals(self):
        self.channel_definer.detectorsChanged.connect(self._init_channels_from_wizard)

        # hook up save/load
        self.toolButton_save_fit.clicked.connect(self.save_fit)

        # --- Clear buttons ---
        clear_buttons = {
            self.toolButton_clear_burst: self.burst_files_list,
            self.toolButton_clear_irf: self.irf_file_widgets,
            self.toolButton_clear_bg: self.bg_file_widgets,
        }
        for button, widget_group in clear_buttons.items():
            button.clicked.connect(lambda _, wg=widget_group: self.clear_files(wg))

        # --- Burst processing & navigation ---
        self.pushButton_process_bursts.clicked.connect(self.process_bursts)
        # Stop button functionality is deprecated in favor of modal progress dialog cancel
        try:
            self.pushButton_stop.hide()
            self.pushButton_stop.setEnabled(False)
        except Exception:
            pass
        self.comboBox_window.currentTextChanged.connect(self._on_channel_changed)
        self.spinBox_current_file_idx.valueChanged.connect(self.update_current_file)

        # --- Fit‐parameter controls (internal) ---

        # Connect detector change to update_internal_fit_parameters since g_factor, l1, and l2 are now detector-specific
        self.comboBox_window.currentTextChanged.connect(self.update_internal_fit_parameters)

        # --- Micro‐time range → update decay + fit ---
        self.channel_definer.micro_binning_combo.currentTextChanged.connect(self._update_max_bins_from_tttr)
        self.channel_definer.micro_binning_combo.currentTextChanged.connect(self.on_micro_time_range_changed)
        self.spinBox_micro_time_start.valueChanged.connect(
            lambda val: self.spinBox_micro_time_stop.setMinimum(val + 1)
        )
        for sb in (self.spinBox_micro_time_start, self.spinBox_micro_time_stop):
            sb.valueChanged.connect(self.on_micro_time_range_changed)

        # --- Fit‐parameter controls (variable) ---
        variable_spins = (
            self.doubleSpinBox_tau,
            self.doubleSpinBox_gamma,
            self.doubleSpinBox_r0,
            self.doubleSpinBox_rho,
        )
        for spin in variable_spins:
            spin.valueChanged.connect(self.update_variable_fit_parameters)

        variable_checks = (
            self.checkBox_fix_tau,
            self.checkBox_fix_gamma,
            self.checkBox_fix_r0,
            self.checkBox_fix_rho,
            self.checkBox_2IStar,
            self.checkBox_BIFL_scatter
        )
        for chk in variable_checks:
            chk.stateChanged.connect(self.update_variable_fit_parameters)

        # --- Other parameter updates ---
        self.spinBox_min_photons.valueChanged.connect(self.update_parameters)
        # persist min_photons per detector into channel_settings on change
        self.spinBox_min_photons.valueChanged.connect(lambda val: self._save_min_photons_for_current_detector(val))

        # --- IRF parameter controls ---
        irf_controls = [
            self.doubleSpinBox_irf_threshold_vv,
            self.doubleSpinBox_irf_threshold_vh,
            self.checkBox_irf_one_for_all,
            self.comboBox_irf_select,
            self.doubleSpinBox_shift,
            self.doubleSpinBox_shift_sp,
            self.doubleSpinBox_shift_ss,
            self.spinBox_irf_start,
            self.spinBox_irf_stop,
        ]
        for ctrl in irf_controls:
            # use currentTextChanged or stateChanged automatically based on widget type
            signal = (getattr(ctrl, 'valueChanged', None) or
                      getattr(ctrl, 'stateChanged', None) or
                      getattr(ctrl, 'currentTextChanged'))
            signal.connect(self.on_irf_parameters_changed)

        # whenever the user finishes (or re-configures) the DetectorWizardPage,
        # rebuild all the IRF/BG lists and window combobox
        self.channel_definer.detectorsChanged.connect(self._init_channels_from_wizard)

        # --- UI actions ---
        self.tabWidget.currentChanged.connect(self._on_tab_changed)
        # Start hyperparameter optimization
        try:
            self.toolButton_hyper_opt.clicked.connect(
                lambda: self.optimize_hyperparameters(n_iter=int(self.spinBox_n_h_opt.value()))
            )
        except Exception:
            pass

    def initialize_ui_values(self):
        # No need to initialize comboBox_tttr_file_type as we're using channel_definer.filetype instead
        self.micro_time_range = (0, 4096)

        self.tau = 4.0
        self.gamma = 0.0
        self.r0 = 0.38
        self.rho = 0.25
        self.fix_tau = False
        self.fix_gamma = False
        self.fix_r0 = True
        self.fix_rho = False
        self.min_photons = 10
        self.irf_threshold_vv = 0.02
        self.irf_threshold_vh = 0.02

    def browse_files(self, list_widget):
        dialog = QtWidgets.QFileDialog(self, "Select Files")
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if dialog.exec_():
            for file in dialog.selectedFiles():
                list_widget.add_file(file)

    def clear_files(self, list_widget):
        if isinstance(list_widget, dict):
            for key, value in list_widget.items():
                value.clear()
        if list_widget == self.burst_files_list:
            self.df_bursts = None
            self.tttrs.clear()  # Properly clear the LazyTTTRDict
            self._tttr_paths.clear()  # Clear the paths dictionary
            self.burst_files_list.clear()
            # No need to reset comboBox_tttr_file_type as we're using channel_definer.filetype instead
        elif list_widget == self.irf_file_widgets:
            self.irf_np.clear()
        else:
            self.bg_np.clear()
        self.combined_plot.clear()

    def update_burst_files(self):
        files = self.burst_files_list.get_selected_files()
        self.current_file_idx = 0
        max_idx = max(0, len(files) - 1)
        self.spinBox_current_file_idx.setMaximum(max_idx)
        if files:
            self.current_filename = str(files[0])
            self.lineEdit_current_filename.setText(self.current_filename)

    def update_irf_files(self, detector=None):
        self._update_hist_files(
            widgets_dict=self.irf_file_widgets,
            np_dict=self.irf_np,
            one_for_all=self.one_for_all_irf,
            normalize=2,
            threshold=-1,
            state_key='irf',
            detector=detector
        )

    def update_bg_files(self, detector=None):
        self._update_hist_files(
            widgets_dict=self.bg_file_widgets,
            np_dict=self.bg_np,
            one_for_all=self.one_for_all_bg,
            normalize=3,
            state_key='bg',
            detector=detector
        )
        # refresh the spinbox whenever bg changes
        self.update_scatter_count_rate_ui()

    def update_current_file(self, index):
        files = self.burst_files_list.get_selected_files()
        if not files or not (0 <= index < len(files)):
            return
        self.current_file_idx = index
        self.current_filename = str(files[index])
        self.update_decay_of_detector()
        self.update_fit()

    def update_scatter_count_rate_ui(self):
        """
        Push the current scatter_count_rate into the spinbox.
        """
        self.doubleSpinBox_scatter_Countrate.setValue(self.scatter_count_rate)

    def load_burst_data(self):
        files = self.burst_files_list.get_selected_files()
        if not files:
            self.df_bursts = None
            self.tttrs.clear()  # Properly clear the LazyTTTRDict
            self._tttr_paths.clear()  # Clear the paths dictionary

            # Disable IRF and BG file drops when all burst files are removed
            self._set_irf_bg_widgets_enabled(False)
        else:
            paris = files[0].parent
            if self.df_bursts is None:
                self.df_bursts = pd.DataFrame()
            df, tttrs = self.read_burst_analysis(paris.parent)
            self.tttrs = tttrs
            self.df_bursts = pd.concat([self.df_bursts, df], ignore_index=True, sort=False)

            # Enable IRF and BG file drops after burst files are loaded
            self._set_irf_bg_widgets_enabled(True)

            # Update excitation period from TTTR header if available
            if self.tttrs:
                # Get the first TTTR in our dict
                tttr = next(iter(self.tttrs.values()))
                try:
                    # Use macro_time_resolution directly (in seconds)
                    # Convert to repetition rate in MHz (1/seconds * 1e-6)
                    repetition_rate = 1.0 / tttr.header.macro_time_resolution * 1e-6

                    if repetition_rate > 0:
                        # Convert repetition rate (MHz) to excitation period (ns)
                        excitation_period = 1000.0 / repetition_rate
                        chisurf.logging.info(f"Updated excitation period to {excitation_period} ns based on repetition rate {repetition_rate} MHz")
                except (AttributeError, ValueError) as e:
                    chisurf.logging.info(f"Could not extract repetition rate from header: {e}")

        self._update_max_bins_from_tttr()

    def _save_min_photons_for_current_detector(self, val: int):
        """Persist min_photons in channel_settings for the current detector."""
        try:
            det = self.current_detector
            st = self.channel_settings.get(det, {})
            st['min_photons'] = int(val)
            self.channel_settings[det] = st
        except Exception:
            pass

    def on_micro_time_range_changed(self, _=None):
        # When micro-time window or binning changes: recompute decays and update plots/fit
        self.update_decay_of_detector()
        self._fit = None
        self.update_fit()

        # Additionally refresh per-burst histogram plots
        if self.df_bursts is not None and self.tttrs:
            try:
                self.inspect_bursts(self.burst_idx, embed=True)
            except Exception:
                # keep UI responsive even if burst plot refresh fails
                pass

        # — now *save* the new micro-time state for this detector —
        det = self.current_detector
        st = self.channel_settings.get(det, {})
        st['micro_time_start'], st['micro_time_stop'] = self.micro_time_range
        st['micro_time_binning'] = self.micro_time_binning
        self.channel_settings[det] = st

    def get_current_jordis(self):
        if self.df_bursts is None:
            chisurf.logging.info("No burst DataFrame loaded.")
            return

        # gather detector channels and microtime settings
        detector_info = getattr(self.channel_definer, 'detectors', {}).get(self.current_detector, {})
        chs = detector_info.get('chs', [])
        if not chs:
            chisurf.logging.info('Channels not found')
            return

        mt_bin = self.micro_time_binning

        # 1) Which .bur file is selected in the UI?
        curr_bur = Path(self.current_filename).name

        # 2) Check if 'burst_file' column exists in the DataFrame
        if 'burst_file' not in self.df_bursts.columns:
            chisurf.logging.info(f"'burst_file' column not found in DataFrame. Available columns: {list(self.df_bursts.columns)}")
            # Try to use the first file if burst_file column doesn't exist
            if len(self.df_bursts) > 0:
                df_this = self.df_bursts
                chisurf.logging.info(f"Using all rows in DataFrame as fallback")
            else:
                chisurf.logging.info("DataFrame is empty")
                return
        else:
            # Filter df_bursts to just its rows
            df_this = self.df_bursts[self.df_bursts["burst_file"] == curr_bur]
            if df_this.empty:
                chisurf.logging.info(f"No bursts found for {curr_bur!r}")
                return

        # 3) Now grab the TTTR filename from the first row of that subset
        tttr_name = df_this.loc[df_this.index[0], "First File"]

        # 4) Load or retrieve the TTTR
        key = Path(tttr_name).stem
        chisurf.logging.debug(f"Looking for TTTR with key: {key}")
        tttr = self.tttrs.get(key)
        if tttr is None:
            chisurf.logging.info(f"TTTR with key {key} not found.")
            return

        # get the list of photon‐indices for *all* bursts in this file
        indices = self.get_burst_indices_for_current_file()
        if not indices:
            chisurf.logging.info('No indices found')
            return
        indices = np.array(indices)

        # slice the TTTR down to just burst photons
        burst_tttr = tttr[indices]

        # build the decay histogram over every burst in the file
        jordis = self.make_jordi(
            [burst_tttr],
            chs,
            self.micro_time_range,
            mt_bin,
            normalize_counts=-1
        )

        return jordis

    def pass_photon_threshold(self, data, gui: bool = False):
        # check photon threshold
        s = np.sum(data)
        r = s >= self.min_photons
        if not r and gui:
            QtWidgets.QMessageBox.warning(self, f"Not Enough Photons: {int(s)} < {self.min_photons}")
        return r

    def update_decay_of_detector(self):
        jordis = self.get_current_jordis()

        if jordis is None:
            return
        data = np.sum(jordis, axis=0)
        self.pass_photon_threshold(data)
        self.decay_of_current_file = data

    def update_fit_ui(self, res: dict):
        # Use property setters to avoid direct widget access and keep side-effects consistent
        try:
            x = res.get('x', res['x'])
        except Exception:
            x = res['x']
        self.tau_result = float(x[0])
        self.gamma_result = float(x[1])
        self.r0_result = float(x[2])
        self.rho_result = float(x[3])
        # twoIstar may be absent depending on fit; guard accordingly
        if 'twoIstar' in res:
            self.twoIstar_result = float(res['twoIstar'])
        if len(x) > 6:
            self.r_scatter_result = float(x[6])
        if len(x) > 7:
            self.r_exp_result = float(x[7])

    def update_window_combobox(self):
        dets = list(self.channel_definer.detectors.keys())

        self.comboBox_window.clear()
        self.comboBox_window.addItems(dets)
        self.comboBox_window.setCurrentIndex(0)
        self.update_selected_window()

        for cb in (self.comboBox_irf_select, self.comboBox_background_select):
            cb.clear()
            cb.addItems(dets)
            cb.setCurrentIndex(0)

        self._switch_filewidget(self.irf_file_widgets, dets[0])
        self._switch_filewidget(self.bg_file_widgets, dets[0])

    def update_selected_window(self, detectors=None):
        if detectors is None:
            detectors = [self.current_detector]
        elif detectors == "all":
            detectors = self.channel_definer.detectors.keys()

        for detector in detectors:
            info = getattr(self.channel_definer, 'detectors', {}).get(detector, {})
            chs = info.get('chs', [])
            if chs:
                ranges = info.get('micro_time_ranges', [])
                if ranges:
                    raw_start, raw_stop = ranges[0]
                    bin_start = raw_start // self.micro_time_binning
                    bin_stop = raw_stop // self.micro_time_binning
                    self.micro_time_range = (bin_start, bin_stop)
                    self.channel_settings[detector] = self._capture_current_ui_state()

    def stop_burst_processing(self):
        """
        Stop the burst processing when the stop button is clicked.
        """
        self.stop_processing = True
        chisurf.logging.info("Stop button clicked, stopping burst processing")
        
    def update_parameters(self):
        self._fit = None
        self.update_fit()

    def _find_detector_row(self, detector_name):
        """
        Helper method to find the row index of a detector in the detectors_form table.

        Args:
            detector_name (str): The name of the detector to find.

        Returns:
            int: The row index of the detector, or -1 if not found.
        """
        for row in range(self.channel_definer.detectors_form.rowCount()):
            if self.channel_definer.detectors_form.item(row, 0).text() == detector_name:
                return row
        return -1

    def clear_fit(self):
        self._fit = None

    def create_fit_instance(self):
        sb, eb = self.micro_time_range
        # basic params
        dt = self.dt_effective
        period = self.excitation_period
        gf = self.g_factor
        l1 = self.l1
        l2 = self.l2
        irf = self.irf
        bg = self.bg

        # Ensure arrays are float64 without modifying their shapes or window content
        irf = irf.astype(np.float64, copy=True)
        bg = bg.astype(np.float64, copy=True)

        # finally, build the fit
        fit = tttrlib.Fit23(
            dt=dt,
            irf=irf,
            background=bg,
            period=period,
            g_factor=gf,
            l1=l1,
            l2=l2,
            p2s_twoIstar_flag=self.p2s_twoIstar,
            soft_bifl_scatter_flag=self.BIFL_scatter
        )
        return fit

    def update_fit(self):
        x0, fixed = self.fit_parameters
        sb, eb = self.micro_time_range
        det = self.current_detector
        decay = self.decay_of_current_file

        if det not in self.irf_np or det not in self.bg_np:
            return
        if decay is None:
            return

        self._fit = None

        # use full-length decay (already zeroed outside window)
        d = decay.astype(np.float64, copy=False)
        res = self.fit(data=d, initial_values=x0, fixed=fixed)
        self.plot_fit_result(res)

    def _get_channel_ranges_bins(self):
        """Return per-channel (vv, vh) start/stop in histogram bins for current detector.
        Falls back to the global micro_time_range if specific ranges are not available.
        """
        # default fallback
        sb_def, eb_def = self.micro_time_range
        sb_vv = sb_def
        eb_vv = eb_def
        sb_vh = sb_def
        eb_vh = eb_def
        try:
            info = getattr(self.channel_definer, 'detectors', {}).get(self.current_detector, {})
            ranges = info.get('micro_time_ranges', None)
            if ranges and len(ranges) >= 2:
                raw_vv = ranges[0]
                raw_vh = ranges[1]
                # convert raw (in native bins) to our current binned indices
                binning = max(1, int(self.micro_time_binning))
                sb_vv = int(raw_vv[0] // binning)
                eb_vv = int(raw_vv[1] // binning)
                sb_vh = int(raw_vh[0] // binning)
                eb_vh = int(raw_vh[1] // binning)
        except Exception:
            pass
        return sb_vv, eb_vv, sb_vh, eb_vh

    def plot_fit_result(self, fit_result):
        chisurf.logging.info("plot fit result")
        # clear both panels
        self.combined_plot.clear()
        self.residual_plot.clear()
        sb, eb = self.micro_time_range

        # only plot if we actually loaded IRF *and* BG for this detector
        det = self.current_detector
        if det not in self.irf_np or det not in self.bg_np:
            return

        # plot data and model in the bottom panel, but only within channel-specific ranges
        data_full = np.asarray(self.fit.data)
        model_full = np.asarray(self.fit.model)
        n = len(data_full) // 2
        vv_sb, vv_eb, vh_sb, vh_eb = self._get_channel_ranges_bins()
        # clamp
        vv_sb = max(0, int(vv_sb)); vv_eb = min(n, int(vv_eb)) if vv_eb is not None else n
        vh_sb = max(0, int(vh_sb)); vh_eb = min(n, int(vh_eb)) if vh_eb is not None else n
        data_vv = data_full[0:n][vv_sb:vv_eb]
        data_vh = data_full[n:2*n][vh_sb:vh_eb]
        model_vv = model_full[0:n][vv_sb:vv_eb]
        model_vh = model_full[n:2*n][vh_sb:vh_eb]
        data_rng = np.hstack([data_vv, data_vh])
        model_rng = np.hstack([model_vv, model_vh])
        self.combined_plot.plot(data_rng,
                                pen=None,
                                symbol='o',
                                symbolSize=3)
        self.combined_plot.plot(model_rng, pen='g')

        # Plot IRF & BG within per-channel windows
        irf_full = self.irf.astype(np.float64, copy=True)
        bg_full = self.bg.astype(np.float64, copy=True)
        # apply background scaling by acquisition time
        bg_full *= self.total_burst_time_seconds

        n = len(irf_full) // 2
        vv_sb, vv_eb, vh_sb, vh_eb = self._get_channel_ranges_bins()
        vv_sb = max(0, int(vv_sb)); vv_eb = min(n, int(vv_eb)) if vv_eb is not None else n
        vh_sb = max(0, int(vh_sb)); vh_eb = min(n, int(vh_eb)) if vh_eb is not None else n

        irf_rng = np.hstack([irf_full[0:n][vv_sb:vv_eb], irf_full[n:2*n][vh_sb:vh_eb]])
        bg_rng = np.hstack([bg_full[0:n][vv_sb:vv_eb], bg_full[n:2*n][vh_sb:vh_eb]])

        # scale IRF to visible data amplitude for plotting
        m_irf = np.max(irf_rng) if irf_rng.size else 0.0
        m_dat = np.max(data_rng) if data_rng.size else 0.0
        if m_irf > 0 and m_dat > 0:
            irf_rng = irf_rng / m_irf * m_dat

        self.combined_plot.plot(irf_rng, pen='r', name='IRF')
        self.combined_plot.plot(bg_rng, pen='b', name='Background')

        # compute & plot weighted residuals
        data = np.asarray(self.fit.data, dtype=float)
        model = np.asarray(self.fit.model, dtype=float)

        resid = np.zeros_like(data, dtype=float)
        mask = data > 0
        resid[mask] = (data[mask] - model[mask]) / np.sqrt(data[mask])

        # draw residuals in the top panel only within per-channel ranges
        pen = pg.mkPen(color=(200, 20, 20), width=1)
        resid = np.asarray(resid)
        n = len(resid) // 2
        vv_sb, vv_eb, vh_sb, vh_eb = self._get_channel_ranges_bins()
        vv_sb = max(0, int(vv_sb)); vv_eb = min(n, int(vv_eb)) if vv_eb is not None else n
        vh_sb = max(0, int(vh_sb)); vh_eb = min(n, int(vh_eb)) if vh_eb is not None else n
        resid_rng = np.hstack([resid[0:n][vv_sb:vv_eb], resid[n:2*n][vh_sb:vh_eb]])

        self.residual_plot.plot(resid_rng,
                                pen=pen,
                                symbol='o',
                                symbolSize=3)

        self.update_fit_ui(fit_result)

    def _build_irf_bg_cache(self):
        irf_cache = {}
        bg_cache = {}
        for det in self.channel_definer.detectors.keys():
            st = self._ensure_channel_state(det)
            # Start from stored IRF/BG arrays (unshifted, unwindowed ideally)
            raw_irf = np.array(st.get('irf', []), dtype=np.float64, copy=True)
            raw_bg = np.array(st.get('bg', []), dtype=np.float64, copy=True)

            # Process IRF: 1) global VH shift; 2) sub-bin shifts; 3) IRF range; 4) thresholding
            if raw_irf.size > 0:
                half = raw_irf.size // 2
                sp = raw_irf[:half].astype(np.float64, copy=True)
                ss = raw_irf[half:].astype(np.float64, copy=True)
                # 1) global integer shift on VH
                if self.shift != 0:
                    ss = np.roll(ss, self.shift)
                # 2) individual sub-bin shifts
                sp = interpolate_shift(sp, self.shift_sp)
                ss = interpolate_shift(ss, self.shift_ss)
                # 3) IRF range windowing
                start = int(self.irf_start)
                stop = int(self.irf_stop)
                if start >= 0:
                    sp[:max(0, start)] = 0
                    ss[:max(0, start)] = 0
                if stop >= 0 and stop + 1 < sp.size:
                    sp[stop + 1:] = 0
                if stop >= 0 and stop + 1 < ss.size:
                    ss[stop + 1:] = 0
                # 4) thresholding per channel
                th_vv = float(self.irf_threshold_vv)
                th_vh = float(self.irf_threshold_vh)
                if th_vv > 0 and sp.size and sp.max() > 0:
                    sp[sp < th_vv * sp.max()] = 0
                if th_vh > 0 and ss.size and ss.max() > 0:
                    ss[ss < th_vh * ss.max()] = 0
                irf_cache[det] = np.hstack([sp, ss])
            else:
                irf_cache[det] = raw_irf

            # Process BG: apply only global VH shift (background is not thresholded/ranged here)
            if raw_bg.size > 0:
                half_bg = raw_bg.size // 2
                vv = raw_bg[:half_bg].astype(np.float64, copy=True)
                vh = raw_bg[half_bg:].astype(np.float64, copy=True)
                if self.shift != 0:
                    vh = np.roll(vh, self.shift)
                bg_cache[det] = np.hstack([vv, vh])
            else:
                bg_cache[det] = raw_bg
        return irf_cache, bg_cache

        if self.df_bursts is None or not self.tttrs:
            QtWidgets.QMessageBox.warning(self, "No Data", "No burst data loaded.")
            return

        # Reset stop flag
        self.stop_processing = False

        # Progress UI
        total_bursts = len(self.df_bursts)
        progress = QProgressDialog("Processing bursts...", "Cancel", 0, total_bursts, self)
        progress.setWindowTitle("Processing bursts")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setValue(0)
        progress.show()

        # ---- Per-detector static caches ----
        irf_cache, bg_cache = self._build_irf_bg_cache()
        settings_cache = {det: self._ensure_channel_state(det) for det in self.channel_definer.detectors.keys()}

        # Single global micro-time binning
        # (verify uniformity; if not, fall back to current UI value)
        det_mbs = {int(st['micro_time_binning']) for st in settings_cache.values()}
        if len(det_mbs) == 1:
            global_mb = int(next(iter(det_mbs)))
        else:
            chisurf.logging.warning(
                f"Detectors have differing micro_time_binning {det_mbs}; falling back to UI {self.micro_time_binning}"
            )
            global_mb = int(self.micro_time_binning)

        # Windows (sb, eb) per detector (in *binned* indices)
        window_cache = {}
        for det, st in settings_cache.items():
            sb = int(st['micro_time_start'])
            eb = int(st['micro_time_stop'])
            if eb <= sb:
                sb_fix, eb_fix = self.micro_time_range
                sb, eb = int(sb_fix), int(eb_fix)
            window_cache[det] = (sb, eb)

        # Per-detector routing-channel lists and LUTs
        channels_cache = {}
        rc_max_seen = 0
        for det, info in self.channel_definer.detectors.items():
            chs = info.get('chs', [])
            if len(chs) >= 2:
                pchs = chs[::2]
                schs = chs[1::2]
            else:
                pchs = chs
                schs = chs
            channels_cache[det] = (np.asarray(pchs, dtype=int), np.asarray(schs, dtype=int))
            if len(chs):
                rc_max_seen = max(rc_max_seen, int(np.max(chs)))

        # Pre-instantiate one fitter per detector (IRF/BG fixed for this run)
        fitters = {}
        for det, st in settings_cache.items():
            fitters[det] = tttrlib.Fit23(
                dt=st['dt'],
                irf=irf_cache[det],
                background=bg_cache[det],
                period=st['excitation_period'],
                g_factor=st['g_factor'],
                l1=st['l1'],
                l2=st['l2'],
                p2s_twoIstar_flag=st['p2s_twoIstar'],
                soft_bifl_scatter_flag=st['BIFL_scatter']
            )

        # Helper for default records
        metrics = ['2I* ', 'Tau', 'gamma', 'r0', 'rho', 'BIFL scatter?', '2I*: P+2S?', 'r Scatter', 'r Experimental']
        results = []

        def default_record(fname, det, cp_sum=0, cs_sum=0):
            color = det.lower()
            rec = {
                'First File': fname,
                'Detector': det,
                'Ng-p-all': int(cp_sum),
                'Ng-s-all': int(cs_sum),
                f'Number of Photons (fit window) ({color})': int(cp_sum + cs_sum)
            }
            for m in metrics:
                rec[f'{m} ({color})'] = float('nan')
            results.append(rec)

        # ---- Per-file cache: one binned micro-time array + routing LUTs ----
        file_cache = {}

        def maybe_pump_ui(k: int) -> None:
            if (k % 25) == 0:
                QtWidgets.QApplication.processEvents()

        for i, row in self.df_bursts.iterrows():
            if self.stop_processing or progress.wasCanceled():
                chisurf.logging.info("Burst processing stopped by user")
                break
            progress.setValue(i + 1)
            maybe_pump_ui(i + 1)

            fname = row['First File']
            first_ph = int(row['First Photon'])
            last_ph = int(row['Last Photon'])
            key = Path(fname).stem

            if first_ph < 0 or last_ph < 0:
                for det in self.channel_definer.detectors.keys():
                    default_record(fname, det, -1, -1)
                continue

            # Build file cache on first encounter
            if key not in file_cache:
                tttr = self.tttrs.get(key)
                if tttr is None:
                    for det in self.channel_definer.detectors.keys():
                        default_record(fname, det, -1, -1)
                    continue

                rc_full = np.asarray(tttr.routing_channels)
                mt_full = np.asarray(tttr.micro_times)

                # One global binned micro-time array for this file
                if global_mb <= 1:
                    mt_bins_full = mt_full.astype(np.int32, copy=True)
                else:
                    mt_bins_full = (mt_full // global_mb).astype(np.int32, copy=False)

                # LUTs per detector for channel membership
                rc_max = int(rc_full.max()) if rc_full.size else rc_max_seen
                det_luts = {}
                for det, (pchs, schs) in channels_cache.items():
                    is_p = np.zeros(rc_max + 1, dtype=bool)
                    is_s = np.zeros(rc_max + 1, dtype=bool)
                    if pchs.size:
                        is_p[pchs] = True
                    if schs.size:
                        is_s[schs] = True
                    det_luts[det] = (is_p, is_s)

                file_cache[key] = {
                    'rc_full': rc_full,
                    'mt_bins_full': mt_bins_full,
                    'det_luts': det_luts
                }

            fc = file_cache[key]
            rc_slice = fc['rc_full'][first_ph:last_ph]
            mt_bins = fc['mt_bins_full'][first_ph:last_ph]

            for det in self.channel_definer.detectors.keys():
                st = settings_cache[det]
                sb, eb = window_cache[det]
                irf_half_len = max(1, irf_cache[det].size // 2)

                # Fast masks from LUTs
                is_p_lut, is_s_lut = fc['det_luts'][det]
                m_p = is_p_lut[rc_slice]
                m_s = is_s_lut[rc_slice]

                # Histograms per half (no TTTR slicing)
                cp = np.bincount(mt_bins[m_p], minlength=irf_half_len).astype(np.float64, copy=False)
                cs = np.bincount(mt_bins[m_s], minlength=irf_half_len).astype(np.float64, copy=False)

                # Photon threshold
                cp_sum = float(cp.sum());
                cs_sum = float(cs.sum())
                if (cp_sum + cs_sum) < st['min_photons']:
                    default_record(fname, det, cp_sum, cs_sum)
                    continue

                # Integer VH shift then windowing
                if self.shift != 0:
                    cs = np.roll(cs, int(self.shift))
                if sb > 0:
                    cp[:sb] = 0.0;
                    cs[:sb] = 0.0
                if eb < cp.size:
                    cp[eb:] = 0.0;
                    cs[eb:] = 0.0

                # Assemble decay
                decay = np.empty(cp.size + cs.size, dtype=np.float64)
                decay[:cp.size] = cp
                decay[cp.size:] = cs

                # Fit using pre-made fitter
                fitter = fitters[det]
                res = fitter(data=decay, initial_values=st['initial_x0'], fixed=st['fixed_flags'])

                color = det.lower()
                results.append({
                    'First File': fname,
                    'Detector': det,
                    'Ng-p-all': int(cp_sum),
                    'Ng-s-all': int(cs_sum),
                    f'Number of Photons (fit window) ({color})': int(cp_sum + cs_sum),
                    f'2I*  ({color})': res.get('twoIstar', 0.0),
                    f'Tau ({color})': res['x'][0],
                    f'gamma ({color})': res['x'][1],
                    f'r0 ({color})': res['x'][2],
                    f'rho ({color})': res['x'][3],
                    f'BIFL scatter? ({color})': int(st['BIFL_scatter']),
                    f'2I*: P+2S? ({color})': int(st['p2s_twoIstar']),
                    f'r Scatter ({color})': res['x'][6] if len(res['x']) > 6 else float('nan'),
                    f'r Experimental ({color})': res['x'][7] if len(res['x']) > 7 else float('nan'),
                })

        try:
            progress.close()
        except Exception:
            pass

        result_df = pd.DataFrame(results)
        self._save_burst_results_fast(result_df)

    @staticmethod
    def _hist2_split(mt_bins: np.ndarray,
                     rc_slice: np.ndarray,
                     is_p_lut: np.ndarray,
                     is_s_lut: np.ndarray,
                     half_len: int):
        """
        One-pass P/S histogram:
          - classify photons as 0(P) / 1(S) via LUTs
          - bincount(mt*2 + cls, minlength=2*half_len)
          - deinterleave to cp/cs
        Returns uint32 arrays; cast to float only when filling the decay buffer.
        """
        cls = np.full(rc_slice.shape[0], -1, dtype=np.int8)
        pm = is_p_lut[rc_slice]
        sm = is_s_lut[rc_slice]
        cls[pm] = 0
        cls[sm] = 1
        valid = cls >= 0
        if not np.any(valid):
            return (np.zeros(half_len, dtype=np.uint32),
                    np.zeros(half_len, dtype=np.uint32))
        b = mt_bins[valid]
        c = cls[valid].astype(np.int32, copy=False)
        h2 = np.bincount(b * 2 + c, minlength=2 * half_len)
        return (h2[0::2].astype(np.uint32, copy=False),
                h2[1::2].astype(np.uint32, copy=False))

    def process_bursts(self):
        import os
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        from multiprocessing import shared_memory
        from chisurf.plugins.burst_mle_analysis._mp_worker import process_one_file_worker

        if self.df_bursts is None or not self.tttrs:
            QtWidgets.QMessageBox.warning(self, "No Data", "No burst data loaded.")
            return

        # UI
        self.stop_processing = False
        total_bursts = len(self.df_bursts)
        progress = QProgressDialog("Processing bursts...", "Cancel", 0, total_bursts, self)
        progress.setWindowTitle("Processing bursts")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setValue(0)
        progress.show()

        def ui_pump(k: int):
            if (k % 20) == 0:
                QtWidgets.QApplication.processEvents()

        # Per-detector constants
        irf_cache, bg_cache = self._build_irf_bg_cache()
        settings_cache = {det: self._ensure_channel_state(det) for det in self.channel_definer.detectors.keys()}
        det_order = list(self.channel_definer.detectors.keys())

        # uniform binning
        det_mbs = {int(st['micro_time_binning']) for st in settings_cache.values()}
        global_mb = int(next(iter(det_mbs))) if len(det_mbs) == 1 else int(self.micro_time_binning)

        # windows, channels, rc max
        window_cache = {}
        channels_cache = {}
        rc_max_seen = 0
        for det, info in self.channel_definer.detectors.items():
            st = settings_cache[det]
            sb = int(st['micro_time_start']);
            eb = int(st['micro_time_stop'])
            if eb <= sb:
                sb, eb = map(int, self.micro_time_range)
            window_cache[det] = (sb, eb)

            chs = info.get('chs', [])
            pchs = chs[::2] if len(chs) >= 2 else chs
            schs = chs[1::2] if len(chs) >= 2 else chs
            pchs = np.asarray(pchs, dtype=int);
            schs = np.asarray(schs, dtype=int)
            channels_cache[det] = (pchs, schs)
            if len(chs):
                rc_max_seen = max(rc_max_seen, int(np.max(chs)))

        # Build per-file jobs with shared memory
        jobs = []
        shm_blocks = []  # to unlink at end
        for fname, df_file in self.df_bursts.groupby('First File', sort=False):
            key = Path(fname).stem
            tttr = self.tttrs.get(key)

            if tttr is None:
                jobs.append((fname,
                             list(df_file[['First Photon', 'Last Photon']].itertuples(index=False, name=None)),
                             None, None, None, None, None, None,
                             det_order, {}, int(self.shift or 0)))
                continue

            rc_full = np.asarray(tttr.routing_channels)
            mt_full = np.asarray(tttr.micro_times)
            mt_bins_full = (mt_full // global_mb).astype(np.int32, copy=False) if global_mb > 1 else mt_full.astype(
                np.int32, copy=True)

            # compact dtypes to reduce bandwidth
            if rc_full.dtype != np.uint16 and int(rc_full.max(initial=0)) <= 65535:
                rc_full = rc_full.astype(np.uint16, copy=False)
            if mt_bins_full.dtype != np.uint16 and int(mt_bins_full.max(initial=0)) <= 65535:
                mt_bins_full = mt_bins_full.astype(np.uint16, copy=False)

            # Shared memory blocks (parent owns lifecycle)
            rc_shm = shared_memory.SharedMemory(create=True, size=rc_full.nbytes)
            np.ndarray(rc_full.shape, dtype=rc_full.dtype, buffer=rc_shm.buf)[:] = rc_full
            mt_shm = shared_memory.SharedMemory(create=True, size=mt_bins_full.nbytes)
            np.ndarray(mt_bins_full.shape, dtype=mt_bins_full.dtype, buffer=mt_shm.buf)[:] = mt_bins_full
            shm_blocks.extend([rc_shm, mt_shm])

            # Per-detector config (use class LUT: -1 ignore, 0=P, 1=S)
            rc_max = int(rc_full.max(initial=rc_max_seen)) if rc_full.size else rc_max_seen
            perdet_cfg = {}
            for det in det_order:
                st = settings_cache[det]
                pchs, schs = channels_cache[det]
                class_lut = np.full(rc_max + 1, -1, dtype=np.int8)
                if pchs.size: class_lut[pchs] = 0
                if schs.size: class_lut[schs] = 1
                half_len = max(1, irf_cache[det].size // 2)
                perdet_cfg[det] = {
                    'sb': int(window_cache[det][0]),
                    'eb': int(window_cache[det][1]),
                    'half_len': half_len,
                    'dt': float(st['dt']),
                    'period': float(st['excitation_period']),
                    'g_factor': float(st['g_factor']),
                    'l1': float(st['l1']), 'l2': float(st['l2']),
                    'p2s_twoIstar': bool(st['p2s_twoIstar']),
                    'BIFL_scatter': bool(st['BIFL_scatter']),
                    'min_photons': int(st['min_photons']),
                    'x0': np.asarray(st['initial_x0'], dtype=np.float64),
                    'fixed': np.asarray(st['fixed_flags'], dtype=np.int32),
                    'irf': np.asarray(irf_cache[det], dtype=np.float64),
                    'bg': np.asarray(bg_cache[det], dtype=np.float64),
                    'class_lut': class_lut,
                }

            bursts = list(df_file[['First Photon', 'Last Photon']].itertuples(index=False, name=None))
            jobs.append((fname, bursts,
                         rc_shm.name, rc_full.shape, str(rc_full.dtype),
                         mt_shm.name, mt_bins_full.shape, str(mt_bins_full.dtype),
                         det_order, perdet_cfg, int(self.shift or 0)))

        # Processes (leave one core for UI; cap by #files)
        ctx = mp.get_context('spawn')
        max_workers = max(1, min(os.cpu_count() or 8, len(jobs)) - 1)
        results = []
        processed = 0
        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                futs = [ex.submit(process_one_file_worker, j) for j in jobs]
                for fut in as_completed(futs):
                    try:
                        out, nbursts = fut.result()
                    except Exception as e:
                        chisurf.logging.error(f"Worker failed: {e}")
                        out, nbursts = [], 0
                    results.extend(out)
                    processed += nbursts
                    progress.setValue(min(processed, total_bursts))
                    ui_pump(processed)
        finally:
            try:
                progress.close()
            except Exception:
                pass
            # cleanup shared memory
            for block in shm_blocks:
                try:
                    block.close();
                    block.unlink()
                except Exception:
                    pass

        if self.stop_processing or (processed < total_bursts and progress.wasCanceled()):
            QtWidgets.QMessageBox.information(self, "Canceled", "Burst processing was canceled.")
            return

        result_df = pd.DataFrame(results)
        self._save_burst_results_fast(result_df)

    def make_jordi(
            self,
            tttr_list: typing.List[tttrlib.TTTR],
            detector_chs: typing.List[int],
            micro_time_range: typing.Tuple[int, int],
            micro_time_binning: int,
            save_files: bool = False,
            normalize_counts: int = 1,
            threshold: typing.Union[float, typing.Tuple[float, float]] = -1,
            minlength: int = -1,
            apply_vh_shift: bool = True
    ) -> typing.List[np.ndarray]:
        jordis = list()
        # Determine per-channel ranges: fall back to provided micro_time_range for both
        sb_def, eb_def = micro_time_range
        # Try to get detector-specific ranges from the channel_definer
        vv_sb = sb_def; vv_eb = eb_def
        vh_sb = sb_def; vh_eb = eb_def
        
        info = getattr(self.channel_definer, 'detectors', {}).get(self.current_detector, {})
        ranges = info.get('micro_time_ranges', None)
        if ranges and len(ranges) >= 2:
            raw_vv = ranges[0]
            raw_vh = ranges[1]
            binning = max(1, int(micro_time_binning))
            vv_sb = int(raw_vv[0] // binning)
            vv_eb = int(raw_vv[1] // binning)
            vh_sb = int(raw_vh[0] // binning)
            vh_eb = int(raw_vh[1] // binning)

        for idx, tttr in enumerate(tttr_list):
            # Use filter_tttr helper to select relevant events
            if len(detector_chs) >= 2:
                tp = self.filter_tttr(tttr, micro_time_range, detector_chs[::2])
                ts = self.filter_tttr(tttr, micro_time_range, detector_chs[1::2])
            else:
                tp = ts = self.filter_tttr(tttr, micro_time_range, detector_chs)

            # Build full microtime histograms (default uses full range)
            cp = tp.get_microtime_histogram(micro_time_binning)[0].astype(np.float64, copy=False)
            cs = ts.get_microtime_histogram(micro_time_binning)[0].astype(np.float64, copy=False)

            # Apply integer VH shift BEFORE any other operation
            if apply_vh_shift and self.shift != 0:
                cs = np.roll(cs, self.shift)

            # Now zero out-of-window bins per channel (after shift), only when shifting/windowing is desired
            if apply_vh_shift:
                if vv_sb > 0:
                    cp[:vv_sb] = 0
                if vv_eb < cp.size:
                    cp[vv_eb:] = 0
                if vh_sb > 0:
                    cs[:vh_sb] = 0
                if vh_eb < cs.size:
                    cs[vh_eb:] = 0

                # Apply thresholds
                th_vv = th_vh = -1.0
                if isinstance(threshold, (tuple, list)) and len(threshold) >= 2:
                    th_vv = float(threshold[0]) if threshold[0] is not None else -1.0
                    th_vh = float(threshold[1]) if threshold[1] is not None else -1.0
                if th_vv > 0:
                    if cp.size and cp.max() > 0:
                        cp[cp < th_vv * cp.max()] = 0
                if th_vh > 0:
                    if cs.size and cs.max() > 0:
                        cs[cs < th_vh * cs.max()] = 0

            # Optional normalization
            if normalize_counts == 1:
                # Normalize by average count rate
                ct = (cp.sum() + cs.sum()) / 2.0
                if ct > 0:
                    cp /= ct
                    cs /= ct
            elif normalize_counts == 2:
                # Normalize individually
                cp_sum = cp.sum()
                cs_sum = cs.sum()
                if cp_sum > 0:
                    cp = cp / cp_sum
                if cs_sum > 0:
                    cs = cs / cs_sum
            elif normalize_counts == 3:
                # Normalize by acquisition time
                acquisition_time = (tttr.macro_times[-1] - tttr.macro_times[0]) * tttr.header.macro_time_resolution
                if acquisition_time > 0:
                    cs /= acquisition_time
                    cp /= acquisition_time

            # now build the JORDI vector
            j = np.hstack([cp, cs])
            jordis.append(j)

            # Optional save
            if save_files:
                basename = getattr(tttr, 'filename', None)
                if basename:
                    base = Path(basename).with_suffix('').as_posix()
                else:
                    base = f"jordi_{idx}"
                out_name = f"{base}_{''.join(map(str, detector_chs))}.dat"
                write_jordi(j, out_name)

        return np.array(jordis)

    def filter_tttr(self, tttr, micro_time_range, detector_chs):
        """
        Return a TTTR slice filtered only by routing channels.
        Note: We intentionally ignore micro_time_range to keep full micro-time coverage.
        Zeroing outside [start, stop] is applied later at the histogram level.
        """
        ch = tttr.routing_channels
        mask = np.isin(ch, detector_chs)
        return tttr[np.where(mask)[0]]

    def get_burst_indices_for_current_file(self) -> list[int]:
        if not self.current_filename or self.df_bursts is None:
            return []

        # lazy-compute stems if missing
        if "stem" not in self.df_bursts:
            self.df_bursts["stem"] = (
                self.df_bursts["First File"]
                .str.split(r"[\\/]").str[-1]
                .str.rsplit(".", n=1).str[0]
            )

        # select only the bursts for this file
        curr_stem = Path(self.current_filename).stem
        df_file = self.df_bursts.loc[self.df_bursts["stem"] == curr_stem]
        if df_file.empty:
            return []

        # pull start/stop as int arrays
        starts = df_file["First Photon"].to_numpy(dtype=np.int32)
        stops = df_file["Last Photon"].to_numpy(dtype=np.int32)

        # build a single “difference” event array with bincount
        # - at each start index we +1, at each (stop+1) we -1
        idxs = np.concatenate([starts, stops + 1])
        weights = np.concatenate([
            np.ones_like(starts, dtype=np.int32),
            -np.ones_like(stops + 1, dtype=np.int32),
        ])
        max_len = idxs.max() + 1
        events = np.bincount(idxs, weights, minlength=max_len)

        # cumulative sum >0 gives a boolean mask of covered photons
        coverage = np.cumsum(events)[:-1] > 0

        # return all covered indices
        return np.nonzero(coverage)[0].tolist()

    def read_burst_analysis(
            self,
            paris_path: Path,
            pattern: str = "**/*.bur",
            row_stride: int = 2
    ) -> tuple[pd.DataFrame, dict[str, tttrlib.TTTR]]:
        print("def read_burst_analysis")
        # 1) Locate and sanity-check
        bur_files = sorted(paris_path.glob(pattern))
        if not bur_files:
            raise ValueError(f"No burst files found in {paris_path!s}")

        # Check for JSON file in Info folder of the burst folder
        info_directory = paris_path / 'Info'
        json_file_path = info_directory / "photon_selection_parameters.json"

        # Use safe_open_file to read the JSON file if it exists
        from chisurf.settings.file_utils import safe_open_file
        import json

        setup_info = None
        json_data = safe_open_file(
            json_file_path,
            processor=json.load,
            default_value=None,
            error_message=f"Could not read setup information from {json_file_path}"
        )

        if json_data:
            chisurf.logging.info(f"Found setup information in {json_file_path}")
            # Extract setup information from JSON
            setup_info = json_data.get("setup_info")
            if setup_info:
                chisurf.logging.info("Using setup information from JSON file")
                # If we have setup information, we can use it to configure the wizard
                # For example, we could set channel settings, detector settings, etc.
                # This will depend on what's available in the JSON and what's needed by the wizard

                # If the channel_definer is available, we can update its settings
                if hasattr(self, 'channel_definer') and setup_info.get("windows"):
                    self.channel_definer.windows = setup_info.get("windows", {})
                    self.channel_definer.detectors = setup_info.get("detectors", {})
                    chisurf.logging.info("Updated channel definitions from JSON file")

        # 2) Sample first file to infer which cols are numeric and build robust dtype spec
        sample = pd.read_csv(
            bur_files[0],
            sep="\t",
            header=0,
            skiprows=[1],
            nrows=100,
            engine="c",
            low_memory=False
        )
        # Numeric columns from sample
        num_cols = sample.select_dtypes(include="number").columns
        dtype_spec: dict[str, str] = {col: "float64" for col in num_cols}
        # Ensure file/path-like columns are treated as strings across all files
        string_like_cols = set()
        for col in sample.columns:
            if col.strip() == "" or "File" in col or col in ("First File", "Last File", "BID File", "burst_file"):
                string_like_cols.add(col)
        # Explicitly include common string columns even if not present in the sample
        string_like_cols.update({"First File", "Last File", "BID File", "burst_file", ""})
        for col in string_like_cols:
            dtype_spec[col] = "string"
        # BID Index should be integer if present (use pandas nullable integer)
        if "BID Index" in sample.columns:
            dtype_spec["BID Index"] = "Int64"
        else:
            # add proactively; ignored for files without the column
            dtype_spec["BID Index"] = "Int64"

        # 3) Read each file and concat
        file_dfs = []
        for fn in bur_files:
            df_part = pd.read_csv(
                fn,
                sep="\t",
                header=0,
                skiprows=[1],
                dtype=dtype_spec,
                engine="c",
                low_memory=False
            )
            # Track source .bur file name
            df_part["burst_file"] = str(getattr(fn, 'name', fn))
            file_dfs.append(df_part)
        df = pd.concat(file_dfs, ignore_index=True)

        # 4) One-time down-sampling
        if row_stride > 1:
            df = df.iloc[::row_stride].reset_index(drop=True)

        raw_files = df["First File"].dropna().unique()
        base_dir = paris_path.parent
        # clear any old registrations
        print("self._tttr_paths:", self._tttr_paths)
        for fn in raw_files:
            # Extract just the filename part from the "First File" column
            filename = Path(fn).name
            stem = Path(filename).stem
            # The TTTR file should be in the parent directory of the burst file
            self._tttr_paths[stem] = base_dir / filename

        # now self.tttrs is set up, but no TTTR objects created yet
        return df, self.tttrs

    def save_fit(self):
        """
        Save micro_time_start, micro_time_stop, irf_threshold_vv, irf_threshold_vh, shift, shift_sp, shift_ss,
        p2s_twoIstar, BIFL_scatter, fix_tau, fix_gamma, fix_r0, fix_rho
        to detector_setups.json file in the currently selected setup.
        """
        # Get the current detector
        current_detector = self.current_detector

        # Get the current setup name
        setup_name = self.channel_definer.setup_combo.currentText()
        if not setup_name:
            QMessageBox.warning(self, "Warning", "No setup selected. Please select a setup first.")
            return

        # Get the detector_setups.json file path
        setups_file = self.channel_definer.current_setups_file

        # Load existing setups
        setups = load_detector_setups(setups_file)

        # Check if the setup exists
        if setup_name not in setups.get("setups", {}):
            QMessageBox.warning(self, "Warning", f"Setup '{setup_name}' not found.")
            return

        # Get the parameters for the current detector
        detector_params = {
            "micro_time_start": self.micro_time_range[0],
            "micro_time_stop": self.micro_time_range[1],
            "irf_threshold_vv": self.irf_threshold_vv,
            "irf_threshold_vh": self.irf_threshold_vh,
            "shift": self.shift,
            "shift_sp": self.shift_sp,
            "shift_ss": self.shift_ss,
            "irf_start": self.irf_start,
            "irf_stop": self.irf_stop,
            "p2s_twoIstar": self.p2s_twoIstar,
            "BIFL_scatter": self.BIFL_scatter,
            "fix_tau": self.fix_tau,
            "fix_gamma": self.fix_gamma,
            "fix_r0": self.fix_r0,
            "fix_rho": self.fix_rho,
            "min_photons": int(self.min_photons)
        }

        # Get the setup data
        setup_data = setups["setups"][setup_name]

        # Add or update the MLE settings for the current detector
        if "detectors" not in setup_data:
            setup_data["detectors"] = {}

        # Check if the detector exists in the setup
        if current_detector not in setup_data["detectors"]:
            QMessageBox.warning(self, "Warning", f"Detector '{current_detector}' not found in setup '{setup_name}'.")
            return

        # Add MLE settings to the detector
        if "mle_settings" not in setup_data["detectors"][current_detector]:
            setup_data["detectors"][current_detector]["mle_settings"] = {}

        # Update the MLE settings (merge with existing instead of overwrite)
        existing = setup_data["detectors"][current_detector].get("mle_settings", {})
        if not isinstance(existing, dict):
            existing = {}
        existing.update(detector_params)
        setup_data["detectors"][current_detector]["mle_settings"] = existing

        # Save the updated setups
        if save_detector_setups(setups, setups_file):
            QMessageBox.information(self, "Saved", f"MLE settings for detector '{current_detector}' saved to setup '{setup_name}'.")
        else:
            QMessageBox.critical(self, "Error", f"Could not save MLE settings to setup '{setup_name}'.")

    def optimize_hyperparameters(
            self,
            n_iter: int = 40,
            bounds: dict | None = None,
            seed: int | None = None,
            weights: dict | None = None
    ):
        """Delegate to external HPO function to keep this file lean."""
        try:
            from chisurf.plugins.burst_mle_analysis.utils import optimize_hyperparameters as _opt_hpo
            return _opt_hpo(self, n_iter=n_iter, bounds=bounds, seed=seed, weights=weights)
        except Exception as e:
            QMessageBox.critical(self, "HPO error", f"{e}")
            return None

    def save_settings(self):
        """
        Dump TTTR file‐type, per‐channel settings, AND
        DetectorWizardPage windows + detectors in the exact JSON shape
        that load_data_into_tables expects.
        """
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save All Settings",
            str(Path.home() / "mle_wizard_settings.json"),
            "JSON Files (*.json)"
        )
        if not path:
            return

        detwiz = self.channel_definer

        # Ensure current detector state is captured before saving
        try:
            cur_det = self.current_detector
            if cur_det:
                self.channel_settings[cur_det] = self._capture_current_ui_state()
        except Exception:
            pass

        payload = {
            "tttr_file_type": self.channel_definer.filetype or "Auto",
            "channel_settings": self.channel_settings,
            "detector_settings": detwiz.get_settings(),
            "micro_time_binning": self.micro_time_binning
        }

        with open(path, 'w') as f:
            json.dump(payload, f, indent=4, cls=NumpyEncoder)

        # show the file in both line edits
        QMessageBox.information(self, "Saved", f"All settings saved to:\n{path}")

    def load_settings(self):
        """
        Read that JSON, restore:
          - TTTR file‐type combo
          - per‐channel UI state
          - DetectorWizardPage windows + detectors
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load All Settings",
            str(Path.home()),
            "JSON Files (*.json)"
        )
        if not path:
            return

        # 1) load payload
        with open(path, 'r') as f:
            payload = json.load(f)

        # Micro time binning
        mtb = payload.get("micro_time_binning", None)
        if mtb is not None:
            # Update micro_time_binning in DetectorWizardPage
            self.channel_definer.micro_binning_combo.setCurrentText(str(int(mtb)))

        # 2) TTTR file‐type is now handled by DetectorWizardPage
        # The tttr_file_type will be set when we load the detector settings below

        # 3) reflect path
        self.lineEdit_settings_file.setText(path)
        # self.channel_definer.file_path_line_edit.setText(path)

        # 4) channel_settings
        self.channel_settings = payload.get("channel_settings", {})
        # Make sure each detector state is complete
        try:
            for det in list(self.channel_settings.keys()):
                self._ensure_channel_state(det)
        except Exception:
            pass

        # 5) detector_settings → load into DetectorWizardPage
        det_data = payload.get("detector_settings", {})
        if det_data:
            # 5a) disconnect the one slot so it won’t fire automatically
            self.channel_definer.detectorsChanged.disconnect(self._init_channels_from_wizard)

            # 5b) repopulate the wizard page
            self.channel_definer.load_data_into_tables(det_data)

            # 5c) re-attach and manually kick off exactly one rebuild
            self.channel_definer.detectorsChanged.connect(self._init_channels_from_wizard)
            self._init_channels_from_wizard()

        # grab the *names* we just loaded directly from JSON,
        # so we don’t invoke the broken .detectors property
        valid_dets = set(det_data.get("detectors", {}).keys())

        # 6) re‐apply per‐detector UI state only to those names
        current = self.comboBox_window.currentText()
        for det, state in self.channel_settings.items():
            if det not in valid_dets:
                continue
            self.comboBox_window.setCurrentText(det)
            widgets = (
                self.spinBox_micro_time_start,
                self.spinBox_micro_time_stop,
                self.doubleSpinBox_irf_threshold_vv,
                self.doubleSpinBox_irf_threshold_vh,
                self.doubleSpinBox_shift,
                self.doubleSpinBox_shift_sp,
                self.doubleSpinBox_shift_ss,
            )
            self.block_widget_signals(widgets)
            self._apply_ui_state(state)
            self.unblock_widget_signals(widgets)

        # restore whatever was selected originally
        self.comboBox_window.setCurrentText(current)

        # 7) refresh
        self.update_decay_of_detector()
        self.update_fit()

        QMessageBox.information(self, "Loaded", f"All settings loaded from:\n{path}")


if __name__ == 'plugin':
    mle = MLELifetimeAnalysisWizard()
    mle.show()

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    mle = MLELifetimeAnalysisWizard()
    mle.setWindowTitle('MLE Lifetime Analysis')
    mle.show()
    sys.exit(app.exec_())