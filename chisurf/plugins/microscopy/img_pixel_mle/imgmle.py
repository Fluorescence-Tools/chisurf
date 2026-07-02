"""
Lifetime MLE Analysis Wizard (reactive/cached)

This version makes the GUI more *reactive* by avoiding unnecessary
recomputation of histograms/decays. In particular, changing the micro-time
window (start/stop) now only re-slices cached full histograms instead of
rebuilding them, and IRF preparation is cached and only recomputed when its
own parameters change (thresholds/shifts/binning/channels/IRF file).

Key ideas:
- Cache full micro-time histograms for P/S once per (TTTR, channels, binning)
  and only slice them on start/stop changes.
- Cache raw IRF histograms and a *prepared* IRF for the current
  thresholds/shifts. Re-prepare IRF only when those parameters change.
- Background from file is cached as full histograms, then sliced at use-time.
- Update routing in `update_parameters()` detects which control changed and
  triggers the minimal required work: slice-only, IRF-only, or full rebuild.

Everything else is backward-compatible with your previous UI wiring.
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
import contextlib


import chisurf as cs
import chisurf.gui.decorators
import chisurf.core.settings
import chisurf.gui.widgets.wizard

import tttrlib

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c


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


@persist_plugin_state("img_pixel_mle")
class LifetimeMleAnalysisWizard(QtWidgets.QMainWindow):
    """
    Main wizard for Lifetime MLE Analysis.

    This class now includes a small caching layer and minimal recomputation
    rules so that common UI tweaks (e.g., micro-time start/stop) are instant.
    """
    # ==============================
    # Small helpers: IRF preparation
    # ==============================
    def _interpolate_shift(self, arr: np.ndarray, shift: Union[int, float]) -> np.ndarray:
        """
        Shift a 1D array by a given number of bins, supporting fractional shifts.
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
                    shift_ss: float = 0,
                    threshold_vv: Optional[float] = None,
                    threshold_vh: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare IRF by applying threshold, normalization, and shifts.
        """
        irf_p = irf_p.astype(np.float64).copy()
        irf_s = irf_s.astype(np.float64).copy()

        # Thresholds (per-channel if provided)
        t_p = threshold_vv if threshold_vv is not None else threshold
        t_s = threshold_vh if threshold_vh is not None else threshold
        if t_p is not None and t_p > 0:
            mx = irf_p.max() if irf_p.size else 1.0
            irf_p[irf_p < t_p * mx] = 0
        if t_s is not None and t_s > 0:
            mx = irf_s.max() if irf_s.size else 1.0
            irf_s[irf_s < t_s * mx] = 0

        # Normalize
        sp = irf_p.sum()
        if sp > 0:
            irf_p /= sp
        ss = irf_s.sum()
        if ss > 0:
            irf_s /= ss

        # Sub-bin shifts
        irf_p = self._interpolate_shift(irf_p, shift_sp)
        irf_s = self._interpolate_shift(irf_s, shift_ss)

        # Relative integer shift of S
        if shift != 0:
            irf_s = np.roll(irf_s, shift)
            if shift > 0:
                irf_s[:shift] = 0.0
            else:
                irf_s[shift:] = 0.0

        return irf_p, irf_s

    # ==============================
    # Init & caches
    # ==============================
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data / IRF / BG containers
        self.tttr_data = None
        self.irf_tttr = None
        self.clsm_p = None
        self.clsm_s = None
        self.irf_p = None  # RAW IRF P (full length, post-hist but pre-prepare)
        self.irf_s = None  # RAW IRF S (full length, post-hist but pre-prepare)
        self.tau = None
        self.rho = None
        self._fit = None
        self.decay_all_photons = None
        self.micro_time_range = [0, 0]
        self._micro_time_user_override: bool = False
        self._binning_factor_override: Optional[int] = None
        self._excitation_period_override: Optional[float] = None
        self._in_multi_detector_loop: bool = False
        # Per-detector UI/state cache (like Burst MLE)
        self.channel_settings: Dict[str, Dict] = {}
        self._last_detector_selected: Optional[str] = None

        # Background pattern variables (FULL, unsliced)
        self.bg_tttr = None
        self._bg_p_full = None
        self._bg_s_full = None

        # ------------------
        # Caches & signatures
        # ------------------
        # Full histograms of *data* (not IRF):
        self._full_hist_p: Optional[np.ndarray] = None
        self._full_hist_s: Optional[np.ndarray] = None
        self._hist_signature: Optional[tuple] = None  # (id(tttr), ch_p tuple, ch_s tuple, binning, n_channels)

        # IRF raw hist cache (preparation works on these):
        self._irf_hist_signature: Optional[tuple] = None  # (id(irf_tttr), ch_p tuple, ch_s tuple, binning, n_channels)

        # Prepared IRF cache (fully prepared for current thresholds/shifts):
        self._irf_p_prepared_full: Optional[np.ndarray] = None
        self._irf_s_prepared_full: Optional[np.ndarray] = None
        self._irf_prepare_signature: Optional[tuple] = None  # (irf_hist_sig, thr_vv, thr_vh, shift, shift_sp, shift_ss)

        # Load UI
        ui_file = os.path.join(os.path.dirname(__file__), 'imgmle.ui')
        uic.loadUi(ui_file, self)

        # Detector tab
        self.tab_detector = QtWidgets.QWidget()
        self.tab_widget.insertTab(0, self.tab_detector, "Detector Definition")
        self.tab_widget.setCurrentIndex(0)
        self.verticalLayout_detector_tab = QtWidgets.QVBoxLayout(self.tab_detector)
        self.channel_definer = cs.gui.widgets.wizard.DetectorWizardPage(parent=self)
        self.groupBox_detector = QtWidgets.QGroupBox("Detector Configuration")
        self.verticalLayout_detector = QtWidgets.QVBoxLayout(self.groupBox_detector)
        self.verticalLayout_detector.addWidget(self.channel_definer)
        self.verticalLayout_detector_tab.addWidget(self.groupBox_detector)
        self.channel_definer.detectorsChanged.connect(self._on_detectors_updated)

        # Populate & connect
        self._populate_detector_combo()
        self.comboBox_detector_select.currentTextChanged.connect(self._on_detector_changed)

        # Dual IRF thresholds
        self.doubleSpinBox_irf_threshold_vv.setDecimals(4)
        self.doubleSpinBox_irf_threshold_vv.setRange(0.0, 1.0)
        self.doubleSpinBox_irf_threshold_vv.setSingleStep(0.001)
        init_thr = 0.02
        self.doubleSpinBox_irf_threshold_vv.setValue(init_thr)

        self.doubleSpinBox_irf_threshold_vh.setDecimals(4)
        self.doubleSpinBox_irf_threshold_vh.setRange(0.0, 1.0)
        self.doubleSpinBox_irf_threshold_vh.setSingleStep(0.001)
        self.doubleSpinBox_irf_threshold_vh.setValue(init_thr)

        # Flags
        self.checkBox_2IStar.setChecked(True)
        self.checkBox_BIFL_scatter.setChecked(False)

        # File format radio group
        self.file_format_group = QtWidgets.QButtonGroup(self)
        self.file_format_group.addButton(self.radioButton_FileHDF)
        self.file_format_group.addButton(self.radioButton_FileCsv)

        # FileListWidget callbacks
        self.tttr_list.file_added_callback = self.update_tttr_files
        self.irf_list.file_added_callback = self.update_irf_files
        self.bg_list.file_added_callback = self.update_bg_files

        # Buttons
        self.browse_tttr_button.clicked.connect(lambda: self.browse_files(self.tttr_list))
        self.clear_tttr_button.clicked.connect(lambda: self.clear_files(self.tttr_list))
        self.browse_irf_button.clicked.connect(lambda: self.browse_files(self.irf_list))
        self.clear_irf_button.clicked.connect(lambda: self.clear_files(self.irf_list))
        self.browse_bg_button.clicked.connect(lambda: self.browse_files(self.bg_list, "Background Files (*.ht3 *.ptu *.pt3);;All Files (*.*)"))
        self.clear_bg_button.clicked.connect(lambda: self.clear_files(self.bg_list))
        self.process_button.clicked.connect(self.process_data_all_detectors)

        # Background radios
        self.bg_fixed_radio.toggled.connect(self.toggle_background_source)
        self.bg_file_radio.toggled.connect(self.toggle_background_source)

        # File selector
        self.file_selector_combo.currentIndexChanged.connect(self.on_file_selection_changed)

        # Plots
        self.residual_plot.setLabel('left', 'Residuals')
        self.combined_plot.setLabel('bottom', 'Time (ch.)')
        self.combined_plot.setLabel('left', 'Intensity')
        self.combined_plot.setLogMode(y=True)
        self.combined_plot.setYRange(-1, 5)
        self.residual_plot.setXLink(self.combined_plot)

        # Wire signals
        self.connect_signals()

        # UI init
        self.initialize_ui_values()

        # Window size
        self.resize(700, 500)

    # ==============================
    # Wiring / signals
    # ==============================
    def connect_signals(self):
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Fit-affecting numeric params → just update fit
        self.min_photons_spinbox.valueChanged.connect(self.update_parameters)
        self.tau_spinbox.valueChanged.connect(self.update_parameters)
        self.gamma_spinbox.valueChanged.connect(self.update_parameters)
        self.r0_spinbox.valueChanged.connect(self.update_parameters)
        self.rho_spinbox.valueChanged.connect(self.update_parameters)
        self.fix_tau_checkbox.stateChanged.connect(self.update_parameters)
        self.fix_gamma_checkbox.stateChanged.connect(self.update_parameters)
        self.fix_r0_checkbox.stateChanged.connect(self.update_parameters)
        self.fix_rho_checkbox.stateChanged.connect(self.update_parameters)
        self.checkBox_2IStar.stateChanged.connect(self.update_parameters)
        self.checkBox_BIFL_scatter.stateChanged.connect(self.update_parameters)

        # Time-window controls → slice-only
        self.micro_time_start_spinbox.valueChanged.connect(self.update_parameters)
        self.micro_time_stop_spinbox.valueChanged.connect(self.update_parameters)

        # IRF-preparation-only controls → re-prepare IRF then fit
        self.shift_spinbox.valueChanged.connect(self.update_parameters)
        self.shift_sp_spinbox.valueChanged.connect(self.update_parameters)
        self.shift_ss_spinbox.valueChanged.connect(self.update_parameters)
        self.doubleSpinBox_irf_threshold_vv.valueChanged.connect(self.update_parameters)
        self.doubleSpinBox_irf_threshold_vh.valueChanged.connect(self.update_parameters)

        # Background controls → rebuild background only
        self.bg_p_spinbox.valueChanged.connect(self.update_parameters)
        self.bg_s_spinbox.valueChanged.connect(self.update_parameters)
        self.use_bg_checkbox.stateChanged.connect(self.update_parameters)
        self.bg_fixed_radio.toggled.connect(self.update_parameters)
        self.bg_file_radio.toggled.connect(self.update_parameters)

    def initialize_ui_values(self):
        start, stop = self.micro_time_range
        self.micro_time_start_spinbox.setValue(start)
        self.micro_time_stop_spinbox.setValue(stop)

    # ==============================
    # Setup / detector
    # ==============================
    def on_tab_changed(self, index: int):
        try:
            if index != 0 and getattr(self, 'channel_definer', None) is not None:
                self._apply_setup_from_wizard()
        except Exception:
            pass

    def _split_ps_channels(self, chs: typing.List[int]) -> typing.Tuple[typing.List[int], typing.List[int]]:
        if not chs:
            return [], []
        pchs = chs[::2]
        schs = chs[1::2] if len(chs) > 1 else chs
        return pchs, schs

    def _populate_detector_combo(self):
        if getattr(self, 'comboBox_detector_select', None) is None:
            return
        self.comboBox_detector_select.blockSignals(True)
        self.comboBox_detector_select.clear()
        try:
            dets = []
            if getattr(self, 'channel_definer', None) is not None:
                dets = list(self.channel_definer.detectors.keys())
            if dets:
                self.comboBox_detector_select.addItems(dets)
                self._on_detector_changed(dets[0])
        except Exception:
            pass
        finally:
            self.comboBox_detector_select.blockSignals(False)

    def _on_detectors_updated(self):
        try:
            # Initialize per-detector state cache
            if getattr(self, 'channel_definer', None) is not None:
                for det in list(self.channel_definer.detectors.keys()):
                    self._ensure_channel_state(det)
            self._populate_detector_combo()
            self._apply_setup_from_wizard()
        except Exception:
            pass

    def _apply_setup_from_wizard(self):
        try:
            if getattr(self, 'channel_definer', None) is None:
                return
            try:
                st = self.channel_definer.get_settings()
                tttr_read = st.get('tttr_reading', {}) if isinstance(st, dict) else {}
                binf = int(tttr_read.get('micro_time_binning', self._binning_factor_override or 1))
                self._binning_factor_override = binf
                if 'excitation_period' in tttr_read:
                    try:
                        self._excitation_period_override = float(tttr_read.get('excitation_period'))
                    except Exception:
                        self._excitation_period_override = None
            except Exception:
                pass
            det_name = None
            try:
                det_name = self.comboBox_detector_select.currentText() if getattr(self, 'comboBox_detector_select', None) is not None else None
                if not det_name and self.channel_definer.detectors:
                    det_name = list(self.channel_definer.detectors.keys())[0]
            except Exception:
                pass
            if det_name:
                self._on_detector_changed(det_name)
                try:
                    info = self.channel_definer.detectors.get(det_name, {})
                    mtrs = info.get('micro_time_ranges', []) or []
                    if len(mtrs) > 0 and not getattr(self, '_micro_time_user_override', False):
                        binning = max(1, int(self._get_binning_factor()))
                        sb = int(mtrs[0][0] // binning)
                        eb = int(mtrs[0][1] // binning)
                        if eb <= sb:
                            eb = sb + 1
                        try:
                            if getattr(self, 'tttr_data', None) is not None:
                                n_channels = int(self.tttr_data.header.number_of_micro_time_channels // binning)
                                sb = max(0, min(sb, max(0, n_channels - 1)))
                                eb = max(1, min(eb, n_channels))
                        except Exception:
                            pass
                        self.micro_time_start_spinbox.setValue(sb)
                        self.micro_time_stop_spinbox.setValue(eb)
                        self.micro_time_range = [sb, eb]
                        # Slice-only if we already have caches
                        self._update_slice_and_fit()
                except Exception:
                    pass
        except Exception:
            pass

    def _get_binning_factor(self) -> int:
        try:
            if getattr(self, 'channel_definer', None) is not None:
                st = self.channel_definer.get_settings()
                if isinstance(st, dict):
                    val = st.get('tttr_reading', {}).get('micro_time_binning')
                    if val is not None:
                        return int(val)
        except Exception:
            pass
        if getattr(self, '_binning_factor_override', None):
            try:
                return int(self._binning_factor_override)
            except Exception:
                pass
        try:
            if getattr(self, 'binning_factor_spinbox', None) is not None:
                return int(self.binning_factor_spinbox.value())
        except Exception:
            pass
        return 1

    def _get_excitation_period_ns(self) -> Optional[float]:
        try:
            if getattr(self, 'channel_definer', None) is not None:
                st = self.channel_definer.get_settings()
                if isinstance(st, dict):
                    val = st.get('tttr_reading', {}).get('excitation_period')
                    if val is not None:
                        return float(val)
        except Exception:
            pass
        try:
            if self._excitation_period_override is not None:
                return float(self._excitation_period_override)
        except Exception:
            pass
        try:
            if hasattr(self, 'tttr_data') and self.tttr_data is not None:
                header = self.tttr_data.header
                if hasattr(header, 'laser_period') and header.laser_period:
                    return float(header.laser_period) * 1e9
                if hasattr(header, 'tttr_info') and 'SyncRate' in header.tttr_info:
                    sync_rate = float(header.tttr_info['SyncRate'])
                    if sync_rate > 0:
                        return 1e9 / sync_rate
                nch = int(header.number_of_micro_time_channels)
                dt_ns = float(header.micro_time_resolution) * 1e9
                if nch > 0 and dt_ns > 0:
                    return nch * dt_ns
        except Exception:
            pass
        return None

    def _get_effective_detector_params(self) -> Tuple[List[int], List[int], int, Optional[Tuple[int, int]], float, float, float]:
        ch_p_list: List[int] = [0]
        ch_s_list: List[int] = [1]
        binning = self._get_binning_factor()
        mtr = None
        g = 1.0
        l1 = 1.0 / 3.0
        l2 = 1.0 / 3.0
        try:
            if getattr(self, 'channel_definer', None) is not None and self.channel_definer.detectors:
                det_name = None
                if getattr(self, 'comboBox_detector_select', None) is not None:
                    det_name = self.comboBox_detector_select.currentText()
                if not det_name:
                    det_name = list(self.channel_definer.detectors.keys())[0]
                info = self.channel_definer.detectors.get(det_name, {})
                if 'ch_p' in info or 'ch_s' in info:
                    try:
                        cp = info.get('ch_p', []) or []
                        ch_s = info.get('ch_s', []) or []
                        ch_p_list = [int(cp)] if isinstance(cp, int) else [int(x) for x in cp]
                        ch_s_list = [int(ch_s)] if isinstance(ch_s, int) else [int(x) for x in ch_s]
                    except Exception:
                        pass
                else:
                    chs = info.get('chs', [])
                    pchs, schs = self._split_ps_channels(chs)
                    if pchs:
                        ch_p_list = [int(x) for x in pchs]
                    if schs:
                        ch_s_list = [int(x) for x in schs]
                mtrs = info.get('micro_time_ranges', []) or []
                if len(mtrs) > 0:
                    mtr = (int(mtrs[0][0]), int(mtrs[0][1]))
                g = float(info.get('g_factor', g))
                l1 = float(info.get('l1', l1))
                l2 = float(info.get('l2', l2))
        except Exception:
            pass
        return ch_p_list, ch_s_list, binning, mtr, g, l1, l2

    def _capture_current_ui_state(self) -> Dict:
        """Capture current UI state relevant per detector (like in Burst MLE)."""
        try:
            start = int(self.micro_time_start_spinbox.value())
            stop = int(self.micro_time_stop_spinbox.value())
        except Exception:
            start, stop = (self.micro_time_range[0], self.micro_time_range[1])
        state = {
            'micro_time_start': start,
            'micro_time_stop': stop,
            'micro_time_binning': int(self._get_binning_factor()),
            'irf_threshold_vv': float(self.irf_threshold_vv),
            'irf_threshold_vh': float(self.irf_threshold_vh),
            'shift': int(getattr(self, 'shift_spinbox', None).value() if getattr(self, 'shift_spinbox', None) is not None else 0),
            'shift_sp': float(getattr(self, 'shift_sp_spinbox', None).value() if getattr(self, 'shift_sp_spinbox', None) is not None else 0.0),
            'shift_ss': float(getattr(self, 'shift_ss_spinbox', None).value() if getattr(self, 'shift_ss_spinbox', None) is not None else 0.0),
            # Per-detector flags
            'p2s_twoIstar': bool(self.p2s_twoIstar) if hasattr(self, 'p2s_twoIstar') else bool(getattr(self, 'checkBox_2IStar', None).isChecked()) if getattr(self, 'checkBox_2IStar', None) is not None else True,
            'BIFL_scatter': bool(self.BIFL_scatter) if hasattr(self, 'BIFL_scatter') else bool(getattr(self, 'checkBox_BIFL_scatter', None).isChecked()) if getattr(self, 'checkBox_BIFL_scatter', None) is not None else False,
        }
        return state

    def _ensure_channel_state(self, det: str) -> Dict:
        """Ensure a per-detector state exists; initialize from setup if missing."""
        st = self.channel_settings.get(det, {}).copy()
        # binning
        if 'micro_time_binning' not in st:
            st['micro_time_binning'] = int(self._get_binning_factor())
        # thresholds default from detector mle_settings or sensible defaults
        try:
            info = getattr(self.channel_definer, 'detectors', {}).get(det, {})
            mle = (info.get('mle_settings', {}) or {}) if isinstance(info, dict) else {}
        except Exception:
            info = {}
            mle = {}
        st.setdefault('irf_threshold_vv', float(mle.get('irf_threshold_vv', 0.02)))
        st.setdefault('irf_threshold_vh', float(mle.get('irf_threshold_vh', st['irf_threshold_vv'])))
        st.setdefault('shift', 0)
        st.setdefault('shift_sp', 0.0)
        st.setdefault('shift_ss', 0.0)
        # Flags: default from mle_settings or current UI
        default_twoI = bool(mle.get('p2s_twoIstar', getattr(self, 'p2s_twoIstar', True)))
        default_bifl = bool(mle.get('BIFL_scatter', getattr(self, 'BIFL_scatter', False)))
        st.setdefault('p2s_twoIstar', default_twoI)
        st.setdefault('BIFL_scatter', default_bifl)
        # micro-time start/stop from detector ranges if not present
        if 'micro_time_start' not in st or 'micro_time_stop' not in st:
            try:
                mtrs = info.get('micro_time_ranges', []) or []
            except Exception:
                mtrs = []
            sb = 0; eb = max(1, self.micro_time_range[1])
            if mtrs:
                binning = max(1, int(st['micro_time_binning']))
                try:
                    sb = int(mtrs[0][0] // binning)
                    eb = int(mtrs[0][1] // binning)
                except Exception:
                    sb, eb = 0, max(1, self.micro_time_range[1])
            # clamp to TTTR size if available
            try:
                if getattr(self, 'tttr_data', None) is not None:
                    n_tot = int(self.tttr_data.header.number_of_micro_time_channels // max(1, st['micro_time_binning']))
                    sb = max(0, min(sb, max(0, n_tot - 1)))
                    eb = max(sb + 1, min(eb, n_tot))
            except Exception:
                pass
            if eb <= sb:
                eb = sb + 1
            st['micro_time_start'] = int(sb)
            st['micro_time_stop'] = int(eb)
        # Save back and return
        self.channel_settings[det] = st
        return st

    def _apply_channel_state(self, st: Dict):
        """Apply per-detector state to UI, blocking signals to avoid redundant recomputation."""
        try:
            # Block signals while setting values
            for w in [getattr(self, 'micro_time_start_spinbox', None), getattr(self, 'micro_time_stop_spinbox', None),
                      getattr(self, 'doubleSpinBox_irf_threshold_vv', None), getattr(self, 'doubleSpinBox_irf_threshold_vh', None),
                      getattr(self, 'shift_spinbox', None), getattr(self, 'shift_sp_spinbox', None), getattr(self, 'shift_ss_spinbox', None),
                      getattr(self, 'checkBox_2IStar', None), getattr(self, 'checkBox_BIFL_scatter', None)]:
                if w is not None:
                    w.blockSignals(True)
            # Set values
            if getattr(self, 'micro_time_start_spinbox', None) is not None:
                self.micro_time_start_spinbox.setValue(int(st.get('micro_time_start', 0)))
            if getattr(self, 'micro_time_stop_spinbox', None) is not None:
                self.micro_time_stop_spinbox.setValue(int(st.get('micro_time_stop', max(1, self.micro_time_range[1]))))
            self.micro_time_range = [int(st.get('micro_time_start', 0)), int(st.get('micro_time_stop', max(1, self.micro_time_range[1])))]
            # thresholds
            self.irf_threshold_vv = float(st.get('irf_threshold_vv', 0.02))
            self.irf_threshold_vh = float(st.get('irf_threshold_vh', self.irf_threshold_vv))
            # shifts
            if getattr(self, 'shift_spinbox', None) is not None:
                self.shift_spinbox.setValue(int(st.get('shift', 0)))
            if getattr(self, 'shift_sp_spinbox', None) is not None:
                self.shift_sp_spinbox.setValue(float(st.get('shift_sp', 0.0)))
            if getattr(self, 'shift_ss_spinbox', None) is not None:
                self.shift_ss_spinbox.setValue(float(st.get('shift_ss', 0.0)))
            # flags
            if getattr(self, 'checkBox_2IStar', None) is not None:
                self.checkBox_2IStar.setChecked(bool(st.get('p2s_twoIstar', getattr(self, 'p2s_twoIstar', True))))
            if getattr(self, 'checkBox_BIFL_scatter', None) is not None:
                self.checkBox_BIFL_scatter.setChecked(bool(st.get('BIFL_scatter', getattr(self, 'BIFL_scatter', False))))
        finally:
            for w in [getattr(self, 'micro_time_start_spinbox', None), getattr(self, 'micro_time_stop_spinbox', None),
                      getattr(self, 'doubleSpinBox_irf_threshold_vv', None), getattr(self, 'doubleSpinBox_irf_threshold_vh', None),
                      getattr(self, 'shift_spinbox', None), getattr(self, 'shift_sp_spinbox', None), getattr(self, 'shift_ss_spinbox', None),
                      getattr(self, 'checkBox_2IStar', None), getattr(self, 'checkBox_BIFL_scatter', None)]:
                if w is not None:
                    w.blockSignals(False)

    def _sync_current_ui_to_channel_settings(self):
        """Persist current UI values into per-detector state cache."""
        try:
            det = self._current_detector_name()
            if not det:
                return
            st = self._ensure_channel_state(det)
            cur = self._capture_current_ui_state()
            st.update(cur)
            self.channel_settings[det] = st
        except Exception:
            pass

    def _on_detector_changed(self, det_name: str):
        try:
            if not det_name or getattr(self, 'channel_definer', None) is None:
                return
            # Save previous detector state
            if self._last_detector_selected:
                try:
                    prev = self._last_detector_selected
                    if prev in getattr(self.channel_definer, 'detectors', {}).keys():
                        self.channel_settings[prev] = self._capture_current_ui_state()
                except Exception:
                    pass
            info = self.channel_definer.detectors.get(det_name)
            if not info:
                return
            # Ensure and apply per-detector state (start/stop, thresholds, shifts)
            st = self._ensure_channel_state(det_name)
            self._apply_channel_state(st)
            # Avoid setup overwrite of window in subsequent loading
            self._micro_time_user_override = True
            # Changing detector likely changes channels → full recompute
            self.update_irf_files()
            self.update_bg_files()
            self.load_data_and_compute_decays()  # ensures caches for new detector
            self._last_detector_selected = det_name
        except Exception:
            pass

    # ==============================
    # Caches: builders & updaters
    # ==============================
    def _current_hist_signature(self, ch_p: List[int], ch_s: List[int], binning: int, n_channels: int) -> tuple:
        return (id(self.tttr_data), tuple(ch_p), tuple(ch_s), int(binning), int(n_channels))

    def _ensure_full_hists(self, force: bool = False):
        if self.tttr_data is None:
            return
        ch_p_list, ch_s_list, binning_factor, _mtr_unused, *_ = self._get_effective_detector_params()
        try:
            n_channels = int(self.tttr_data.header.number_of_micro_time_channels // max(1, binning_factor))
        except Exception:
            n_channels = 0
        sig = self._current_hist_signature(ch_p_list, ch_s_list, binning_factor, n_channels)
        if force or self._full_hist_p is None or self._hist_signature != sig:
            micro_times = self.tttr_data.micro_times // max(1, binning_factor)
            idx_p = self.tttr_data.get_selection_by_channel(ch_p_list)
            idx_s = self.tttr_data.get_selection_by_channel(ch_s_list)
            self._full_hist_p = np.bincount(micro_times[idx_p], minlength=n_channels)
            self._full_hist_s = np.bincount(micro_times[idx_s], minlength=n_channels)
            self._hist_signature = sig

    def _current_irf_hist_signature(self, ch_p: List[int], ch_s: List[int], binning: int, n_channels: int) -> tuple:
        return (id(self.irf_tttr), tuple(ch_p), tuple(ch_s), int(binning), int(n_channels))

    def _ensure_irf_hist(self, force: bool = False):
        if self.irf_tttr is None:
            return
        ch_p_list, ch_s_list, binning_factor, _mtr_unused, *_ = self._get_effective_detector_params()
        try:
            n_channels = int(self.irf_tttr.header.number_of_micro_time_channels // max(1, binning_factor))
        except Exception:
            n_channels = 0
        sig = self._current_irf_hist_signature(ch_p_list, ch_s_list, binning_factor, n_channels)
        if force or self.irf_p is None or self._irf_hist_signature != sig:
            irf_data_p = self.irf_tttr[self.irf_tttr.get_selection_by_channel(ch_p_list)]
            irf_data_s = self.irf_tttr[self.irf_tttr.get_selection_by_channel(ch_s_list)]
            self.irf_p, _ = irf_data_p.get_microtime_histogram(binning_factor)
            self.irf_s, _ = irf_data_s.get_microtime_histogram(binning_factor)
            # Align size to n_channels just in case
            if self.irf_p.size > n_channels:
                self.irf_p = self.irf_p[:n_channels]
            if self.irf_s.size > n_channels:
                self.irf_s = self.irf_s[:n_channels]
            self._irf_hist_signature = sig
            # Invalidate prepared IRF (depends on raw IRF)
            self._irf_prepare_signature = None

    def _current_irf_prepare_signature(self) -> Optional[tuple]:
        if self._irf_hist_signature is None:
            return None
        return (
            self._irf_hist_signature,
            float(self.irf_threshold_vv),
            float(self.irf_threshold_vh),
            int(self.shift_spinbox.value()),
            float(self.shift_sp_spinbox.value()),
            float(self.shift_ss_spinbox.value()),
        )

    def _ensure_prepared_irf(self, force: bool = False):
        # Ensure raw IRF hist exists
        self._ensure_irf_hist()
        sig = self._current_irf_prepare_signature()
        if sig is None:
            return
        if force or self._irf_p_prepared_full is None or self._irf_prepare_signature != sig:
            p, s = self.prepare_irf(
                self.irf_p if self.irf_p is not None else np.array([]),
                self.irf_s if self.irf_s is not None else np.array([]),
                threshold=-1,
                shift=sig[3],
                shift_sp=sig[4],
                shift_ss=sig[5],
                threshold_vv=sig[1],
                threshold_vh=sig[2]
            )
            self._irf_p_prepared_full = p
            self._irf_s_prepared_full = s
            self._irf_prepare_signature = sig

    def _key_cols(self, df: pd.DataFrame) -> list:
        keys = [c for c in ['FileIndex', 'Z pixel', 'Y pixel', 'X pixel', 'Pixel Number'] if c in df.columns]
        return keys

    def _normalize_coord_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize typical variants => single canonical name, then drop dups
        canonical = {
            'xpixel': 'X pixel', 'x': 'X pixel',
            'ypixel': 'Y pixel', 'y': 'Y pixel',
            'zpixel': 'Z pixel', 'z': 'Z pixel',
            'pixelnumber': 'Pixel Number', 'pixel': 'Pixel Number', 'pix': 'Pixel Number',
            'fileindex': 'FileIndex', 'file': 'FileIndex', 'file_id': 'FileIndex',
        }
        ren = {}
        for col in list(df.columns):
            key = col.replace(' ', '').lower()
            if key in canonical:
                ren[col] = canonical[key]
        if ren:
            df = df.rename(columns=ren)
        # drop duplicate columns (keeps the first)
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def _rename_measurement_cols(self, df: pd.DataFrame, det_name: str) -> pd.DataFrame:
        keys = set(self._key_cols(df))
        rename = {}
        for c in df.columns:
            if c in keys:
                continue
            rename[c] = f"{c} [{det_name}]"
        return df.rename(columns=rename)

    def _combine_multi_detector_results_as_columns(self) -> pd.DataFrame:
        frames = []
        for det_block in getattr(self, 'results_list_all_detectors', []) or []:
            if not det_block:
                continue
            det_name = det_block[0].get('detector', self._current_detector_name())
            rows = []
            for entry in det_block:
                fidx = entry.get('file_index', 0)
                for r in (entry.get('results') or []):
                    rr = dict(r)
                    rr['FileIndex'] = fidx
                    rows.append(rr)
            if not rows:
                continue
            df = self._build_dataframe_from_results(rows)
            df = self._normalize_coord_columns(df)
            df = self._rename_measurement_cols(df, det_name)
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        base = frames[0]
        keys = self._key_cols(base)
        for df in frames[1:]:
            base = base.merge(df, on=keys, how='outer', suffixes=('', '_dup'))
            # any accidental dup columns from pandas suffixes
            dup_cols = [c for c in base.columns if c.endswith('_dup')]
            if dup_cols:
                base = base.drop(columns=dup_cols)

        # order: keys first, then the detector-annotated metrics
        return base[keys + sorted([c for c in base.columns if c not in keys])]

    # ==============================
    # UI helpers
    # ==============================
    def update_parameters(self):
        """Smart parameter update routing to minimize recomputation."""
        if not hasattr(self, 'tttr_data') or self.tttr_data is None:
            return

        sender = self.sender()
        # Update cached micro time range from UI
        self.micro_time_range = [self.micro_time_start_spinbox.value(), self.micro_time_stop_spinbox.value()]
        # Persist current UI to per-detector cache
        self._sync_current_ui_to_channel_settings()

        # 1) Micro-time window changed → slice-only
        if sender in [getattr(self, 'micro_time_start_spinbox', None), getattr(self, 'micro_time_stop_spinbox', None)]:
            self._micro_time_user_override = True
            self._update_slice_and_fit()
            return

        # 2) IRF-preparation-only inputs → re-prepare IRF and refit (no data hist rebuild)
        if sender in [
            getattr(self, 'shift_spinbox', None),
            getattr(self, 'shift_sp_spinbox', None),
            getattr(self, 'shift_ss_spinbox', None),
            getattr(self, 'doubleSpinBox_irf_threshold_vv', None),
            getattr(self, 'doubleSpinBox_irf_threshold_vh', None)
        ]:
            # Just invalidate prepared IRF and refit
            self._ensure_prepared_irf(force=True)
            self.update_fit()
            return

        # 3) Background controls → refit only (background built in get_settings)
        if sender in [
            getattr(self, 'bg_p_spinbox', None),
            getattr(self, 'bg_s_spinbox', None),
            getattr(self, 'use_bg_checkbox', None),
            getattr(self, 'bg_fixed_radio', None),
            getattr(self, 'bg_file_radio', None)
        ]:
            self.update_fit()
            return

        # 4) Fit seeds/fixes → refit only
        if sender in [
            getattr(self, 'min_photons_spinbox', None),
            getattr(self, 'tau_spinbox', None),
            getattr(self, 'gamma_spinbox', None),
            getattr(self, 'r0_spinbox', None),
            getattr(self, 'rho_spinbox', None),
            getattr(self, 'fix_tau_checkbox', None),
            getattr(self, 'fix_gamma_checkbox', None),
            getattr(self, 'fix_r0_checkbox', None),
            getattr(self, 'fix_rho_checkbox', None),
            getattr(self, 'checkBox_2IStar', None),
            getattr(self, 'checkBox_BIFL_scatter', None)
        ]:
            self.update_fit()
            return

        # Unknown sender or structural changes → be safe and rebuild
        self.load_data_and_compute_decays()

    def toggle_background_source(self):
        if self.bg_file_radio.isChecked():
            self.load_background_pattern()
        self.update_fit()

    # ==============================
    # File handling
    # ==============================
    def browse_files(self, list_widget, name_filter="TTTR Files (*.ht3 *.ptu *.pt3);;All Files (*.*)"):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter(name_filter)
        if file_dialog.exec_():
            file_names = file_dialog.selectedFiles()
            for file_name in file_names:
                list_widget.add_file(file_name)

    def clear_files(self, list_widget):
        list_widget.clear()
        if list_widget in [self.tttr_list, self.irf_list]:
            if list_widget == self.tttr_list:
                self.tttr_data = None
                self.clsm_p = None
                self.clsm_s = None
                self.decay_all_photons = None
                # Invalidate caches
                self._full_hist_p = None
                self._full_hist_s = None
                self._hist_signature = None
            elif list_widget == self.irf_list:
                self.irf_tttr = None
                self.irf_p = None
                self.irf_s = None
                self._irf_hist_signature = None
                self._irf_p_prepared_full = None
                self._irf_s_prepared_full = None
                self._irf_prepare_signature = None
            self._fit = None
            if hasattr(self, 'combined_plot') and self.combined_plot is not None:
                self.combined_plot.clear()
            if hasattr(self, 'residual_plot') and self.residual_plot is not None:
                self.residual_plot.clear()
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
        elif list_widget == self.bg_list:
            self.bg_tttr = None
            self._bg_p_full = None
            self._bg_s_full = None
            self.update_fit()

    def update_tttr_files(self):
        self.load_data_and_compute_decays()

    @property
    def p2s_twoIstar(self) -> bool:
        return self.checkBox_2IStar.isChecked()

    @p2s_twoIstar.setter
    def p2s_twoIstar(self, value: bool):
        self.checkBox_2IStar.setChecked(value)

    @property
    def BIFL_scatter(self) -> bool:
        return bool(self.checkBox_BIFL_scatter.isChecked())

    @BIFL_scatter.setter
    def BIFL_scatter(self, value: bool):
        self.checkBox_BIFL_scatter.setChecked(value)

    @property
    def irf_threshold_vv(self) -> float:
        try:
            if getattr(self, 'doubleSpinBox_irf_threshold_vv', None) is not None:
                return float(self.doubleSpinBox_irf_threshold_vv.value())
        except Exception:
            pass
        return 0.02

    @irf_threshold_vv.setter
    def irf_threshold_vv(self, v: float):
        try:
            if getattr(self, 'doubleSpinBox_irf_threshold_vv', None) is not None:
                self.doubleSpinBox_irf_threshold_vv.setValue(v)
                return
        except Exception:
            pass

    @property
    def irf_threshold_vh(self) -> float:
        try:
            if getattr(self, 'doubleSpinBox_irf_threshold_vh', None) is not None:
                return float(self.doubleSpinBox_irf_threshold_vh.value())
        except Exception:
            pass
        return self.irf_threshold_vv

    @irf_threshold_vh.setter
    def irf_threshold_vh(self, v: float):
        try:
            if getattr(self, 'doubleSpinBox_irf_threshold_vh', None) is not None:
                self.doubleSpinBox_irf_threshold_vh.setValue(v)
                return
        except Exception:
            pass
        self.irf_threshold_vv = v

    def update_irf_files(self):
        self.load_data_and_compute_decays()

    def update_bg_files(self):
        self.load_background_pattern()
        self.update_fit()

    # ==============================
    # Background
    # ==============================
    def load_background_pattern(self):
        if not self.bg_file_radio.isChecked():
            return
        bg_files = self.bg_list.get_selected_files()
        if not bg_files:
            self.bg_tttr = None
            self._bg_p_full = None
            self._bg_s_full = None
            return
        try:
            ch_p_list, ch_s_list, binning_factor, _mtr_unused, *_ = self._get_effective_detector_params()
            fn_bg = bg_files[0]
            self.bg_tttr = tttrlib.TTTR(fn_bg)
            bg_data_p = self.bg_tttr[self.bg_tttr.get_selection_by_channel(ch_p_list)]
            bg_data_s = self.bg_tttr[self.bg_tttr.get_selection_by_channel(ch_s_list)]
            self._bg_p_full, _ = bg_data_p.get_microtime_histogram(binning_factor)
            self._bg_s_full, _ = bg_data_s.get_microtime_histogram(binning_factor)
            cs.logging.info(f"Loaded background pattern from {fn_bg}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading background pattern: {str(e)}")
            self.bg_tttr = None
            self._bg_p_full = None
            self._bg_s_full = None

    # ==============================
    # Data & decays
    # ==============================
    def load_data_and_compute_decays(self):
        ch_p_list, ch_s_list, binning_factor, mtr, _, _, _ = self._get_effective_detector_params()
        tttr_files = self.tttr_list.get_selected_files()
        irf_files = self.irf_list.get_selected_files()
        if not tttr_files or not irf_files:
            return
        try:
            # Load TTTR (first sets context; keep others in list for batch processing)
            self.tttr_data_list = []
            fn_clsm = tttr_files[0]
            self.tttr_data = tttrlib.TTTR(fn_clsm)
            self.tttr_data_list.append(self.tttr_data)
            for i in range(1, len(tttr_files)):
                self.tttr_data_list.append(tttrlib.TTTR(tttr_files[i]))

            # IRF TTTR
            fn_irf = irf_files[0]
            self.irf_tttr = tttrlib.TTTR(fn_irf)

            # Ensure caches
            self._ensure_full_hists(force=True)
            self._ensure_irf_hist(force=True)
            self._ensure_prepared_irf(force=True)

            # Set default micro-time window from setup if available (and not overridden)
            if mtr is not None and not getattr(self, '_micro_time_user_override', False):
                bfac = max(1, int(binning_factor))
                sb = int(mtr[0] // bfac)
                eb = int(mtr[1] // bfac)
                if eb <= sb:
                    eb = sb + 1
                try:
                    n_tot = int(self.tttr_data.header.number_of_micro_time_channels // bfac)
                    sb = max(0, min(sb, max(0, n_tot - 1)))
                    eb = max(1, min(eb, n_tot))
                except Exception:
                    pass
                self.micro_time_start_spinbox.setValue(sb)
                self.micro_time_stop_spinbox.setValue(eb)
                self.micro_time_range = [sb, eb]

            # Update slice & fit
            self._update_slice_and_fit()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading data: {str(e)}")
            return

    def _update_slice_and_fit(self):
        if self.tttr_data is None:
            return
        self._ensure_full_hists()
        start, stop = self.micro_time_range
        # Guard against empty caches
        if self._full_hist_p is None or self._full_hist_s is None:
            return
        # Slice-only
        hist_p = self._full_hist_p[start:stop]
        hist_s = self._full_hist_s[start:stop]
        self.decay_all_photons = np.hstack([hist_p, hist_s])
        # Prepared IRF is sliced during get_settings()/plot
        self.update_fit()

    # ==============================
    # Batch processing (unchanged heavy path)
    # ==============================
    @contextlib.contextmanager
    def _suppress_qmessagebox(self, default_answer=None):
        """
        Temporarily suppress QMessageBox popups during batch operations.
        All messages are logged instead. 'question' returns default_answer or QMessageBox.Yes.
        """
        try:
            MB = QMessageBox
            orig_info = MB.information
            orig_warn = MB.warning
            orig_crit = MB.critical
            orig_question = MB.question

            def _noop_info(parent, title, text, *args, **kwargs):
                try:
                    cs.logging.info(f"[info suppressed] {title}: {text}")
                except Exception:
                    pass
                return MB.Ok

            def _noop_warn(parent, title, text, *args, **kwargs):
                try:
                    cs.logging.warning(f"[warning suppressed] {title}: {text}")
                except Exception:
                    pass
                return MB.Ok

            def _noop_crit(parent, title, text, *args, **kwargs):
                try:
                    cs.logging.error(f"[critical suppressed] {title}: {text}")
                except Exception:
                    pass
                return MB.Ok

            def _noop_question(parent, title, text, buttons=MB.Yes | MB.No, default_button=MB.No):
                try:
                    cs.logging.info(f"[question suppressed] {title}: {text}")
                except Exception:
                    pass
                return default_answer if default_answer is not None else MB.Yes

            MB.information = _noop_info
            MB.warning = _noop_warn
            MB.critical = _noop_crit
            MB.question = _noop_question
            yield
        finally:
            try:
                MB.information = orig_info
                MB.warning = orig_warn
                MB.critical = orig_crit
                MB.question = orig_question
            except Exception:
                pass

    def process_data(self):
        # Keep heavy per-pixel path unchanged; users trigger via button.
        self.load_data_and_compute_decays()
        all_settings = self.get_settings()
        ch_p = all_settings['ch_p']
        ch_s = all_settings['ch_s']
        if isinstance(ch_p, int):
            ch_p = [ch_p]
        if isinstance(ch_s, int):
            ch_s = [ch_s]
        binning_factor = all_settings['binning_factor']
        minimum_n_photons = all_settings['min_photons']
        if not hasattr(self, 'tttr_data_list') or not self.tttr_data_list:
            return
        irf = all_settings['irf']
        x0 = np.array([all_settings['tau'], all_settings['gamma'], all_settings['r0'], all_settings['rho']])
        fixed = np.array([
            1 if all_settings['fix_tau'] else 0,
            1 if all_settings['fix_gamma'] else 0,
            1 if all_settings['fix_r0'] else 0,
            1 if all_settings['fix_rho'] else 0
        ])
        self.results_list = []
        self.tau_list = []
        self.rho_list = []
        self.results = []
        tttr_files = self.tttr_list.get_selected_files()
        total_files = len(self.tttr_data_list)
        progress_dialog = CombinedProgressDialog(self)
        progress_dialog.set_file_progress(1, total_files)
        progress_dialog.show()
        QApplication.processEvents()
        time_start = time.time()
        start, stop = self.micro_time_range
        for file_idx, tttr_data in enumerate(self.tttr_data_list):
            progress_dialog.set_file_progress(file_idx, total_files)
            QApplication.processEvents()
            self.tttr_data = tttr_data
            file_results = []
            self.results_list.append(file_results)
            self.clsm_p = tttrlib.CLSMImage(self.tttr_data, channels=ch_p, fill=True)
            self.clsm_s = tttrlib.CLSMImage(self.tttr_data, channels=ch_s, fill=True)
            if all_settings['stack_frames']:
                self.clsm_p.stack_frames()
                self.clsm_s.stack_frames()
            n_channels = self.tttr_data.header.number_of_micro_time_channels // binning_factor
            settings = {
                'dt': self.tttr_data.header.micro_time_resolution * 1e9 * binning_factor,
                'g_factor': all_settings['g_factor'],
                'l1': all_settings['l1'],
                'l2': all_settings['l2'],
                'convolution_stop': -1,
                'irf': irf,
                'period': all_settings['period'],
                'background': all_settings['background'],
                'p2s_twoIstar_flag': all_settings['p2s_twoIstar'],
                'soft_bifl_scatter_flag': all_settings['BIFL_scatter']
            }
            fit23 = tttrlib.Fit23(**settings)
            intensity = self.clsm_p.intensity
            micro_times = self.tttr_data.micro_times // binning_factor
            n_channels = self.tttr_data.header.number_of_micro_time_channels // binning_factor
            tau_array = np.zeros_like(intensity, dtype=np.float32)
            rho_array = np.zeros_like(intensity, dtype=np.float32)
            n_frames, n_lines, n_pixel = self.clsm_p.shape
            progress_dialog.set_frame_progress(0, n_frames)
            progress_dialog.set_line_progress(0, n_lines)
            QApplication.processEvents()
            hist_p_template = np.zeros(stop - start, dtype=np.int64)
            hist_s_template = np.zeros(stop - start, dtype=np.int64)
            batch_results = []
            for i in range(n_frames):
                progress_dialog.set_frame_progress(i + 1, n_frames)
                QApplication.processEvents()
                for j in range(n_lines):
                    progress_dialog.set_line_progress(j + 1, n_lines)
                    QApplication.processEvents()
                    line_data = []
                    for k in range(n_pixel):
                        idx_p = self.clsm_p[i][j][k].tttr_indices
                        idx_s = self.clsm_s[i][j][k].tttr_indices
                        n_p = len(idx_p)
                        n_s = len(idx_s)
                        total_photons = n_p + n_s
                        line_data.append({
                            'idx_p': np.array(idx_p, copy=True) if len(idx_p) > 0 else idx_p,
                            'idx_s': np.array(idx_s, copy=True) if len(idx_s) > 0 else idx_s,
                            'n_p': n_p,
                            'n_s': n_s,
                            'total_photons': total_photons,
                            'pixel_idx': k,
                        })
                    for pixel_data in line_data:
                        k = pixel_data['pixel_idx']
                        n_p = pixel_data['n_p']
                        n_s = pixel_data['n_s']
                        total_photons = pixel_data['total_photons']
                        idx_p = pixel_data['idx_p']
                        idx_s = pixel_data['idx_s']
                        if total_photons < minimum_n_photons:
                            # Below the fit threshold: emit only MLE fit columns
                            # (NaN). Per-pixel intensity/count columns come from the
                            # Intensity tool, not the MLE, so this stays fit-only.
                            result_dict = {
                                'Y pixel': j,
                                'X pixel': k,
                                'Pixel Number': j * n_pixel + k,
                                f'Number of Photons (fit window)': 0,
                                f'tau': np.nan,
                                f'gamma': np.nan,
                                f'r0': np.nan,
                                f'rho': np.nan,
                                f'BIFL scatter fit?': 0,
                                f'2I*: P+2S?': 0,
                                f'rS': np.nan,
                                f'rE': np.nan,
                                f'2I*': np.nan,
                            }
                            if n_frames > 1:
                                result_dict['Z pixel'] = i
                            batch_results.append(result_dict)
                            continue
                        if n_p > 0:
                            hist_p = np.bincount(micro_times[idx_p], minlength=n_channels)[start:stop] if start < n_channels else hist_p_template.copy()
                            fit_window_photons_p = np.sum(hist_p)
                        else:
                            hist_p = hist_p_template.copy()
                            fit_window_photons_p = 0
                        if n_s > 0:
                            hist_s = np.bincount(micro_times[idx_s], minlength=n_channels)[start:stop] if start < n_channels else hist_s_template.copy()
                            fit_window_photons_s = np.sum(hist_s)
                        else:
                            hist_s = hist_s_template.copy()
                            fit_window_photons_s = 0
                        fit_window_photons = fit_window_photons_p + fit_window_photons_s
                        hist = np.concatenate([hist_p, hist_s])
                        r = fit23(hist, x0, fixed)
                        tau_array[i, j, k] = r['x'][0]
                        rho_array[i, j, k] = r['x'][3]
                        # MLE-only output: fit parameters + coordinates. Per-pixel
                        # intensity/count-rate columns are produced by the Intensity
                        # tool and merged into the shared imaging HDF5.
                        result_dict = {
                            'Y pixel': j,
                            'X pixel': k,
                            'Pixel Number': j * n_pixel + k,
                            f'Number of Photons (fit window)': fit_window_photons,
                            f'tau': r['x'][0],
                            f'gamma': r['x'][1],
                            f'r0': r['x'][2],
                            f'rho': r['x'][3],
                            f'BIFL scatter fit?': int(all_settings['BIFL_scatter']),
                            f'2I*: P+2S?': all_settings['p2s_twoIstar'],
                            f'rS': np.nan,
                            f'rE': np.nan,
                            f'2I*': r.get('twoIstar', -1),
                        }
                        if n_frames > 1:
                            result_dict['Z pixel'] = i
                        batch_results.append(result_dict)
                    file_results.extend(batch_results)
                    self.results.extend(batch_results)
                    batch_results = []
            self.tau_list.append(tau_array)
            self.rho_list.append(rho_array)
            self.tau = tau_array
            self.rho = rho_array
            progress_dialog.set_frame_progress(0, 1)
            progress_dialog.set_line_progress(0, 1)
            QApplication.processEvents()
        time_stop = time.time()
        progress_dialog.set_file_progress(total_files, total_files)
        QApplication.processEvents()
        progress_dialog.hide()
        self.display_results()
        self.update_fit()
        if not getattr(self, '_in_multi_detector_loop', False):
            QMessageBox.information(self, "Processing Complete", f"Processing completed in {time_stop - time_start:.2f} seconds.")
        if not getattr(self, '_in_multi_detector_loop', False):
            self.auto_export_results()

    def process_data_all_detectors(self):
        """
        Process either the currently selected detector or all detectors if defined.
        When multiple detectors are processed, suppress modal message boxes so the run
        is not interrupted. If Auto Export is enabled, export once at the end.
        """
        try:
            if getattr(self, 'channel_definer', None) is not None and self.channel_definer.detectors:
                det_names = list(self.channel_definer.detectors.keys())
                self.results_list_all_detectors = []
                self._in_multi_detector_loop = True
                with self._suppress_qmessagebox():
                    for det in det_names:
                        if getattr(self, 'comboBox_detector_select', None) is not None:
                            idx = self.comboBox_detector_select.findText(det)
                            if idx >= 0:
                                self.comboBox_detector_select.setCurrentIndex(idx)
                            else:
                                self._on_detector_changed(det)
                        else:
                            self._on_detector_changed(det)
                        self.process_data()
                        tagged_per_file = []
                        for f_idx, file_results in enumerate(getattr(self, 'results_list', [])):
                            tagged_per_file.append({'detector': det, 'file_index': f_idx, 'results': file_results})
                        self.results_list_all_detectors.append(tagged_per_file)
                self._in_multi_detector_loop = False
                # Single final export always
                self.auto_export_results()
            else:
                self.process_data()
        except Exception as e:
            try:
                QMessageBox.warning(self, "Processing Error", f"Failed to process all detectors: {str(e)}")
            except Exception:
                try:
                    cs.logging.error(f"Failed to process all detectors: {e}")
                except Exception:
                    pass

    # ==============================
    # Display / plots
    # ==============================
    def on_file_selection_changed(self, index):
        if index < 0 or not hasattr(self, 'tau_list') or not self.tau_list:
            return
        if index < len(self.tau_list):
            self.display_file_results(index)

    def display_file_results(self, file_index):
        if file_index < 0 or not hasattr(self, 'tau_list') or file_index >= len(self.tau_list):
            return
        tau = self.tau_list[file_index]
        self.image_view.setImage(tau[0], levels=(0, 5))
        self.hist_plot.clear()
        y, x = np.histogram(tau[0].flatten(), bins=131, range=(0.01, 5))
        self.hist_plot.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        self.hist_plot.setLabel('left', 'Count')
        self.hist_plot.setLabel('bottom', 'Lifetime (ns)')

    def display_results(self):
        if not hasattr(self, 'tau_list') or not self.tau_list:
            return
        self.file_selector_combo.blockSignals(True)
        self.file_selector_combo.clear()
        tttr_files = self.tttr_list.get_selected_files()
        for i, _ in enumerate(self.tau_list):
            if i < len(tttr_files):
                file_name = os.path.basename(tttr_files[i])
            else:
                file_name = f"File {i+1}"
            self.file_selector_combo.addItem(file_name)
        self.file_selector_combo.blockSignals(False)
        if self.tau_list:
            self.display_file_results(0)

    # ==============================
    # Export
    # ==============================
    def _current_detector_name(self) -> str:
        try:
            if getattr(self, 'comboBox_detector_select', None) is not None:
                det = self.comboBox_detector_select.currentText().strip()
                if det:
                    return det
            if getattr(self, 'channel_definer', None) is not None and self.channel_definer.detectors:
                return list(self.channel_definer.detectors.keys())[0]
        except Exception:
            pass
        return 'detector'

    def _build_dataframe_from_results(self, results) -> pd.DataFrame:
        df = pd.DataFrame(results)

        # detector-color rename kept
        det_name = self._current_detector_name()
        color = det_name.lower()
        rename_map = {c: c.replace('(green)', f'({color})') for c in list(df.columns) if '(green)' in c}
        if rename_map:
            df = df.rename(columns=rename_map)

        # ints for flags
        try:
            bifl_col = f'BIFL scatter fit?'
            if bifl_col in df.columns:
                df[bifl_col] = df[bifl_col].astype(int)
        except Exception:
            pass

        # normalize coordinate columns & dtypes
        df = self._normalize_coord_columns(df)
        if 'FileIndex' in df.columns:
            with contextlib.suppress(Exception):
                df['FileIndex'] = pd.to_numeric(df['FileIndex'], errors='coerce').fillna(0).astype(int)

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
        return df

    def _compose_base_name(self, base_name: str, n_frames: int, binning_factor: int, min_photons: int, micro_time_start: int, micro_time_stop: int) -> str:
        return f"{base_name}_Frames_{n_frames}_#23_BinFactor_{binning_factor}_MinPh#{min_photons}_MicroTime_{micro_time_start}-{micro_time_stop}"

    def auto_export_results(self):
        if not hasattr(self, 'results_list') or not self.results_list:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return
        all_settings = self.get_settings()
        tttr_files = self.tttr_list.get_selected_files()
        if not tttr_files:
            return
        if len(tttr_files) == 1:
            base_name = os.path.splitext(os.path.basename(tttr_files[0]))[0]
        else:
            base_name = "MultipleFiles"
        min_photons = all_settings['min_photons']
        binning_factor = all_settings['binning_factor']
        micro_time_start, micro_time_stop = all_settings['micro_time_range']
        n_frames = 0
        if hasattr(self, 'clsm_p') and self.clsm_p is not None:
            n_frames = self.clsm_p.shape[0]
        composed_base = self._compose_base_name(base_name, n_frames, binning_factor, min_photons, micro_time_start, micro_time_stop)
        file_dir = os.path.dirname(tttr_files[0])

        # Multi-detector handling: if we processed all detectors, write a single HDF5 with groups per detector
        multi_det = hasattr(self, 'results_list_all_detectors') and self.results_list_all_detectors
        if multi_det:
            df_all = self._combine_multi_detector_results_as_columns()
            if df_all.empty:
                QMessageBox.warning(self, "Export Error", "No multi-detector results to export.")
                return

            # one flat table, no HDF5 groups
            if all_settings['file_format_hdf']:
                file_name = f"{composed_base}_allDetectors.h5"
                file_path = os.path.join(file_dir, file_name)
                try:
                    df_all.to_hdf(file_path, key='results', mode='w',
                                  complevel=9, complib='blosc', format='table')
                except Exception as e:
                    QMessageBox.warning(self, "Export Error", f"Failed writing HDF5: {e}")
                    return
                QMessageBox.information(self, "Auto Export Complete", f"Results automatically exported to {file_path}")
            else:
                file_name = f"{composed_base}_allDetectors.pg4"
                file_path = os.path.join(file_dir, file_name)
                try:
                    with open(file_path, 'w') as f:
                        f.write('\t'.join(df_all.columns) + '\n')
                        for _, row in df_all.iterrows():
                            f.write(
                                '\t'.join([f"{val:.8f}" if isinstance(val, float) else f"{val}" for val in row]) + '\n')
                except Exception as e:
                    QMessageBox.warning(self, "Export Error", f"Failed writing CSV: {e}")
                    return
                QMessageBox.information(self, "Auto Export Complete", f"Results automatically exported to {file_path}")
            return

        # Fallbacks: single detector or CSV
        if multi_det and not all_settings['file_format_hdf']:
            # CSV cannot hold groups; write per-detector CSV files into same directory
            try:
                for det_block in self.results_list_all_detectors:
                    if not det_block:
                        continue
                    det_name = det_block[0].get('detector', self._current_detector_name())
                    rows = []
                    for entry in det_block:
                        fidx = entry.get('file_index', 0)
                        res = entry.get('results', [])
                        for r in res:
                            r = dict(r)
                            r['FileIndex'] = fidx
                            rows.append(r)
                    df_det = self._build_dataframe_from_results(rows)
                    fn = f"{composed_base}_{det_name}.pg4"
                    fp = os.path.join(file_dir, fn)
                    with open(fp, 'w') as f:
                        f.write('\t'.join(df_det.columns) + '\n')
                        for _, row in df_det.iterrows():
                            f.write('\t'.join([f"{val:.8f}" if isinstance(val, float) else f"{val}" for val in row]) + '\n')
                QMessageBox.information(self, "Auto Export Complete", f"Per-detector CSVs exported to {file_dir}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Failed writing CSVs: {e}")
            return

        # Single-detector case
        filtered_results = [item for sublist in self.results_list for item in sublist]
        df = self._build_dataframe_from_results(filtered_results)
        det_name = self._current_detector_name()
        if all_settings['file_format_hdf']:
            file_name = f"{composed_base}_{det_name}.h5"
            file_path = os.path.join(file_dir, file_name)
            df.to_hdf(file_path, key='results', mode='w', complevel=9, complib='blosc', format='table')
        else:
            file_name = f"{composed_base}_{det_name}.pg4"
            file_path = os.path.join(file_dir, file_name)
            with open(file_path, 'w') as f:
                f.write('\t'.join(df.columns) + '\n')
                for _, row in df.iterrows():
                    f.write('\t'.join([f"{val:.8f}" if isinstance(val, float) else f"{val}" for val in row]) + '\n')
        QMessageBox.information(self, "Auto Export Complete", f"Results automatically exported to {file_path}")

    # ==============================
    # Fit & plots
    # ==============================
    def fit_parameters(self):
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
        use_bg = use_bg if use_bg is not None else self.use_bg_checkbox.isChecked()
        bg_fixed = bg_fixed if bg_fixed is not None else self.bg_fixed_radio.isChecked()
        bg_file = bg_file if bg_file is not None else self.bg_file_radio.isChecked()
        bg_p = bg_p if bg_p is not None else self.bg_p_spinbox.value()
        bg_s = bg_s if bg_s is not None else self.bg_s_spinbox.value()
        if not use_bg:
            return np.zeros_like(irf)
        n_half = len(irf) // 2
        background = np.zeros_like(irf)
        # Slice full background to current window
        start, stop = self.micro_time_range
        if bg_file and self._bg_p_full is not None and self._bg_s_full is not None:
            p = self._bg_p_full[start:stop]
            s = self._bg_s_full[start:stop]
            if p.size > n_half:
                p = p[:n_half]
            if s.size > n_half:
                s = s[:n_half]
            if p.size < n_half:
                p = np.pad(p, (0, n_half - p.size))
            if s.size < n_half:
                s = np.pad(s, (0, n_half - s.size))
            background[:n_half] = p
            background[n_half:] = s
            cs.logging.info("Using background pattern from file (sliced)")
        else:
            background[:n_half] = bg_p
            background[n_half:] = bg_s
            cs.logging.info(f"Using fixed background values: P={bg_p}, S={bg_s}")
        return background

    def update_fit(self):
        if not hasattr(self, 'tttr_data') or self.tttr_data is None:
            return
        if self.decay_all_photons is None:
            return
        # Ensure prepared IRF (depends on thresholds/shifts) is ready
        self._ensure_prepared_irf()
        all_settings = self.get_settings()
        x0 = np.array([all_settings['tau'], all_settings['gamma'], all_settings['r0'], all_settings['rho']])
        fixed = np.array([
            1 if all_settings['fix_tau'] else 0,
            1 if all_settings['fix_gamma'] else 0,
            1 if all_settings['fix_r0'] else 0,
            1 if all_settings['fix_rho'] else 0
        ])
        settings = {
            'dt': all_settings['dt'],
            'g_factor': all_settings['g_factor'],
            'l1': all_settings['l1'],
            'l2': all_settings['l2'],
            'convolution_stop': -1,
            'irf': all_settings['irf'],
            'period': all_settings['period'],
            'background': all_settings['background'],
            'p2s_twoIstar_flag': all_settings['p2s_twoIstar'],
            'soft_bifl_scatter_flag': all_settings['BIFL_scatter']
        }
        self._fit = tttrlib.Fit23(**settings)
        res = self._fit(data=self.decay_all_photons, initial_values=x0, fixed=fixed)
        self.plot_fit_result(res)

    def plot_fit_result(self, fit_result):
        self.combined_plot.clear()
        self.residual_plot.clear()
        if self._fit is None:
            return
        # Data & model
        self.combined_plot.plot(self._fit.data, pen=None, symbol='o', symbolSize=3)
        self.combined_plot.plot(self._fit.model, pen='g')
        # IRF slice from prepared cache
        start, stop = self.micro_time_range
        self._ensure_prepared_irf()
        irf_p_range = (self._irf_p_prepared_full[start:stop]
                       if self._irf_p_prepared_full is not None else np.array([]))
        irf_s_range = (self._irf_s_prepared_full[start:stop]
                       if self._irf_s_prepared_full is not None else np.array([]))
        irf = np.hstack([irf_p_range, irf_s_range]) if irf_p_range.size or irf_s_range.size else np.array([])
        if irf.size:
            max_irf = np.max(irf) if np.max(irf) > 0 else 1
            max_data = max(self._fit.data) if max(self._fit.data) > 0 else 1
            irf_scaled = irf * (max_data / max_irf)
            self.combined_plot.plot(irf_scaled, pen='r', name='IRF')
        # Weighted residuals
        data = np.asarray(self._fit.data, dtype=float)
        model = np.asarray(self._fit.model, dtype=float)
        resid = np.zeros_like(data, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            mask = data > 0
            resid[mask] = (data[mask] - model[mask]) / np.sqrt(data[mask])
        try:
            resid = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
        except TypeError:
            resid = np.nan_to_num(resid)
        pen = pg.mkPen(color=(200, 20, 20), width=1)
        self.residual_plot.plot(resid, pen=pen, symbol='o', symbolSize=3)
        self.update_fit_ui(fit_result)

    def update_fit_ui(self, fit_result):
        if fit_result is None:
            return
        self.tau_label.setText(f"{fit_result['x'][0]:.3f}")
        self.gamma_label.setText(f"{fit_result['x'][1]:.3f}")
        self.r0_label.setText(f"{fit_result['x'][2]:.3f}")
        self.rho_label.setText(f"{fit_result['x'][3]:.3f}")
        if hasattr(self._fit, 'chi_square'):
            chi2 = self._fit.chi_square
        else:
            data = self._fit.data
            model = self._fit.model
            with np.errstate(divide='ignore', invalid='ignore'):
                chi2 = np.sum(((data - model) ** 2) / np.maximum(model, 1))
        self.chi2_label.setText(f"Chi²: {chi2:.3f}")

    # ==============================
    # Settings IO
    # ==============================
    def get_settings(self) -> Dict:
        ch_p_eff, ch_s_eff, binning_factor, _mtr_eff, g_eff, l1_eff, l2_eff = self._get_effective_detector_params()
        start, stop = self.micro_time_range
        # Use prepared IRF cache and slice here
        self._ensure_prepared_irf()
        if self._irf_p_prepared_full is not None and self._irf_s_prepared_full is not None:
            irf_p_range = self._irf_p_prepared_full[start:stop]
            irf_s_range = self._irf_s_prepared_full[start:stop]
            irf = np.hstack([irf_p_range, irf_s_range])
        else:
            irf = None
        dt = None
        if hasattr(self, 'tttr_data') and self.tttr_data is not None:
            dt = self.tttr_data.header.micro_time_resolution * 1e9 * binning_factor
        x0, fixed = self.fit_parameters()
        settings = {
            'dt': dt,
            'g_factor': g_eff,
            'l1': l1_eff,
            'l2': l2_eff,
            'convolution_stop': -1,
            'irf': irf,
            'period': self._get_excitation_period_ns(),
            'background': self.create_background(irf) if irf is not None else None,
            'binning_factor': binning_factor,
            'micro_time_range': self.micro_time_range,
            'ch_p': ch_p_eff,
            'ch_s': ch_s_eff,
            'min_photons': self.min_photons_spinbox.value(),
            'tau': x0[0],
            'gamma': x0[1],
            'r0': x0[2],
            'rho': x0[3],
            'fix_tau': self.fix_tau_checkbox.isChecked(),
            'fix_gamma': self.fix_gamma_checkbox.isChecked(),
            'fix_r0': self.fix_r0_checkbox.isChecked(),
            'fix_rho': self.fix_rho_checkbox.isChecked(),
            'irf_threshold_vv': self.irf_threshold_vv,
            'irf_threshold_vh': self.irf_threshold_vh,
            'shift': self.shift_spinbox.value(),
            'shift_sp': self.shift_sp_spinbox.value(),
            'shift_ss': self.shift_ss_spinbox.value(),
            'use_bg': self.use_bg_checkbox.isChecked(),
            'bg_fixed': self.bg_fixed_radio.isChecked(),
            'bg_file': self.bg_file_radio.isChecked(),
            'bg_p': self.bg_p_spinbox.value(),
            'bg_s': self.bg_s_spinbox.value(),
            'stack_frames': self.checkBoxStackFrames.isChecked(),
            'file_format_hdf': self.radioButton_FileHDF.isChecked(),
            'file_format_csv': self.radioButton_FileCsv.isChecked(),
            'p2s_twoIstar': self.p2s_twoIstar,
            'BIFL_scatter': self.BIFL_scatter
        }
        return settings

    def load_settings_from_dict(self, settings: Dict):
        try:
            if 'g_factor' in settings and hasattr(self, 'g_factor_spinbox'):
                self.g_factor_spinbox.setValue(settings['g_factor'])
            if 'l1' in settings and hasattr(self, 'l1_spinbox'):
                self.l1_spinbox.setValue(settings['l1'])
            if 'l2' in settings and hasattr(self, 'l2_spinbox'):
                self.l2_spinbox.setValue(settings['l2'])
        except Exception:
            pass
        if 'period' in settings:
            try:
                self._excitation_period_override = float(settings['period'])
            except Exception:
                self._excitation_period_override = None
        if 'binning_factor' in settings:
            try:
                self._binning_factor_override = int(settings['binning_factor'])
            except Exception:
                self._binning_factor_override = None
        if 'micro_time_range' in settings:
            self.micro_time_start_spinbox.setValue(settings['micro_time_range'][0])
            self.micro_time_stop_spinbox.setValue(settings['micro_time_range'][1])
            self.micro_time_range = settings['micro_time_range']
            self._micro_time_user_override = True
        if 'ch_p' in settings and getattr(self, 'ch_p_spinbox', None) is not None:
            try:
                self.ch_p_spinbox.setValue(settings['ch_p'])
            except Exception:
                pass
        if 'ch_s' in settings and getattr(self, 'ch_s_spinbox', None) is not None:
            try:
                self.ch_s_spinbox.setValue(settings['ch_s'])
            except Exception:
                pass
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
        if 'irf_threshold_vv' in settings:
            self.irf_threshold_vv = settings['irf_threshold_vv']
        if 'irf_threshold_vh' in settings:
            self.irf_threshold_vh = settings['irf_threshold_vh']
        if 'irf_threshold' in settings:
            self.irf_threshold_vv = settings['irf_threshold']
            try:
                if getattr(self, 'doubleSpinBox_irf_threshold_vh', None) is None:
                    self.irf_threshold_vh = settings['irf_threshold']
            except Exception:
                pass
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
        # After loading: update minimal path
        self._update_slice_and_fit()



