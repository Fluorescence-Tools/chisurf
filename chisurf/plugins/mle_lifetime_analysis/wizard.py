import typing

from typing import Union
from pathlib import Path

from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import pandas as pd

from guidata.widgets.dataframeeditor import DataFrameEditor

import tttrlib
import json

import chisurf
import chisurf.gui.decorators
import chisurf.settings
import chisurf.gui.widgets.wizard



class FileListWidget(QtWidgets.QListWidget):
    """
    A QListWidget subclass that accepts file drops and maintains a list of file paths.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    file_added_callback : callable, optional
        Function to call when files are added.
    process_on_drop : bool, optional
        Whether to process files immediately on drop.
    """
    def __init__(self, parent=None, file_added_callback=None, process_on_drop=False):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.file_added_callback = file_added_callback
        self.process_on_drop = process_on_drop
        self.setMaximumHeight(100)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """
        Handle drag enter events to accept file URLs.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        """
        Handle drag move events to accept file URLs.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        """
        Handle drop events, extract file paths, and add them to the list.
        """
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        file_paths: typing.List[str] = []
        for url in event.mimeData().urls():
            local = Path(url.toLocalFile())
            if local.is_file():
                file_paths.append(str(local))
            elif local.is_dir():
                bursts = list(local.glob('**/*.bur'))
                if bursts:
                    file_paths.extend(str(f) for f in bursts)
                else:
                    for ext in tttrlib.get_supported_filetypes():
                        file_paths.extend(str(f) for f in local.glob(f'**/*{ext}'))
        file_paths.sort()
        self.blockSignals(True)
        for fp in file_paths:
            self.add_file(fp)
        self.blockSignals(False)
        if self.file_added_callback:
            self.file_added_callback()
        event.acceptProposedAction()

    def add_file(self, file_path: str):
        """
        Add a file path to the list as a checkable item.

        Parameters
        ----------
        file_path : str
            Path of the file to add.
        """
        item = QtWidgets.QListWidgetItem(file_path, self)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        self.addItem(item)

    def get_selected_files(self) -> typing.List[Path]:
        """
        Get the list of currently selected (checked) files.

        Returns
        -------
        List[Path]
            Paths of selected files.
        """
        return [Path(self.item(i).text()) for i in range(self.count())
                if self.item(i).checkState() == QtCore.Qt.Checked]



class MLELifetimeAnalysisWizard(QtWidgets.QMainWindow):

    def _on_tab_changed(self, index: int):
        """
        Handle tab change events, updating parameters tab when selected.

        Parameters
        ----------
        index : int
            Index of the newly selected tab.
        """
        if self.tabWidget.widget(index) is self.tab_parameters:
            current_idx = self.spinBox_current_file_idx.value()
            self.spinBox_current_file_idx.blockSignals(True)
            self.spinBox_current_file_idx.setValue(current_idx)
            self.spinBox_current_file_idx.blockSignals(False)
            self.update_current_file(current_idx)

    def _update_max_bins_from_tttr(self):
        """
        Update maximum micro-time bins from loaded TTTR header.
        """
        if not getattr(self, 'tttrs', None):
            return

        tttr = next(iter(self.tttrs.values()))
        try:
            total_channels = tttr.header.number_of_micro_time_channels
        except Exception:
            return

        binning = self.spinBox_micro_time_binning.value()
        max_bins = total_channels // binning
        sb = self.spinBox_micro_time_stop
        sb.setMaximum(max_bins)
        start = self.spinBox_micro_time_start.value()
        sb.setMinimum(start + 1)
        val = sb.value()
        if val < start + 1:
            sb.setValue(start + 1)
        elif val > max_bins:
            sb.setValue(max_bins)

    def _capture_current_ui_state(self):
        """Read out all the widgets you want to preserve."""
        x0, fixed = self.fit_parameters
        start_bin, stop_bin = self.micro_time_range
        return {
            'micro_time_start': start_bin,
            'micro_time_stop': stop_bin,
            'micro_time_binning': self.micro_time_binning,
            'irf_threshold': self.irf_threshold,
            'shift': self.shift,
            'shift_sp': self.shift_sp,
            'shift_ss': self.shift_ss,
            'dt': self.dt_effective,
            'excitation_period': self.excitation_period,
            'g_factor': self.g_factor,
            'l1': self.l1,
            'l2': self.l2,
            # capture the full IRF & BG histograms *as lists* for JSON serialization
            'irf': self.irf.tolist(),
            'bg': self.bg.tolist(),
            'initial_x0': x0.tolist(),
            'fixed_flags': fixed.astype(int).tolist()
        }

    def _apply_ui_state(self, state):
        """Push a saved state back into the widgets."""
        # — micro-time controls —
        self.spinBox_micro_time_start.setValue(state['micro_time_start'])
        self.spinBox_micro_time_stop.setValue(state['micro_time_stop'])
        self.spinBox_micro_time_binning.setValue(state['micro_time_binning'])

        # — IRF threshold & shifts —
        self.doubleSpinBox_irf_threshold.setValue(state['irf_threshold'])
        self.doubleSpinBox_shift.setValue(state['shift'])
        self.doubleSpinBox_shift_sp.setValue(state['shift_sp'])
        self.doubleSpinBox_shift_ss.setValue(state['shift_ss'])

        # — effective dt —
        self.doubleSpinBox_dt_effective.setValue(state['dt'])

        # — “internal” fit parameters —
        # use the property setters so the UI stays in sync
        self.excitation_period = state['excitation_period']
        self.g_factor          = state['g_factor']
        self.l1                = state['l1']
        self.l2                = state['l2']

        # — initial‐guess & fixed flags —
        x0    = state['initial_x0']
        fixed = state['fixed_flags']
        self.doubleSpinBox_tau.setValue  (x0[0])
        self.doubleSpinBox_gamma.setValue(x0[1])
        self.doubleSpinBox_r0.setValue   (x0[2])
        self.doubleSpinBox_rho.setValue  (x0[3])

        self.checkBox_fix_tau .setChecked(bool(fixed[0]))
        self.checkBox_fix_gamma.setChecked(bool(fixed[1]))
        self.checkBox_fix_r0   .setChecked(bool(fixed[2]))
        self.checkBox_fix_rho  .setChecked(bool(fixed[3]))

        # — restore cached IRF/BG arrays so your `.irf` & `.bg` props pick them up —
        det = self.current_detector
        self.irf_np[det] = state['irf'].copy()
        self.bg_np [det] = state['bg'].copy()

    def _on_channel_changed(self, new_detector):
        old = getattr(self, '_last_detector', None)
        if old is not None:
            # save old‐channel UI state
            self.channel_settings[old] = self._capture_current_ui_state()

        # pull in the channel‐definer info for the new detector
        info = self.channel_definer.detectors.get(new_detector, {})
        chs = info.get('chs', [])

        # update the parallel/perpendicular channel textfields
        if chs:
            self.lineEdit_parallel_channels.setText(','.join(map(str, chs[::2])))
            self.lineEdit_perpendicular_channels.setText(
                ','.join(map(str, chs[1::2] if len(chs) > 1 else [0]))
            )

        # set the spinboxes to any channel‐specific micro‐time defaults
        ranges = info.get('micro_time_ranges', [])
        if ranges:
            start, stop = ranges[0]
            # block signals so we don't trigger micro-time‐range callbacks
            for sb in (self.spinBox_micro_time_start, self.spinBox_micro_time_stop):
                sb.blockSignals(True)
            self.spinBox_micro_time_start.setValue(start // self.micro_time_binning)
            self.spinBox_micro_time_stop.setValue(stop // self.micro_time_binning)
            for sb in (self.spinBox_micro_time_start, self.spinBox_micro_time_stop):
                sb.blockSignals(False)

        # restore any previously‐saved UI state for this detector
        state = self.channel_settings.get(new_detector)
        if state is not None:
            widgets = (
                self.spinBox_micro_time_start,
                self.spinBox_micro_time_stop,
                self.spinBox_micro_time_binning,
                self.doubleSpinBox_irf_threshold,
                self.doubleSpinBox_shift,
                self.doubleSpinBox_shift_sp,
                self.doubleSpinBox_shift_ss,
            )
            for w in widgets:
                w.blockSignals(True)
            self._apply_ui_state(state)
            for w in widgets:
                w.blockSignals(False)

        # remember where we are now
        self._last_detector = new_detector

        # this covers everything update_selected_window used to do:
        self.update_irf_files()
        self.update_bg_files()
        self.update_decay_of_detector()
        self.update_scatter_count_rate_ui()
        self._fit = None
        self.update_fit()

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

    def _switch_filewidget(self, widgets_dict: dict, active: str):
        """
        Show only the FileListWidget corresponding to the active detector.
        """
        for det, fw in widgets_dict.items():
            fw.setVisible(det == active)

    def _prepare_irf_bg_widgets(self):
        """
        Initialize IRF and background file selection widgets, populate selectors,
        and synchronize widget visibility across detectors.
        """
        dets = list(self.channel_definer.detectors.keys())

        # Create IRF and background FileListWidgets for each detector
        for det in dets:
            irf_fw = FileListWidget(parent=self, file_added_callback=self.update_irf_files)
            irf_fw.hide()
            self.verticalLayout_irf_files.addWidget(irf_fw)
            self.irf_file_widgets[det] = irf_fw

            bg_fw = FileListWidget(parent=self, file_added_callback=self.update_bg_files)
            bg_fw.hide()
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

    def _update_hist_files(self, widgets_dict, np_dict, one_for_all: bool, normalize: int, threshold: float = -1):
        # if norm = 1 --> norm global (vv+vh is normalized)
        # if norm = 2 --> norm individually (vv norm, vh norm)
        det = self.current_detector
        fw = widgets_dict.get(det)
        files = fw.get_selected_files() if fw else []

        # propagate “one for all” if requested
        if one_for_all:
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

        tttr_inputs = []
        for fp in files:
            key = fp.name
            tttr = self.tttrs.get(key)
            if tttr is None:
                tttr = tttrlib.TTTR(str(fp), self.tttr_file_type)
                self.tttrs[key] = tttr
            tttr_inputs.append(tttr)

        # Create detector list (parallel, perpendicular)
        det_chs = [x for pair in zip(self.detector_channels[0], self.detector_channels[1]) for x in pair]

        # build jordi histograms
        if tttr_inputs:
            jordi = self.make_jordi(
                tttr_list=tttr_inputs,
                detector_chs=det_chs,
                micro_time_range=self.micro_time_range,
                micro_time_binning=self.micro_time_binning,
                save_files=self.save_jordis,
                normalize_counts=normalize,
                threshold=threshold
            )
            np_dict[det] = np.sum(jordi, axis=0)
        # redraw without fitting
        self.update_decay_of_detector()

    def inspect_bursts(self, idx: int, embed: bool = False):
        if self.df_bursts is None or not self.tttrs:
            print("No burst data loaded.")
            return

        row = self.df_bursts.iloc[idx]
        key = Path(row['First File']).stem
        tttr = self.tttrs.get(key)
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

            tp = self.filter_tttr(burst, self.micro_time_range, pchs)
            ts = self.filter_tttr(burst, self.micro_time_range, schs)
            cp = tp.get_microtime_histogram(self.micro_time_binning)[0]
            cs = ts.get_microtime_histogram(self.micro_time_binning)[0]
            data = np.hstack([cp, cs])

            # plot it
            p.plot(data, pen=None, symbol='o', symbolSize=4)

            # move to next row in the grid
            self.burst_layout.nextRow()

    @chisurf.gui.decorators.init_with_ui(
        "mle_lifetime_analysis/wizard.ui",
        path=chisurf.settings.plugin_path
    )
    def __init__(self, *args, **kwargs):
        # Core attributes
        self.df_bursts = None
        self._fit = None
        self.tttrs = {}
        self.irf_np = {}
        self.bg_np = {}
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

        # Detector definition tab
        self.tab_detector = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tab_detector, "Detector Definition")
        self.verticalLayout_detector_tab = QtWidgets.QVBoxLayout(self.tab_detector)
        self.channel_definer = chisurf.gui.widgets.wizard.DetectorWizardPage(parent=self)
        self.groupBox_detector = QtWidgets.QGroupBox("Detector Configuration")
        self.verticalLayout_detector = QtWidgets.QVBoxLayout(self.groupBox_detector)
        self.verticalLayout_detector.addWidget(self.channel_definer)
        self.verticalLayout_detector_tab.addWidget(self.groupBox_detector)

        # Prepare irf, bg widgets
        self.irf_file_widgets = {}
        self.bg_file_widgets = {}
        self._prepare_irf_bg_widgets()

        # Inspect a single burst, as chosen by the burst_idx property
        self.spinBox_burst_idx.valueChanged.connect(lambda val: self.inspect_bursts(val, embed=True))

        self.channel_settings = {}
        # initialize per-channel defaults
        for det in self.channel_definer.detectors:
            self.channel_settings[det] = self._capture_current_ui_state()

        self.setup_plots()
        self.update_window_combobox()
        self.initialize_ui_values()
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

        # Decay plot selected burst
        self.groupBox_burst_plot = QtWidgets.QGroupBox("Individual selection")
        self.verticalLayout_4.addWidget(self.groupBox_burst_plot)
        self.verticalLayout_burst_plot = QtWidgets.QVBoxLayout(self.groupBox_burst_plot)

        # this GraphicsLayoutWidget will hold one PlotItem per detector
        self.burst_layout = pg.GraphicsLayoutWidget()
        self.verticalLayout_burst_plot.addWidget(self.burst_layout)

    def update_variable_fit_parameters(self):
        print("update initial parameters. BLANK")
        self.update_fit()

    def update_internal_fit_parameters(self):
        print("update internal fit parameters")
        # Set fit to none to force recreation of fit
        self._fit = None
        self.update_fit()

    def on_irf_parameters_changed(self, _=None):
        """
        Called whenever any IRF parameter (threshold,
        'one‐for‐all' toggle, detector selector, or your
        shift / shift_sp / shift_ss) changes.
        """
        # 1) clear any *cached* IRF data
        self.irf_np.clear()

        # 2) recompute IRFs from the files (this will use your new shift/shift_sp/shift_ss
        self.update_irf_files()

        # 3) toss out the old Fit23 so we’ll build a fresh one with the new IRF
        self._fit = None

        # 4) refresh the decay and re‐run the fit
        self.update_decay_of_detector()
        self.update_fit()

    def connect_signals(self):
        # --- Burst file browser (static list) ---
        self.toolButton_burst_files.clicked.connect(
            lambda: self.browse_files(self.burst_files_list)
        )

        # --- IRF file browser (current detector) ---
        self.toolButton_irf_files.clicked.connect(
            lambda: self.browse_files(
                self.irf_file_widgets[self.comboBox_irf_select.currentText()]
            )
        )

        # --- BG file browser (current detector) ---
        self.toolButton_bg_files.clicked.connect(
            lambda: self.browse_files(
                self.bg_file_widgets[self.comboBox_background_select.currentText()]
            )
        )

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
        self.comboBox_window.currentTextChanged.connect(self._on_channel_changed)
        self.spinBox_current_file_idx.valueChanged.connect(self.update_current_file)

        # --- Fit‐parameter controls (internal) ---
        internal_params = (
            self.doubleSpinBox_dt,
            self.doubleSpinBox_excitation_period,
            self.doubleSpinBox_g_factor,
            self.doubleSpinBox_l1,
            self.doubleSpinBox_l2,
        )
        for spin in internal_params:
            spin.valueChanged.connect(self.update_internal_fit_parameters)

        # dt update shortcut
        self.spinBox_micro_time_binning.valueChanged.connect(self.update_dt)

        # --- Micro‐time range → update decay + fit ---
        self.spinBox_micro_time_binning.valueChanged.connect(self._update_max_bins_from_tttr)
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
        )
        for chk in variable_checks:
            chk.stateChanged.connect(self.update_variable_fit_parameters)

        # --- Other parameter updates ---
        self.spinBox_min_photons.valueChanged.connect(self.update_parameters)

        # --- IRF parameter controls ---
        irf_controls = [
            self.doubleSpinBox_irf_threshold,
            self.checkBox_irf_one_for_all,
            self.comboBox_irf_select,
            self.doubleSpinBox_shift,
            self.doubleSpinBox_shift_sp,
            self.doubleSpinBox_shift_ss,
        ]
        for ctrl in irf_controls:
            # use currentTextChanged or stateChanged automatically based on widget type
            signal = (getattr(ctrl, 'valueChanged', None) or
                      getattr(ctrl, 'stateChanged', None) or
                      getattr(ctrl, 'currentTextChanged'))
            signal.connect(self.on_irf_parameters_changed)

        # --- UI actions ---
        self.pushButton_show_df.clicked.connect(self.show_dataframe_editor)
        self.tabWidget.currentChanged.connect(self._on_tab_changed)

    def show_dataframe_editor(self):
        if self.df_bursts is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "No burst-analysis DataFrame loaded.")
            return
        dlg = DataFrameEditor(self)
        if not dlg.setup_and_check(self.df_bursts, title="Burst Data"):
            return
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.df_bursts = dlg.get_value()

    def initialize_ui_values(self):
        self.comboBox_tttr_file_type.clear()
        self.comboBox_tttr_file_type.addItems(['Auto'] + list(tttrlib.TTTR.get_supported_container_names()))
        self.lineEdit_parallel_channels.setText("8")
        self.lineEdit_perpendicular_channels.setText("0")
        self.spinBox_micro_time_start.setValue(0)
        self.spinBox_micro_time_stop.setValue(4096)
        self.spinBox_micro_time_binning.setValue(16)
        self.doubleSpinBox_dt.setValue(0.004069)
        self.doubleSpinBox_excitation_period.setValue(13.6)
        self.doubleSpinBox_g_factor.setValue(1.08316)
        self.doubleSpinBox_l1.setValue(0.03080)
        self.doubleSpinBox_l2.setValue(0.03680)
        self.doubleSpinBox_tau.setValue(4.0)
        self.doubleSpinBox_gamma.setValue(0.1)
        self.doubleSpinBox_r0.setValue(0.38)
        self.doubleSpinBox_rho.setValue(1.22)
        self.checkBox_fix_tau.setChecked(False)
        self.checkBox_fix_gamma.setChecked(False)
        self.checkBox_fix_r0.setChecked(True)
        self.checkBox_fix_rho.setChecked(False)
        self.spinBox_min_photons.setValue(60)
        self.doubleSpinBox_irf_threshold.setValue(0.02)

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
            self.df_bursts, self.tttrs = None, {}
            self.burst_files_list.clear()
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

    def update_irf_files(self):
        self._update_hist_files(
            widgets_dict=self.irf_file_widgets,
            np_dict=self.irf_np,
            one_for_all=self.one_for_all_irf,
            normalize=2,
            threshold=self.irf_threshold
        )

    def update_bg_files(self):
        self._update_hist_files(
            widgets_dict=self.bg_file_widgets,
            np_dict=self.bg_np,
            one_for_all=self.one_for_all_bg,
            normalize=3
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
            self.df_bursts, self.tttrs = None, {}
        else:
            paris = files[0].parent
            if self.df_bursts is None:
                self.df_bursts = pd.DataFrame()
            df, tttrs = self.read_burst_analysis(paris, self.tttr_file_type)
            self.tttrs = tttrs
            self.df_bursts = pd.concat([self.df_bursts, df], ignore_index=True, sort=False)

        if self.n_bursts == 0:
            # no bursts: lock at 0
            self.spinBox_burst_idx.setRange(0, 0)
            self.spinBox_burst_idx.setValue(0)
            self.spinBox_burst_idx.setEnabled(False)
        else:
            # set 0…n_bursts–1
            self.spinBox_burst_idx.setEnabled(True)
            self.spinBox_burst_idx.setRange(0, self.n_bursts - 1)
            # reset to first burst
            self.burst_idx = 0
            # immediately draw the first burst
            self.inspect_bursts(self.burst_idx, embed=True)

        self._update_max_bins_from_tttr()

    def on_micro_time_range_changed(self, _=None):
        """
        Called whenever micro_time_start or micro_time_stop changes.
        Clears cached IRF/BG histograms, recomputes them, updates decay and fit.
        """
        # 1) clear any precomputed histograms
        self.irf_np.clear()
        self.bg_np.clear()

        # 2) rebuild IRF and BG for the current (and, if “one‐for‐all” is set, all) detectors
        self.update_irf_files()
        self.update_bg_files()

        # 3) recompute the decay of the current file
        self.update_decay_of_detector()

        # 4) reset and rerun your fit
        self._fit = None
        self.update_fit()

    def get_current_jordis(self):
        # gather detector channels and microtime settings
        detector_info = getattr(self.channel_definer, 'detectors', {}) \
            .get(self.current_detector, {})
        chs = detector_info.get('chs', [])
        if not chs:
            print('Channels not found')
            return

        mt_range = self.micro_time_range
        mt_bin = self.micro_time_binning

        # retrieve the TTTR for the current file
        key = Path(self.current_filename).stem
        tttr = self.tttrs.get(key)
        if tttr is None:
            print('No tttr found')
            return

        # get the list of photon‐indices for *all* bursts in this file
        indices = self.get_burst_indices_for_current_file()
        if not indices:
            print('No indices found')
            return
        indices = np.array(indices)

        # slice the TTTR down to just burst photons
        burst_tttr = tttr[indices]

        # build the decay histogram over every burst in the file
        jordis = self.make_jordi(
            [burst_tttr],
            chs,
            mt_range,
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
        self.doubleSpinBox_tau_result.setValue(res['x'][0])
        self.doubleSpinBox_gamma_result.setValue(res['x'][1])
        self.doubleSpinBox_r0_result.setValue(res['x'][2])
        self.doubleSpinBox_rho_result.setValue(res['x'][3])
        self.doubleSpinBox_twoIstar_result.setValue(res['twoIstar'])
        self.doubleSpinBox_r_scatter_result.setValue(res['x'][6])
        self.doubleSpinBox_r_exp_result.setValue(res['x'][7])

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

    def update_selected_window(self):
        print("update_selected_window")
        info = getattr(self.channel_definer, 'detectors', {}).get(self.current_detector, {})
        chs = info.get('chs', [])
        if chs:
            self.lineEdit_parallel_channels.setText(','.join(map(str, chs[::2])))
            self.lineEdit_perpendicular_channels.setText(','.join(map(str, chs[1::2] if len(chs)>1 else [0])))
            ranges = info.get('micro_time_ranges', [])
            if ranges:
                self.spinBox_micro_time_start.setValue(ranges[0][0] // self.micro_time_binning)
                self.spinBox_micro_time_stop.setValue(ranges[0][1] // self.micro_time_binning)
            # recompute IRF/BG for the newly selected detector…
            self.update_irf_files()
            self.update_bg_files()
            # …then redraw decay and update count rate
            self.update_decay_of_detector()
            self.update_scatter_count_rate_ui()
            self.update_fit()

    def update_dt(self):
        effective = self.doubleSpinBox_dt.value() * self.spinBox_micro_time_binning.value()
        self.doubleSpinBox_dt_effective.setValue(effective)

    def update_parameters(self):
        self._fit = None
        self.update_fit()

    @property
    def dt_effective(self):
        return self.doubleSpinBox_dt_effective.value()

    @dt_effective.setter
    def dt_effective(self, value):
        self.doubleSpinBox_dt_effective.setValue(value)

    @property
    def n_bursts(self) -> int:
        """Number of bursts currently loaded."""
        return len(self.df_bursts) if self.df_bursts is not None else 0

    @property
    def detector_channels(self):
        parallel = [int(x) for x in self.lineEdit_parallel_channels.text().split(',')]
        perp = [int(x) for x in self.lineEdit_perpendicular_channels.text().split(',')]
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

    @property
    def micro_time_binning(self):
        return self.spinBox_micro_time_binning.value()

    @property
    def tttr_file_type(self):
        txt = self.comboBox_tttr_file_type.currentText()
        return None if txt == 'Auto' else txt

    @property
    def fit_parameters(self):
        tau = self.doubleSpinBox_tau.value()
        gamma = self.doubleSpinBox_gamma.value()
        r0 = self.doubleSpinBox_r0.value()
        rho = self.doubleSpinBox_rho.value()
        fixed = [
            int(self.checkBox_fix_tau.isChecked()),
            int(self.checkBox_fix_gamma.isChecked()),
            int(self.checkBox_fix_r0.isChecked()),
            int(self.checkBox_fix_rho.isChecked()),
        ]
        return np.array([tau, gamma, r0, rho]), np.array(fixed)

    @property
    def current_filename(self) -> str:
        return self.lineEdit_current_filename.text()

    @current_filename.setter
    def current_filename(self, value: str):
        self.lineEdit_current_filename.setText(value)

    @property
    def burst_idx(self) -> int:
        return self.spinBox_burst_idx.value()

    @burst_idx.setter
    def burst_idx(self, idx: int):
        self.spinBox_burst_idx.setValue(idx)

    @property
    def current_detector(self):
        return self.comboBox_window.currentText()

    @property
    def excitation_period(self) -> float:
        return float(self.doubleSpinBox_excitation_period.value())

    @excitation_period.setter
    def excitation_period(
            self,
            v: float
    ):
        self.doubleSpinBox_excitation_period.setValue(v)

    @property
    def g_factor(self) -> float:
        return float(self.doubleSpinBox_g_factor.value())

    @g_factor.setter
    def g_factor(
            self,
            v: float
    ):
        self.doubleSpinBox_g_factor.setValue(v)

    @property
    def l1(self) -> float:
        return float(self.doubleSpinBox_l1.value())

    @l1.setter
    def l1(
            self,
            v: float
    ):
        self.doubleSpinBox_l1.setValue(v)

    @property
    def l2(self) -> float:
        return float(self.doubleSpinBox_l2.value())

    @l2.setter
    def l2(
            self,
            v: float
    ):
        self.doubleSpinBox_l2.setValue(v)

    @property
    def scatter_count_rate(self) -> float:
        """
        Total background count‐rate (in counts per second)
        for the currently selected detector.
        """
        # self.bg is an array of counts/sec per channel bin
        return float(np.sum(self.bg))

    @property
    def irf_threshold(self) -> float:
        return float(self.doubleSpinBox_irf_threshold.value())

    @irf_threshold.setter
    def irf_threshold(
            self,
            v: float
    ):
        self.doubleSpinBox_irf_threshold.setValue(v)

    @property
    def min_photons(self) -> float:
        return float(self.spinBox_min_photons.value())

    @min_photons.setter
    def min_photons(
            self,
            v: float
    ):
        self.spinBox_min_photons.setValue(v)

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

    @one_for_all_irf.setter
    def one_for_all_bg(
            self,
            v: bool
    ):
        self.checkBox_bg_one_for_all.setChecked(v)

    @property
    def shift(self) -> int:
        """Integer shift of the second (ss) decay relative to the first (sp)."""
        return int(self.doubleSpinBox_shift.value())

    @property
    def shift_sp(self) -> float:
        """Sub-channel (fractional) shift to apply to the sp IRF."""
        return float(self.doubleSpinBox_shift_sp.value())

    @property
    def shift_ss(self) -> float:
        """Sub-channel (fractional) shift to apply to the ss IRF."""
        return float(self.doubleSpinBox_shift_ss.value())

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

        # apply sub-bin shifts
        sp = self._interpolate_shift(sp, self.shift_sp)
        ss = self._interpolate_shift(ss, self.shift_ss)

        # apply integer relative shift to the second decay
        if self.shift != 0:
            ss = np.roll(ss, self.shift)

        # reassemble
        return np.hstack([sp, ss])

    @property
    def bg(self) -> np.ndarray:
        """
        Return the background array for the currently selected detector.
        If none was loaded, return a zeros default matching the IRF length.
        """
        det = self.current_detector
        arr = self.bg_np.get(det)
        if arr is None:
            # match IRF length
            return np.zeros_like(self.irf)
        return arr

    @property
    def fit(self):
        if self._fit is None:
            self._fit = self.create_fit_instance()
        return self._fit

    @property
    def save_jordis(self):
        return self.checkBox_save_jordis.isChecked()

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

    def clear_fit(self):
        self._fit = None

    def create_fit_instance(self):
        # basic params
        dt     = self.dt_effective
        period = self.excitation_period
        gf     = self.g_factor
        l1     = self.l1
        l2     = self.l2
        irf = self.irf
        bg = self.bg

        # finally, build the fit
        fit = tttrlib.Fit23(
            dt=dt,
            irf=irf,
            background=bg,
            period=period,
            g_factor=gf,
            l1=l1,
            l2=l2
        )
        return fit

    def update_fit(self):
        x0, fixed = self.fit_parameters
        decay = self.decay_of_current_file
        det = self.current_detector
        if det not in self.irf_np or det not in self.bg_np:
            return
        if decay is None:
            return
        res = self.fit(data=decay, initial_values=x0, fixed=fixed)
        self.plot_fit_result(res)

    def plot_fit_result(self, fit_result):
        # clear both panels
        self.combined_plot.clear()
        self.residual_plot.clear()

        # only plot if we actually loaded IRF *and* BG for this detector
        det = self.current_detector
        if det not in self.irf_np or det not in self.bg_np:
            return

        # plot data and model in the bottom panel
        self.combined_plot.plot(self.fit.data,
                                pen=None,
                                symbol='o',
                                symbolSize=3)
        self.combined_plot.plot(self.fit.model, pen='g')

        irf = np.copy(self.irf)
        irf /= np.max(irf)
        irf *= max(self.fit.data)
        self.combined_plot.plot(irf, pen='r', name='IRF')
        self.combined_plot.plot(self.bg * self.total_burst_time_seconds, pen='b', name='Background')

        # compute & plot weighted residuals
        data  = self.fit.data
        model = self.fit.model
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            resid = (data - model) / np.sqrt(model)
        # fall‐back zeros where model==0
        resid = np.nan_to_num(resid)

        # draw residuals in the top panel
        pen = pg.mkPen(color=(200, 20, 20), width=1)
        self.residual_plot.plot(resid,
                                pen=pen,
                                symbol='o',
                                symbolSize=3)

        self.update_fit_ui(fit_result)
    def process_bursts(self):
        if self.df_bursts is None or not self.tttrs:
            QtWidgets.QMessageBox.warning(self, "No Data", "No burst data loaded.")
            return

        n = len(self.df_bursts)
        self.progressBar.setRange(0, n)
        results = []

        # Loop over every burst
        for idx, row in self.df_bursts.iterrows():
            self.progressBar.setValue(idx + 1)
            QtWidgets.QApplication.processEvents()

            key = Path(row['First File']).stem
            tttr = self.tttrs.get(key)
            if tttr is None:
                continue
            burst = tttr[int(row['First Photon']):int(row['Last Photon'])]

            # Now loop each detector, using its own saved UI state (no widget writes)
            for det, info in self.channel_definer.detectors.items():
                state = self.channel_settings[det]
                # unpack everything:
                start_bin = state['micro_time_start']
                stop_bin = state['micro_time_stop']
                micro_bin = state['micro_time_binning']
                dt = state['dt']
                period = state['excitation_period']
                gf = state['g_factor']
                l1 = state['l1']
                l2 = state['l2']
                irf = state['irf']
                bg = state['bg']
                x0 = np.array(state['initial_x0'])
                fixed = np.array(state['fixed_flags'], dtype=int)

                chs = info.get('chs', [])
                if not chs:
                    continue

                pchs = chs[::2]
                schs = chs[1::2] if len(chs) > 1 else chs

                # Build micro-time histograms using those ranges/binning
                cp = self.filter_tttr(burst, [start_bin, stop_bin], pchs).get_microtime_histogram(micro_bin)[0]
                cs = self.filter_tttr(burst, [start_bin, stop_bin], schs).get_microtime_histogram(micro_bin)[0]

                # Stack to form the JORDI vector
                c_jordi = np.hstack([cp[start_bin:stop_bin], cs[start_bin:stop_bin]])

                # Create a fresh Fit23 instance using this detector’s IRF/BG
                fit = tttrlib.Fit23(
                    dt=dt,
                    irf=irf,
                    background=bg,
                    period=period,
                    g_factor=gf,
                    l1=l1,
                    l2=l2
                )
                res = fit(data=c_jordi, initial_values=x0, fixed=fixed)

                # Collect fit results
                color = det.lower()
                results.append({
                    'First File': row['First File'],
                    'Detector': det,
                    'Ng-p-all': int(cp.sum()),
                    'Ng-s-all': int(cs.sum()),
                    f'Number of Photons (fit window) ({color})': int(cp.sum() + cs.sum()),
                    f'2I* ({color})': res.get('twoIstar', 0.0),
                    f'Tau ({color})': res['x'][0],
                    f'gamma ({color})': res['x'][1],
                    f'r0 ({color})': res['x'][2],
                    f'rho ({color})': res['x'][3],
                    f'BIFL scatter? ({color})': res['x'][6],
                    f'2I*: P+2S? ({color})': res.get('P2S_ratio', 0.0),
                    f'r Scatter ({color})': res['x'][6],
                    f'r Experimental ({color})': res['x'][7],
                })

        # Build a DataFrame of all results
        result_df = pd.DataFrame(results)

        # Write out per-file, per-detector into b*4 folders
        written_dirs = set()
        for file_path in self.burst_files_list.get_selected_files():
            stem = file_path.stem
            for det in self.channel_definer.detectors:
                color = det.lower()
                letter = color[0]
                out_dir = file_path.parent.parent / f"b{letter}4"
                out_dir.mkdir(parents=True, exist_ok=True)

                # Filter results for this file + detector
                mask = (
                        result_df['First File']
                        .map(lambda fn: Path(fn).stem)
                        .eq(stem)
                        & result_df['Detector'].eq(det)
                )
                df_file = result_df[mask]
                if df_file.empty:
                    continue

                cols = [
                    'Ng-p-all', 'Ng-s-all',
                    f'Number of Photons (fit window) ({color})',
                    f'2I* ({color})', f'Tau ({color})', f'gamma ({color})',
                    f'r0 ({color})', f'rho ({color})',
                    f'BIFL scatter? ({color})', f'2I*: P+2S? ({color})',
                    f'r Scatter ({color})', f'r Experimental ({color})'
                ]

                # Interleave zero-rows and data rows
                zero_template = {c: 0.0 for c in cols}
                records = []
                for _, data_row in df_file.iterrows():
                    records.append(zero_template.copy())
                    records.append({c: data_row[c] for c in cols})
                records.append(zero_template.copy())

                df_out = pd.DataFrame(records)
                out_file = out_dir / f"{stem}.b{letter}4"
                df_out.to_csv(
                    out_file,
                    sep='\t',
                    index=False,
                    float_format='%.6f',
                    columns=cols
                )

                # Save channel_settings alongside the output
                settings_file = out_dir / 'channel_settings.json'
                with open(settings_file, 'w') as sf:
                    json.dump(self.channel_settings, sf, indent=4)
                written_dirs.add(out_dir.name)

        folder_list = ", ".join(sorted(written_dirs)) if written_dirs else "(no data)"
        QtWidgets.QMessageBox.information(
            self,
            "Done",
            f"Burst-fit results saved in folders: {folder_list}"
        )

    def make_jordi(
        self,
        tttr_list: typing.List[tttrlib.TTTR],
        detector_chs: typing.List[int],
        micro_time_range: typing.List[int],
        micro_time_binning: int,
        save_files: bool = False,
        normalize_counts: int = 1,
        threshold: float = -1
    ) -> typing.List[np.ndarray]:
        jordis = list()
        start_bin, stop_bin = micro_time_range
        for idx, tttr in enumerate(tttr_list):
            # Use filter_tttr helper to select relevant events
            if len(detector_chs) >= 2:
                tp = self.filter_tttr(tttr, micro_time_range, detector_chs[::2])
                ts = self.filter_tttr(tttr, micro_time_range, detector_chs[1::2])
            else:
                tp = ts = self.filter_tttr(tttr, micro_time_range, detector_chs)

            # Build microtime histograms
            cp = tp.get_microtime_histogram(micro_time_binning)[0][start_bin:stop_bin]
            cs = ts.get_microtime_histogram(micro_time_binning)[0][start_bin:stop_bin]

            # Apply a threshold
            if threshold > 0:
                cp[cp < threshold * cp.max()] = 0
                cs[cs < threshold * cs.max()] = 0

            # Optional normalization
            if normalize_counts == 1:
                # Normalize by average count rate
                ct = (cp.sum() + cs.sum()) / 2.0
                if ct > 0:
                    cp /= ct
                    cs /= ct
            elif normalize_counts == 2:
                # Normalize individually
                if cp.sum() > 0:
                    cp = cp / cp.sum()
                if cs.sum() > 0:
                    cs = cs / cs.sum()
            elif normalize_counts == 3:
                # Normalize by acquisition time
                acquisition_time = (tttr.macro_times[-1] - tttr.macro_times[0]) * tttr.header.macro_time_resolution
                cs /= acquisition_time
                cp /= acquisition_time

            # apply integer shift of the second decay relative to the first
            if self.shift != 0:
                cs = np.roll(cs, self.shift)

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
                np.savetxt(out_name, j)

        return np.array(jordis)

    def filter_tttr(self, tttr, micro_time_range, detector_chs):
        """
        micro_time_range is [start_bin, stop_bin].
        Convert to raw micro-time before masking.
        """
        start_bin, stop_bin = micro_time_range
        bin_size = self.micro_time_binning

        # convert bins → raw micro-time units
        raw_start = start_bin * bin_size
        raw_stop = stop_bin * bin_size

        mt = tttr.micro_times
        ch = tttr.routing_channels

        mask = (
                (mt >= raw_start) &
                (mt <= raw_stop) &
                np.isin(ch, detector_chs)
        )
        return tttr[np.where(mask)[0]]

    def get_burst_indices_for_current_file(self) -> typing.List[int]:
        # current_filename might be "path/xxx.bur" but the TTTR could be "path/xxx.yyy"
        current = self.current_filename
        if not current or self.df_bursts is None:
            return []

        # compare on the stem ("xxx"), not the full name+ext
        curr_stem = Path(current).stem

        # compute stems of all entries in 'First File'
        stems = self.df_bursts['First File'].apply(lambda fn: Path(fn).stem)

        # select bursts whose First File stem matches
        df_file = self.df_bursts[stems == curr_stem]
        if df_file.empty:
            return []

        # collect all photon indices over those bursts
        indices = set()
        for _, row in df_file.iterrows():
            start = int(row['First Photon'])
            stop = int(row['Last Photon'])
            indices.update(range(start, stop + 1))

        return sorted(indices)

    def read_burst_analysis(self, paris_path: Path, tttr_file_type, pattern='bur', row_stride=2):
        info_path = paris_path / 'Info'
        data_path = paris_path.parent
        dfs = []
        bur_files = sorted(paris_path.glob('**/*.bur'))
        if bur_files:
            for fn in bur_files:
                lines = open(fn).read().splitlines()
                headers = lines[0].split('	')
                data = [l.split('	') for l in lines[2::row_stride]]
                dfs.append(pd.DataFrame(data, columns=headers))
        if not dfs:
            raise ValueError(f"No burst files in {paris_path}")
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df = df.apply(lambda col: pd.to_numeric(col, errors='ignore'))

        tttrs = {}
        for ff in df['First File']:
            fn = data_path / ff
            key = Path(fn).stem
            if key not in tttrs:
                tttrs[key] = tttrlib.TTTR(str(fn), tttr_file_type)
        return df, tttrs


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
