import typing
import faulthandler
faulthandler.enable(all_threads=True)

from typing import Union

from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
import pyqtgraph as pg
import numpy as np
import pandas as pd

from guidata.widgets.dataframeeditor import DataFrameEditor

import json

import chisurf
import chisurf.gui.decorators
import chisurf.settings
import chisurf.gui.widgets.wizard


import collections.abc
from pathlib import Path
import tttrlib
from typing import Callable, Dict, Iterator


class LazyTTTRDict(collections.abc.MutableMapping):
    """
    A dict-like that maps a key (file‐stem) → TTTR object,
    but only calls tttrlib.TTTR(path, file_type) on first access.
    """
    def __init__(
        self,
        path_map: Dict[str, Path],
        file_type_getter: Callable[[], str]
    ):
        """
        Parameters
        ----------
        path_map : Dict[str, Path]
            Maps file‐stem (no extension) → full Path to .ptu/.ht3/etc.
        file_type_getter : () → str
            A zero‐argument callable returning current TTTR file‐type (e.g. self.tttr_file_type).
        """
        self._paths = path_map
        self._cache: Dict[str, tttrlib.TTTR] = {}
        self._file_type_getter = file_type_getter
        self._warning_shown = False

    def __getitem__(self, key: str) -> tttrlib.TTTR:
        if key not in self._paths:
            raise KeyError(f"No TTTR path for key {key!r}")
        if key not in self._cache:
            path = self._paths[key]
            # Check if _file_type_getter is None
            if self._file_type_getter is None:
                if not self._warning_shown:
                    QMessageBox.warning(
                        None,
                        "Warning",
                        "The file type getter is None. This may cause issues with TTTR file loading."
                    )
                    self._warning_shown = True
                # Use a default file type or try to infer it
                file_type = tttrlib.inferTTTRFileType(str(path))
            else:
                file_type = self._file_type_getter()
            # instantiate on first use
            self._cache[key] = tttrlib.TTTR(str(path), file_type)
        return self._cache[key]

    def __setitem__(self, key: str, value: tttrlib.TTTR):
        # allow manual override if you really want
        self._cache[key] = value

    def __delitem__(self, key: str):
        self._paths.pop(key, None)
        self._cache.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def add_path(self, key: str, path: Path):
        """
        Register a new TTTR file to be loaded on demand.
        """
        self._paths[key] = path

    def clear(self):
        self._paths.clear()
        self._cache.clear()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Convert to list (or you could serialize differently)
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return super().default(obj)



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

    def _init_channels_from_wizard(self):
        """
        Called whenever the DetectorWizardPage has a new set of detectors;
        populates IRF/BG selectors, window combobox, and per-channel state.
        """
        dets = list(self.channel_definer.detectors.keys())
        chisurf.logging.info('_init_channels_from_wizard')
        # reset our per-channel state cache
        self.channel_settings.clear()

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

        # finally, run the same logic as update_selected_window()
        self.update_selected_window(detectors="all")

    def _on_tab_changed(self, index: int):
        if self.tabWidget.widget(index) is self.tab_parameters:
            current_idx = self.spinBox_current_file_idx.value()
            self.spinBox_current_file_idx.blockSignals(True)
            self.spinBox_current_file_idx.setValue(current_idx)
            self.spinBox_current_file_idx.blockSignals(False)
            self.update_current_file(current_idx)
            self.update_bg_files()

        if self.tabWidget.widget(index) is self.tab_process:
            # Display settings
            chisurf.logging.info("Update: tab_process")
            self.plainTextEdit.setReadOnly(True)
            s = json.dumps(self.channel_settings, indent=1, cls=NumpyEncoder)
            self.plainTextEdit.setPlainText(s)

    def _update_max_bins_from_tttr(self):
        try:
            tttr = next(iter(self.tttrs.values()))
            total_channels = tttr.header.number_of_micro_time_channels
        except Exception:
            return

        max_bins: int = total_channels // self.micro_time_binning
        self.micro_time_range = 0, max_bins
        self.full_range = (0, max_bins)

    def _capture_current_ui_state(self):
        chisurf.logging.info("_capture_current_ui_state")
        x0, fixed = self.fit_parameters
        start_bin, stop_bin = self.micro_time_range
        d = {
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
        self.spinBox_micro_time_start.setValue(state['micro_time_start'])
        self.spinBox_micro_time_stop.setValue(state['micro_time_stop'])
        self.spinBox_micro_time_binning.setValue(state['micro_time_binning'])

        # — IRF threshold & shifts —
        self.doubleSpinBox_irf_threshold.setValue(state['irf_threshold'])
        self.doubleSpinBox_shift.setValue(state['shift'])
        self.doubleSpinBox_shift_sp.setValue(state['shift_sp'])
        self.doubleSpinBox_shift_ss.setValue(state['shift_ss'])

        self.min_photons = state['min_photons']
        self.p2s_twoIstar = state['p2s_twoIstar']
        self.BIFL_scatter = state['BIFL_scatter']

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
            for sb in (self.spinBox_micro_time_start, self.spinBox_micro_time_stop):
                sb.blockSignals(True)
            self.micro_time_range = (bin_start, bin_stop)
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

    def _update_hist_files(
        self,
        widgets_dict: dict,
        np_dict: dict,
        one_for_all: bool,
        normalize: int,
        threshold: float = -1,
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
            print('det:', det)
            fw = widgets_dict.get(det)
            files: typing.List[Path] = fw.get_selected_files() if fw else []
            print('files:', files)

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
                    threshold=threshold
                )
                arr = np.sum(jordi, axis=0)
                np_dict[det] = arr

                if state_key is not None:
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

        # Detector definition tab
        self.tab_detector = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tab_detector, "Detector Definition")
        self.verticalLayout_detector_tab = QtWidgets.QVBoxLayout(self.tab_detector)
        self.channel_definer = chisurf.gui.widgets.wizard.DetectorWizardPage(parent=self)
        self.groupBox_detector = QtWidgets.QGroupBox("Detector Configuration")
        self.verticalLayout_detector = QtWidgets.QVBoxLayout(self.groupBox_detector)
        self.verticalLayout_detector.addWidget(self.channel_definer)
        self.verticalLayout_detector_tab.addWidget(self.groupBox_detector)
        # ← as soon as the user finishes defining detectors, re-build all our channel UIs

        # Prepare irf, bg widgets
        self._prepare_irf_bg_widgets()

        # Inspect a single burst, as chosen by the burst_idx property
        self.spinBox_burst_idx.valueChanged.connect(lambda val: self.inspect_bursts(val, embed=True))

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

        # Decay plot selected burst
        self.groupBox_burst_plot = QtWidgets.QGroupBox("Individual selection")
        self.verticalLayout_4.addWidget(self.groupBox_burst_plot)
        self.verticalLayout_burst_plot = QtWidgets.QVBoxLayout(self.groupBox_burst_plot)

        # this GraphicsLayoutWidget will hold one PlotItem per detector
        self.burst_layout = pg.GraphicsLayoutWidget()
        self.verticalLayout_burst_plot.addWidget(self.burst_layout)

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
        self.toolButton_save_settings.clicked.connect(self.save_settings)
        self.toolButton_load_settings.clicked.connect(self.load_settings)

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
            self.doubleSpinBox_l2
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
            self.checkBox_2IStar,
            self.checkBox_BIFL_scatter
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

        # whenever the user finishes (or re-configures) the DetectorWizardPage,
        # rebuild all the IRF/BG lists and window combobox
        self.channel_definer.detectorsChanged.connect(self._init_channels_from_wizard)

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
        self.spinBox_min_photons.setValue(10)
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
            self.df_bursts = None
            self.tttrs.clear()  # Properly clear the LazyTTTRDict
            self._tttr_paths.clear()  # Clear the paths dictionary
            self.burst_files_list.clear()
            # Reset the comboBox_tttr_file_type to Auto (first index)
            self.comboBox_tttr_file_type.setCurrentIndex(0)
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
            threshold=self.irf_threshold,
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
        else:
            paris = files[0].parent
            if self.df_bursts is None:
                self.df_bursts = pd.DataFrame()
            df, tttrs = self.read_burst_analysis(paris.parent)
            self.tttrs = tttrs
            self.df_bursts = pd.concat([self.df_bursts, df], ignore_index=True, sort=False)

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
                        self.doubleSpinBox_excitation_period.setValue(excitation_period)
                        chisurf.logging.info(f"Updated excitation period to {excitation_period} ns based on repetition rate {repetition_rate} MHz")
                except (AttributeError, ValueError) as e:
                    chisurf.logging.info(f"Could not extract repetition rate from header: {e}")

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
        # once we know which TTTR is loaded, recompute effective dt
        self.update_dt()

    def on_micro_time_range_changed(self, _=None):
        # existing work: clear hist, rebuild IRF/BG, update decay & fit…
        self.update_irf_files(detector=[self.current_detector])
        # self.update_bg_files(detector=[self.current_detector])
        # self.update_decay_of_detector()
        self._fit = None
        self.update_fit()

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

        # 2) Filter df_bursts to just its rows
        df_this = self.df_bursts[self.df_bursts["burst_file"] == curr_bur]
        if df_this.empty:
            chisurf.logging.info(f"No bursts found for {curr_bur!r}")
            return

        # 3) Now grab the TTTR filename from the first row of that subset
        tttr_name = df_this.loc[df_this.index[0], "First File"]

        # 4) Load or retrieve the TTTR
        key = Path(tttr_name).stem
        chisurf.logging.info(f"Looking for TTTR with key: {key}")
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
            self.full_range,
            mt_bin,
            normalize_counts=-1,
            minlength=self.full_range[1]
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

    def update_dt(self):
        """
        Compute the per-channel Δt by pulling micro_time_resolution
        from any loaded TTTR header (they’re all identical),
        otherwise fall back to the manual dt spin-box.
        """
        binning = self.spinBox_micro_time_binning.value()

        if self.tttrs:
            # grab the first TTTR in our dict; all should share the same resolution
            tttr = next(iter(self.tttrs.values()))
            micro_res = tttr.header.micro_time_resolution * 1e9 # use nano seconds
            self.doubleSpinBox_dt.setValue(micro_res)
        else:
            micro_res = self.doubleSpinBox_dt.value()

        effective = micro_res * binning
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
        #self.spinBox_micro_time_start.setMinimum(start)
        #self.spinBox_micro_time_start.setMaximum(stop)
        #self.spinBox_micro_time_stop.setMinimum(start)
        #self.spinBox_micro_time_stop.setMaximum(stop)

    @property
    def micro_time_binning(self):
        return self.spinBox_micro_time_binning.value()

    @property
    def tttr_file_type(self):
        txt = self.comboBox_tttr_file_type.currentText()
        if txt == 'Auto':
            # Try to use the first tttr path from the LazyTTTRDict
            if self._tttr_paths:
                # Get the first tttr path from the LazyTTTRDict
                first_path = next(iter(self._tttr_paths.values()))
                if first_path:
                    file_type_int = tttrlib.inferTTTRFileType(str(first_path))
                    # Update comboBox_tttr_file_type if a file type is recognized
                    if file_type_int is not None and file_type_int >= 0:
                        # Get the list of supported container names
                        container_names = tttrlib.TTTR.get_supported_container_names()
                        if 0 <= file_type_int < len(container_names):
                            # Add 1 to the index to account for 'Auto' at index 0
                            idx = file_type_int + 1
                            if 0 <= idx < self.comboBox_tttr_file_type.count():
                                self.comboBox_tttr_file_type.setCurrentIndex(idx)
                    return file_type_int

            # Fall back to current_filename if no tttr paths are available
            filename = self.current_filename
            if filename:
                file_type_int = tttrlib.inferTTTRFileType(filename)
                # Update comboBox_tttr_file_type if a file type is recognized
                if file_type_int is not None and file_type_int >= 0:
                    # Get the list of supported container names
                    container_names = tttrlib.TTTR.get_supported_container_names()
                    if 0 <= file_type_int < len(container_names):
                        # Add 1 to the index to account for 'Auto' at index 0
                        idx = file_type_int + 1
                        if 0 <= idx < self.comboBox_tttr_file_type.count():
                            self.comboBox_tttr_file_type.setCurrentIndex(idx)
                return file_type_int
            return None
        return txt

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
        chisurf.logging.info(f"property irf.shape1: {arr.shape}")

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
        irf = np.hstack([sp, ss])
        chisurf.logging.info(f"property irf.shape2: {irf.shape}")
        return irf

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
        sb, eb = self.micro_time_range
        # basic params
        dt     = self.dt_effective
        period = self.excitation_period
        gf     = self.g_factor
        l1     = self.l1
        l2     = self.l2
        irf = self.irf
        bg = self.bg

        half = irf.size // 2
        irf = np.hstack([irf[:half][sb:eb],irf[half:][sb:eb]])
        bg = np.hstack([bg[:half][sb:eb],bg[half:][sb:eb]])

        # finally, build the fit
        fit = tttrlib.Fit23(
            dt=dt,
            irf=irf,
            background=bg,
            period=period,
            g_factor=gf,
            l1=l1,
            l2=l2,
            p2s_twoIstar_flag = self.p2s_twoIstar,
            soft_bifl_scatter_flag = self.BIFL_scatter
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

        half = decay.size // 2
        d = np.hstack([decay[:half][sb:eb],decay[half:][sb:eb]])
        res = self.fit(data=d, initial_values=x0, fixed=fixed)
        self.plot_fit_result(res)

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

        # plot data and model in the bottom panel
        self.combined_plot.plot(self.fit.data,
                                pen=None,
                                symbol='o',
                                symbolSize=3)
        self.combined_plot.plot(self.fit.model, pen='g')

        # Plot IRF & BG

        irf = self.irf
        half = irf.size // 2
        irf = np.hstack([irf[:half][sb:eb],irf[half:][sb:eb]])
        irf /= np.max(irf)
        irf *= max(self.fit.data)

        bg = self.bg
        bg = np.hstack([bg[:half][sb:eb],bg[half:][sb:eb]])
        bg *= self.total_burst_time_seconds

        self.combined_plot.plot(irf, pen='r', name='IRF')
        self.combined_plot.plot(bg , pen='b', name='Background')

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

    def process_bursts_new(self):
        if self.df_bursts is None or not self.tttrs:
            QtWidgets.QMessageBox.warning(self, "No Data", "No burst data loaded.")
            return

        # prepare progress bar
        total_bursts = len(self.df_bursts)
        self.progressBar.setRange(0, total_bursts)

        # cache IRF/BG per detector
        irf_cache = {}
        bg_cache = {}
        for det, info in self.channel_definer.detectors.items():
            st = self.channel_settings[det]
            sb, eb = st['micro_time_start'], st['micro_time_stop']
            half = st['irf'].size // 2
            irf_cache[det] = np.hstack([st['irf'][:half][sb:eb], st['irf'][half:][sb:eb]])
            bg_cache[det] = np.hstack([st['bg'][:half][sb:eb], st['bg'][half:][sb:eb]])

        # helper to emit a default-zero record
        metrics = ['2I*', 'Tau', 'gamma', 'r0', 'rho', 'BIFL scatter?', '2I*: P+2S?', 'r Scatter', 'r Experimental']
        results = []

        def default_record(fname, det, cp_sum=0, cs_sum=0):
            color = det.lower()
            rec = {
                'First File': fname,
                'Detector': det,
                'Ng-p-all': cp_sum,
                'Ng-s-all': cs_sum,
                f'Number of Photons (fit window) ({color})': 0
            }
            for m in metrics:
                rec[f'{m} ({color})'] = 0.0
            results.append(rec)

        # --- main loop ---
        for idx, row in self.df_bursts.iterrows():
            self.progressBar.setValue(idx + 1)
            QtWidgets.QApplication.processEvents()

            fname = row['First File']
            first_ph = int(row['First Photon'])
            last_ph  = int(row['Last Photon'])
            key      = Path(fname).stem
            tttr     = self.tttrs.get(key)

            bad_index = first_ph < 0 or last_ph < 0 or tttr is None

            # --- build a single mask of in-burst photons via bincount ---
            if bad_index:
                burst_tttr = None
            else:
                start = max(0, first_ph)
                stop  = min(last_ph, len(tttr) - 1)
                # +1 at start, -1 at stop+1 → cumsum > 0
                edges   = np.array([start, stop + 1], dtype=np.int64)
                weights = np.array([1, -1], dtype=np.int32)
                counts  = np.bincount(edges, weights, minlength=len(tttr) + 1)
                mask    = np.cumsum(counts)[:-1] > 0
                burst_tttr = tttr[np.nonzero(mask)[0]]

            # now one burst_tttr slice for all detectors
            for det, info in self.channel_definer.detectors.items():
                if bad_index or burst_tttr is None:
                    default_record(fname, det)
                    continue

                st     = self.channel_settings[det]
                sb, eb = st['micro_time_start'], st['micro_time_stop']
                mb     = st['micro_time_binning']
                irf    = irf_cache[det]
                bg     = bg_cache[det]
                pchs   = info['chs'][::2]
                schs   = info['chs'][1::2] if len(info['chs']) > 1 else info['chs']

                cp = (
                    self.filter_tttr(burst_tttr, [sb, eb], pchs)
                        .get_microtime_histogram(mb, minlength=self.micro_time_range[1])[0]
                        [sb:eb]
                )
                cs = (
                    self.filter_tttr(burst_tttr, [sb, eb], schs)
                        .get_microtime_histogram(mb, minlength=self.micro_time_range[1])[0]
                        [sb:eb]
                )
                cp_sum, cs_sum = int(cp.sum()), int(cs.sum())

                # length mismatch or too few photons → default
                if irf.size != cp.size + cs.size or not self.pass_photon_threshold(np.hstack([cp, cs])):
                    default_record(fname, det, cp_sum, cs_sum)
                    continue

                # perform fit
                fit = tttrlib.Fit23(
                    dt=st['dt'],
                    irf=irf,
                    background=bg,
                    period=st['excitation_period'],
                    g_factor=st['g_factor'],
                    l1=st['l1'],
                    l2=st['l2'],
                    p2s_twoIstar_flag = st['p2s_twoIstar'],
                    soft_bifl_scatter_flag = st['BIFL_scatter']
                )
                res = fit(data=np.hstack([cp, cs]), initial_values=st['initial_x0'], fixed=st['fixed_flags'])

                # build result record
                color = det.lower()
                rec = {
                    'First File': fname,
                    'Detector': det,
                    'Ng-p-all': cp_sum,
                    'Ng-s-all': cs_sum,
                    f'Number of Photons (fit window) ({color})': cp_sum + cs_sum,
                    f'2I* ({color})': res.get('twoIstar', 0.0),
                    f'Tau ({color})': res['x'][0],
                    f'gamma ({color})': res['x'][1],
                    f'r0 ({color})': res['x'][2],
                    f'rho ({color})': res['x'][3],
                    f'BIFL scatter? ({color})': res['x'][6],
                    f'2I*: P+2S? ({color})': res.get('P2S_ratio', 0.0),
                    f'r Scatter ({color})': res['x'][6],
                    f'r Experimental ({color})': res['x'][7],
                }
                results.append(rec)

        result_df = pd.DataFrame(results)
        chisurf.logging.info("results_df", result_df)

        # Write out per-file, per-detector
        written_dirs = set()
        for file_path in self.burst_files_list.get_selected_files():
            stem = file_path.stem
            for det in self.channel_definer.detectors:
                color = det.lower()
                letter = color[0]
                out_dir = file_path.parent.parent / f"b{letter}4"
                out_dir.mkdir(parents=True, exist_ok=True)

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

                settings_file = out_dir / 'channel_settings.json'
                with open(settings_file, 'w') as sf:
                    json.dump(self.channel_settings, sf, indent=4, cls=NumpyEncoder)
                written_dirs.add(out_dir.name)

        folder_list = ", ".join(sorted(written_dirs)) if written_dirs else "(no data)"
        QtWidgets.QMessageBox.information(
            self,
            "Done",
            f"Burst-fit results saved in folders: {folder_list}"
        )


    def process_bursts(self):
        if self.df_bursts is None or not self.tttrs:
            QtWidgets.QMessageBox.warning(self, "No Data", "No burst data loaded.")
            return

        # prepare progress bar
        total_bursts = len(self.df_bursts)
        self.progressBar.setRange(0, total_bursts)

        # cache IRF/BG per detector
        irf_cache = {}
        bg_cache = {}
        for det, info in self.channel_definer.detectors.items():
            st = self.channel_settings[det]
            sb, eb = st['micro_time_start'], st['micro_time_stop']
            half = st['irf'].size // 2
            irf_cache[det] = np.hstack([st['irf'][:half][sb:eb], st['irf'][half:][sb:eb]])
            bg_cache[det] = np.hstack([st['bg'][:half][sb:eb], st['bg'][half:][sb:eb]])

        # helper to emit a default-zero record
        metrics = ['2I* ', 'Tau', 'gamma', 'r0', 'rho', 'BIFL scatter?', '2I*: P+2S?', 'r Scatter', 'r Experimental']
        results = []

        tttrs = self.tttrs
        channel_settings = self.channel_settings
        channels = self.channel_definer.detectors.items()

        def default_record(fname, det, cp_sum=0, cs_sum=0):
            color = det.lower()
            rec = {
                'First File': fname,
                'Detector': det,
                'Ng-p-all': cp_sum,
                'Ng-s-all': cs_sum,
                f'Number of Photons (fit window) ({color})': cp_sum + cs_sum
            }
            for m in metrics:
                rec[f'{m} ({color})'] = 0
            results.append(rec)

        for idx, row in self.df_bursts.iterrows():
            self.progressBar.setValue(idx + 1)
            QtWidgets.QApplication.processEvents()

            fname = row['First File']
            first_ph = int(row['First Photon'])
            last_ph = int(row['Last Photon'])
            key = Path(fname).stem

            bad_index = first_ph < 0 or last_ph < 0
            tttr = tttrs.get(key)
            burst = None if bad_index else tttr[first_ph:last_ph]

            for det, info in channels:

                if bad_index:
                    default_record(fname, det, -1, -1)
                    continue

                st = channel_settings[det]
                sb, eb = st['micro_time_start'], st['micro_time_stop']
                mb = st['micro_time_binning']
                irf = irf_cache[det]
                bg = bg_cache[det]

                pchs = info['chs'][::2]
                schs = info['chs'][1::2] if len(info['chs']) > 1 else info['chs']

                cp = self.filter_tttr(burst, [sb, eb], pchs) \
                         .get_microtime_histogram(mb, minlength=self.micro_time_range[1])[0][sb:eb]
                cs = self.filter_tttr(burst, [sb, eb], schs) \
                         .get_microtime_histogram(mb, minlength=self.micro_time_range[1])[0][sb:eb]
                cp_sum, cs_sum = int(cp.sum()), int(cs.sum())
                decay = np.hstack([cp, cs])

                # length mismatch or too few photons → default
                decay_size = cp.size + cs.size
                if (irf.size != cp.size + cs.size):
                    chisurf.logging.info(f"det: {det} - IRF mismatch, irf.size: {irf.size}, decay_size: {decay_size}")
                    default_record(fname, det, cp_sum, cs_sum)
                    continue
                if (cp_sum + cs_sum < st['min_photons']):
                    chisurf.logging.info(f"Skip: {Path(fname).name} Burst: {idx} Detector: {det} NPh: {cp_sum + cs_sum} < {st['min_photons']}")
                    default_record(fname, det, cp_sum, cs_sum)
                    continue

                # perform fit
                fit = tttrlib.Fit23(
                    dt=st['dt'],
                    irf=irf,
                    background=bg,
                    period=st['excitation_period'],
                    g_factor=st['g_factor'],
                    l1=st['l1'],
                    l2=st['l2'],
                    p2s_twoIstar_flag=st['p2s_twoIstar'],
                    soft_bifl_scatter_flag=st['BIFL_scatter']
                )
                res = fit(data=decay, initial_values=st['initial_x0'], fixed=st['fixed_flags'])

                # build result record
                color = det.lower()
                rec = {
                    'First File': fname,
                    'Detector': det,
                    'Ng-p-all': cp_sum,
                    'Ng-s-all': cs_sum,
                    f'Number of Photons (fit window) ({color})': cp_sum + cs_sum,
                    f'2I*  ({color})': res.get('twoIstar', 0.0),
                    f'Tau ({color})': res['x'][0],
                    f'gamma ({color})': res['x'][1],
                    f'r0 ({color})': res['x'][2],
                    f'rho ({color})': res['x'][3],
                    f'BIFL scatter? ({color})': int(st['BIFL_scatter']),
                    f'2I*: P+2S? ({color})': int(st['p2s_twoIstar']),
                    f'r Scatter ({color})': res['x'][6],
                    f'r Experimental ({color})': res['x'][7],
                }
                results.append(rec)

        result_df = pd.DataFrame(results)

        files = self.burst_files_list.get_selected_files()
        dets = list(self.channel_definer.detectors.keys())
        total_tasks = len(files) * len(dets)
        progress = QProgressDialog("Saving burst-fit results...", "Cancel", 0, total_tasks, self)
        progress.setWindowTitle("Saving burst-fit results")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()
        current_task = 0

        # write out per-file, per-detector
        written_dirs = set()
        for file_path in files:
            stem = file_path.stem
            for det in dets:
                color = det.lower()
                letter = color[0]
                out_dir = file_path.parent.parent / f"b{letter}4"
                out_dir.mkdir(parents=True, exist_ok=True)

                mask = (
                        result_df['First File'].map(lambda fn: Path(fn).stem).eq(stem)
                        & result_df['Detector'].eq(det)
                )
                df_file = result_df[mask]
                if df_file.empty:
                    continue

                cols = [
                    'Ng-p-all', 'Ng-s-all',
                    f'Number of Photons (fit window) ({color})',
                    f'2I*  ({color})', f'Tau ({color})', f'gamma ({color})',
                    f'r0 ({color})', f'rho ({color})', f'BIFL scatter? ({color})',
                    f'2I*: P+2S? ({color})', f'r Scatter ({color})', f'r Experimental ({color})'
                ]
                zero_template = {c: 0.0 for c in cols}
                records = []
                for _, data_row in df_file.iterrows():
                    records.append(zero_template.copy())
                    records.append({c: data_row[c] for c in cols})
                records.append(zero_template.copy())

                df_out = pd.DataFrame(records)
                out_file = out_dir / f"{stem}.b{letter}4"
                with open(out_file, 'w', newline='') as f:
                    f.write('\t'.join(cols) + '\t\n')
                    df_out.to_csv(
                        f,
                        sep='\t',
                        index=False,
                        header=False,
                        float_format='%.6f',
                        columns=cols
                    )

                settings_file = out_dir / 'channel_settings.json'
                with open(settings_file, 'w') as sf:
                    json.dump(self.channel_settings, sf, indent=4, cls=NumpyEncoder)
                written_dirs.add(out_dir.name)

                current_task += 1
                progress.setValue(current_task)
                QtWidgets.QApplication.processEvents()
                if progress.wasCanceled():
                    progress.close()
                    QMessageBox.information(self, "Canceled", "Save operation was canceled.")
                    return

        progress.close()
        folder_list = ", ".join(sorted(written_dirs)) if written_dirs else "(no data)"
        QMessageBox.information(self, "Done", f"Burst-fit results saved in folders: {folder_list}")

    def make_jordi(
        self,
        tttr_list: typing.List[tttrlib.TTTR],
        detector_chs: typing.List[int],
        micro_time_range: typing.List[int],
        micro_time_binning: int,
        save_files: bool = False,
        normalize_counts: int = 1,
        threshold: float = -1,
        minlength: int = -1
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
            cp = tp.get_microtime_histogram(micro_time_binning, minlength=minlength)[0][start_bin:stop_bin]
            cs = ts.get_microtime_histogram(micro_time_binning, minlength=minlength)[0][start_bin:stop_bin]

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

        # 2) Sample first file to infer which cols are numeric
        sample = pd.read_csv(
            bur_files[0],
            sep="\t",
            header=0,
            skiprows=[1],
            nrows=100,
            engine="c",
            low_memory=False
        )
        num_cols = sample.select_dtypes(include="number").columns
        dtype_spec = {col: "float64" for col in num_cols}
        dtype_spec["First File"] = "string"

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
            df_part["burst_file"] = fn.name
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

    def save_settings(self):
        """
        Dump TTTR file type, per‐channel settings, AND the DetectorWizardPage settings
        into a single JSON, and show the path in lineEdit_settings_file.
        """
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save All Settings",
            str(Path.home() / "mle_wizard_settings.json"),
            "JSON Files (*.json)"
        )
        if not path:
            return

        payload = {
            "tttr_file_type": self.tttr_file_type,
            "channel_settings": self.channel_settings,
            "detector_settings": self.channel_definer.settings,
            "micro_time_binning": self.micro_time_binning,
        }

        try:
            with open(path, 'w') as f:
                json.dump(payload, f, indent=4, cls=NumpyEncoder)
            # keep the file path visible
            self.lineEdit_settings_file.setText(path)
            # also update the DetectorWizardPage’s own line‐edit
            self.channel_definer.file_path_line_edit.setText(path)
            QMessageBox.information(self, "Saved", f"All settings saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save settings:\n{e}")

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

        payload = {
            "tttr_file_type": self.comboBox_tttr_file_type.currentText(),
            "channel_settings": self.channel_settings,
            "detector_settings": detwiz.get_settings(),
            "micro_time_binning": self.micro_time_binning
        }

        with open(path, 'w') as f:
            json.dump(payload, f, indent=4, cls=NumpyEncoder)

        # show the file in both line edits
        self.lineEdit_settings_file.setText(path)
        detwiz.file_path_line_edit.setText(path)
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
            self.spinBox_micro_time_binning.setValue(int(mtb))

        # 2) TTTR file‐type
        tttr_type = payload.get("tttr_file_type", "Auto")
        idx = self.comboBox_tttr_file_type.findText(tttr_type)
        if idx != -1:
            self.comboBox_tttr_file_type.setCurrentIndex(idx)

        # 3) reflect path
        self.lineEdit_settings_file.setText(path)
        #self.channel_definer.file_path_line_edit.setText(path)

        # 4) channel_settings
        self.channel_settings = payload.get("channel_settings", {})

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
            for w in (
                self.spinBox_micro_time_start,
                self.spinBox_micro_time_stop,
                self.spinBox_micro_time_binning,
                self.doubleSpinBox_irf_threshold,
                self.doubleSpinBox_shift,
                self.doubleSpinBox_shift_sp,
                self.doubleSpinBox_shift_ss,
            ):
                w.blockSignals(True)
            self._apply_ui_state(state)
            for w in (
                self.spinBox_micro_time_start,
                self.spinBox_micro_time_stop,
                self.spinBox_micro_time_binning,
                self.doubleSpinBox_irf_threshold,
                self.doubleSpinBox_shift,
                self.doubleSpinBox_shift_sp,
                self.doubleSpinBox_shift_ss,
            ):
                w.blockSignals(False)

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
