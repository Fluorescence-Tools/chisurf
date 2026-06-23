import os
import pathlib
import typing
import zipfile
import shutil
from datetime import datetime

import tttrlib
import json
import time
import numpy as np
import pandas as pd

import pyqtgraph as pg
import matplotlib

from chisurf import logging
import chisurf.core.fio as io
import chisurf.core.fio.fluorescence
import chisurf.core.math
import chisurf.gui.decorators
import chisurf.core.fluorescence.burst
from chisurf.gui import QtGui, QtWidgets, QtCore, uic
from chisurf.core.math.signal import fill_small_gaps_in_array
from chisurf.core.settings.path_utils import get_path
from chisurf.core.settings.file_utils import safe_open_file
from ..tttr_channeldefinition.tttr_detector_setups import save_detector_setups, load_detector_setups
from .tttr_photon_filter_support import CommaSeparatedIntegersValidator
from .tttr_photon_filter_mode import install_filter_mode_visibility
from .tttr_photon_filter_file_drop import install_file_drop
from .tttr_photon_filter_plots import create_plots, place_plots
from .tttr_photon_filter_connections import setup_connections as _setup_connections
from chisurf.gui.widgets.progress import EnhancedProgressDialog
from chisurf.core.fluorescence.burst.utils import create_array_with_ones


colors = chisurf.core.settings.gui['plot']['colors']


class WizardTTTRPhotonFilter(QtWidgets.QWizardPage):
    """
    WizardTTTRPhotonFilter loads TTTR files and applies photon filtering
    based on count-rate or burst-search criteria, channel selection,
    microtime ranges, and optional gap-filling.

    Parameters
    ----------
    windows : dict
        Dictionary of predefined PIE windows. Passed to the relevant UI comboBox.
    detectors : dict
        Dictionary of predefined detectors. Passed to the relevant UI comboBox.
    show_dT : bool, default=True
        Whether to show the dT plot (between consecutive photons).
    show_filter : bool, default=True
        Whether to show the filter/selection plot.
    show_mcs : bool, default=True
        Whether to show the intensity trace (MCS) plot.
    show_decay : bool, default=True
        Whether to show the decay plot (microtime histogram).
    show_burst : bool, default=True
        Whether to show the burst histogram plot.
    default_dT_min : float, default=0.0001
        Initial lower bound for delta macro-time filter (in ms).
    default_dT_max : float, default=0.15
        Initial upper bound for delta macro-time filter (in ms).
    use_dT_min : bool, default=True
        Whether the lower bound of delta macro-time filter is active initially.
    use_dT_max : bool, default=True
        Whether the upper bound of delta macro-time filter is active initially.
    default_photon_threshold : int, default=160
        Initial threshold for count rate or burst search.
    default_count_rate_window_ms : float, default=1.0
        Initial time window (ms) for count-rate based filtering.
    invert_filter : bool, default=False
        Whether to invert the filter initially (applies to all filter types).
    default_filter_mode : str, default='count_rate'
        Selects which filter mode is active at initialization.
        Valid options: 'count_rate', 'burst', 'bocpd', or 'kalman'.
        - If 'count_rate', the "Count rate" option is selected in the combobox.
        - If 'burst', the "Burst" option is selected in the combobox.
        - If 'bocpd', the "BOCPD Burst" option is selected in the combobox.
        - If 'kalman', the "Kalman Burst" option is selected in the combobox.

    Notes
    -----
    The filter mode can also be changed by the user at runtime. The
    current mode can be queried from the `used_filter` property.
    """

    # Signal for status messages to be displayed in the main window's statusbar
    status_message = QtCore.Signal(str, int)

    @property
    def photon_number_threshold(self):
        return self.spinBox.value()

    @property
    def target_path(self) -> pathlib.Path:
        return pathlib.Path(self.lineEdit_2.text())

    @property
    def filename(self) -> pathlib.Path:
        return pathlib.Path(self.lineEdit.text())

    @property
    def filetype(self) -> str | None:
        setup_name = self.comboBox.currentText()
        if not setup_name or setup_name == "No setups available":
            # Display warning message if no setup is selected
            QtWidgets.QMessageBox.warning(
                self,
                "No Setup Selected",
                "Please define a setup first in the Detector Configuration page."
            )
            return None

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the file type from the setup's tttr_reading section
            setup_data = setups["setups"][setup_name]
            if "tttr_reading" in setup_data and "file_type" in setup_data["tttr_reading"]:
                file_type = setup_data["tttr_reading"]["file_type"]

                # If file type is Auto, try to infer from current file
                if file_type == 'Auto':
                    current_file = self.current_tttr_filename
                    if current_file and pathlib.Path(current_file).exists():
                        file_type_int = tttrlib.inferTTTRFileType(current_file)
                        if file_type_int is not None and file_type_int >= 0:
                            container_names = tttrlib.TTTR.get_supported_container_names()
                            if 0 <= file_type_int < len(container_names):
                                return container_names[file_type_int]
                    return None
                return file_type

        # If setup doesn't exist or doesn't have file type, return None without showing warning
        # This prevents the error message from appearing on startup
        return None

    @property
    def trace_bin_width(self) -> float:
        return float(self.doubleSpinBox_4.value())

    @property
    def use_lower(self) -> bool:
        return bool(self.checkBox_2.isChecked())

    @property
    def use_upper(self) -> bool:
        return bool(self.checkBox_3.isChecked())

    @property
    def use_gap_fill(self):
        return bool(self.checkBox_5.isChecked())

    @property
    def plot_min(self):
        return self.spinBox_2.value()

    @property
    def plot_max(self):
        return self.spinBox_3.value()

    @plot_max.setter
    def plot_max(self, v):
        self.spinBox_3.blockSignals(True)
        self.spinBox_3.setValue(v)
        self.spinBox_3.blockSignals(False)

    @property
    def current_tttr_filename(self):
        v = self.spinBox_4.value()
        try:
            filenames = self.settings.get('tttr_filenames', [])
        except Exception:
            filenames = []
        if 0 <= v < len(filenames):
            return filenames[v]
        else:
            # Return empty string for safe use in setText and truthiness checks
            return ""

    @property
    def decay_coarse(self):
        return self.spinBox_5.value()

    @property
    def max_gap(self):
        if not self.use_gap_fill:
            return 0
        return self.spinBox_7.value()

    @property
    def number_of_burst_bins(self):
        return self.spinBox_6.value()

    @property
    def channels(self) -> typing.List[int]:
        s = self.lineEdit_4.text()
        if len(s) > 0:
            return [int(x) for x in s.split(',')]
        return []

    @property
    def min_ph(self):
        """
        The minimum number of photons for burst detection (or any other usage).
        This property gets the current spinBox value.
        """
        return self.spinBox.value()

    @min_ph.setter
    def min_ph(self, value):
        """
        Sets the spinBox value to the specified integer/float.
        """
        self.spinBox.setValue(value)

    @property
    def ph_window(self):
        """
        The photon window size for burst detection (or other usage).
        This property gets the current spinBox_8 value.
        """
        return self.spinBox_8.value()

    @ph_window.setter
    def ph_window(self, value):
        """
        Sets the spinBox_8 value to the specified integer/float.
        """
        self.spinBox_8.setValue(value)

    @property
    def bocpd_prior_count(self):
        """
        The prior photon count before data for BOCPD Gamma prior.
        This property gets the current doubleSpinBox_5 value.
        """
        return self.doubleSpinBox_5.value()

    @bocpd_prior_count.setter
    def bocpd_prior_count(self, value):
        """
        Sets the doubleSpinBox_5 value to the specified float.
        """
        self.doubleSpinBox_5.setValue(value)

    # For backward compatibility
    @property
    def bocpd_alpha(self):
        """
        Deprecated: Use bocpd_prior_count instead.
        The alpha parameter for BOCPD Gamma prior.
        """
        return self.bocpd_prior_count

    @bocpd_alpha.setter
    def bocpd_alpha(self, value):
        """
        Deprecated: Use bocpd_prior_count instead.
        Sets the alpha parameter for BOCPD Gamma prior.
        """
        self.bocpd_prior_count = value

    @property
    def bocpd_prior_duration(self):
        """
        The time window assumed for prior_count for BOCPD Gamma prior.
        This property gets the current doubleSpinBox_6 value.
        """
        return self.doubleSpinBox_6.value()

    @bocpd_prior_duration.setter
    def bocpd_prior_duration(self, value):
        """
        Sets the doubleSpinBox_6 value to the specified float.
        """
        self.doubleSpinBox_6.setValue(value)

    # For backward compatibility
    @property
    def bocpd_beta(self):
        """
        Deprecated: Use bocpd_prior_duration instead.
        The beta parameter for BOCPD Gamma prior.
        """
        return self.bocpd_prior_duration

    @bocpd_beta.setter
    def bocpd_beta(self, value):
        """
        Deprecated: Use bocpd_prior_duration instead.
        Sets the beta parameter for BOCPD Gamma prior.
        """
        self.bocpd_prior_duration = value

    @property
    def bocpd_changepoint_prob(self):
        """
        The probability of burst start in any bin for BOCPD.
        This property gets the current doubleSpinBox_7 value.
        """
        return self.doubleSpinBox_7.value()

    @bocpd_changepoint_prob.setter
    def bocpd_changepoint_prob(self, value):
        """
        Sets the doubleSpinBox_7 value to the specified float.
        """
        self.doubleSpinBox_7.setValue(value)

    # For backward compatibility
    @property
    def bocpd_hazard(self):
        """
        Deprecated: Use bocpd_changepoint_prob instead.
        The hazard rate (probability of change point) for BOCPD.
        """
        return self.bocpd_changepoint_prob

    @bocpd_hazard.setter
    def bocpd_hazard(self, value):
        """
        Deprecated: Use bocpd_changepoint_prob instead.
        Sets the hazard rate (probability of change point) for BOCPD.
        """
        self.bocpd_changepoint_prob = value

    @property
    def cusum_bg_rate(self):
        """The background rate parameter for CUSUM filter.
        Uses doubleSpinBox_5.
        """
        return self.doubleSpinBox_5.value()

    @cusum_bg_rate.setter
    def cusum_bg_rate(self, value):
        self.doubleSpinBox_5.setValue(value)

    @property
    def cusum_sb_ratio(self):
        """The signal-to-background ratio parameter for CUSUM filter.
        Uses doubleSpinBox_6.
        """
        return self.doubleSpinBox_6.value()

    @cusum_sb_ratio.setter
    def cusum_sb_ratio(self, value):
        self.doubleSpinBox_6.setValue(value)

    @property
    def cusum_alpha(self):
        """The false alarm probability parameter for CUSUM filter.
        Uses doubleSpinBox_7.
        """
        return self.doubleSpinBox_7.value()

    @cusum_alpha.setter
    def cusum_alpha(self, value):
        self.doubleSpinBox_7.setValue(value)

    @property
    def cusum_beta(self):
        """The missed detection probability parameter for CUSUM filter.
        Uses doubleSpinBox_8.
        """
        return self.doubleSpinBox_8.value()

    @cusum_beta.setter
    def cusum_beta(self, value):
        self.doubleSpinBox_8.setValue(value)

    @property
    def kalman_q(self):
        """
        The process noise parameter for Kalman filter.
        This property gets the current doubleSpinBox_8 value.
        """
        return self.doubleSpinBox_8.value()

    @kalman_q.setter
    def kalman_q(self, value):
        """
        Sets the doubleSpinBox_8 value to the specified float.
        """
        self.doubleSpinBox_8.setValue(value)

    @property
    def kalman_r_scale(self):
        """
        The measurement noise scaling parameter for Kalman filter.
        This property gets the current doubleSpinBox_9 value.
        """
        return self.doubleSpinBox_9.value()

    @kalman_r_scale.setter
    def kalman_r_scale(self, value):
        """
        Sets the doubleSpinBox_9 value to the specified float.
        """
        self.doubleSpinBox_9.setValue(value)

    @property
    def kalman_z_thresh(self):
        """
        The threshold for burst detection in Kalman filter.
        This property gets the current doubleSpinBox_10 value.
        """
        return self.doubleSpinBox_10.value()

    @kalman_z_thresh.setter
    def kalman_z_thresh(self, value):
        """
        Sets the doubleSpinBox_10 value to the specified float.
        """
        self.doubleSpinBox_10.setValue(value)

    @property
    def kalman_min_len(self):
        """
        The minimum burst length in bins for Kalman filter.
        This property gets the current spinBox_9 value.
        """
        return self.spinBox_9.value()

    @kalman_min_len.setter
    def kalman_min_len(self, value):
        """
        Sets the spinBox_9 value to the specified integer.
        """
        self.spinBox_9.setValue(value)

    @property
    def kalman_merge_gap(self):
        """
        The maximum gap between bursts to merge them in Kalman filter.
        This property gets the current spinBox_7 value.
        """
        return self.spinBox_7.value()

    @kalman_merge_gap.setter
    def kalman_merge_gap(self, value):
        """
        Sets the spinBox_7 value to the specified integer.
        """
        self.spinBox_7.setValue(value)

    @property
    def microtime_ranges(self) -> typing.Optional[typing.List[typing.Tuple[int, int]]]:
        s = self.lineEdit_5.text()

        # Check if the input string is empty
        if not s:
            return None

        try:
            text = str(s).strip()
            if not text:
                return None

            # Allow both ';' and ',' as range separators.
            segments = []
            for item in text.replace(',', ';').split(';'):
                item = item.strip()
                if item:
                    segments.append(item)

            if not segments:
                chisurf.logging.log(0, "::microtime_ranges: No usable ranges after parsing.")
                return None

            ranges: typing.List[typing.Tuple[int, int]] = []
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue

                # Support ':' or '-' as min-max separator while allowing negative values.
                if ':' in seg:
                    a_txt, b_txt = seg.split(':', 1)
                else:
                    pos = seg.rfind('-')
                    if pos <= 0:
                        a_txt = seg
                        b_txt = seg
                    else:
                        a_txt = seg[:pos]
                        b_txt = seg[pos + 1 :]

                a = int(a_txt.strip())
                b = int(b_txt.strip())
                if a <= b:
                    ranges.append((a, b))
                else:
                    ranges.append((b, a))

            return ranges if ranges else None

        except (ValueError, TypeError):
            chisurf.logging.log(1, "::microtime_ranges: Invalid values in microsecond ranges.")
            return None

    @property
    def dT(self):
        if isinstance(self.tttr, tttrlib.TTTR):
            mT = self.tttr.get_macro_times()
            d = np.diff(mT, prepend=mT[0])
            h = self.tttr.header
            d = d * h.macro_time_resolution * 1000.0
            return d

    @property
    def dT_min(self) -> float:
        return self._dT_min

    @property
    def dT_max(self) -> float:
        return self._dT_max

    @property
    def used_filter(self):
        """
        Returns the currently active filter mode based on comboBox_burst_filter selection:
        - 'count_rate' if "Count rate" is selected
        - 'burst' if "Burst" is selected
        - 'bocpd' if "BOCPD Burst" is selected
        - 'kalman' if "Kalman Burst" is selected
        """
        current_text = self.comboBox_burst_filter.currentText()
        if current_text == "Count rate":
            return 'count_rate'
        elif current_text == "Burst":
            return 'burst'
        elif current_text == "BOCPD Burst":
            return 'bocpd'
        elif current_text == "Kalman Burst":
            return 'kalman'
        elif current_text == "CUSUM Burst":
            return 'cusum'

    @property
    def selected(self):
        """
        Returns a boolean mask (as np.uint8 array) of selected photons
        based on all applied filters: channel/microtime filter,
        dT thresholding, count-rate filter (if active),
        or burst filter (if active), plus optional gap-filling.
        """
        start_time = time.time()
        dT = self.dT
        tttr = self.tttr
        if dT is None:
            return list()

        s = np.ones_like(dT, dtype=bool)
        chs = self.channels
        if len(chs) > 0:
            mask = tttrlib.TTTRMask()
            mask.select_channels(tttr, chs, mask=True)
            m = mask.get_mask()
            s = np.logical_and(s, m)

        if self.microtime_ranges:
            mask = tttrlib.TTTRMask()
            mask.select_microtime_ranges(tttr, self.microtime_ranges)
            mask.flip()
            m = mask.get_mask()
            s = np.logical_and(s, m)

        if self.use_lower:
            s = np.logical_and(s, dT >= self.dT_min)
        if self.use_upper:
            s = np.logical_and(s, dT <= self.dT_max)

        # Apply filter depending on combobox selection and filter_active setting
        if not self.settings.get('filter_active', True):
            # If filter is not active, skip filter application
            pass
        elif self.used_filter == 'count_rate':
            filter_options = self.settings['count_rate_filter']
            selection_idx = chisurf.core.fluorescence.burst.count_rate_filter(
                tttr=self.tttr,
                n_ph_max=filter_options['n_ph_max'],
                time_window=filter_options['time_window'],
                invert=self.settings.get('invert_filter', False),
                make_mask=True
            )
            s = np.logical_and(s, selection_idx >= 0)

        elif self.used_filter == 'burst':
            min_ph = self.min_ph
            ph_window = self.ph_window
            tw = self.dT_max / 1000.0
            sel = chisurf.core.fluorescence.burst.burst_filter(
                tttr=tttr,
                min_ph=min_ph,
                ph_window=ph_window,
                time_window=tw
            )
            s = np.logical_and(s, sel)

        elif self.used_filter == 'bocpd':
            # Get channels
            channel_list = self.channels

            # Get macro times and routing channels
            macro_times = tttr.macro_times
            time_unit = tttr.header.macro_time_resolution
            timestamps = macro_times * time_unit  # Convert to seconds
            channels = tttr.routing_channels

            # If no channels are selected, use all available channels
            if len(channel_list) < 1:
                channel_list = tttr.get_used_routing_channels()

            # Extract timestamps for each channel
            timestamps_list = []
            for channel in channel_list:
                channel_timestamps = timestamps[channels == channel]
                timestamps_list.append(channel_timestamps)

                if len(channel_timestamps) == 0:
                    chisurf.logging.log(1, f"No photons found for channel {channel}")
                    return s.astype(dtype=np.uint8)

            # Get BOCPD parameters from UI
            prior_count = self.bocpd_prior_count
            prior_duration = self.bocpd_prior_duration
            changepoint_prob = self.bocpd_changepoint_prob
            min_counts = self.min_ph  # Use min_ph as min_counts
            max_run = 256  # Default max run length

            # Run BOCPD burst detection with multiple channels
            bursts, _, _, _, _ = chisurf.core.fluorescence.burst.bocpd_burst_detection_multi(
                timestamps_list,
                dt=self.trace_bin_width / 1000.0,  # bin width from UI trace_bin_width
                prior_count=prior_count,
                prior_duration=prior_duration,
                changepoint_prob=changepoint_prob,
                max_run=max_run,
                min_counts=min_counts
            )

            # Convert bursts to start-stop indices
            start_stop = chisurf.core.fluorescence.burst.bocpd.convert_bursts_to_start_stop(bursts, tttr)

            if len(start_stop) == 0:
                return s.astype(dtype=np.uint8)

            # Create mask
            n = len(tttr)
            sel = create_array_with_ones(start_stop, n)
            s = np.logical_and(s, sel)

        elif self.used_filter == 'kalman':
            # Get channels
            channel_list = self.channels

            # Get macro times and routing channels
            macro_times = tttr.macro_times
            time_unit = tttr.header.macro_time_resolution
            timestamps = macro_times * time_unit  # Convert to seconds
            channels = tttr.routing_channels

            # If no channels are selected, use all available channels
            if len(channel_list) < 1:
                channel_list = tttr.get_used_routing_channels()

            # Extract timestamps for each channel
            timestamps_list = []
            for channel in channel_list:
                channel_timestamps = timestamps[channels == channel]
                timestamps_list.append(channel_timestamps)

                if len(channel_timestamps) == 0:
                    chisurf.logging.log(1, f"No photons found for channel {channel}")
                    return s.astype(dtype=np.uint8)

            # Get Kalman filter parameters
            q = self.kalman_q
            r_scale = self.kalman_r_scale
            z_thresh = self.kalman_z_thresh
            min_len = self.kalman_min_len
            merge_gap = self.kalman_merge_gap
            min_counts = self.min_ph  # Use min_ph as min_counts

            # Run Kalman filter burst detection with multiple channels
            bursts, _, _, _, _ = chisurf.core.fluorescence.burst.kalman_burst_detection_multi(
                timestamps_list,
                dt=self.trace_bin_width / 1000.0,  # bin width from UI trace_bin_width
                q=q,
                r_scale=r_scale,
                z_thresh=z_thresh,
                min_len=min_len,
                merge_gap=merge_gap,
                min_counts=min_counts
            )

            # Convert bursts to start-stop indices
            start_stop = chisurf.core.fluorescence.burst.kalman.convert_bursts_to_start_stop(bursts, tttr)

            if len(start_stop) == 0:
                return s.astype(dtype=np.uint8)

            # Create mask
            n = len(tttr)
            sel = create_array_with_ones(start_stop, n)
            s = np.logical_and(s, sel)

        elif self.used_filter == 'cusum':
            # Get CUSUM parameters from UI
            min_ph = self.min_ph
            bg_rate = int(self.cusum_bg_rate)
            sb_ratio = self.cusum_sb_ratio
            alpha = self.cusum_alpha
            beta = self.cusum_beta

            # Run CUSUM burst detection
            import chisurf.core.fluorescence.burst.cusum as cusum_mod
            sel = cusum_mod.cusum_filter(
                tttr=tttr,
                min_ph=min_ph,
                background_rate=bg_rate,
                sb_ratio=sb_ratio,
                alpha=alpha,
                beta=beta
            )
            s = np.logical_and(s, sel)

        # Apply invert logic if the invert checkbox is checked (for all filter modes)
        # First check top-level setting, then fall back to count_rate_filter for backward compatibility
        if self.settings.get('invert_filter', False) and self.used_filter != 'count_rate':
            s = ~s

        if self.max_gap > 0 and self.use_gap_fill:
            s = fill_small_gaps_in_array(s, max_gap=self.max_gap)

        return s.astype(dtype=np.uint8)

    @property
    def burst_start_stop(self):
        """
        Returns the start-stop indices of bursts in the selected mask,
        allowing for small gaps with max_gap=4 by default.
        """
        selected = self.selected
        max_gap = self.max_gap
        if len(selected) > max_gap:
            return chisurf.core.math.signal.find_bursts(selected, max_gap=max_gap)
        else:
            return np.array([], dtype=np.uint64)

    @property
    def burst_lengths(self):
        brst = self.burst_start_stop
        if len(brst) > 2:
            n = brst.T[1] - brst.T[0]
        else:
            n = np.array([], dtype=np.uint64)
        return n

    @property
    def save_sl5(self):
        return self.checkBox_6.isChecked()

    @property
    def save_bur(self):
        return self.checkBox_7.isChecked()

    @property
    def save_hdf5(self):
        # For testing purposes, return True
        # In a production environment, this would check a checkbox or configuration setting
        return True  # Change to False to disable HDF5 output

    def update_filter_plot(self):
        """
        Update the plot showing which photons are selected (1) vs unselected (0).
        """
        try:
            selected = self.selected
            if isinstance(selected, (np.ndarray, list)) and len(selected) > 0:
                n_min = self.plot_min
                n_max = self.plot_max
                x = np.arange(n_min, n_max)
                y = selected[n_min:n_max]
                self.plot_select.setData(x=x, y=y)
            else:
                # If selected is not available, clear the plot
                self.plot_select.setData([], [])
        except (AttributeError, IndexError, TypeError):
            # Handle the case when self.selected is not available or returns an error
            self.plot_select.setData([], [])

    def update_dt_plot(self):
        """
        Update the plot of dT vs photon index, highlighting selected vs unselected points.
        """
        try:
            dT = self.dT
            if isinstance(dT, np.ndarray):
                n_min = self.plot_min
                n_max = self.plot_max
                mask = self.selected[n_min:n_max].astype(bool)
                y = dT[n_min:n_max]
                x = np.arange(n_min, n_max)

                mx = np.ma.masked_array(x, mask=~mask)
                my = np.ma.masked_array(y, mask=~mask)
                self.plot_selected.setData(x=mx.compressed(), y=my.compressed())

                mx = np.ma.masked_array(x, mask=mask)
                my = np.ma.masked_array(y, mask=mask)
                self.plot_unselected.setData(x=mx.compressed(), y=my.compressed())
            else:
                # If dT is not available, clear the plots
                self.plot_selected.setData([], [])
                self.plot_unselected.setData([], [])
        except AttributeError:
            # Handle the case when self.dT is not available
            self.plot_selected.setData([], [])
            self.plot_unselected.setData([], [])

    @property
    def do_mcs_plot(self) -> bool:
        return self.toolButton_2.isChecked()

    @do_mcs_plot.setter
    def do_mcs_plot(self, v):
        self.toolButton_2.setChecked(v)

    @property
    def do_burst_photon_plot(self) -> bool:
        return self.toolButton_7.isChecked()

    @do_burst_photon_plot.setter
    def do_burst_photon_plot(self, v):
        self.toolButton_7.setChecked(v)

    @property
    def do_microtime_plot(self) -> bool:
        return self.toolButton_3.isChecked()

    @do_microtime_plot.setter
    def do_microtime_plot(self, v):
        self.toolButton_3.setChecked(v)

    def update_mcs_plot(self):
        """
        Update the intensity trace plot (MCS) for both all photons and the selected subset.
        """
        if self.do_mcs_plot:
            try:
                tttr = self.tttr
                if isinstance(tttr, tttrlib.TTTR):
                    try:
                        selected = self.selected
                        if isinstance(selected, (np.ndarray, list)) and len(selected) > 0:
                            idx = np.where(selected)[0]
                            tw = self.trace_bin_width / 1000.0
                            trace_selected = tttr[idx].get_intensity_trace(time_window_length=tw)
                            x1 = np.arange(len(trace_selected)) * tw
                            self.plot_mcs_selected.setData(x1, trace_selected)
                        else:
                            self.plot_mcs_selected.setData([], [])
                    except (AttributeError, IndexError, TypeError):
                        self.plot_mcs_selected.setData([], [])

                    trace_all = tttr.get_intensity_trace(time_window_length=self.trace_bin_width / 1000.0)
                    x2 = np.arange(len(trace_all)) * (self.trace_bin_width / 1000.0)
                    self.plot_mcs_all.setData(x2, trace_all)
                else:
                    self.plot_mcs_all.setData([], [])
                    self.plot_mcs_selected.setData([], [])
            except (AttributeError, IndexError, TypeError):
                self.plot_mcs_all.setData([], [])
                self.plot_mcs_selected.setData([], [])
        else:
            self.plot_mcs_all.setData(x=[1.0], y=[1.0])
            self.plot_mcs_selected.setData(x=[1.0], y=[1.0])

    def update_burst_histogram(self):
        """
        Update the histogram of burst lengths (only relevant if burst mode is active).
        """
        if self.do_burst_photon_plot:
            try:
                burst_lengths = self.burst_lengths
                if isinstance(burst_lengths, np.ndarray) and len(burst_lengths) > 0:
                    num_bins = self.number_of_burst_bins
                    hist, bin_edges = np.histogram(burst_lengths, bins=num_bins)

                    self.pw_burst_histogram.clear()
                    self.pw_burst_histogram.addItem(
                        pg.BarGraphItem(
                            x0=bin_edges[:-1],
                            x1=bin_edges[1:],
                            y0=0,
                            y1=hist,
                            brush='b',
                            pen='w'
                        )
                    )
                    self.plot_burst_histogram = self.pw_burst_histogram.plot(
                        bin_edges,
                        hist,
                        pen='b',
                        stepMode=True
                    )

                    total_bursts = np.sum(hist)
                    pos_x = 0.5 * (bin_edges[0] + bin_edges[-1]) if bin_edges.size > 1 else 0
                    pos_y = (hist.max() * 0.9) if hist.size > 0 else 1.0
                    self.total_bursts_label = pg.TextItem(
                        f"Total bursts: {total_bursts}",
                        anchor=(0, 0),
                        color='w'
                    )
                    self.total_bursts_label.setPos(pos_x, pos_y)
                    self.pw_burst_histogram.addItem(self.total_bursts_label)
                    self.pw_burst_histogram.setYRange(0.0, max(hist) if hist.size > 0 else 1.0)
                    self.pw_burst_histogram.setXRange(0.0, max(bin_edges) if bin_edges.size > 0 else 1.0)
                else:
                    self.pw_burst_histogram.clear()
                    self.pw_burst_histogram.addItem(
                        pg.TextItem(
                            "No burst data available",
                            anchor=(0.5, 0.5),
                            color='w'
                        )
                    )
            except (AttributeError, IndexError, TypeError, ValueError):
                self.pw_burst_histogram.clear()
                self.pw_burst_histogram.addItem(
                    pg.TextItem(
                        "Error processing burst data",
                        anchor=(0.5, 0.5),
                        color='w'
                    )
                )

    def update_decay_plot(self):
        """
        Update the microtime decay plot (histogram), comparing all photons vs selected photons.
        """
        if self.do_microtime_plot:
            try:
                tttr = self.tttr
                if isinstance(tttr, tttrlib.TTTR):
                    try:
                        selected = self.selected
                        if isinstance(selected, (np.ndarray, list)) and len(selected) > 0:
                            idx = np.where(selected)[0]
                            y, x = tttr[idx].get_microtime_histogram(self.decay_coarse)
                            if np.any(y > 0):
                                idx_max = np.where(y > 0)[0][-1]
                                x = x[:idx_max]
                                y = y[:idx_max]
                                x *= 1e9
                                self.plot_decay_selected.setData(x=x, y=y)
                            else:
                                self.plot_decay_selected.setData([], [])
                        else:
                            self.plot_decay_selected.setData([], [])
                    except (AttributeError, IndexError, TypeError):
                        self.plot_decay_selected.setData([], [])

                    y, x = tttr.get_microtime_histogram(self.decay_coarse)
                    if np.any(y > 0):
                        idx_max = np.where(y > 0)[0][-1]
                        x = x[:idx_max]
                        y = y[:idx_max]
                        x *= 1e9
                        self.plot_decay_all.setData(x=x, y=y)
                    else:
                        self.plot_decay_all.setData([], [])
                else:
                    self.plot_decay_all.setData([], [])
                    self.plot_decay_selected.setData([], [])
            except (AttributeError, IndexError, TypeError):
                self.plot_decay_all.setData([], [])
                self.plot_decay_selected.setData([], [])
        else:
            self.plot_decay_all.setData(x=[1.0], y=[1.0])
            self.plot_decay_selected.setData(x=[1.0], y=[1.0])

    def _rebuild_channel_labels(self):
        """Rebuild per-detector labels in the info panel from user-defined detectors."""
        tttr = self.tttr
        old_labels = self._channel_labels
        self._channel_labels = {}

        # Remove old labels from the form layout
        for lbl in old_labels.values():
            lbl.setParent(None)
            lbl.deleteLater()
        old_labels.clear()

        if tttr is None:
            return

        # Create one label per detector name
        for det_name in sorted(self.detectors.keys()):
            lbl = QtWidgets.QLabel("—")
            lbl.setToolTip(f"Mean photons per burst in detector '{det_name}'")
            self._channel_labels[det_name] = lbl
            self._info_form.addRow(f"Mean ph ({det_name}):", lbl)

    def update_burst_info(self):
        """Refresh the burst info panel with current burst statistics."""
        try:
            tttr = self.tttr
            start_stop = self.burst_start_stop
            if tttr is None:
                self.label_burst_count.setText("0")
                self.label_mean_duration.setText("—")
                self.label_mean_photons.setText("—")
                for lbl in self._channel_labels.values():
                    lbl.setText("—")
                return

            # Rebuild labels when detectors change
            current_detectors = set(self.detectors.keys()) if self.detectors else set()
            if current_detectors != set(self._channel_labels.keys()):
                self._rebuild_channel_labels()

            if start_stop is None or len(start_stop) == 0:
                self.label_burst_count.setText("0")
                self.label_mean_duration.setText("—")
                self.label_mean_photons.setText("—")
                for lbl in self._channel_labels.values():
                    lbl.setText("—")
                return

            n_bursts = len(start_stop)
            self.label_burst_count.setText(str(n_bursts))

            macro_times = tttr.macro_times
            mt_res = tttr.header.macro_time_resolution

            durations = (macro_times[start_stop[:, 1]] - macro_times[start_stop[:, 0]]) * mt_res * 1000.0
            photon_counts = start_stop[:, 1] - start_stop[:, 0] + 1

            self.label_mean_duration.setText(f"{np.mean(durations):.3f} ms")
            self.label_mean_photons.setText(f"{np.mean(photon_counts):.1f}")

            routing = tttr.routing_channels
            for det_name, lbl in self._channel_labels.items():
                det_chs = list(self.detectors[det_name].get('chs', []))
                if not det_chs:
                    lbl.setText("—")
                    continue
                per_burst = []
                for s, e in start_stop:
                    mask = np.isin(routing[s:e + 1], det_chs)
                    per_burst.append(np.sum(mask))
                lbl.setText(f"{np.mean(per_burst):.1f}" if per_burst else "—")
        except Exception:
            pass

    def _create_burst_info_panel(self):
        """Build the burst info QGroupBox and insert it next to the Filter groupBox."""
        self.burst_info_group = QtWidgets.QGroupBox("Info")
        self._info_form = QtWidgets.QFormLayout(self.burst_info_group)
        self._info_form.setContentsMargins(4, 8, 4, 4)
        self._info_form.setSpacing(2)

        self.label_burst_count = QtWidgets.QLabel("0")
        self.label_mean_duration = QtWidgets.QLabel("—")
        self.label_mean_photons = QtWidgets.QLabel("—")

        self._info_form.addRow("Bursts:", self.label_burst_count)
        self._info_form.addRow("Mean duration:", self.label_mean_duration)
        self._info_form.addRow("Mean photons/burst:", self.label_mean_photons)

        self._channel_labels = {}
        self._info_container = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(self._info_container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)

        parent = self.groupBox.parentWidget()
        layout = parent.layout()
        pos = None
        if isinstance(layout, QtWidgets.QGridLayout):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item is not None and item.widget() == self.groupBox:
                    pos = layout.getItemPosition(i)
                    layout.removeWidget(self.groupBox)
                    break

        hbox.addWidget(self.groupBox)
        hbox.addWidget(self.burst_info_group)

        if pos is not None:
            layout.addWidget(self._info_container, pos[0], pos[1], pos[2], pos[3])

    def update_plots(self, selection: str = "all"):
        """
        Convenience method to update different sets of plots
        ('mcs', 'decay', 'dT', 'filter', 'burst').
        """
        if 'mcs' in selection:
            self.update_mcs_plot()
        if 'decay' in selection:
            self.update_decay_plot()
        if 'dT' in selection:
            self.update_dt_plot()
        if 'filter' in selection:
            self.update_filter_plot()
        else:
            self.update_dt_plot()
            self.update_decay_plot()
            self.update_mcs_plot()
            self.update_filter_plot()
            self.update_burst_histogram()
        self.update_burst_info()

    def read_tttr(self):
        """
        Called when the user changes the active file.
        Looks up the preloaded TTTR object in self.tttr_objects.
        """
        fn = self.current_tttr_filename
        if not fn:
            return

        p = pathlib.Path(fn).resolve()
        p_str = str(p)

        if p_str in self.tttr_objects:
            try:
                self.tttr = self.tttr_objects[p_str]
                self.plot_max = len(self.tttr) - 1
                header = self.tttr.get_header()
                s = header.json
                d = json.loads(s)
                self.settings['header'] = d
                self.update_plots()
            except Exception as e:
                # If there's an error accessing the TTTR object's properties,
                # display an error message and exit early
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Reading File",
                    f"Failed to read file '{p.name}' with the selected setup.\n\n"
                    f"Error: {str(e)}\n\n"
                    f"Please check that you have selected the correct setup for this file type."
                )
                # Remove the problematic TTTR object from the dictionary
                if p_str in self.tttr_objects:
                    del self.tttr_objects[p_str]
                self.tttr_objects = dict()
                self.onClearFiles()
                return  # Exit early to prevent undefined state

    def update_output_path(self):
        """
        Update the suggested output path (lineEdit_2) based on the current settings.
        """
        if len(self.channels) > 0:
            chs = ','.join([str(x) for x in self.channels])
        else:
            chs = 'All'

        # Determine path format based on filter mode
        if self.used_filter == "count_rate":
            path_prefix = "countrate"
        elif self.used_filter == "bocpd":
            path_prefix = "bocpd"
        elif self.used_filter == "kalman":
            path_prefix = "kalman"
        elif self.used_filter == "cusum":
            path_prefix = "cusum"
        else:
            path_prefix = "burstwise"

        s = f"{path_prefix}_{chs} {self.dT_max:.4f}#{self.photon_number_threshold}"
        self.lineEdit_2.setText(s)

    def update_parameter(self):
        """
        Update internal settings whenever the user changes filters or region selectors.
        Robust to early initialization when region_selector/settings may not yet exist.
        """
        # Ensure settings dicts exist (early init safety)
        if not hasattr(self, 'settings') or not isinstance(self.settings, dict):
            self.settings = {}
        self.settings.setdefault('count_rate_filter', {})
        self.settings.setdefault('delta_macro_time_filter', {})

        # Determine lb/ub even if region_selector is not yet created
        try:
            lb, ub = self.region_selector.getRegion()
        except Exception:
            # Fallback to spin boxes or sane defaults
            try:
                lb = float(self.doubleSpinBox_2.value())
                ub = float(self.doubleSpinBox_3.value())
            except Exception:
                lb, ub = 0.0001, 0.15
            # Convert to log space if axis is in log mode (to match later back-conversion)
            try:
                if self.pw_dT.getAxis('left').logMode:
                    lb = np.log10(lb) if lb > 0 else -4
                    ub = np.log10(ub) if ub > 0 else 0
            except Exception:
                pass

        self.settings['filter_active'] = self.checkBox_4.isChecked()
        self.settings['count_rate_filter']['n_ph_max'] = int(self.spinBox.value())
        self.settings['count_rate_filter']['time_window'] = float(self.doubleSpinBox.value()) * 1e-3
        # Store invert setting at top level for all filter types
        self.settings['invert_filter'] = bool(self.checkBox.isChecked())
        # For backward compatibility, also store in count_rate_filter
        self.settings['count_rate_filter']['invert'] = bool(self.checkBox.isChecked())

        # Map from (maybe log) lb/ub back to linear if axis is log
        try:
            is_log = self.pw_dT.getAxis('left').logMode
        except Exception:
            is_log = False

        self.settings['delta_macro_time_filter']['dT_min'] = 10.0 ** lb if is_log else lb
        self.settings['delta_macro_time_filter']['dT_max'] = 10.0 ** ub if is_log else ub
        self.settings['delta_macro_time_filter']['dT_min_active'] = self.checkBox_2.isChecked()
        self.settings['delta_macro_time_filter']['dT_max_active'] = self.checkBox_3.isChecked()

        # Avoid updating plots too early if plot widgets not ready
        try:
            self.update_plots()
        except Exception:
            pass
        try:
            self.update_output_path()
        except Exception:
            pass

    def onClearFiles(self):
        """
        Clears the list of filenames, resets the spinBox, clears the lineEdit,
        unsets the current TTTR object, and also clears all plots.
        """
        lst = self.settings.get('tttr_filenames')
        if isinstance(lst, list):
            lst.clear()
        # Reset index spinbox safely
        self.spinBox_4.blockSignals(True)
        self.spinBox_4.setMaximum(0)
        self.spinBox_4.setValue(0)
        self.spinBox_4.blockSignals(False)
        self.comboBox.setEnabled(True)
        self.lineEdit.clear()
        self.tttr = None
        # Reset resolved output directory cache to avoid stale paths
        self._resolved_output_dir = None
        # Also clear any cached TTTR objects
        try:
            if hasattr(self, 'tttr_objects') and isinstance(self.tttr_objects, dict):
                self.tttr_objects.clear()
        except Exception:
            pass

        # Clear each plot item
        self.plot_unselected.setData([], [])
        self.plot_selected.setData([], [])
        self.plot_mcs_all.setData([], [])
        self.plot_mcs_selected.setData([], [])
        self.plot_decay_all.setData([], [])
        self.plot_decay_selected.setData([], [])
        self.plot_select.setData([], [])

        # Clear the burst histogram entirely (removes bars/text)
        self.pw_burst_histogram.clear()

    def updateUI(self):
        """
        Updates the lineEdit with the currently active file name.
        Safe when there is no current file.
        """
        fn = self.current_tttr_filename
        self.lineEdit.setText(fn if isinstance(fn, str) else "")

    def onRegionUpdate(self):
        """
        Sync the numeric spinBoxes (doubleSpinBox_2/3) with the region item in the dT plot.
        """
        lb, ub = self.doubleSpinBox_2.value(), self.doubleSpinBox_3.value()
        if self.pw_dT.getAxis('left').logMode:
            lb, ub = np.log10(lb), np.log10(ub)
        self.region_selector.setRegion(rgn=(lb, ub))

    def get_unique_folder_path(self, base_path: pathlib.Path) -> pathlib.Path:
        """
        Return a path that is unique w.r.t. both the directory and a sibling .zip file.
        E.g., if 'burstwise_All#60' folder OR 'burstwise_All#60.zip' exists, try suffixes.
        """

        def name_taken(p: pathlib.Path) -> bool:
            return p.exists() or (p.parent / f"{p.name}.zip").exists()

        if not name_taken(base_path):
            return base_path

        counter = 0
        while True:
            candidate = base_path.parent / f"{base_path.name}_{counter}"
            if not name_taken(candidate):
                return candidate
            counter += 1

    @property
    def parent_directories(self) -> typing.List[pathlib.Path]:
        """
        Resolve and cache a unique analysis output directory once, then reuse it.
        - If a unique output directory was already chosen (e.g., during saving), reuse it (stable).
        - Otherwise, compute the unique directory based on the first TTTR file and cache it.
        """
        # Reuse the resolved directory if available (ensures consistency across calls)
        if getattr(self, "_resolved_output_dir", None):
            return [self._resolved_output_dir for _ in self.settings['tttr_filenames']]
        # Compute and cache based on the first file only (all outputs are stored together)
        if not self.settings['tttr_filenames']:
            return []
        first = self.settings['tttr_filenames'][0].replace('\x00', '')
        fn = pathlib.Path(first).absolute()
        base_path = fn.parent / self.target_path
        self._resolved_output_dir = self.get_unique_folder_path(base_path)
        return [self._resolved_output_dir for _ in self.settings['tttr_filenames']]

    @property
    def original_directories(self) -> typing.List[pathlib.Path]:
        """
        For each TTTR filename, get the parent directory with the configured target path
        without adding numeric suffixes. These are the original folder names.
        """
        r = []
        for filename in self.settings['tttr_filenames']:
            filename = filename.replace('\x00', '')
            fn = pathlib.Path(filename).absolute()
            base_path = fn.parent / self.target_path
            r.append(base_path)
        return r

    def save_selection(self, output_types=None, zip_output=False, remove_folder=False):
        """
        Save the selection data in .bur or .json.gz, depending on user checkboxes or specified output_types,
        and display a progress bar while saving. Optionally zip the output folder after saving and remove
        the original folder if requested. Also saves a JSON file with all parameters to an info folder.

        Parameters:
        -----------
        output_types : set, optional
            Set of output types to save. If provided, this overrides the checkbox settings.
            Possible values: "bur", "sl5", "hdf5"
        zip_output : bool, optional
            Whether to zip the output folder after saving. Default is False.
        remove_folder : bool, optional
            Whether to remove the original folder after zipping. Default is False.
            This parameter is only used if zip_output is True.
        """
        logger = logging.getLogger(__name__)
        logger.debug("Called save_selection with output_types=%s, zip_output=%s, remove_folder=%s",
                     output_types, zip_output, remove_folder)

        # --------------------------------------------------------------------
        # Reserve all unique output directories once, before writing any files
        from pathlib import Path
        fn = self.settings['tttr_filenames'][0]
        fn_path = Path(fn.replace('\x00', '')).resolve()
        base_path = fn_path.parent / self.target_path
        unique_path = self.get_unique_folder_path(base_path)
        # Cache the resolved output directory for consistent access by other pages
        self._resolved_output_dir = unique_path
        # --------------------------------------------------------------------

        total_files = len(self.settings['tttr_filenames'])
        # Determine output types
        if output_types is None:
            output_types = set()
            if self.save_bur:
                output_types.add("bur")
            if self.save_sl5:
                output_types.add("sl5")
            if self.save_hdf5:
                output_types.add("hdf5")
        logger.debug("Output types resolved to: %s", output_types)

        # Calculate total tasks
        total_tasks = 0
        if "bur" in output_types:
            total_tasks += total_files
        if "sl5" in output_types:
            total_tasks += total_files
        if "hdf5" in output_types:
            total_tasks += 1  # single combined HDF5
        if zip_output:
            total_tasks += 2  # zip and optional removal
        logger.debug("Total tasks calculated: %d", total_tasks)

        # Warn if remove_folder true without zip
        if remove_folder and not zip_output:
            logger.warning("remove_folder=True but zip_output=False; remove_folder will be ignored.")

        # Initialize progress dialog
        progress = EnhancedProgressDialog("Saving Selection", "Initializing...", 0, total_tasks, self)
        progress.show()
        current_task = 0
        all_dfs = []

        # Write .bur and collect for HDF5
        for filename in self.settings['tttr_filenames']:
            fn = Path(filename)
            self.tttr = self.tttr_objects[str(fn.resolve())]
            logger.debug("Processing file %s", fn.name)

            include_zeros = "bur" in output_types
            df = io.fluorescence.burst.generate_burst_dataframe(
                start_stop=self.burst_start_stop,
                filename=fn,
                tttr=self.tttr,
                windows=self.windows,
                detectors=self.detectors,
                include_interleaved_zeros=include_zeros
            )
            logger.debug("Generated DataFrame for %s; rows=%d", fn.name, len(df) if df is not None else 0)

            if "bur" in output_types:
                bur_dir = unique_path / 'bi4_bur'
                bur_dir.mkdir(parents=True, exist_ok=True)
                bur_file = bur_dir / f"{fn.stem}.bur"
                progress.update_text(f"Saving BUR file: {fn.stem}.bur")
                io.fluorescence.burst.write_dataframe_to_bur(df, bur_file)
                mt = self.tttr.macro_times[-1] * self.tttr.header.macro_time_resolution
                io.fluorescence.burst.write_mti_summary(
                    filename=fn,
                    analysis_dir=unique_path,
                    max_macro_time=mt,
                    append=True
                )
                current_task += 1
                progress.update_progress(current_task)
                logger.debug("Saved BUR file: %s", fn.stem)
                if progress.wasCanceled():
                    logger.warning("Operation canceled by user during BUR saving.")
                    return

            if "hdf5" in output_types and df is not None:
                if not "bur" in output_types and include_zeros:
                    # drop all-zero rows
                    df = df.loc[~(df.select_dtypes(include=['number']) == 0).all(axis=1)]
                df_copy = df.copy()
                df_copy['Source File'] = str(fn)
                all_dfs.append(df_copy)

        # Combined HDF5

        if "hdf5" in output_types and all_dfs:
            progress.update_text("Creating combined HDF5 file (compact)...")
            combined = pd.concat(all_dfs, ignore_index=True)

            # --- drop empty columns
            combined = combined.dropna(axis=1, how="all")

            # --- downcast numerics
            for c in combined.select_dtypes(include=["integer"]).columns:
                if (combined[c] >= 0).all():
                    combined[c] = pd.to_numeric(combined[c], downcast="unsigned")
                else:
                    combined[c] = pd.to_numeric(combined[c], downcast="integer")
            for c in combined.select_dtypes(include=["floating"]).columns:
                combined[c] = combined[c].astype(np.float32)

            # --- encode strings/objects as categorical codes (to keep format='fixed')
            #     store mapping for reconstruction; -1 will represent NaN
            cat_map = {}
            obj_cols = combined.select_dtypes(include=["object"]).columns
            # strongly recommend encoding these (very repetitive paths)
            must_encode = {"Source File", "First File", "Last File"} & set(obj_cols)

            for col in obj_cols:
                # Decide if encoding is worth it: always for must_encode, else low-cardinality
                nunique = combined[col].nunique(dropna=False)
                if (col in must_encode) or (nunique <= 0.5 * len(combined)):
                    cat = pd.Categorical(combined[col], ordered=False)
                    cat_map[col] = cat.categories.tolist()
                    # use smallest int that can hold -1 and max code
                    codes = cat.codes.astype(np.int32)
                    combined[col] = codes
                else:
                    # If you really want to keep these as strings, *convert to bytes*
                    # or skip entirely; better is to encode everything to codes for size.
                    cat = pd.Categorical(combined[col], ordered=False)
                    cat_map[col] = cat.categories.tolist()
                    combined[col] = cat.codes.astype(np.int32)

            # pick compression with fallback
            complib = "bzip2"
            try:
                # test if available on this build
                pd.HDFStore(str(unique_path / '___tmp__.h5'), mode='w', complib=complib).close()
                (unique_path / '___tmp__.h5').unlink(missing_ok=True)
            except Exception:
                complib = "blosc"  # fast & good ratio
                try:
                    pd.HDFStore(str(unique_path / '___tmp__.h5'), mode='w', complib=complib).close()
                    (unique_path / '___tmp__.h5').unlink(missing_ok=True)
                except Exception:
                    complib = "zlib"  # last resort, always available

            hdf5_dir = unique_path / 'hdf5'
            hdf5_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            h5_file = hdf5_dir / f"burst_data_{timestamp}.h5"

            progress.update_text(f"Writing HDF5 file: {h5_file.name}")

            # Write with FIXED format (dense), no index
            with pd.HDFStore(h5_file, mode='w', complib=complib, complevel=9) as store:
                store.put('results', combined, format='fixed', index=False)
                st = store.get_storer('results')
                # Save mapping; -1 in codes = NaN
                st.attrs.category_map = json.dumps(cat_map)

            current_task += 1
            progress.update_progress(current_task, "HDF5 file created (compact)")
            logger.debug("HDF5 compact file written: %s (complib=%s)", h5_file, complib)
            if progress.wasCanceled():
                logger.warning("Operation canceled by user during HDF5 writing.")
                return

        # SL5 output
        if "sl5" in output_types:
            sl5_dir = unique_path / 'sl5'
            for filename in self.settings['tttr_filenames']:
                fn = Path(filename)
                sl5_dir.mkdir(parents=True, exist_ok=True)
                progress.update_text(f"Saving SL5 file: {fn.stem}.json.gz")
                data = {
                    'filename': os.path.relpath(fn, unique_path),
                    'filetype': self.filetype,
                    'count_rate_filter': self.settings['count_rate_filter'],
                    'delta_macro_time_filter': self.settings['delta_macro_time_filter'],
                    'filter': chisurf.core.fio.compress_numpy_array(self.selected)
                }
                output_file = sl5_dir / f"{fn.stem}.json.gz"
                with io.open_maybe_zipped(output_file, 'w') as f:
                    f.write(json.dumps(data))
                current_task += 1
                progress.update_progress(current_task)
                logger.debug("Saved SL5 file: %s", fn)
                if progress.wasCanceled():
                    logger.warning("Operation canceled by user during SL5 saving.")
                    return

        # Save parameters Info
        info_dir = unique_path / 'Info'
        info_dir.mkdir(parents=True, exist_ok=True)
        params = self.get_burst_selection_parameters()
        params.update({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'selected_setup': self.comboBox.currentText(),
            'channels': self.channels,
            'decay_coarse': self.decay_coarse,
            **({"microtime_ranges": self.microtime_ranges} if self.microtime_ranges else {}),
            'files': [Path(f).name for f in self.settings['tttr_filenames']]
        })
        with open(info_dir / 'photon_selection_parameters.json', 'w') as f:
            json.dump(params, f, indent=4)
        with open(info_dir / 'datetime.txt', 'w') as f:
            now = datetime.now()
            f.write(f"Date: {now.strftime('%Y-%m-%d')}\nTime: {now.strftime('%H:%M:%S')}\n")
        current_task += 1
        progress.update_progress(current_task)

        # Finish or zip
        if not zip_output:
            progress.finish("Selection saved successfully")
        else:
            # Zip the first output directory
            zip_target = unique_path
            progress.update_text(f"Zipping output folder: {zip_target}")
            current_task += 1
            progress.update_progress(current_task)
            zip_file = self.zip_output_folder(zip_target, progress)
            if zip_file and remove_folder:
                shutil.rmtree(zip_target)
            progress.finish("ZIP archive completed")

    def fill_pie_windows(self, k):
        self.windows = k
        self.comboBox_3.addItems(k.keys())

    def zip_output_folder(self, output_folder, existing_progress=None, add_timestamp=False):
        """
        Zip the output folder and its contents.

        Parameters:
        -----------
        output_folder : pathlib.Path
            Path to the folder to be zipped.
        existing_progress : QtWidgets.QProgressDialog, optional
            An existing progress dialog to use instead of creating a new one.
        add_timestamp : bool, optional
            Whether to add a timestamp to the zip filename. Default is False.

        Returns:
        --------
        pathlib.Path
            Path to the created zip file.
        """
        if not output_folder.exists() or not output_folder.is_dir():
            return None

        def unique_zip_path(base: pathlib.Path) -> pathlib.Path:
            """Return a unique zip path alongside base (base.name.zip, or _N.zip if taken)."""
            first = base.parent / f"{base.name}.zip"
            if not first.exists():
                return first
            i = 0
            while True:
                cand = base.parent / f"{base.name}_{i}.zip"
                if not cand.exists():
                    return cand
                i += 1

        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            zip_filename = output_folder.parent / f"{output_folder.name}_{timestamp}.zip"
            # belt & suspenders in case a same-timestamp file exists:
            if zip_filename.exists():
                zip_filename = unique_zip_path(output_folder)
        else:
            zip_filename = unique_zip_path(output_folder)

        # Use existing progress dialog if provided, otherwise create a new one
        using_existing_progress = existing_progress is not None
        if not using_existing_progress:
            progress = EnhancedProgressDialog("Creating ZIP Archive", "Zipping output folder...", 0, 100, self)
            progress.show()
        else:
            progress = existing_progress
            progress.update_text("Zipping output folder...")

        progress.setValue(10)  # Show some initial progress
        QtWidgets.QApplication.processEvents()

        try:
            # Create the zip file
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Get total number of files for better progress tracking
                total_files = sum([len(files) for _, _, files in os.walk(output_folder)])
                processed_files = 0

                # Walk through all files and subdirectories in the output folder
                for root, dirs, files in os.walk(output_folder):
                    # Convert root path to a pathlib.Path for easier manipulation
                    root_path = pathlib.Path(root)

                    # Add each file to the zip
                    for file in files:
                        file_path = root_path / file
                        # Calculate the relative path for the file in the zip
                        rel_path = file_path.relative_to(output_folder)
                        # Add the file to the zip
                        zipf.write(file_path, rel_path)

                        # Update progress based on files processed
                        processed_files += 1
                        progress_value = 10 + int(80 * processed_files / total_files) if total_files > 0 else 90
                        if isinstance(progress, EnhancedProgressDialog):
                            progress.update_progress(progress_value, f"Zipping: {rel_path}")
                        else:
                            progress.setValue(progress_value)
                            progress.setLabelText(f"Zipping: {rel_path}")
                            QtWidgets.QApplication.processEvents()

                        if progress.wasCanceled():
                            return None

            # Final progress update
            if isinstance(progress, EnhancedProgressDialog):
                progress.update_progress(100, "ZIP archive completed")
            else:
                progress.setValue(100)
                progress.setLabelText("ZIP archive completed")
                QtWidgets.QApplication.processEvents()

            return zip_filename

        except Exception as e:
            # Update progress dialog instead of showing a message box
            error_message = f"Error creating ZIP: {str(e)}"
            if isinstance(progress, EnhancedProgressDialog):
                progress.update_text(error_message)
                QtWidgets.QApplication.processEvents()
                time.sleep(2)  # Give user time to see the error
            elif using_existing_progress:
                progress.setLabelText(error_message)
                QtWidgets.QApplication.processEvents()
                time.sleep(2)  # Give user time to see the error
            else:
                # Only show message box if we created our own progress dialog and it's not enhanced
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Creating ZIP",
                    f"Failed to create ZIP archive: {str(e)}"
                )
            return None
        finally:
            # Close the progress dialog only if we created it
            if not using_existing_progress:
                progress.finish("ZIP operation completed")

    def fill_detectors(self, k):
        self.detectors = k
        # Add "All" option at the beginning
        self.comboBox_2.addItem("All")
        self.comboBox_2.addItems(k.keys())

    def update_detectors(self):
        """
        Sync the comboBox_2 selection to the channel lineEdit_4.
        If "All" is selected, empty the channel numbers line widget.
        """
        key = self.comboBox_2.currentText()
        if key == "All":
            # Empty the channel numbers line widget
            self.lineEdit_4.setText("")
        else:
            # Set the channel numbers from the selected detector
            s = ", ".join([str(i) for i in self.detectors[key]["chs"]])
            self.lineEdit_4.setText(s)
        self.update_parameter()

    def update_pie_windows(self):
        """
        Sync the comboBox_3 selection to the lineEdit_5 for microtime ranges.
        """
        key = self.comboBox_3.currentText()
        pie_win = self.windows[key]

        # Check if pie_win is a list/tuple or a single integer
        if isinstance(pie_win, (list, tuple)):
            # Handle list/tuple of windows
            formatted_windows = []
            # If pie_win is a list with exactly 2 elements and both are integers,
            # it's likely a single window definition [start, end]
            if len(pie_win) == 2 and all(isinstance(x, int) for x in pie_win):
                formatted_windows.append(f"{pie_win[0]}-{pie_win[1]}")
            else:
                # Otherwise, process each item in the list
                for i in pie_win:
                    if isinstance(i, (list, tuple)) and len(i) >= 2:
                        # Normal case: i is a tuple/list with at least two elements
                        formatted_windows.append(f"{i[0]}-{i[1]}")
                    elif isinstance(i, int):
                        # Handle case where i is a single integer
                        formatted_windows.append(f"{i}-{i}")
                    else:
                        # Skip invalid entries
                        continue
            s = ";".join(formatted_windows)
        elif isinstance(pie_win, int):
            # Handle case where pie_win is a single integer
            s = f"{pie_win}-{pie_win}"
        else:
            # Default to empty string for unexpected types
            s = ""

        self.lineEdit_5.setText(s)
        self.update_parameter()

    def load_available_setups(self):
        """
        Load available setups from the detector setups file and populate the comboBox.
        If no setups are available, display a message in the comboBox.
        """
        self.comboBox.clear()

        # Load setups from the detector setups file
        setups = load_detector_setups()
        setup_names = list(setups.get("setups", {}).keys())

        if not setup_names:
            # If no setups are available, add a placeholder item
            self.comboBox.addItem("No setups available")
        else:
            # Add all available setups to the comboBox
            self.comboBox.addItems(setup_names)

            # If there's a last used setup, select it
            if setups.get("last_used") and setups["last_used"] in setup_names:
                index = self.comboBox.findText(setups["last_used"])
                if index >= 0:
                    self.comboBox.setCurrentIndex(index)

    def get_burst_selection_parameters(self):
        """
        Collect all burst selection parameters from the UI.

        Returns:
            dict: A dictionary containing all burst selection parameters.
        """
        # Get the region selector values
        lb, ub = self.region_selector.getRegion()
        dT_min = 10.0 ** lb if self.pw_dT.getAxis('left').logMode else lb
        dT_max = 10.0 ** ub if self.pw_dT.getAxis('left').logMode else ub

        # Collect all parameters
        params = {
            "dT_min": dT_min,
            "dT_max": dT_max,
            "use_dT_min": self.checkBox_2.isChecked(),
            "use_dT_max": self.checkBox_3.isChecked(),
            "photon_threshold": self.spinBox.value(),
            "count_rate_window_ms": self.doubleSpinBox.value(),
            "invert_filter": self.checkBox.isChecked(),
            "filter_mode": self.used_filter,
            "filter_active": self.checkBox_4.isChecked(),
            "use_gap_fill": self.checkBox_5.isChecked(),
            "max_gap": self.spinBox_7.value(),
            "trace_bin_width": self.doubleSpinBox_4.value(),
            "number_of_burst_bins": self.spinBox_6.value(),
            "channels": self.channels,
            "decay_coarse": self.decay_coarse,
            "ph_window": self.ph_window
        }

        # Add microtime ranges if available
        if self.microtime_ranges:
            params["microtime_ranges"] = self.microtime_ranges

        # Add BOCPD parameters if BOCPD is selected
        if self.used_filter == 'bocpd':
            params.update({
                "bocpd_prior_count": self.bocpd_prior_count,
                "bocpd_prior_duration": self.bocpd_prior_duration,
                "bocpd_changepoint_prob": self.bocpd_changepoint_prob
            })

        # Add Kalman filter parameters if Kalman is selected
        if self.used_filter == 'kalman':
            params.update({
                "kalman_q": self.kalman_q,
                "kalman_r_scale": self.kalman_r_scale,
                "kalman_z_thresh": self.kalman_z_thresh,
                "kalman_min_len": self.kalman_min_len,
                "kalman_merge_gap": self.kalman_merge_gap
            })

        # Add CUSUM filter parameters if CUSUM is selected
        if self.used_filter == 'cusum':
            params.update({
                "cusum_bg_rate": self.cusum_bg_rate,
                "cusum_sb_ratio": self.cusum_sb_ratio,
                "cusum_alpha": self.cusum_alpha,
                "cusum_beta": self.cusum_beta
            })

        return params

    def save_burst_selection_parameters(self):
        """
        Save the burst selection parameters to the current setup.
        """
        logger = logging.getLogger(__name__)
        setup_name = self.comboBox.currentText()
        if not setup_name or setup_name == "No setups available":
            logger.warning("No setup selected - cannot save burst selection parameters")
            self.status_message.emit("No setup selected - please select a setup first", 5000)
            return False

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the burst selection parameters
            burst_params = self.get_burst_selection_parameters()

            # Add the burst selection parameters to the setup
            setup_data = setups["setups"][setup_name]
            setup_data["burst_selection"] = burst_params

            # Save the updated setups
            if save_detector_setups(setups):
                logger.info(f"Burst selection parameters saved to setup '{setup_name}' successfully")
                self.status_message.emit(f"Burst selection parameters saved to setup '{setup_name}'", 3000)
                return True
            else:
                logger.error(f"Failed to save burst selection parameters to setup '{setup_name}'")
                self.status_message.emit(f"Failed to save burst selection parameters to setup '{setup_name}'", 5000)
                return False
        else:
            logger.warning(f"The selected setup '{setup_name}' does not exist")
            self.status_message.emit(f"Setup '{setup_name}' does not exist", 5000)
            return False

    def update_micro_time_binning(self, setup_name=None):
        """
        Update the micro time binning (Decay bin) from the selected setup.

        Args:
            setup_name (str, optional): The name of the setup to use. If None, uses the currently selected setup.
        """
        if setup_name is None:
            setup_name = self.comboBox.currentText()

        if not setup_name or setup_name == "No setups available":
            return

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the micro time binning from the setup's tttr_reading section
            setup_data = setups["setups"][setup_name]
            if "tttr_reading" in setup_data and "micro_time_binning" in setup_data["tttr_reading"]:
                micro_time_binning = setup_data["tttr_reading"]["micro_time_binning"]
                # Update the spinBox_5 value (decay_coarse)
                self.spinBox_5.setValue(micro_time_binning)
                # Update the plots
                self.update_plots()

    def update_channel_routing(self, setup_name=None):
        """
        Update the channel routing (detectors) from the selected setup.

        Args:
            setup_name (str, optional): The name of the setup to use. If None, uses the currently selected setup.
        """
        if setup_name is None:
            setup_name = self.comboBox.currentText()

        if not setup_name or setup_name == "No setups available":
            return

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the detectors from the setup
            setup_data = setups["setups"][setup_name]
            if "detectors" in setup_data:
                # Update the detectors attribute
                self.detectors = setup_data["detectors"]

                # Clear and repopulate comboBox_2
                self.comboBox_2.blockSignals(True)
                self.comboBox_2.clear()

                # Add "All" option at the beginning
                self.comboBox_2.addItem("All")

                # Add detector names
                self.comboBox_2.addItems(self.detectors.keys())

                # Select the first detector by default (or "All" if no detectors)
                if self.comboBox_2.count() > 0:
                    self.comboBox_2.setCurrentIndex(0)

                self.comboBox_2.blockSignals(False)

                # Update the channel numbers in lineEdit_4
                self.update_detectors()

    def update_pie_windows_from_setup(self, setup_name=None):
        """
        Update the PIE windows from the selected setup.

        Args:
            setup_name (str, optional): The name of the setup to use. If None, uses the currently selected setup.
        """
        if setup_name is None:
            setup_name = self.comboBox.currentText()

        if not setup_name or setup_name == "No setups available":
            return

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the windows from the setup
            setup_data = setups["setups"][setup_name]
            if "windows" in setup_data:
                # Update the windows attribute
                self.windows = setup_data["windows"]

                # Clear and repopulate comboBox_3
                self.comboBox_3.blockSignals(True)
                self.comboBox_3.clear()

                # Add window names
                self.comboBox_3.addItems(self.windows.keys())

                # Select the first window by default
                if self.comboBox_3.count() > 0:
                    self.comboBox_3.setCurrentIndex(0)

                self.comboBox_3.blockSignals(False)

                # Update the microtime ranges in lineEdit_5
                self.update_pie_windows()

    def update_spinbox_7_state(self):
        """
        Update the enabled state of spinBox_7 based on the state of checkBox_5.
        When gap filling is disabled (checkBox_5 is unchecked), spinBox_7 should be disabled.
        """
        self.spinBox_7.setEnabled(self.checkBox_5.isChecked())

    def setup_connections(self):
        _setup_connections(self)

    def update_burst_selection_parameters(self, setup_name=None):
        """
        Update the burst selection parameters from the selected setup.

        Args:
            setup_name (str, optional): The name of the setup to use. If None, uses the currently selected setup.
        """
        if setup_name is None:
            setup_name = self.comboBox.currentText()

        if not setup_name or setup_name == "No setups available":
            return

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the burst selection parameters from the setup
            setup_data = setups["setups"][setup_name]
            if "burst_selection" in setup_data:
                burst_params = setup_data["burst_selection"]

                # Update the UI with the burst selection parameters
                if "dT_min" in burst_params and "dT_max" in burst_params:
                    dT_min = burst_params["dT_min"]
                    dT_max = burst_params["dT_max"]
                    if self.pw_dT.getAxis('left').logMode:
                        dT_min = np.log10(dT_min) if dT_min > 0 else -4  # Default to -4 if dT_min is 0 or negative
                        dT_max = np.log10(dT_max) if dT_max > 0 else 0   # Default to 0 if dT_max is 0 or negative
                    self.region_selector.setRegion((dT_min, dT_max))
                    self._dT_min = burst_params["dT_min"]
                    self._dT_max = burst_params["dT_max"]
                    self.doubleSpinBox_2.setValue(burst_params["dT_min"])
                    self.doubleSpinBox_3.setValue(burst_params["dT_max"])

                if "use_dT_min" in burst_params:
                    self.checkBox_2.setChecked(burst_params["use_dT_min"])

                if "use_dT_max" in burst_params:
                    self.checkBox_3.setChecked(burst_params["use_dT_max"])

                if "photon_threshold" in burst_params:
                    self.spinBox.setValue(burst_params["photon_threshold"])

                if "ph_window" in burst_params:
                    self.ph_window = burst_params["ph_window"]

                if "count_rate_window_ms" in burst_params:
                    self.doubleSpinBox.setValue(burst_params["count_rate_window_ms"])

                if "invert_filter" in burst_params:
                    self.checkBox.setChecked(burst_params["invert_filter"])
                # For backward compatibility
                elif "invert_count_rate_filter" in burst_params:
                    self.checkBox.setChecked(burst_params["invert_count_rate_filter"])

                if "filter_active" in burst_params:
                    self.checkBox_4.setChecked(burst_params["filter_active"])

                if "filter_mode" in burst_params:
                    if burst_params["filter_mode"] == "count_rate":
                        self.comboBox_burst_filter.setCurrentText("Count rate")
                    elif burst_params["filter_mode"] == "burst":
                        self.comboBox_burst_filter.setCurrentText("Burst")
                    elif burst_params["filter_mode"] == "bocpd":
                        self.comboBox_burst_filter.setCurrentText("BOCPD Burst")
                    elif burst_params["filter_mode"] == "kalman":
                        self.comboBox_burst_filter.setCurrentText("Kalman Burst")
                    elif burst_params["filter_mode"] == "cusum":
                        self.comboBox_burst_filter.setCurrentText("CUSUM Burst")
                    else:
                        # Default to burst mode if unknown
                        self.comboBox_burst_filter.setCurrentText("Burst")

                # BOCPD parameters
                if burst_params.get("filter_mode") == "bocpd":
                    if "bocpd_prior_count" in burst_params:
                        self.bocpd_prior_count = burst_params["bocpd_prior_count"]
                    if "bocpd_prior_duration" in burst_params:
                        self.bocpd_prior_duration = burst_params["bocpd_prior_duration"]
                    if "bocpd_changepoint_prob" in burst_params:
                        self.bocpd_changepoint_prob = burst_params["bocpd_changepoint_prob"]

                # Kalman filter parameters
                if burst_params.get("filter_mode") == "kalman":
                    if "kalman_q" in burst_params:
                        self.kalman_q = burst_params["kalman_q"]
                    if "kalman_r_scale" in burst_params:
                        self.kalman_r_scale = burst_params["kalman_r_scale"]
                    if "kalman_z_thresh" in burst_params:
                        self.kalman_z_thresh = burst_params["kalman_z_thresh"]
                    if "kalman_min_len" in burst_params:
                        self.kalman_min_len = burst_params["kalman_min_len"]
                    if "kalman_merge_gap" in burst_params:
                        self.kalman_merge_gap = burst_params["kalman_merge_gap"]

                # CUSUM filter parameters
                if burst_params.get("filter_mode") == "cusum":
                    if "cusum_bg_rate" in burst_params:
                        self.cusum_bg_rate = burst_params["cusum_bg_rate"]
                    if "cusum_sb_ratio" in burst_params:
                        self.cusum_sb_ratio = burst_params["cusum_sb_ratio"]
                    if "cusum_alpha" in burst_params:
                        self.cusum_alpha = burst_params["cusum_alpha"]
                    if "cusum_beta" in burst_params:
                        self.cusum_beta = burst_params["cusum_beta"]

                if "use_gap_fill" in burst_params:
                    self.checkBox_5.setChecked(burst_params["use_gap_fill"])

                if "max_gap" in burst_params:
                    self.spinBox_7.setValue(burst_params["max_gap"])

                if "trace_bin_width" in burst_params:
                    self.doubleSpinBox_4.setValue(burst_params["trace_bin_width"])

                if "number_of_burst_bins" in burst_params:
                    self.spinBox_6.setValue(burst_params["number_of_burst_bins"])

                self.spinBox_5.setValue(burst_params["decay_coarse"])

                # Update the plots
                self.update_plots()

    @chisurf.gui.decorators.init_with_ui("tttr_photon_filter.ui")
    def __init__(
            self,
            *args,
            windows,
            detectors,
            show_dT: bool = True,
            show_filter: bool = True,
            show_mcs: bool = True,
            show_decay: bool = True,
            show_burst: bool = True,
            default_mcs_dT: float = 1.0,
            default_dT_min: float = 0.0001,
            default_dT_max: float = 0.15,
            use_dT_min: bool = False,
            use_dT_max: bool = True,
            default_photon_threshold: int = 60,
            default_count_rate_window_ms: float = 1.0,
            invert_filter: bool = True,
            default_filter_mode: str = 'burst',
            use_gap_fill: bool = False,
            default_max_gap: int = 3,
            **kwargs
    ):
        """
        Initialize the photon-filter wizard.

        Parameters
        ----------
        windows : dict
            Dictionary of predefined PIE windows. Passed to the relevant UI comboBox.
        detectors : dict
            Dictionary of predefined detectors. Passed to the relevant UI comboBox.
        show_dT : bool, default=True
            Whether to show the dT plot (between consecutive photons).
        show_filter : bool, default=True
            Whether to show the filter/selection plot.
        show_mcs : bool, default=True
            Whether to show the intensity trace (MCS) plot.
        show_decay : bool, default=True
            Whether to show the microtime histogram (decay plot).
        show_burst : bool, default=True
            Whether to show the burst histogram.
        default_dT_min : float, default=0.0001
            Initial lower bound for delta macro-time filter (in ms).
        default_dT_max : float, default=0.15
            Initial upper bound for delta macro-time filter (in ms).
        default_mcs_dT : float, default=1.0
            Default / initial bin width value of intensity trace (in ms).
        use_dT_min : bool, default=False
            Whether the lower bound of delta macro-time filter is active initially.
        use_dT_max : bool, default=True
            Whether the upper bound of delta macro-time filter is active initially.
        default_photon_threshold : int, default=60
            Initial threshold for count rate or burst search.
        default_count_rate_window_ms : float, default=1.0
            Initial time window (ms) for count-rate based filtering.
        invert_filter : bool, default=True
            Whether to invert the filter initially (applies to all filter types).
        default_filter_mode : str, default='burst'
            Which filter mode radio button is selected by default.
            Valid: 'count_rate', 'burst', or 'bocpd'.
        use_gap_fill : bool, default=False
            Whether gap-filling is enabled by default.
        default_max_gap : int, default=3
            Maximum number of consecutive unselected photons that can be 'filled'
            when gap-filling is applied.

        Returns
        -------
        None. Initializes the UI and sets default widget states.
        """
        self.setTitle("Photon filter")
        self.windows = windows
        self.detectors = detectors
        self.fill_detectors(detectors)
        self.fill_pie_windows(windows)

        # Dictionary to store all TTTR objects
        self.tttr_objects = dict()

        # Load available setups for comboBox
        self.load_available_setups()

        # Main settings
        self.settings: dict = {}
        self.settings['tttr_filenames'] = []
        self.settings['count_rate_filter'] = {}
        self.settings['delta_macro_time_filter'] = {}
        # Initialize top-level invert_filter setting
        self.settings['invert_filter'] = invert_filter
        self.filter_data_saved = False

        # Cached resolved output directory for stable access across pages
        self._resolved_output_dir = None

        install_filter_mode_visibility(self, default_filter_mode)
        install_file_drop(self)

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.setSizePolicy(sizePolicy)

        # Internal TTTR reference
        self.tttr = None

        # Initialize dT_min/dT_max
        self._dT_min = default_dT_min
        self._dT_max = default_dT_max

        create_plots(self, colors)
        place_plots(self)

        # Build the burst info panel next to the Filter groupBox
        self._create_burst_info_panel()

        # Setup all signal-slot connections
        self.setup_connections()

        # Custom validator
        validator = CommaSeparatedIntegersValidator()
        self.lineEdit_4.setValidator(validator)

        # Initialize defaults
        self.spinBox.setValue(default_photon_threshold)
        self.doubleSpinBox.setValue(default_count_rate_window_ms)
        self.checkBox.setChecked(invert_filter)
        self.checkBox_2.setChecked(use_dT_min)
        self.checkBox_3.setChecked(use_dT_max)
        self.doubleSpinBox_4.setValue(default_mcs_dT)

        # -- NEW: set default gap-fill checkbox/spinbox --
        self.checkBox_5.setChecked(use_gap_fill)  # <--- gap-fill checkbox
        self.spinBox_7.setValue(default_max_gap)  # <--- max-gap spinbox
        self.update_spinbox_7_state()  # Update spinBox_7 enabled state based on checkBox_5

        # Control initial plot visibility
        self.pw_dT.setVisible(show_dT)
        self.toolButton_4.setChecked(show_filter)
        self.toolButton_2.setChecked(show_mcs)
        self.toolButton_3.setChecked(show_decay)
        self.toolButton_7.setChecked(show_burst)

        # -- Set the default filter mode --
        # install_filter_mode_visibility above already set the combobox
        # based on default_filter_mode. Nothing more to do here.

        # Update micro time binning and burst selection parameters from the selected setup
        self.update_micro_time_binning()
        self.update_burst_selection_parameters()

        # Final initial update
        self.update_parameter()
