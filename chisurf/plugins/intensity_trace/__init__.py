"""
Intensity Trace Analysis for Single-Molecule Data

This plugin provides tools for analyzing fluorescence intensity time traces from 
single-molecule experiments. It enables researchers to extract dynamic information 
from photon counting data, particularly for studying conformational changes, 
molecular interactions, and reaction kinetics at the single-molecule level.

Features:
- Loading and displaying Time-Tagged Time-Resolved (TTTR) data as intensity traces
- Histogram analysis of photon counts with customizable binning
- Hidden Markov Model (HMM) analysis for state detection and classification
- Bayesian Information Criterion (BIC) calculation for optimal state number determination
- Dwell time analysis for extracting kinetic information and rate constants
- FRET efficiency calculation and state-specific distribution analysis
- Transition probability matrix visualization and analysis
- Exponential fitting of dwell time distributions
- Interactive visualization with adjustable parameters
- Support for multi-channel data analysis (donor/acceptor channels)

The plugin implements a comprehensive workflow for single-molecule state analysis:
1. Load TTTR data and convert to binned intensity traces
2. Visualize traces and photon count distributions
3. Apply HMM to identify discrete states in noisy data
4. Analyze state transitions and dwell times to extract kinetic information
5. For FRET data, calculate efficiency distributions for each state

Ideal for analyzing single-molecule FRET, protein folding/unfolding, enzyme dynamics,
ligand binding, blinking behavior, or any other dynamic processes that can be 
observed in fluorescence intensity traces. The HMM approach is particularly powerful
for detecting states in noisy data with overlapping distributions.
"""

name = "Single-Molecule:Intensity trace"



import sys
import pathlib
import numpy as np
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QDialog, QCheckBox, QGridLayout
)
from PyQt5.QtGui import QPainterPath, QBrush, QColor, QPen
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import TextItem

import tttrlib
from PyQt5.QtWidgets import QGraphicsPathItem
from hmmlearn.hmm import GaussianHMM
from pyqtgraph import ImageItem, colormap


def save_burst_ids(hmm_states, time_axis, time_window_s, tttr_obj, output_dir="."):
    burst_ids = {}

    for state in np.unique(hmm_states):
        mask = hmm_states == state
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # Calculate burst intervals
        bursts = []
        start_idx = indices[0]

        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                stop_idx = indices[i - 1]
                bursts.append((start_idx, stop_idx))
                start_idx = indices[i]

        # Append the last burst
        bursts.append((start_idx, indices[-1]))

        burst_ids[state] = bursts

        # Optionally save burst IDs to file
        output_file = pathlib.Path(output_dir) / f"burst_ids_state_{state}.bst"
        with open(output_file, 'w') as f:
            for start_bin, stop_bin in bursts:
                # Convert bin indices back to TTTR indices
                start_time = time_axis[start_bin]
                stop_time = time_axis[stop_bin] + time_window_s

                macro_time_resolution = tttr_obj.header.macro_time_resolution
                start_tttr_idx = np.searchsorted(tttr_obj.macro_times, start_time / macro_time_resolution)
                stop_tttr_idx = np.searchsorted(tttr_obj.macro_times, stop_time / macro_time_resolution)

                f.write(f"{start_tttr_idx}\t{stop_tttr_idx}\n")

    return burst_ids


def compute_bic_curve(data, max_states=10):
    bics = []
    n_samples, n_features = data.shape

    for n in range(1, max_states + 1):
        try:
            model = GaussianHMM(n_components=n, covariance_type="full", n_iter=1000, random_state=0)
            model.fit(data)
            logL = model.score(data)
            n_params = n * (n - 1) + n * n_features * 2  # transitions + mean + cov
            bic = np.log(n_samples) * n_params - 2 * logL
            bics.append((n, bic))
        except Exception as e:
            print(f"HMM fit failed for {n} states: {e}")
            bics.append((n, np.nan))
    return bics


def compute_dwell_times(state_sequence, time_step=1.0):
    """
    Compute dwell times for each state in a state sequence.

    Parameters:
    -----------
    state_sequence : np.ndarray
        Array of inferred HMM state labels (1D).
    time_step : float
        Duration represented by each step in the state sequence (e.g., in seconds).

    Returns:
    --------
    dwell_times : dict
        Dictionary mapping state index to a list of dwell times in units of `time_step`.
    """
    dwell_times = {}
    if len(state_sequence) == 0:
        return dwell_times

    current_state = state_sequence[0]
    dwell_count = 1

    for s in state_sequence[1:]:
        if s == current_state:
            dwell_count += 1
        else:
            if current_state not in dwell_times:
                dwell_times[current_state] = []
            dwell_times[current_state].append(dwell_count * time_step)
            current_state = s
            dwell_count = 1

    # Save the last dwell
    if current_state not in dwell_times:
        dwell_times[current_state] = []
    dwell_times[current_state].append(dwell_count * time_step)

    return dwell_times


class DistPlotWindow(QDialog):
    def __init__(self, fret_distributions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FRET Efficiency Distributions by State")
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)

        plots = []
        y_max = 0
        all_bins = np.linspace(0, 1, 51)
        bin_edges = all_bins

        # First pass: compute max y for shared scaling
        hists = {}
        for state, frets in fret_distributions.items():
            if len(frets) == 0:
                continue
            hist, _ = np.histogram(frets, bins=bin_edges, density=True)
            hists[state] = hist
            y_max = max(y_max, np.max(hist))

        # Second pass: plot, with state labels
        for idx, (state, hist) in enumerate(hists.items()):
            plot = self.plot_widget.addPlot(row=idx, col=0)
            # — add the state as the plot title —
            plot.setTitle(f"State {state}")

            plot.setYRange(0, y_max)
            plot.setXRange(0, 1)

            x_vals = bin_edges[:-1]
            y_vals = hist
            color = (state * 50 % 255, 100, 180, 150)
            plot.plot(x_vals, y_vals, stepMode=False, fillLevel=0, brush=color)

            # clean up axes (we're showing titles instead)
            plot.hideAxis('left')
            if idx == len(hists) - 1:
                plot.setLabel('bottom', 'FRET Efficiency')
            else:
                plot.hideAxis('bottom')

            plots.append(plot)

        # Link all axes
        for plot in plots[1:]:
            plot.setXLink(plots[0])
            plot.setYLink(plots[0])


class ElbowPlotWindow(QDialog):
    def __init__(self, bics, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HMM State Selection - BIC Elbow Plot")

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.plot_widget = pg.PlotWidget(title="BIC vs Number of States")
        layout.addWidget(self.plot_widget)

        states, bic_values = zip(*bics)
        bic_values = np.array(bic_values)
        valid = ~np.isnan(bic_values)
        self.plot_widget.plot(np.array(states)[valid], bic_values[valid], pen='b', symbol='o')
        self.plot_widget.setLabel('bottom', 'Number of States')
        self.plot_widget.setLabel('left', 'BIC (lower is better)')


class DwellTimeWindow(QDialog):
    def __init__(self, dwell_times_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dwell Time Distributions")
        self.dwell_times_dict = dwell_times_dict

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Controls
        control_layout = QHBoxLayout()

        self.min_bin_box = QDoubleSpinBox()
        self.min_bin_box.setRange(0.0, 90000.0)
        self.min_bin_box.setSuffix(" ms")
        self.min_bin_box.setValue(0.0)
        control_layout.addWidget(QLabel("Min Bin:"))
        control_layout.addWidget(self.min_bin_box)

        self.max_bin_box = QDoubleSpinBox()
        self.max_bin_box.setRange(0.1, 90000.0)
        self.max_bin_box.setSuffix(" ms")
        self.max_bin_box.setValue(1000.0)
        control_layout.addWidget(QLabel("Max Bin:"))
        control_layout.addWidget(self.max_bin_box)

        self.bin_spinner = QSpinBox()
        self.bin_spinner.setRange(10, 200)
        self.bin_spinner.setValue(31)
        control_layout.addWidget(QLabel("Number of Bins:"))
        control_layout.addWidget(self.bin_spinner)

        self.normalize_checkbox = QCheckBox("Normalize Histogram")
        control_layout.addWidget(self.normalize_checkbox)

        save_button = QPushButton("Save Histograms and Fits")
        save_button.clicked.connect(self.save_histograms)
        control_layout.addWidget(save_button)

        main_layout.addLayout(control_layout)

        # Plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.plot_widget)

        # Storage for histograms and fit results
        self.histograms = {}
        self.fits = {}

        # Connect updates
        self.bin_spinner.valueChanged.connect(self.update_plots)
        self.min_bin_box.valueChanged.connect(self.update_plots)
        self.max_bin_box.valueChanged.connect(self.update_plots)
        self.normalize_checkbox.stateChanged.connect(self.update_plots)

        self.update_plots()

    @staticmethod
    def _exp_func(x, A, tau):
        return A * np.exp(-x / tau)

    def update_plots(self):
        self.plot_widget.clear()
        n_bins = self.bin_spinner.value()
        min_bin = self.min_bin_box.value()
        max_bin = self.max_bin_box.value()
        normalize = self.normalize_checkbox.isChecked()

        if max_bin <= min_bin:
            return

        bins = np.linspace(min_bin, max_bin, n_bins + 1)
        self.histograms.clear()
        self.fits.clear()
        plots = []

        for state, dwell_times in self.dwell_times_dict.items():
            if len(dwell_times) == 0:
                continue

            plot = self.plot_widget.addPlot()
            plots.append(plot)

            # Histogram data
            hist_data = np.array(dwell_times) * 1e3
            y, x = np.histogram(hist_data, bins=bins)
            if normalize:
                y = y / y.sum() if y.sum() > 0 else y
            x_center = 0.5 * (x[:-1] + x[1:])
            self.histograms[state] = (x_center, y)

            # Exponential fit
            mask = y > 0
            try:
                popt, _ = curve_fit(
                    self._exp_func,
                    x_center[mask],
                    y[mask],
                    p0=(y.max(), (x_center * y).sum() / y.sum())
                )
                A_fit, tau_fit = popt
                self.fits[state] = (A_fit, tau_fit)

                # Annotate fit on plot with margin
                fit_text = f"τ = {tau_fit:.1f} ms"
                text_item = TextItem(fit_text, anchor=(0, 0))
                plot.addItem(text_item)

                vb = plot.getViewBox()
                rect = vb.viewRect()
                x_min, x_max = rect.left(), rect.right()
                y_min, y_max = rect.bottom(), rect.top()
                # 15% margin inside the view
                x_offset = 0.15 * (x_max - x_min)
                y_offset = 0.15 * (y_max - y_min)
                text_item.setPos(x_max - x_offset, y_max - y_offset)

                # Plot fit curve
                x_fit = np.linspace(min_bin, max_bin, 200)
                y_fit = self._exp_func(x_fit, *popt)
                plot.plot(x_fit, y_fit, pen=pg.mkPen('r', width=2))
            except Exception:
                pass

            color = (state * 40 % 255, 100, 150, 150)
            plot.plot(x_center, y, stepMode=False, fillLevel=0, brush=color)
            plot.setLabel('bottom', f'Dwell Time (ms) - State {state}')
            plot.hideAxis('left')

        if len(plots) > 1:
            for p in plots[1:]:
                p.setXLink(plots[0])
            if normalize:
                for p in plots[1:]:
                    p.setYLink(plots[0])

    def save_histograms(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Dwell Time Histograms and Fits", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        with open(file_path, 'w') as f:
            f.write("State,BinCenter,Count\n")
            for state, (x_center, y) in self.histograms.items():
                for xc, yc in zip(x_center, y):
                    f.write(f"{state},{xc},{yc}\n")
            f.write("\nState,Parameter,Value\n")
            for state, (A_fit, tau_fit) in self.fits.items():
                f.write(f"{state},A,{A_fit}\n")
                f.write(f"{state},tau,{tau_fit}\n")
        print(f"Histograms and fits saved to {file_path}")


class TransitionMatrixWindow(QDialog):
    def __init__(self, matrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HMM Transition Matrix")
        layout = QVBoxLayout()
        plot_widget = pg.PlotWidget()
        img = ImageItem(matrix.T)
        cmap = colormap.get("viridis")
        img.setLookupTable(cmap.getLookupTable())
        img.setLevels([0, np.max(matrix)])
        plot_widget.addItem(img)
        plot_widget.setLabel('left', 'To State')
        plot_widget.setLabel('bottom', 'From State')
        plot_widget.getViewBox().invertY(True)
        layout.addWidget(plot_widget)
        self.setLayout(layout)

class IntensityPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.plot_widget)
        self.plots = []

    def _create_log_hist_plot(self, linked_y_plot, show_x_axis):
        log_axis = pg.AxisItem(orientation='bottom', logMode=True)
        hist_plot = pg.PlotItem(axisItems={'bottom': log_axis})
        hist_plot.setYLink(linked_y_plot)
        hist_plot.getViewBox().invertX(False)
        hist_plot.hideAxis('left')
        if not show_x_axis:
            hist_plot.hideAxis('bottom')
        else:
            hist_plot.setLabel('bottom', 'Counts (log)')
        return hist_plot

    def _add_fill_between_yaxis_and_curve(self, plot, x_data, y_data, color=(255, 0, 0, 80)):
        path = QPainterPath()
        path.moveTo(0, y_data[0])
        for x, y in zip(x_data, y_data):
            path.lineTo(x, y)
        path.lineTo(0, y_data[-1])
        path.closeSubpath()

        item = QGraphicsPathItem(path)
        item.setBrush(QBrush(QColor(*color)))
        item.setPen(QPen(Qt.NoPen))
        plot.addItem(item)

    def plot_trace_and_histogram(
        self, time_axis, traces, channel_labels=None,
        bin_count=100, time_window_ms=10.0,
        hist_min=None, hist_max=None, hmm_states=None
    ):
        self.plot_widget.clear()
        self.plots.clear()

        self.plot_widget.ci.layout.setColumnStretchFactor(0, 2)
        self.plot_widget.ci.layout.setColumnStretchFactor(1, 1)

        n_channels = traces.shape[1]
        if channel_labels is None:
            channel_labels = [f"{i}" for i in range(n_channels)]

        for i in range(n_channels):
            trace = traces[:, i]
            label = channel_labels[i]
            show_x = (i == n_channels)

            trace_plot = self.plot_widget.addPlot(row=i, col=0)
            trace_plot.plot(time_axis, trace, pen='b')
            trace_plot.setLabel('left', f'Ch {label}\nCounts / {int(time_window_ms)} ms')
            if not show_x:
                trace_plot.hideAxis('bottom')
            else:
                trace_plot.setLabel('bottom', 'Time', units='s')

            hist_plot = self._create_log_hist_plot(trace_plot, show_x)
            self.plot_widget.addItem(hist_plot, row=i, col=1)

            data = trace[trace > 0]
            if hist_min is not None and hist_max is not None:
                data = data[(data >= hist_min) & (data <= hist_max)]

            if len(data) > 0:
                counts, bins = np.histogram(data, bins=bin_count, density=False)
                centers = 0.5 * (bins[:-1] + bins[1:])
                hist_plot.addItem(pg.PlotCurveItem(counts, centers[1:], pen='r', stepMode=True))
                self._add_fill_between_yaxis_and_curve(hist_plot, counts, centers)

            self.plots.append((trace_plot, hist_plot))

        combined = traces.sum(axis=1)
        trace_plot = self.plot_widget.addPlot(row=n_channels, col=0)
        trace_plot.plot(time_axis, combined, pen='g')
        trace_plot.setLabel('left', f'Sum\nCounts / {int(time_window_ms)} ms')
        trace_plot.setLabel('bottom', 'Time', units='s')

        hist_plot = self._create_log_hist_plot(trace_plot, True)
        self.plot_widget.addItem(hist_plot, row=n_channels, col=1)

        data = combined[combined > 0]
        if hist_min is not None and hist_max is not None:
            data = data[(data >= hist_min) & (data <= hist_max)]

        if len(data) > 0:
            counts, bins = np.histogram(data, bins=bin_count, density=False)
            centers = 0.5 * (bins[:-1] + bins[1:])
            curve = pg.PlotCurveItem(counts, centers[1:], pen='r', stepMode=True)
            hist_plot.addItem(curve)
            self._add_fill_between_yaxis_and_curve(hist_plot, counts, centers)

        self.plots.append((trace_plot, hist_plot))

        if hmm_states is not None:
            state_plot = self.plot_widget.addPlot(row=n_channels+1, col=0, colspan=1)
            state_plot.setXLink(self.plots[0][0])
            state_plot.plot(time_axis, hmm_states, pen='y')
            state_plot.setLabel('left', 'Hidden States')
            state_plot.setLabel('bottom', 'Time', units='s')

            combined_vals = traces.sum(axis=1)
            means = []
            for s in range(int(np.max(hmm_states)) + 1):
                means.append(np.mean(combined_vals[hmm_states == s]))
            order = np.argsort(means)
            remap = np.zeros_like(order)
            for new_idx, old in enumerate(order):
                remap[old] = new_idx
            sorted_states = remap[hmm_states]

            state_plot.clear()
            state_plot.plot(time_axis, sorted_states, pen='y')

            counts = np.bincount(sorted_states, minlength=len(order))
            state_hist = self.plot_widget.addPlot(row=n_channels+1, col=1, colspan=1)
            bg = pg.BarGraphItem(x=np.arange(len(order)), height=counts, width=0.8, brush=(200, 200, 100, 200))
            state_hist.addItem(bg)
            state_hist.setLabel('bottom', 'State (sorted)')
            state_hist.setLabel('left', 'Count')

            if traces.shape[1] >= 2:
                ch0 = traces[:, 0]
                total = np.clip(traces.sum(axis=1), 1e-6, None)
                fret_eff = ch0 / total

                mask = ch0 >= (hist_min or 0.0)
                fret_by_state = {}
                weights = {}

                if hmm_states is not None:
                    for s in range(int(np.max(hmm_states)) + 1):
                        valid = (hmm_states == s) & mask
                        state_fret = fret_eff[valid]
                        if len(state_fret) > 0:
                            fret_by_state[s] = state_fret
                            weights[s] = len(state_fret) / np.sum(
                                [np.sum((hmm_states == ss) & mask) for ss in fret_by_state.keys()])

                    # Create combined histogram
                    bins = np.linspace(0, 1, 51)
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    combined_hist = np.zeros_like(bin_centers)

                    fret_plot = self.plot_widget.addPlot(row=n_channels + 2, col=0)
                    fret_plot.setXLink(self.plots[0][0])
                    fret_plot.setLabel('left', 'FRET Hist\nper State')
                    fret_plot.setLabel('bottom', 'FRET Efficiency')

                    hist_plot = pg.PlotItem()
                    self.plot_widget.addItem(hist_plot, row=n_channels + 2, col=1)

                    for state, state_fret in fret_by_state.items():
                        hist, _ = np.histogram(state_fret, bins=bins, density=True)
                        combined_hist += weights[state] * hist
                        hist_plot.plot(bin_centers, hist[1:], stepMode=True, fillLevel=0,
                                       brush=(state * 50 % 255, 100, 180, 100))

                    # Add combined histogram
                    hist_plot.plot(bin_centers, combined_hist[1:], stepMode=True, pen='k')

                    fret_plot.plot(time_axis, fret_eff, pen='m')

        for plot, _ in self.plots[1:]:
            plot.setXLink(self.plots[0][0])
        for _, hist in self.plots[1:]:
            hist.setXLink(self.plots[0][1])

class IntensityTrace(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTTR Intensity Trace Viewer")

        # 1) Instantiate the plot widget
        self.plot_widget = IntensityPlotWidget(self)

        # 2) Instantiate labels and buttons
        self.file_label = QLabel("No file selected")

        self.load_button    = QPushButton("Load TTTR")
        self.load_button.clicked.connect(self.load_file)

        self.save_button    = QPushButton("Save Traces")
        self.save_button.clicked.connect(self.save_output)

        self.dist_button    = QPushButton("FRET Distributions")
        self.dist_button.clicked.connect(self.show_fret_distributions)

        self.elbow_button   = QPushButton("BIC Elbow")
        self.elbow_button.clicked.connect(self.show_elbow_plot)

        self.hmm_button     = QPushButton("HMM")
        self.hmm_button.clicked.connect(self.perform_hmm)

        self.matrix_button  = QPushButton("HMM Matrix")
        self.matrix_button.clicked.connect(self.show_matrix)

        self.dwell_button   = QPushButton("Dwell Times")
        self.dwell_button.clicked.connect(self.show_dwell_times)

        # 3) Instantiate input fields and spin boxes
        self.window_input = QLineEdit("10")  # time window in ms
        self.channel_input = QLineEdit("0,2")

        self.bin_spinner = QSpinBox()
        self.bin_spinner.setRange(10, 500)
        self.bin_spinner.setValue(41)

        self.hist_min_input = QDoubleSpinBox()
        self.hist_min_input.setRange(0.0, 10000.0)
        self.hist_min_input.setDecimals(2)
        self.hist_min_input.setValue(20.0)

        self.hist_max_input = QDoubleSpinBox()
        self.hist_max_input.setRange(0.0, 10000.0)
        self.hist_max_input.setDecimals(2)
        self.hist_max_input.setValue(1000.0)

        self.hmm_components_spinner = QSpinBox()
        self.hmm_components_spinner.setRange(1, 15)
        self.hmm_components_spinner.setValue(9)

        # 4) Layout: use QGridLayout for controls
        controls_layout = QGridLayout()

        # Row 0: action buttons
        controls_layout.addWidget(self.load_button,   0, 0)
        controls_layout.addWidget(self.hmm_button,    0, 1)
        controls_layout.addWidget(self.dist_button,   0, 2)
        controls_layout.addWidget(self.dwell_button,  0, 3)
        controls_layout.addWidget(self.matrix_button, 0, 4)
        controls_layout.addWidget(self.save_button,   0, 5)
        controls_layout.addWidget(self.elbow_button,  0, 6)

        # Row 1: first batch of parameters
        controls_layout.addWidget(QLabel("Time window (ms):"), 1, 0)
        controls_layout.addWidget(self.window_input,           1, 1)
        controls_layout.addWidget(QLabel("Channels:"),        1, 2)
        controls_layout.addWidget(self.channel_input,          1, 3)
        controls_layout.addWidget(QLabel("Hist bins:"),       1, 4)
        controls_layout.addWidget(self.bin_spinner,            1, 5)

        # Row 2: second batch of parameters
        controls_layout.addWidget(QLabel("Min:"),             2, 0)
        controls_layout.addWidget(self.hist_min_input,         2, 1)
        controls_layout.addWidget(QLabel("Max:"),             2, 2)
        controls_layout.addWidget(self.hist_max_input,         2, 3)
        controls_layout.addWidget(QLabel("HMM Components:"),  2, 4)
        controls_layout.addWidget(self.hmm_components_spinner, 2, 5)

        # 5) Compose the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.file_label)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.plot_widget)
        self.setLayout(main_layout)

        self.current_data = None

        self.bin_spinner.valueChanged.connect(self.update_plot)
        self.hist_min_input.valueChanged.connect(self.update_plot)
        self.hist_max_input.valueChanged.connect(self.update_plot)

    def show_fret_distributions(self):
        if not self.current_data or 'hmm_states' not in self.current_data:
            return

        traces = self.current_data['padded']
        states = self.current_data['hmm_states']
        if traces.shape[1] < 2:
            print("Need at least two channels for FRET efficiency.")
            return

        fret_eff = traces[:, 0] / np.clip(traces.sum(axis=1), 1e-6, None)
        fret_by_state = {}

        for s in range(np.max(states) + 1):
            fret_by_state[s] = fret_eff[states == s]

        dlg = DistPlotWindow(fret_by_state, self)
        dlg.exec_()

    def show_elbow_plot(self):
        if not self.current_data:
            return
        padded = self.current_data['padded']
        max_states = self.hmm_components_spinner.maximum()
        bics = compute_bic_curve(padded, max_states=max_states)
        dlg = ElbowPlotWindow(bics, self)
        dlg.exec_()

    def show_dwell_times(self):
        if not self.current_data or 'hmm_states' not in self.current_data:
            return
        time_step = self.current_data['window_ms'] / 1000.0
        dwell_times = compute_dwell_times(self.current_data['hmm_states'], time_step)
        dlg = DwellTimeWindow(dwell_times, self)
        dlg.exec_()

    def show_matrix(self):
        if self.current_data and 'transmat' in self.current_data:
            matrix = self.current_data['transmat']
            dlg = TransitionMatrixWindow(matrix, self)
            dlg.exec_()

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PTU File", "", "PTU Files (*.ptu)")
        if not file_path:
            return

        self.file_label.setText(f"Selected file: {file_path}")
        time_window_ms = float(self.window_input.text().strip())
        time_window_s = time_window_ms / 1000.0

        channel_text = self.channel_input.text().strip()
        selected_channels = list(map(int, channel_text.split(','))) if channel_text else None

        time_axis, padded, all_chs = self.process_ptu(pathlib.Path(file_path), time_window_s, selected_channels)
        self.current_data = {'time_axis': time_axis, 'padded': padded, 'channels': all_chs, 'window_ms': time_window_ms, 'hmm_states': None}
        self.update_plot()

    def perform_hmm(self):
        if not self.current_data:
            return
        n_comp = self.hmm_components_spinner.value()
        traces = self.current_data['padded']
        hmm_states, transmat = self.apply_hmm(traces, n_components=n_comp)
        self.current_data['hmm_states'] = hmm_states
        self.current_data['transmat'] = transmat

        # Save burst IDs
        output_dir = QFileDialog.getExistingDirectory(self, "Select Directory for Burst IDs")
        if output_dir:
            save_burst_ids(hmm_states, self.current_data['time_axis'],
                           self.current_data['window_ms'] / 1000.0,
                           tttrlib.TTTR(self.file_label.text().replace("Selected file: ", "")),
                           output_dir=output_dir)

        self.update_plot()

    def save_output(self):
        pass  # Skipped to save space

    def update_plot(self):
        if not self.current_data:
            return
        bin_count = self.bin_spinner.value()
        hist_min = self.hist_min_input.value()
        hist_max = self.hist_max_input.value()
        self.plot_widget.plot_trace_and_histogram(
            self.current_data['time_axis'], self.current_data['padded'], self.current_data['channels'],
            bin_count=bin_count, time_window_ms=self.current_data['window_ms'],
            hist_min=hist_min, hist_max=hist_max, hmm_states=self.current_data.get('hmm_states')
        )

    def process_ptu(self, ptu_file, time_window_length, selected_chs):
        tttr_obj = tttrlib.TTTR(str(ptu_file))
        all_chs = sorted(tttr_obj.get_used_routing_channels())
        sel_chs = [ch for ch in selected_chs if ch in all_chs] if selected_chs else all_chs
        traces = []
        for ch in sel_chs:
            idxs = np.where(tttr_obj.routing_channels == ch)[0]
            sub_tttr = tttr_obj[idxs]
            counts = sub_tttr.get_intensity_trace(time_window_length)
            traces.append(counts)
        num_bins = max(len(t) for t in traces)
        padded = np.zeros((num_bins, len(traces)))
        for i, t in enumerate(traces):
            padded[:len(t), i] = t
        time_axis = np.arange(num_bins) * time_window_length
        return time_axis, padded, sel_chs

    def apply_hmm(self, traces, n_components=2):
        model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000)
        model.fit(traces)
        states = model.predict(traces)
        transmat = model.transmat_

        combined = traces.sum(axis=1)
        means = [np.mean(combined[states == s]) for s in range(n_components)]
        order = np.argsort(means)
        remap = np.zeros(n_components, int)
        for new_label, old_label in enumerate(order):
            remap[old_label] = new_label
        sorted_states = remap[states]
        transmat_sorted = transmat[np.ix_(order, order)]
        return sorted_states, transmat_sorted

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntensityTrace()
    window.show()
    sys.exit(app.exec_())

elif __name__ == "plugin":
    window = IntensityTrace()
    window.show()
