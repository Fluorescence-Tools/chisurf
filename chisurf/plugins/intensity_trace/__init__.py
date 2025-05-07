import sys
import pathlib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtGui import QPainterPath, QBrush, QColor, QPen
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import tttrlib
from PyQt5.QtWidgets import QGraphicsPathItem


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
        path.moveTo(0, y_data[0])  # Start at y-axis
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
        hist_min=None, hist_max=None
    ):
        self.plot_widget.clear()
        self.plots.clear()

        self.plot_widget.ci.layout.setColumnStretchFactor(0, 3)
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

        for plot, _ in self.plots[1:]:
            plot.setXLink(self.plots[0][0])
        for _, hist in self.plots[1:]:
            hist.setXLink(self.plots[0][1])


class IntensityTrace(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTTR Intensity Trace Viewer")

        self.plot_widget = IntensityPlotWidget()
        self.file_label = QLabel("No file selected")
        self.window_input = QLineEdit("10")  # ms
        self.channel_input = QLineEdit("0,2")

        self.bin_spinner = QSpinBox()
        self.bin_spinner.setMinimum(10)
        self.bin_spinner.setMaximum(500)
        self.bin_spinner.setValue(41)

        self.hist_min_input = QDoubleSpinBox()
        self.hist_min_input.setRange(0.0, 10000.0)
        self.hist_min_input.setDecimals(2)
        self.hist_min_input.setValue(20.0)

        self.hist_max_input = QDoubleSpinBox()
        self.hist_max_input.setRange(0.0, 10000.0)
        self.hist_max_input.setDecimals(2)
        self.hist_max_input.setValue(1000.0)

        load_button = QPushButton("Load .ptu File")
        load_button.clicked.connect(self.load_file)

        save_button = QPushButton("Save Traces")
        save_button.clicked.connect(self.save_output)

        # Controls layout
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Time window (ms):"))
        controls_layout.addWidget(self.window_input)
        controls_layout.addWidget(QLabel("Channels:"))
        controls_layout.addWidget(self.channel_input)
        controls_layout.addWidget(QLabel("Hist bins:"))
        controls_layout.addWidget(self.bin_spinner)
        controls_layout.addWidget(QLabel("Min:"))
        controls_layout.addWidget(self.hist_min_input)
        controls_layout.addWidget(QLabel("Max:"))
        controls_layout.addWidget(self.hist_max_input)
        controls_layout.addWidget(load_button)
        controls_layout.addWidget(save_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.file_label)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.plot_widget)
        self.setLayout(main_layout)

        self.current_data = None

        # auto-update when user changes histogram settings
        self.bin_spinner.valueChanged.connect(self.update_plot)
        self.hist_min_input.valueChanged.connect(self.update_plot)
        self.hist_max_input.valueChanged.connect(self.update_plot)

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

        # store for re-plotting on parameter change
        self.current_data = {
            'time_axis': time_axis,
            'padded': padded,
            'channels': all_chs,
            'window_ms': time_window_ms
        }

        self.update_plot()

    def save_output(self):
        if not self.current_data:
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Output")
        if not save_dir:
            return

        save_dir = pathlib.Path(save_dir)
        time_axis = self.current_data['time_axis']
        padded = self.current_data['padded']
        chs = self.current_data['channels']

        # Save trace CSV
        trace_csv = save_dir / "intensity_traces.csv"
        header = "Time(s)," + ",".join(f"Ch{ch}" for ch in chs)
        data = np.column_stack((time_axis, padded))
        np.savetxt(trace_csv, data, delimiter=",", header=header, comments='')
        print(f"Saved: {trace_csv}")

        # Save histogram CSV
        bin_count = self.bin_spinner.value()
        hist_min = self.hist_min_input.value()
        hist_max = self.hist_max_input.value()

        hist_data = []
        for i in range(padded.shape[1]):
            data_i = padded[:, i]
            data_i = data_i[(data_i > 0) & (data_i >= hist_min) & (data_i <= hist_max)]
            counts, bins = np.histogram(data_i, bins=bin_count, density=False)
            centers = 0.5 * (bins[:-1] + bins[1:])
            if i == 0:
                hist_data.append(centers)
            hist_data.append(counts)

        hist_csv = save_dir / "histograms.csv"
        hist_data = np.column_stack(hist_data)
        header = "BinCenter," + ",".join(f"Ch{ch}" for ch in chs)
        np.savetxt(hist_csv, hist_data, delimiter=",", header=header, comments='')
        print(f"Saved: {hist_csv}")

        # Save Screenshot
        screenshot_path = save_dir / "intensity_viewer.png"
        pixmap = self.grab()
        pixmap.save(str(screenshot_path))
        print(f"Saved screenshot: {screenshot_path}")

    def update_plot(self):
        if not self.current_data:
            return

        bin_count = self.bin_spinner.value()
        hist_min = self.hist_min_input.value()
        hist_max = self.hist_max_input.value()

        self.plot_widget.plot_trace_and_histogram(
            self.current_data['time_axis'],
            self.current_data['padded'],
            self.current_data['channels'],
            bin_count=bin_count,
            time_window_ms=self.current_data['window_ms'],
            hist_min=hist_min,
            hist_max=hist_max
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntensityTrace()
    window.show()
    sys.exit(app.exec_())

elif __name__ == "plugin":
    window = IntensityTrace()
    window.show()
