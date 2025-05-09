import sys
import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QScrollArea, QSpinBox
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import tttrlib  # ensure tttrlib is in your PYTHONPATH

class FileLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drag TTTR file or type path...")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.setText(path)
            main_win = self.window()
            if hasattr(main_win, 'load_file'):
                main_win.load_file(path)
        event.acceptProposedAction()

class AlexPTUCreator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALEX PTU Creator")
        self.resize(800, 600)

        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        self.setCentralWidget(central)

        # Controls panel
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(2, 2, 2, 2)
        controls_layout.setSpacing(2)
        main_layout.addWidget(controls, stretch=1)

        # Load/Save
        load_layout = QHBoxLayout()
        load_layout.setContentsMargins(0, 0, 0, 0)
        load_layout.setSpacing(2)
        btn_load = QPushButton("Load…")
        btn_load.clicked.connect(self.open_file_dialog)
        load_layout.addWidget(btn_load)
        self.file_edit = FileLineEdit(self)
        self.file_edit.returnPressed.connect(lambda: self.load_file(self.file_edit.text()))
        load_layout.addWidget(self.file_edit)
        controls_layout.addLayout(load_layout)

        # ALEX Period input
        period_layout = QHBoxLayout()
        period_layout.setContentsMargins(0, 0, 0, 0)
        period_layout.setSpacing(2)
        period_label = QLabel("ALEX Period:")
        period_layout.addWidget(period_label)
        self.period_spinbox = QSpinBox()
        self.period_spinbox.setRange(1, 1000000)
        self.period_spinbox.setValue(8000)  # Default value from the notebook
        self.period_spinbox.valueChanged.connect(self.update_histogram)
        period_layout.addWidget(self.period_spinbox)
        controls_layout.addLayout(period_layout)

        # Period Shift input
        shift_layout = QHBoxLayout()
        shift_layout.setContentsMargins(0, 0, 0, 0)
        shift_layout.setSpacing(2)
        shift_label = QLabel("Period Shift:")
        shift_layout.addWidget(shift_label)
        self.shift_spinbox = QSpinBox()
        self.shift_spinbox.setRange(-1000000, 1000000)
        self.shift_spinbox.setValue(0)  # Default value
        self.shift_spinbox.valueChanged.connect(self.update_histogram)
        shift_layout.addWidget(self.shift_spinbox)
        controls_layout.addLayout(shift_layout)

        # Save button
        save_layout = QHBoxLayout()
        save_layout.setContentsMargins(0, 0, 0, 0)
        save_layout.setSpacing(2)
        self.btn_save = QPushButton("Save PTU…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_file)
        save_layout.addWidget(self.btn_save)
        controls_layout.addLayout(save_layout)

        # Add stretch to push controls to the top
        controls_layout.addStretch()

        # Plot
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        plot_layout.setSpacing(2)
        main_layout.addWidget(plot_container, stretch=3)

        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Micro-time bin')
        self.plot.setLabel('left', 'Counts')
        plot_layout.addWidget(self.plot)

        # state
        self.tttr = None
        self.tttr_path = None
        self.orig_mt = None
        self.routing = None

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open TTTR file", "", "TTTR Files (*.*)")
        if path:
            self.file_edit.setText(path)
            self.load_file(path)

    def load_file(self, path):
        try:
            self.tttr = tttrlib.TTTR(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot load file:\n{e}")
            return

        self.orig_mt = self.tttr.macro_times.copy()
        self.routing = self.tttr.routing_channels.copy()
        self.tttr_path = path
        self.btn_save.setEnabled(True)

        # Update the histogram with current settings
        self.update_histogram()

    def update_histogram(self):
        if self.tttr is None:
            return

        # Get the current ALEX period and shift values
        alex_period = self.period_spinbox.value()
        period_shift = self.shift_spinbox.value()

        # Create a copy of the original TTTR object to avoid modifying the original
        tt = tttrlib.TTTR(self.tttr_path)

        # Apply the ALEX to microtime conversion
        tt.alex_to_microtime(alex_period, period_shift)

        # Get the microtime histogram
        mt_hist = np.bincount(tt.micro_times, minlength=alex_period)

        # Create edges array for step mode plotting (needs to be len(Y) + 1)
        edges = np.arange(alex_period + 1)

        # Plot the histogram
        self.plot.clear()
        self.plot.plot(edges, mt_hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 80))
        self.plot.setTitle(f'ALEX Microtime Histogram (Period: {alex_period}, Shift: {period_shift})')

    def save_file(self):
        if self.tttr is None:
            return

        # Get save path
        d = os.path.dirname(self.tttr_path)
        f = os.path.basename(self.tttr_path)
        base, ext = os.path.splitext(f)
        default_name = f"{base}_alex{ext}"

        sp, _ = QFileDialog.getSaveFileName(
            self, "Save ALEX PTU file", 
            os.path.join(d, default_name), 
            "PTU Files (*.ptu);;All Files (*.*)"
        )

        if not sp:
            return

        try:
            # Create a new TTTR object with the original file
            tt = tttrlib.TTTR(self.tttr_path)

            # Apply the ALEX to microtime conversion
            alex_period = self.period_spinbox.value()
            period_shift = self.shift_spinbox.value()
            tt.alex_to_microtime(alex_period, period_shift)

            # Save the file
            tt.write(sp)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to:\n{sp}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot save:\n{e}")

if __name__ == 'plugin':
    w = AlexPTUCreator(); w.show()
elif __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = AlexPTUCreator(); w.show()
    sys.exit(app.exec_())
