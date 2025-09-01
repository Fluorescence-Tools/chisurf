import sys
import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QScrollArea
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

class MicroTimeShifter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Micro-time Shifter")
        self.resize(800, 400)

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
        main_layout.addWidget(controls)

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
        self.btn_save = QPushButton("Save…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.open_save_dialog)
        load_layout.addWidget(self.btn_save)
        controls_layout.addLayout(load_layout)

        # Shift controls scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls_layout.addWidget(scroll)
        container = QWidget()
        self.shift_layout = QVBoxLayout(container)
        self.shift_layout.setContentsMargins(0, 0, 0, 0)
        self.shift_layout.setSpacing(2)
        scroll.setWidget(container)

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Micro-time bin')
        self.plot.setLabel('left', 'Counts')
        main_layout.addWidget(self.plot, stretch=1)

        # state
        self.global_shift = 0
        self.shifts = {}
        self.orig_mt = None
        self.routing = None
        self.n_mt = 0
        self.tttr_path = None

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open TTTR file", "", "TTTR Files (*.*)")
        if path:
            self.file_edit.setText(path)
            self.load_file(path)

    def load_file(self, path):
        try:
            tt = tttrlib.TTTR(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot load file:\n{e}")
            return
        self.orig_mt = tt.micro_times.copy()
        self.routing = tt.routing_channels.copy()
        self.n_mt = tt.header.get_effective_number_of_micro_time_channels()
        self.tttr_path = path
        chans = tt.get_used_routing_channels()
        self.shifts = {int(c): 0 for c in chans}
        self.global_shift = 0
        self.btn_save.setEnabled(True)
        self.build_shift_controls()
        self.plot_histograms()

    def open_save_dialog(self):
        if not self.tttr_path:
            return
        d = os.path.dirname(self.tttr_path)
        f = os.path.basename(self.tttr_path)
        sp, _ = QFileDialog.getSaveFileName(self, "Save shifted TTTR file", os.path.join(d, f), "TTTR Files (*.*)")
        if sp:
            self.save_file(sp)

    def save_file(self, sp):
        tt = tttrlib.TTTR(self.tttr_path)
        for ch, sv in self.shifts.items():
            tot = (self.global_shift + sv) % self.n_mt
            if tot:
                tt.shift_micro_time_by_channel(ch, tot)
        try:
            tt.write(sp)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to:\n{sp}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot save:\n{e}")

    def build_shift_controls(self):
        # clear
        while self.shift_layout.count():
            w = self.shift_layout.takeAt(0).widget()
            if w: w.setParent(None)
        # global
        self._add_shift_row('All', None)
        for ch in sorted(self.shifts.keys()):
            self._add_shift_row(f"Ch{ch}", ch)
        self.shift_layout.addStretch()

    def _add_shift_row(self, label, channel):
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        L = QLabel(label)
        L.setFixedWidth(30)
        rl.addWidget(L)
        spin = pg.SpinBox(value=(self.global_shift if channel is None else self.shifts[channel]), int=True, step=1, bounds=[-(self.n_mt-1), self.n_mt-1])
        spin.setFixedWidth(80)
        rl.addWidget(spin)
        fn = (self._make_global_fn(spin) if channel is None else self._make_chan_fn(channel, spin))
        spin.editingFinished.connect(fn)
        B = QPushButton('↻')
        B.setFixedWidth(30)
        B.clicked.connect(fn)
        rl.addWidget(B)
        self.shift_layout.addWidget(row)

    def _make_global_fn(self, spin):
        def fn():
            try: self.global_shift = int(spin.value())
            except: return
            self.plot_histograms()
        return fn

    def _make_chan_fn(self, ch, spin):
        def fn():
            try: self.shifts[ch] = int(spin.value())
            except: return
            self.plot_histograms()
        return fn

    def plot_histograms(self):
        self.plot.clear()
        edges = np.arange(self.n_mt+1)
        for i, ch in enumerate(sorted(self.shifts)):
            mask = self.routing==ch
            mt = (self.orig_mt[mask] + (self.global_shift+self.shifts[ch])%self.n_mt) % self.n_mt
            hist = np.bincount(mt, minlength=self.n_mt)
            c = pg.intColor(i, len(self.shifts)); c.setAlpha(100)
            self.plot.plot(edges, hist, pen=c, stepMode=True, fillLevel=0, brush=c, name=str(ch))
        try: self.plot.addLegend()
        except: pass
        self.plot.setTitle('Overlaid Micro-time Histograms')

if __name__ == 'plugin':
    w = MicroTimeShifter(); w.show()
elif __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MicroTimeShifter(); w.show()
    sys.exit(app.exec_())
