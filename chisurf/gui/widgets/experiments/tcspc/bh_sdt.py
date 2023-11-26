from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

import chisurf.fio.fluorescence.sdtfile

import chisurf.fluorescence
import chisurf.fluorescence.tcspc
import chisurf.decorators
import chisurf.data
import chisurf.gui.decorators
from chisurf.experiments.tcspc import TCSPCReader


class TcspcSDTWidget(
    QtWidgets.QWidget
):

    @property
    def name(self) -> str:
        return self.filename + " _ " + str(self.curve_number)

    @property
    def n_curves(self) -> int:
        n_data_curves = len(self._sdt.data)
        return n_data_curves

    @property
    def curve_number(self) -> int:
        return int(self.comboBox.currentIndex())

    @curve_number.setter
    def curve_number(self, v: int):
        self.comboBox.setCurrentIndex(int(v))

    @property
    def filename(self) -> str:
        return str(self.lineEdit.text())

    @filename.setter
    def filename(self, v: str):
        self._sdt = chisurf.fio.fluorescence.sdtfile.SdtFile(v)
        # refresh GUI
        self.comboBox.clear()
        l = [str(i) for i in range(self.n_curves)]
        self.comboBox.addItems(l)
        self.lineEdit.setText(str(v))
        self.lineEdit_3.setText(str(self.rep_rate))

    @property
    def sdt(self):
        if self._sdt is None:
            self.onOpenFile()
        return self._sdt

    @property
    def times(self) -> np.array:
        """
        The time-array in nano-seconds
        """
        x = self._sdt.times[0] * 1e9
        return np.array(x, dtype=np.float64)

    @property
    def ph_counts(self) -> np.array:
        """
        The currently selected fluorescence decay histogram as a numpy array
        """
        y = self._sdt.data[self.curve_number][0]
        return np.array(y, dtype=np.float64)

    @property
    def rep_rate(self) -> float:
        """
        The repetition rate used during the experiment in MHz.
        """
        return 1. / (self._sdt.measure_info[self.curve_number]['rep_t'] * 1e-3)[0]

    @rep_rate.setter
    def rep_rate(self, v):
        pass

    @property
    def curve(self) -> chisurf.data.DataCurve:
        y = self.ph_counts
        ey = chisurf.fluorescence.tcspc.counting_noise(
            decay=y
        )
        d = chisurf.data.DataCurve(
            setup=self,
            x=self.times,
            y=y,
            ey=ey,
            name=self.name
        )
        return d

    def onOpenFile(
            self,
            filename: str = None,
            curve_nbr: int = 0,
            *args,
            **kwargs
    ):

        if filename is None:
            filename = chisurf.gui.widgets.get_filename(
                'Open BH-SDT file',
                'SDT-files (*.sdt)'
            )
        self.filename = filename
        self.curve_number = curve_nbr
        self.textBrowser.setPlainText(str(self.sdt.info))

    @chisurf.gui.decorators.init_with_ui(ui_filename="tcspc_sdt.ui")
    def __init__(self, *args, **kwargs):
        self._sdt = None
        self.actionOpen_SDT_file.triggered.connect(
            self.onOpenFile
        )


class TCSPCSetupSDTWidget(TCSPCReader, QtWidgets.QWidget):

    name = "TCSPC-SDT"

    @property
    def rep_rate(self) -> float:
        return self.tcspcSDT.rep_rate

    @rep_rate.setter
    def rep_rate(self, v):
        pass

    @property
    def dt(self):
        dt = self.tcspcSDT.times[1] - self.tcspcSDT.times[0]
        return dt

    @dt.setter
    def dt(self, v):
        pass

    @property
    def curve_nbr(self):
        return self.tcspcSDT.curve_number

    @curve_nbr.setter
    def curve_nbr(self, v):
        self.tcspcSDT.curve_number = v

    def __init__(self, *args, name: str = 'Becker SDT', **kwargs):
        super().__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        self.tcspcSDT = TcspcSDTWidget()
        self.layout.addWidget(self.tcspcSDT)
        self.name = name
        #self.connect(self.tcspcSDT.actionAdd_curve, QtCore.SIGNAL('triggered()'), )

    def read(self, *args, **kwargs):
        curves = list()
        self.tcspcSDT.onOpenFile(**kwargs)
        for curve_nbr in range(self.tcspcSDT.n_curves):
            self.tcspcSDT.curve_number = curve_nbr
            curves.append(self.tcspcSDT.curve)
        return curves
