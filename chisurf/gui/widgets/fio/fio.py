from __future__ import annotations

import os
from qtpy import QtWidgets

import chisurf.decorators
import chisurf.base
import chisurf.gui.decorators
import chisurf.structure
import chisurf.gui.widgets

from chisurf.fio.fluorescence import photons
from chisurf.fio.fluorescence import tttr


class SpcFileWidget(
    QtWidgets.QWidget
):

    @chisurf.gui.decorators.init_with_ui(
        ui_filename="spcSampleSelectWidget.ui"
    )
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        self._photons = None
        self.filenames = list()
        self.filetypes = tttr.filetypes
        # Actions
        self.actionSample_changed.triggered.connect(self.onSampleChanged)
        self.actionLoad_sample.triggered.connect(self.onLoadSample)

    @property
    def sample_name(self) -> str:
        try:
            return self.filename
        except AttributeError:
            return "--"

    @property
    def dt(self) -> float:
        return float(self.doubleSpinBox.value())

    @dt.setter
    def dt(
            self,
            v: float
    ):
        self.doubleSpinBox.setValue(v)

    def onSampleChanged(self):
        self.dt = float(self._photons.mt_clk / self._photons.n_tac) * 1e6
        self.nTAC = self._photons.n_tac
        self.number_of_photons = self._photons.nPh
        self.measurement_time = self._photons.measurement_time
        self.lineEdit_7.setText("%.2f" % self.count_rate)

    @property
    def measurement_time(
            self
    ) -> float:
        return float(self._photons.measurement_time)

    @measurement_time.setter
    def measurement_time(
            self,
            v: float
    ):
        self.lineEdit_6.setText("%.1f" % v)

    @property
    def number_of_photons(
            self
    ) -> int:
        return int(self.lineEdit_5.value())

    @number_of_photons.setter
    def number_of_photons(
            self,
            v: int
    ):
        self.lineEdit_5.setText(str(v))

    @property
    def rep_rate(
            self
    ) -> float:
        return float(self.doubleSpinBox_2.value())

    @rep_rate.setter
    def rep_rate(
            self,
            v: float
    ):
        self.doubleSpinBox_2.setValue(v)

    @property
    def nTAC(
            self
    ) -> int:
        return int(self.lineEdit.text())

    @nTAC.setter
    def nTAC(
            self,
            v: int
    ):
        self.lineEdit.setText(str(v))

    @property
    def count_rate(
            self
    ) -> float:
        return self._photons.nPh / float(self._photons.measurement_time) / 1000.0

    @property
    def file_type(
            self
    ) -> str:
        return "hdf"

    @property
    def filename(
            self
    ) -> str:
        try:
            return self.filenames[0]
        except:
            return "--"

    def onLoadSample(
            self,
            event,
            filenames: str = None,
            file_type: str = None
    ) -> None:
        if file_type is None:
            file_type = self.file_type
        if filenames is None:
            if file_type in ("hdf"):
                filename = chisurf.gui.widgets.get_filename(
                    'Open Photon-HDF',
                    'Photon-HDF (*.photon.h5)'
                )
                filenames = [filename]
            elif file_type in ("ht3"):
                filename = chisurf.gui.widgets.get_filename(
                    'Open Photon-HDF',
                    'Photon-HDF (*.ht3)'
                )
                filenames = [filename]
            else:
                directory = chisurf.gui.widgets.get_directory()
                filenames = [
                    directory + '/' + s for s in os.listdir(directory)
                ]

        self.lineEdit_2.setText(filenames[0])
        self.filenames = filenames
        self._photons = chisurf.fio.fluorescence.photons.Photons(
            filenames,
            file_type
        )
        #self.samples = self._photons.samples
        #self.comboBox.addItems(self._photons.sample_names)
        self.onSampleChanged()

    @property
    def photons(
            self
    ) -> chisurf.fio.fluorescence.photons.Photons:
        return self._photons


class CsvWidget(
    chisurf.base.Base,
    QtWidgets.QWidget
):

    @chisurf.gui.decorators.init_with_ui(
        ui_filename="csvInput.ui"
    )
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.actionUseHeader.triggered.connect(self.changeCsvParameter)
        self.actionSkiprows.triggered.connect(self.changeCsvParameter)
        self.actionColspecs.triggered.connect(self.changeCsvParameter)
        self.actionCsvType.triggered.connect(self.changeCsvParameter)
        self.actionSetError.triggered.connect(self.changeCsvParameter)
        self.actionColumnsChanged.triggered.connect(
            self.changeCsvParameter
        )
        self.verbose = kwargs.get('verbose', chisurf.verbose)

    def changeCsvParameter(self):
        set_errx_on = bool(self.checkBox_3.isChecked())
        set_erry_on = bool(self.checkBox_4.isChecked())
        colspecs = str(self.lineEdit.text())
        use_header = bool(self.checkBox_2.isChecked())
        n_skip = int(self.spinBox.value())
        if self.radioButton_2.isChecked():
            mode = 'csv'
        elif self.radioButton.isChecked():
            mode = 'fwf'
        else:
            mode = 'yaml'
        chisurf.run(
            "\n".join(
                [
                    "cs.current_setup.error_y_on = %s" % set_erry_on,
                    "cs.current_setup.error_x_on = %s" % set_errx_on,
                    "cs.current_setup.colspecs = '%s'" % colspecs,
                    "cs.current_setup.use_header = %s" % use_header,
                    "cs.current_setup.skiprows = %s" % n_skip,
                    "cs.current_setup.reading_routine = '%s'" % mode,
                    "cs.current_setup.col_ey = %s" % self.spinBox_5.value(),
                    "cs.current_setup.col_ex = %s" % self.spinBox_3.value(),
                    "cs.current_setup.col_x = %s" % self.spinBox_2.value(),
                    "cs.current_setup.col_y = %s" % self.spinBox_4.value()
                ]
            )
        )

    @property
    def filename(self) -> str:
        return str(self.lineEdit_8.text())

    @filename.setter
    def filename(
            self,
            v: str
    ):
        self.lineEdit_8.setText(v)


# To be deleted
#
# class CSVFileWidget(QtWidgets.QWidget):
#
#     def __init__(
#             self,
#             *args,
#             **kwargs
#     ):
#         super().__init__(
#             *args,
#             **kwargs
#         )
#
#         layout = QtWidgets.QVBoxLayout(self)
#         layout.setSpacing(0)
#         layout.setContentsMargins(0, 0, 0, 0)
#
#         self.layout = layout
#         self.csvWidget = CsvWidget(**kwargs)
#         self.layout.addWidget(self.csvWidget)
#
#     def load_data(
#             self,
#             filename: str = None
#     ) -> chisurf.experiment.data.DataCurve:
#         """
#         Loads csv-data into a Curve-object
#         :param filename:
#         :return: Curve-object
#         """
#         d = chisurf.experiment.data.DataCurve(setup=None)
#         if filename is not None:
#             self.csvWidget.load(filename)
#             d.filename = filename
#         else:
#             self.csvWidget.load()
#             d.filename = self.csvWidget.filename
#
#         d.x, d.y = self.csvWidget.data_x, self.csvWidget.data_y
#         if self.weight_calculation is None:
#             d.set_weights(self.csvWidget.error_y)
#         else:
#             d.set_weights(self.weight_calculation(d.y))
#         return d
#
#     def get_data(self, *args, **kwargs):
#         return self.load_data(*args, **kwargs)

