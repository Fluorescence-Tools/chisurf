from __future__ import annotations


import os
from qtpy import  QtWidgets, uic

import mfm
from mfm.io.ascii import Csv
from .photons import Photons
from .tttr import filetypes


class SpcFileWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "spcSampleSelectWidget.ui"
            ),
            self
        )

        self._photons = None
        self.parent = parent
        self.filenames = list()
        self.filetypes = filetypes

        self.actionSample_changed.triggered.connect(self.onSampleChanged)
        self.actionLoad_sample.triggered.connect(self.onLoadSample)
        #self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), self.onFileTypeChanged)

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
        index = 0 # TODO: fix multiple samples per HDF self.comboBox.currentIndex()
        self._photons.sample = self.samples[index]
        self.dt = float(self._photons.mt_clk / self._photons.n_tac) * 1e6
        self.nTAC = self._photons.n_tac
        self.nROUT = self._photons.n_rout
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
    def nROUT(
            self
    ) -> int:
        return int(self.lineEdit_3.text())

    @nROUT.setter
    def nROUT(
            self,
            v: int
    ):
        self.lineEdit_3.setText(str(v))

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

    """
    @property
    def filetypes(self):
        return self._file_types

    @filetypes.setter
    def filetypes(self, v):
        self._file_types = v
        #self.comboBox_2.addItems(list(v.keys()))
    """

    @property
    def count_rate(
            self
    ) -> float:
        return self._photons.nPh / float(self._photons.measurement_time) / 1000.0

    def onFileTypeChanged(self):
        self._photons = None
        #self.comboBox.clear()
        #if self.fileType == "hdf":
        #    self.comboBox.setDisabled(False)
        #else:
        #    self.comboBox.setDisabled(True)

    @property
    def fileType(
            self
    ) -> str:
        return "hdf"
        #return str(self.comboBox_2.currentText())

    @property
    def filename(
            self
    ) -> str:
        try:
            return self.filenames[0]
        except:
            return "--"

    def onLoadSample(
            self
    ) -> None:
        if self.fileType in ("hdf"):
            filename = mfm.widgets.get_filename('Open Photon-HDF', 'Photon-HDF (*.photon.h5)')
            filenames = [filename]
            self.lineEdit_2.setText(filename)
        elif self.fileType in ("ht3"):
            filename = mfm.widgets.open_file('Open Photon-HDF', 'Photon-HDF (*.ht3)')
            filenames = [filename]
        else:
            directory = mfm.widgets.get_directory()
            filenames = [directory + '/' + s for s in os.listdir(directory)]

        self.filenames = filenames
        self._photons = Photons(filenames, self.fileType)
        self.samples = self._photons.samples
        #self.comboBox.addItems(self._photons.sample_names)
        self.onSampleChanged()

    @property
    def photons(
            self
    ) -> Photons:
        return self._photons


class PDBLoad(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PDBLoad, self).__init__()
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "proteinMCLoad.ui"
            ),
            self
        )

        self._data = None
        self._filename = ''

    def load(self, filename=None):
        if filename is None:
            filename = mfm.widgets.get_filename('Open PDB-Structure', 'PDB-file (*.pdb)')
        self.filename = filename
        self.structure = self.filename
        self.lineEdit.setText(str(self.structure.n_atoms))
        self.lineEdit_2.setText(str(self.structure.n_residues))

    @property
    def filename(self):
        return str(self.lineEdit_7.text())

    @filename.setter
    def filename(self, v):
        self.lineEdit_7.setText(v)

    @property
    def calcLookUp(self):
        return self.checkBox.isChecked()

    @property
    def structure(self):
        return self._data

    @structure.setter
    def structure(self, v):
        self._data = mfm.structure.structure.Structure(v, make_coarse=self.calcLookUp)


#class CsvWidget(Csv, QtGui.QWidget):
class CsvWidget(QtWidgets.QWidget):

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "csvInput.ui"
            ),
            self
        )

        #Csv.__init__(self, **kwargs)
        #self.connect(self.spinBox, QtCore.SIGNAL("valueChanged(int)"), self.reload_csv)
        self.actionUseHeader.triggered.connect(self.changeUseHeader)
        self.actionSkiprows.triggered.connect(self.changeSkiprows)
        self.actionColspecs.triggered.connect(self.changeColspecs)
        self.actionCsvType.triggered.connect(self.changeCsvType)
        self.actionSetError.triggered.connect(self.changeError)
        self.verbose = kwargs.get('verbose', mfm.verbose)

    @property
    def col_ex(self) -> int:
        return self.comboBox_3.currentIndex()

    @col_ex.setter
    def col_ex(self, v):
        pass

    def changeError(self):
        set_errx_on = bool(self.checkBox_3.isChecked())
        set_erry_on = bool(self.checkBox_4.isChecked())

        mfm.run("cs.current_setup.error_x_on = %s" % set_errx_on)
        mfm.run("cs.current_setup.error_y_on = %s" % set_erry_on)

    @property
    def x_on(self) -> bool:
        return self.checkBox.isChecked()

    @x_on.setter
    def x_on(
            self,
            v: bool
    ):
        self.checkBox.setChecked(bool(v))

    @property
    def col_x(self) -> int:
        return self.comboBox.currentIndex()

    @col_x.setter
    def col_x(
            self,
            v: int
    ):
        # TODO
        pass

    @property
    def col_y(self) -> int:
        return self.comboBox_2.currentIndex()

    @col_y.setter
    def col_y(
            self,
            v: int
    ):
        # TODO
        pass

    @property
    def data(self):
        return Csv.data.fget(self)

    @data.setter
    def data(self, v):
        Csv.data.fset(self, v)
        self.lineEdit_9.setText("%d" % v.shape[1])
        bx = [self.comboBox, self.comboBox_2, self.comboBox_3, self.comboBox_4]
        if self.n_rows > 0:
            for i, b in enumerate(bx):
                b.clear()
                b.addItems(self.header)
                b.setCurrentIndex(i % self.n_rows)

    def changeSkiprows(self):
        n_skip = int(self.spinBox.value())
        mfm.run("cs.current_setup.skiprows = %s" % n_skip)

    def changeUseHeader(self):
        use_header = bool(self.checkBox_2.isChecked())
        mfm.run("cs.current_setup.use_header = %s" % use_header)

    def changeColspecs(self):
        colspecs = str(self.lineEdit.text())
        mfm.run("cs.current_setup.colspecs = '%s'" % colspecs)

    @property
    def filename(self) -> str:
        return str(self.lineEdit_8.text())

    @filename.setter
    def filename(
            self,
            v: str
    ):
        Csv.filename.fset(self, v)
        self.lineEdit_8.setText(v)

    def changeCsvType(self):
        mode = 'csv' if self.radioButton_2.isChecked() else 'fwf'
        mfm.run("cs.current_setup.file_type = '%s'" % mode)

    def load(self, filename=None, **kwargs):
        if filename is None:
            filename = mfm.widgets.get_filename('Open CSV-File', 'CSV-file (*.*)')
        Csv.load(self, filename, **kwargs)
        self.filename = filename


class CSVFileWidget(QtWidgets.QWidget):

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        self.parent = kwargs.get('parent', None)
        self.name = kwargs.get('name', 'CSV-File')
        self.weight_calculation = kwargs.get('weight_calculation', None)

        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        self.csvWidget = CsvWidget(**kwargs)
        self.layout.addWidget(self.csvWidget)

    def load_data(self, filename=None):
        """
        Loads csv-data into a Curve-object
        :param filename:
        :return: Curve-object
        """
        d = mfm.experiments.data.DataCurve(setup=None)
        if filename is not None:
            self.csvWidget.load(filename)
            d.filename = filename
        else:
            self.csvWidget.load()
            d.filename = self.csvWidget.filename

        d.x, d.y = self.csvWidget.data_x, self.csvWidget.data_y
        if self.weight_calculation is None:
            d.set_weights(self.csvWidget.error_y)
        else:
            d.set_weights(self.weight_calculation(d.y))
        return d

    def get_data(self, *args, **kwargs):
        return self.load_data(*args, **kwargs)
