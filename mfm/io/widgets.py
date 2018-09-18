import os

from PyQt4 import QtGui, uic, QtCore
import mfm
#from mfm.experiments import Setup
import mfm.structure
from mfm.io.ascii import Csv
#from mfm.structure import Structure
import mfm.widgets
from .tttr import filetypes
from .photons import Photons


class SpcFileWidget(QtGui.QWidget):

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/io/spcSampleSelectWidget.ui', self)
        self.parent = parent
        self.filetypes = filetypes

        self.connect(self.actionSample_changed, QtCore.SIGNAL('triggered()'), self.onSampleChanged)
        self.connect(self.actionLoad_sample, QtCore.SIGNAL('triggered()'), self.onLoadSample)
        #self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), self.onFileTypeChanged)

    @property
    def sample_name(self):
        try:
            return self.filename
        except AttributeError:
            return "--"

    @property
    def dt(self):
        return float(self.doubleSpinBox.value())

    @dt.setter
    def dt(self, v):
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
    def measurement_time(self):
        return float(self._photons.measurement_time)

    @measurement_time.setter
    def measurement_time(self, v):
        self.lineEdit_6.setText("%.1f" % v)

    @property
    def number_of_photons(self):
        return int(self.lineEdit_5.value())

    @number_of_photons.setter
    def number_of_photons(self, v):
        self.lineEdit_5.setText(str(v))

    @property
    def rep_rate(self):
        return float(self.doubleSpinBox_2.value())

    @rep_rate.setter
    def rep_rate(self, v):
        self.doubleSpinBox_2.setValue(v)

    @property
    def nROUT(self):
        return int(self.lineEdit_3.text())

    @nROUT.setter
    def nROUT(self, v):
        self.lineEdit_3.setText(str(v))

    @property
    def nTAC(self):
        return int(self.lineEdit.text())

    @nTAC.setter
    def nTAC(self, v):
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
    def count_rate(self):
        return self._photons.nPh / float(self._photons.measurement_time) / 1000.0

    def onFileTypeChanged(self):
        self._photons = None
        #self.comboBox.clear()
        #if self.fileType == "hdf":
        #    self.comboBox.setDisabled(False)
        #else:
        #    self.comboBox.setDisabled(True)

    @property
    def fileType(self):
        return "hdf"
        #return str(self.comboBox_2.currentText())

    @property
    def filename(self):
        try:
            return self.filenames[0]
        except AttributeError:
            return "--"

    def onLoadSample(self):
        if self.fileType in ("hdf"):
            filename = mfm.widgets.open_file('Open Photon-HDF', 'Photon-HDF (*.photon.h5)')
            filenames = [filename]
            self.lineEdit_2.setText(filename)
        """
        elif self.fileType in ("ht3"):
            filename = mfm.widgets.open_file('Open Photon-HDF', 'Photon-HDF (*.ht3)')
            filenames = [filename]
        else:
            directory = mfm.widgets.get_directory()
            filenames = [directory + '/' + s for s in os.listdir(directory)]
        """
        self.filenames = filenames
        self._photons = Photons(filenames, self.fileType)
        self.samples = self._photons.samples
        #self.comboBox.addItems(self._photons.sample_names)
        self.onSampleChanged()

    @property
    def photons(self):
        return self._photons


class PDBLoad(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        uic.loadUi("mfm/ui/proteinMCLoad.ui", self)
        self._data = None
        self._filename = ''

    def load(self, filename=None):
        if filename is None:
            filename = mfm.widgets.open_file('Open PDB-Structure', 'PDB-file (*.pdb)')
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
        self._data = mfm.structure.Structure(v, make_coarse=self.calcLookUp)


#class CsvWidget(Csv, QtGui.QWidget):
class CsvWidget(QtGui.QWidget):

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/experiments/csvInput.ui', self)
        #Csv.__init__(self, **kwargs)
        #self.connect(self.spinBox, QtCore.SIGNAL("valueChanged(int)"), self.reload_csv)
        self.connect(self.actionUseHeader, QtCore.SIGNAL('triggered()'), self.changeUseHeader)
        self.connect(self.actionSkiprows, QtCore.SIGNAL('triggered()'), self.changeSkiprows)
        self.connect(self.actionColspecs, QtCore.SIGNAL('triggered()'), self.changeColspecs)
        self.connect(self.actionCsvType, QtCore.SIGNAL('triggered()'), self.changeCsvType)
        self.connect(self.actionSetError, QtCore.SIGNAL('triggered()'), self.changeError)
        self.connect(self.actionReverseData, QtCore.SIGNAL('triggered()'), self.reverseData)
        self.verbose = kwargs.get('verbose', mfm.verbose)


    @property
    def col_ex(self):
        return self.comboBox_3.currentIndex()

    @col_ex.setter
    def col_ex(self, v):
        pass

    def reverseData(self):
        reverse = bool(self.checkBox_5.isChecked())
        mfm.run("cs.current_setup.reverse = %s" % reverse)

    def changeError(self):
        set_errx_on = bool(self.checkBox_3.isChecked())
        set_erry_on = bool(self.checkBox_4.isChecked())

        mfm.run("cs.current_setup.error_x_on = %s" % set_errx_on)
        mfm.run("cs.current_setup.error_y_on = %s" % set_erry_on)

    # @property
    # def error_y_on(self):
    #     return self.checkBox_4.isChecked()
    #
    # @error_y_on.setter
    # def error_y_on(self, v):
    #     self.checkBox_4.setChecked(bool(v))
    #
    # @property
    # def error_x_on(self):
    #     return self.checkBox_3.isChecked()
    #
    # @error_x_on.setter
    # def error_x_on(self, v):
    #     self.checkBox_3.setChecked(bool(v))

    @property
    def x_on(self):
        return self.checkBox.isChecked()

    @x_on.setter
    def x_on(self, v):
        self.checkBox.setChecked(bool(v))

    @property
    def col_x(self):
        return self.comboBox.currentIndex()

    @col_x.setter
    def col_x(self, v):
        pass

    @property
    def col_y(self):
        return self.comboBox_2.currentIndex()

    @col_y.setter
    def col_y(self, v):
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

    # @property
    # def skiprows(self):
    #     return int(self.spinBox.value())
    #
    # @skiprows.setter
    # def skiprows(self, v):
    #     self.spinBox.setValue(v)

    def changeUseHeader(self):
        use_header = bool(self.checkBox_2.isChecked())
        mfm.run("cs.current_setup.use_header = %s" % use_header)

    # @property
    # def use_header(self):
    #     if self.checkBox_2.isChecked():
    #         return True
    #     else:
    #         return None
    #
    # @use_header.setter
    # def use_header(self, v):
    #     if v is None:
    #         self.checkBox_2.setChecked(False)
    #     else:
    #         self.checkBox_2.setChecked(True)

    def changeColspecs(self):
        colspecs = str(self.lineEdit.text())
        mfm.run("cs.current_setup.colspecs = '%s'" % colspecs)

    # @property
    # def colspecs(self):
    #     return eval(str(self.lineEdit.text()))
    #
    # @colspecs.setter
    # def colspecs(self, v):
    #     self.lineEdit.setText(v)

    @property
    def filename(self):
        return str(self.lineEdit_8.text())

    @filename.setter
    def filename(self, v):
        Csv.filename.fset(self, v)
        self.lineEdit_8.setText(v)

    def changeCsvType(self):
        mode = 'csv' if self.radioButton_2.isChecked() else 'fwf'
        mfm.run("cs.current_setup.mode = '%s'" % mode)

    # @property
    # def mode(self):
    #     mode = 'csv' if self.radioButton_2.isChecked() else 'fwf'
    #     return mode
    #
    # @mode.setter
    # def mode(self, v):
    #     pass

    def load(self, filename=None, **kwargs):
        if filename is None:
            filename = mfm.widgets.open_file('Open CSV-File', 'CSV-file (*.*)')
        Csv.load(self, filename, **kwargs)
        self.lineEdit_8.setText(filename)


class CSVFileWidget(QtGui.QWidget):

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        self.parent = kwargs.get('parent', None)
        self.name = kwargs.get('name', 'CSV-File')
        self.weight_calculation = kwargs.get('weight_calculation', None)

        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.csvWidget = CsvWidget(**kwargs)
        self.layout.addWidget(self.csvWidget)

    def load_data(self, filename=None):
        """
        Loads csv-data into a Curve-object
        :param filename:
        :return: Curve-object
        """
        d = mfm.curve.DataCurve(setup=None)
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