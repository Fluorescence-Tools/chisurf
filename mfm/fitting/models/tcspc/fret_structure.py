from PyQt4 import QtGui, QtCore, uic
import numpy as np
import mfm
import mfm.structure
from . import fret
from mfm.parameter import FittingParameter
from mfm.fitting.models import ModelWidget
from . import tcspc
from mfm.fluorescence.fps import ACV
fret_settings = fret.fret_settings


class FRETStructure(fret.FRETModel):

    name = "FRET: Structure fit"

    x = fret.rda_axis

    @property
    def distance_distribution(self):
        self.p = np.zeros_like(self.x)
        for i, amplitude in enumerate(self.amplitudes):
            p = self.ps[i] * amplitude
            self.p[1:] += p
        self.p /= sum(self.p)

        d = list()
        threshold = max(self.p) * mfm.settings['tcspc']['threshold']
        self.p = np.where(self.p >= threshold, self.p, 0)
        d.append([self.p, self.x])
        d = np.array(d)
        return d

    @property
    def amplitudes(self):
        ampls = np.sqrt(np.array([a.value**2 for a in self._amplitudes]))
        ampls /= sum(ampls)
        return ampls

    def append(self, structure, **kwargs):
        """
        
        :param structure:
        :param labeling_position:
        :return:
        """
        amplitude = kwargs.get('amplitude', 0.5)
        res_1 = kwargs.get('res_1', self.res_1)
        res_2 = kwargs.get('res_2', self.res_2)
        atom_name_1 = kwargs.get('atom_name_1', self.atom_name_1)
        atom_name_2 = kwargs.get('atom_name_2', self.atom_name_2)
        linker_length_1 = kwargs.get('linker_length_1', self.linker_length_1)
        linker_length_2 = kwargs.get('linker_length_2', self.linker_length_2)
        radius1_1 = kwargs.get('radius1_1', self.radius1_1)
        radius1_2 = kwargs.get('radius1_2', self.radius1_2)
        linker_width_1 = kwargs.get('linker_width_1', self.linker_width_1)
        linker_width_2 = kwargs.get('linker_width_2', self.linker_width_2)

        x = self.x
        av1 = ACV(structure, res_1, atom_name_1, None, linker_length_1, linker_width_1, radius1_1)
        av2 = ACV(structure, res_2, atom_name_2, None, linker_length_2, linker_width_2, radius1_2)
        ds = av1.get_random_distance(av2)
        p = np.histogram(ds, x)[0] * amplitude
        self.ps.append(p)
        self.names.append(structure.name)
        amplitude = FittingParameter(value=amplitude, name="x_" + str(len(self.names)), lb=0.0, ub=1.0, bounds_on=True)
        self._amplitudes.append(amplitude)

    def clear(self):
        self._amplitudes = list()
        self.ps = list()
        self.names = list()
        self.p = np.zeros_like(self.x)

    def pop(self):
        self._amplitudes.pop()
        self.ps.pop()

    def __init__(self, fit, **kwargs):
        fret.FRETModel.__init__(self, fit, **kwargs)
        self.names = list()
        self.ps = list()
        self._amplitudes = list()
        self.p = np.zeros_like(self.x)

        self.res_1 = kwargs.get('res_1', 0)
        self.res_2 = kwargs.get('res_2', 0)

        self.atom_name_1 = kwargs.get('atom_name_1', 'CA')
        self.atom_name_2 = kwargs.get('atom_name_2', 'CA')

        self.linker_length_1 = kwargs.get('linker_length_1', 20)
        self.linker_length_2 = kwargs.get('linker_length_2', 20)

        self.radius1_1 = kwargs.get('radius1_1', 4.0)
        self.radius1_2 = kwargs.get('radius1_2', 4.0)

        self.linker_width_1 = kwargs.get('linker_width_1', 4.5)
        self.linker_width_2 = kwargs.get('linker_width_2', 4.5)


class FRETStructureWidget(FRETStructure, ModelWidget):

    def onOpenFilenames(self):
        filenames = mfm.widgets.open_files("Open PDBs", '*.pdb')
        st = ""
        for filename in filenames:
            s = mfm.structure.Structure(filename)
            self.append(s)
            st += filename + "\n"
        self.pdb_line_widget.appendPlainText(st)
        self.folder_line.setText(mfm.working_path)

    def clear(self):
        FRETStructure.clear(self)
        self.pdb_line_widget.setPlainText("")
        self.fraction_widget.setPlainText("")

    def finalize(self):
        fractions = self.amplitudes
        s = ""
        for f in fractions:
            s += "%.3f\n" % f
        self.fraction_widget.setPlainText(s)

    @property
    def linker_width_1(self):
        return float(self.label_widget.doubleSpinBox_2.value())

    @linker_width_1.setter
    def linker_width_1(self, v):
        self.label_widget.doubleSpinBox_2.setValue(v)

    @property
    def linker_width_2(self):
        return float(self.label_widget.doubleSpinBox_5.value())

    @linker_width_2.setter
    def linker_width_2(self, v):
        self.label_widget.doubleSpinBox_5.setValue(v)

    @property
    def radius1_1(self):
        return float(self.label_widget.doubleSpinBox_3.value())

    @radius1_1.setter
    def radius1_1(self, v):
        self.label_widget.doubleSpinBox_3.setValue(v)

    @property
    def radius1_2(self):
        return float(self.label_widget.doubleSpinBox_6.value())

    @radius1_2.setter
    def radius1_2(self, v):
        self.label_widget.doubleSpinBox_6.setValue(v)

    @property
    def linker_length_1(self):
        return float(self.label_widget.doubleSpinBox.value())

    @linker_length_1.setter
    def linker_length_1(self, v):
        self.label_widget.doubleSpinBox.setValue(v)

    @property
    def linker_length_2(self):
        return float(self.label_widget.doubleSpinBox_4.value())

    @linker_length_2.setter
    def linker_length_2(self, v):
        self.label_widget.doubleSpinBox_4.setValue(v)

    @property
    def atom_name_1(self):
        return str(self.label_widget.lineEdit.text())

    @atom_name_1.setter
    def atom_name_1(self, v):
        self.label_widget.lineEdit.setText(str(v))

    @property
    def atom_name_2(self):
        return str(self.label_widget.lineEdit_2.text())

    @atom_name_2.setter
    def atom_name_2(self, v):
        self.label_widget.lineEdit_2.setText(str(v))

    @property
    def res_1(self):
        return int(self.label_widget.spinBox.value())

    @res_1.setter
    def res_1(self, v):
        self.label_widget.spinBox.setValue(v)

    @property
    def res_2(self):
        return int(self.label_widget.spinBox_2.value())

    @res_2.setter
    def res_2(self, v):
        self.label_widget.spinBox_2.setValue(v)

    def __init__(self, fit, **kwargs):
        ModelWidget.__init__(self, fit=fit, icon=QtGui.QIcon(":/icons/icons/TCSPC.png"), **kwargs)
        self.label_widget = uic.loadUi('mfm/ui/fitting/models/tcspc/structure_fit.ui')

        FRETStructure.__init__(self, fit, **kwargs)
        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)

        self.convolve = tcspc.ConvolveWidget(fit=fit, model=self, hide_curve_convolution=True, **kwargs)
        self.donors = tcspc.LifetimeWidget(parent=self, model=self, title='Donor(0)', short='D', name='donors')
        self.generic = tcspc.GenericWidget(fit=fit, model=self, parent=self, **kwargs)

        self.layout.addWidget(self.convolve)
        self.layout.addWidget(self.generic)
        self.layout.addWidget(self.donors)

        self._forster_radius = self._forster_radius.widget()
        self._tau0 = self._tau0.widget()
        self._donly = self._donly.widget()
        self._kappa2 = self._kappa2.widget()

        self.layout.addWidget(self._forster_radius)
        self.layout.addWidget(self._tau0)
        self.layout.addWidget(self._donly)
        self.layout.addWidget(self._kappa2)

        self.layout.addWidget(self.label_widget)

        gb = QtGui.QGroupBox()
        l = QtGui.QVBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)

        gb.setTitle('PDBs')
        l1 = QtGui.QHBoxLayout()
        l1.setSpacing(0)
        l1.setMargin(0)
        label = QtGui.QLabel('Folder:')
        self.folder_line = QtGui.QLineEdit()
        self.folder_line.setEnabled(False)
        open_folder = QtGui.QPushButton()
        open_folder.setText('...')
        self.connect(open_folder, QtCore.SIGNAL("clicked()"), self.onOpenFilenames)
        l1.addWidget(label)
        l1.addWidget(self.folder_line)
        l1.addWidget(open_folder)
        l.addLayout(l1)

        self.pdb_line_widget = QtGui.QPlainTextEdit()
        self.fraction_widget = QtGui.QPlainTextEdit()

        l.addWidget(self.pdb_line_widget)
        l.addWidget(QtGui.QLabel('Fractions:'))
        l.addWidget(self.fraction_widget)
        gb.setLayout(l)

        self.layout.addWidget(gb)
