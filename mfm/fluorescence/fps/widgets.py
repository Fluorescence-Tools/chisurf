# coding=utf-8
from collections import OrderedDict
from PyQt5 import QtCore, QtGui, QtWidgets

import mfm
from mfm.fluorescence.fps.dynamic import ProteinQuenching, Dye, Sticking
from mfm.fitting.widgets import FittingParameterWidget


class ProteinQuenchingWidget(ProteinQuenching, QtWidgets.QGroupBox):
    @property
    def quencher(self):
        p = OrderedDict()
        for qn in str(self.lineEdit_3.text()).split():
            p[qn] = ['CB']
        return p

    @quencher.setter
    def quencher(self, v):
        s = ""
        for resID in list(v.keys()):
            s += " " + resID
        self.lineEdit_3.setText(s)

    @property
    def all_atoms_quench(self):
        return not bool(self.groupBox.isChecked())

    @all_atoms_quench.setter
    def all_atoms_quench(self, v):
        self.groupBox.setChecked(not v)

    @property
    def excluded_atoms(self):
        return str(self.lineEdit_6.text()).split()

    def __init__(self, **kwargs):

        QtWidgets.QGroupBox.__init__(self)
        self.setTitle('Quenching')
        self.lineEdit_6 = QtWidgets.QLineEdit()
        self.lineEdit_6.setText('CA C HA N')
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.lineEdit_3 = QtWidgets.QLineEdit()
        lab = QtWidgets.QLabel('Quenching AA')
        ProteinQuenching.__init__(self, **kwargs)
        self.quencher = kwargs.get('quenching_amino_acids',
                                   {'TRP': ['CB'], 'TYR': ['CB'], 'HIS': ['CB'], 'PRO': ['CB']})
        self._k_quench_scale = FittingParameterWidget(value=kwargs.get('k_quench_protein', 5.0),
                                                      name='kQ', model=self.model)

        self.groupBox = QtWidgets.QGroupBox()
        self.groupBox.setCheckable(True)
        self.groupBox.setDisabled(True)
        self.groupBox.setTitle('Exclude atoms')
        layout.addWidget(self._k_quench_scale, 0, 0, 1, 2)
        layout.addWidget(self.lineEdit_3, 1, 1)
        layout.addWidget(lab, 1, 0)
        layout.addWidget(self.groupBox, 2, 0, 1, 2)

        gl = QtWidgets.QVBoxLayout()
        self.groupBox.setLayout(gl)
        gl.addWidget(self.lineEdit_6)


class DyeWidget(Dye, QtWidgets.QGroupBox):

    @property
    def dye_name(self):
        return str(self.dye_select.currentText())

    @dye_name.setter
    def dye_name(self, v):
        if isinstance(v, str):
            v = 0
        self.dye_select.setCurrentIndex(v)
        self.update_parameter()

    def __init__(self, **kwargs):
        QtWidgets.QGroupBox.__init__(self)
        self.dye_select = QtWidgets.QComboBox()
        self.dye_select.addItems(mfm.fluorescence.fps.dye_names)
        Dye.__init__(self, **kwargs)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        gb = QtWidgets.QGroupBox()
        gb.setTitle(kwargs.get('title', ''))
        layout.addWidget(gb)

        gl = QtWidgets.QGridLayout()
        gl.setSpacing(0)
        gl.setContentsMargins(0, 0, 0, 0)
        gb.setLayout(gl)

        self._critical_distance = FittingParameterWidget(value=7.0, name='RQ', digits=1, model=self.model)
        self._diffusion_coefficient = FittingParameterWidget(value=5.0, name='D[A2/ns]', digits=1, model=self.model)
        self._tau0 = FittingParameterWidget(value=4.0, name='tau0[ns]', digits=2, model=self.model)

        self._av_length = FittingParameterWidget(name='L', value=20.0, digits=1, model=self.model)
        self._av_width = FittingParameterWidget(name='W', value=0.5, digits=1, model=self.model)
        self._av_radius = FittingParameterWidget(name='R', value=5.0, digits=1, model=self.model)

        self.critical_distance = kwargs.get('critical_distance', 7.0)
        self.diffusion_coefficient = kwargs.get('diffusion_coefficient', 5.0)
        self.tau0 = kwargs.get('tau0', 4.0)

        gl.addWidget(self.dye_select, 0, 0, 1, 2)

        gl.addWidget(self._critical_distance, 1, 0)
        gl.addWidget(self._diffusion_coefficient, 2, 0)
        gl.addWidget(self._tau0, 3, 0)

        gl.addWidget(self._av_length, 1, 1)
        gl.addWidget(self._av_width, 2, 1)
        gl.addWidget(self._av_radius, 3, 1)
        self.dye_select.currentIndexChanged[int].connect(self.update_parameter)
        self.update_parameter()


class StickingWidget(Sticking, QtWidgets.QGroupBox):

    @property
    def sticky_mode(self):
        if self.radioButton.isChecked():
            return 'surface'
        elif self.radioButton_2.isChecked():
            return 'quencher'

    @sticky_mode.setter
    def sticky_mode(self, v):
        if v == 'surface':
            self.radioButton.setChecked(True)
            self.radioButton_2.setChecked(False)
        elif v == 'quencher':
            self.radioButton.setChecked(False)
            self.radioButton_2.setChecked(True)

    def __init__(self, **kwargs):
        QtWidgets.QGroupBox.__init__(self)
        self.setTitle('Sticking')
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)
        self.radioButton = QtWidgets.QRadioButton('Surface')
        self.radioButton_2 = QtWidgets.QRadioButton('Quencher')
        Sticking.__init__(self)
        self._slow_radius = FittingParameterWidget(name='Rs[A]', value=10.0, digits=1, model=self.model)
        self._slow_fact = FittingParameterWidget(name='slow fact', value=0.15, digits=3, model=self.model)
        layout.addWidget(self._slow_fact, 0, 1)
        layout.addWidget(self._slow_radius, 1, 1)

        layout.addWidget(self.radioButton, 0, 0)
        layout.addWidget(self.radioButton_2, 1, 0)
