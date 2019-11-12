from __future__ import annotations

import json
import os
import numpy as np
from qtpy import QtWidgets, uic

import chisurf.widgets
import chisurf.decorators
import chisurf.structure
from chisurf.structure.potential.potentials import HPotential, GoPotential, \
    MJPotential, CEPotential, ClashPotential, \
    AvPotential, ASA


class HPotentialWidget(HPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure,
            parent,
            cutoff_ca=8.0,
            cutoff_hbond=3.0
    ):
        QtWidgets.QWidget.__init__(self, parent=parent)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Potential_Hbond_2.ui"
            ),
            self
        )
        self.checkBox.stateChanged[int].connect(self.updateParameter)
        self.checkBox_2.stateChanged[int].connect(self.updateParameter)
        self.checkBox_3.stateChanged[int].connect(self.updateParameter)
        self.checkBox_4.stateChanged[int].connect(self.updateParameter)
        self.actionLoad_potential.triggered.connect(self.onOpenFile)
        self.cutoffCA = cutoff_ca
        self.cutoffH = cutoff_hbond
        super().__init__(
            structure,
            cutoff_ca,
            cutoff_hbond
        )

    def onOpenFile(self):
        filename = chisurf.widgets.get_filename(
            'Open File',
            'NumPy data files (*.npy)'
        )
        self.potential = filename

    @property
    def potential(self):
        return self.hPot

    @potential.setter
    def potential(self, v):
        self._hPot = np.load(v)
        # self._hPot = np.load(
        #     v,
        #     skiprows=1,
        #     dtype=np.float64
        # ).T[1:, :]
        self.hPot = self._hPot
        self.lineEdit_3.setText(str(v))

    @property
    def oh(self):
        return int(self.checkBox.isChecked())

    @oh.setter
    def oh(self, v):
        self.checkBox.setChecked(v)

    @property
    def cn(self):
        return int(self.checkBox_2.isChecked())

    @cn.setter
    def cn(self, v):
        self.checkBox_2.setChecked(v)

    @property
    def ch(self):
        return int(self.checkBox_3.isChecked())

    @ch.setter
    def ch(self, v):
        self.checkBox_3.setChecked(v)

    @property
    def on(self):
        return int(self.checkBox_4.isChecked())

    @on.setter
    def on(self, v):
        self.checkBox_4.setChecked(v)

    @property
    def cutoffH(self):
        return float(self.doubleSpinBox.value())

    @cutoffH.setter
    def cutoffH(self, v):
        self.doubleSpinBox.setValue(float(v))

    @property
    def cutoffCA(self):
        return float(self.doubleSpinBox_2.value())

    @cutoffCA.setter
    def cutoffCA(self, v):
        self.doubleSpinBox_2.setValue(float(v))


class GoPotentialWidget(GoPotential, QtWidgets.QWidget):

    @chisurf.decorators.init_with_ui(
        ui_filename="Potential-CaLJ.ui"
    )
    def __init__(
            self,
            structure: chisurf.structure.Structure = None,
            **kwargs
    ):
        self.lineEdit.textChanged['QString'].connect(self.setGo)
        self.lineEdit_2.textChanged['QString'].connect(self.setGo)
        self.lineEdit_3.textChanged['QString'].connect(self.setGo)

    @property
    def native_cutoff_on(self):
        return bool(self.checkBox.isChecked())

    @property
    def non_native_contact_on(self):
        return bool(self.checkBox_2.isChecked())

    @property
    def epsilon(self):
        return float(self.lineEdit.text())

    @property
    def nnEFactor(self):
        return float(self.lineEdit_2.text())

    @property
    def cutoff(self):
        return float(self.lineEdit_3.text())


class MJPotentialWidget(MJPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            filename: str = './mfm/structure/potential/database/mj.npy',
            ca_cutoff: float =6.5
    ):
        super(MJPotentialWidget, self).__init__(structure, filename, ca_cutoff)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "MJ-resource.ui"
            ),
            self
        )
        self.pushButton.clicked.connect(self.onOpenFile)
        self.potential = filename
        self.ca_cutoff = ca_cutoff

    def onOpenFile(self):
        filename = chisurf.widgets.get_filename(
            'Open MJ-Potential',
            'CSV data files (*.npy)'
        )
        self.potential = filename

    @property
    def potential(self):
        return self.mjPot

    @potential.setter
    def potential(
            self,
            v: str
    ):
        self.mjPot = np.loadtxt(v)
        self.lineEdit.setText(v)

    @property
    def ca_cutoff(
            self
    ) -> float:
        return float(self.lineEdit_2.text())

    @ca_cutoff.setter
    def ca_cutoff(
            self,
            v: float
    ):
        self.lineEdit_2.setText(str(v))


class CEPotentialWidget(CEPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            potential: str = './mfm/structure/potential/database/unres.npy',
            ca_cutoff: float = 25.0
    ):
        super(CEPotentialWidget, self).__init__(
            structure,
            potential=potential,
            ca_cutoff=ca_cutoff
        )
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "unres-cb-resource.ui"
            ),
            self
        )
        self.actionOpen_potential_file.triggered.connect(self.onOpenPotentialFile)
        self.ca_cutoff = ca_cutoff

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, v):
        self.lineEdit.setText(str(v))
        self._potential = np.load(v)

    @property
    def ca_cutoff(self):
        return float(self.doubleSpinBox.value())

    @ca_cutoff.setter
    def ca_cutoff(self, v):
        self.doubleSpinBox.setValue(float(v))

    def onOpenPotentialFile(self):
        filename = chisurf.widgets.get_filename('Open CE-Potential', 'Numpy file (*.npy)')
        self.potential = filename


class ClashPotentialWidget(ClashPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.structure.Structure = None,
            **kwargs
    ):
        super(QtWidgets.QWidget, self).__init__(**kwargs)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "potential-clash.ui"
            ),
            self
        )
        super(ClashPotentialWidget, self).__init__(
            structure=structure,
            **kwargs
        )

    @property
    def clash_tolerance(self):
        return float(self.doubleSpinBox.value())

    @clash_tolerance.setter
    def clash_tolerance(self, v):
        self.doubleSpinBox.setValue(v)

    @property
    def covalent_radius(self):
        return float(self.doubleSpinBox_2.value())

    @covalent_radius.setter
    def covalent_radius(self, v):
        self.doubleSpinBox_2.setValue(v)


class AvPotentialWidget(AvPotential, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.structure.Structure = None,
            parent=None
    ):
        super(AvPotentialWidget, self).__init__()
        QtWidgets.QWidget.__init__(self, parent=parent)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "avWidget.ui"
            ),
            self
        )
        self._filename = None
        self.actionOpenLabeling.triggered.connect(self.onLoadAvJSON)

    def onLoadAvJSON(self):
        self.labeling_file = chisurf.widgets.get_filename(
            description='Open FPS-JSON',
            file_type='FPS-file (*.fps.json)'
        )

    @property
    def labeling_file(self):
        return self._filename

    @labeling_file.setter
    def labeling_file(self, v):
        self._filename = v
        p = json.load(open(v))
        self.distances = p["Distances"]
        self.positions = p["Positions"]
        self.lineEdit_2.setText(v)

    @property
    def n_av_samples(self):
        return int(self.spinBox_2.value())

    @n_av_samples.setter
    def n_av_samples(self, v):
        self.spinBox_2.setValue(int(v))

    @property
    def min_av(self):
        return int(self.spinBox_2.value())

    @min_av.setter
    def min_av(self, v):
        self.spinBox.setValue(int(v))


class AsaWidget(ASA, QtWidgets.QWidget):

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            parent: QtWidgets.QWidget = None,
            **kwargs
    ):
        super(QtWidgets.QWidget, self).__init__(
            parent=parent
        )
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Potential_Asa.ui"
            ),
            self
        )

        self.lineEdit.textChanged['QString'].connect(self.setParameterSphere)
        self.lineEdit_2.textChanged['QString'].connect(self.setParameterProbe)
        self.lineEdit.setText('590')
        self.lineEdit_2.setText('3.5')

        super(AsaWidget, self).__init__(
            structure,
            **kwargs
        )

    def setParameterSphere(self):
        self.n_sphere_point = int(self.lineEdit.text())

    def setParameterProbe(self):
        self.probe = float(self.lineEdit_2.text())


class RadiusGyrationWidget(QtWidgets.QWidget):

    name = 'Radius-Gyration'

    def __init__(
            self,
            structure: chisurf.structure.Structure,
            parent=None
    ):
        super(RadiusGyrationWidget, self).__init__(parent)
        self.structure = structure
        self.parent = parent

    def getEnergy(self, c=None):
        if c is None:
            c = self.structure
        return c.radius_gyration