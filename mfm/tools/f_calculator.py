from PyQt4 import QtCore, QtGui, uic
from scipy.stats import f as fdist
import mfm


class FTestWidget(QtGui.QWidget):

    def read_values(self, target):
        def linkcall():
            fit = target[0]
            self._selected_fit = fit
            n_points = fit.model.n_points
            n_free = fit.model.n_free
            chi2r = fit.chi2r
            target[1].setValue(n_free)
            target[2].setValue(n_points)
            target[3].setValue(chi2r)
        return linkcall

    def read_v(self, target):
        def linkcall():
            fit = target[0]
            t = target[1]
            t.setValue(fit.model.n_points - fit.model.n_free)
            target[2].setValue(fit.chi2r)
        return linkcall

    def read_n(self):
        menu = QtGui.QMenu()
        for f in mfm.fits:
            for fs in f:
                Action = menu.addAction(fs.name)
                Action.triggered.connect(self.read_values((fs, self.spinBox_4, self.spinBox_3, self.doubleSpinBox_5)))
        self.toolButton_3.setMenu(menu)

    def read_n1(self):
        menu = QtGui.QMenu()
        for f in mfm.fits:
            for fs in f:
                Action = menu.addAction(fs.name)
                Action.triggered.connect(self.read_v((fs, self.spinBox, self.doubleSpinBox)))
        self.toolButton.setMenu(menu)

    def read_n2(self):
        menu = QtGui.QMenu()
        for f in mfm.fits:
            for fs in f:
                Action = menu.addAction(fs.name)
                Action.triggered.connect(self.read_v((fs, self.spinBox_2, self.doubleSpinBox_3)))
        self.toolButton_2.setMenu(menu)

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi("mfm/ui/F-Calculator.ui", self)
        self.parent = kwargs.get('parent', None)

        # Upper part of F-Calculator
        self.connect(self.actionN1Changed, QtCore.SIGNAL('triggered()'), self.onN1Changed)
        self.connect(self.actionN2Changed, QtCore.SIGNAL('triggered()'), self.onN2Changed)
        self.connect(self.actionConf_F_Changed, QtCore.SIGNAL('triggered()'), self.onConfChanged)
        self.connect(self.actionChi2_1_F_Changed, QtCore.SIGNAL('triggered()'), self.onChi2_1_Changed)
        self.connect(self.actionChi2_2_F_Changed, QtCore.SIGNAL('triggered()'), self.onChi2_2_Changed)

        # Lower part of F-Calculator
        self.connect(self.actionChi2MinChanged, QtCore.SIGNAL('triggered()'), self.calculate_chi2_max)
        self.connect(self.actionNParameterChanged, QtCore.SIGNAL('triggered()'), self.calculate_chi2_max)
        self.connect(self.actionDofChanged, QtCore.SIGNAL('triggered()'), self.calculate_chi2_max)
        self.connect(self.actionOnConf_2_Changed, QtCore.SIGNAL('triggered()'), self.calculate_chi2_max)

        self.toolButton.clicked.connect(self.read_n1)
        self.toolButton_2.clicked.connect(self.read_n2)
        self.toolButton_3.clicked.connect(self.read_n)

    def calculate_chi2_max(self):
        self.chi2_max = mfm.fitting.chi2_max(self._selected_fit, conf_level=self.conf_level_2,
                                             npars=self.npars, nu=self.dof, chi2_min=self.chi2_min)

    def onChi2_2_Changed(self):
        # recalculate confidence level
        conf_level = fdist.cdf(self.chi2_2 / self.chi2_1, self.n1, self.n2)
        self.doubleSpinBox_2.blockSignals(True)
        self.doubleSpinBox_2.setValue(conf_level)
        self.doubleSpinBox_2.blockSignals(False)

    def onConfChanged(self):
        # recalculate chi2_max
        chi2_2 = self.chi2_1 * self.n2 / self.n1 * fdist.isf(1. - self.conf_level, self.n1, self.n2)
        self.doubleSpinBox_3.blockSignals(True)
        self.doubleSpinBox_3.setValue(chi2_2)
        self.doubleSpinBox_3.blockSignals(False)

    def onChi2_1_Changed(self):
        # recalculate chi2_2
        chi2_2 = self.chi2_1 * self.n2 / self.n1 * fdist.isf(1. - self.conf_level, self.n1, self.n2)
        self.doubleSpinBox_3.blockSignals(True)
        self.doubleSpinBox_3.setValue(chi2_2)
        self.doubleSpinBox_3.blockSignals(False)

    def onN1Changed(self):
        # recalculate confidence level
        conf_level = fdist.cdf(self.chi2_2 / self.chi2_1, self.n1, self.n2)
        self.doubleSpinBox_2.blockSignals(True)
        self.doubleSpinBox_2.setValue(conf_level)
        self.doubleSpinBox_2.blockSignals(False)

    def onN2Changed(self):
        # recalculate confidence level
        conf_level = fdist.cdf(self.chi2_2 / self.chi2_1, self.n1, self.n2)
        self.doubleSpinBox_2.blockSignals(True)
        self.doubleSpinBox_2.setValue(conf_level)
        self.doubleSpinBox_2.blockSignals(False)

    @property
    def n1(self):
        return float(self.spinBox.value())

    @n1.setter
    def n1(self, v):
        self.spinBox.setValue(v)

    @property
    def n2(self):
        return float(self.spinBox_2.value())

    @n2.setter
    def n2(self, v):
        self.spinBox_2.setValue(v)

    @property
    def conf_level(self):
        return float(self.doubleSpinBox_2.value())

    @conf_level.setter
    def conf_level(self, c):
        self.doubleSpinBox_2.setValue(c)

    @property
    def chi2_1(self):
        return float(self.doubleSpinBox.value())

    @chi2_1.setter
    def chi2_1(self, v):
        self.doubleSpinBox.setValue(v)

    @property
    def chi2_2(self):
        return float(self.doubleSpinBox_3.value())

    @chi2_2.setter
    def chi2_2(self, v):
        self.doubleSpinBox_3.setValue(v)

    @property
    def npars(self):
        return int(self.spinBox_4.value())

    @npars.setter
    def npars(self, v):
        self.spinBox_4.setValue(v)

    @property
    def dof(self):
        return int(self.spinBox_3.value())

    @dof.setter
    def dof(self, v):
        self.spinBox_3.setValue(v)

    @property
    def conf_level_2(self):
        return float(self.doubleSpinBox_4.value())

    @conf_level_2.setter
    def conf_level_2(self, c):
        self.doubleSpinBox_4.setValue(c)

    @property
    def chi2_max(self):
        return float(self.lineEdit.text())

    @chi2_max.setter
    def chi2_max(self, c):
        self.lineEdit.setText(str(c))

    @property
    def chi2_min(self):
        return float(self.doubleSpinBox_5.value())

    @chi2_min.setter
    def chi2_min(self, c):
        self.doubleSpinBox_5.setValue(c)
