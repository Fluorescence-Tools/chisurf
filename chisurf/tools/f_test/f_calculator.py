from __future__ import annotations

import sys

from qtpy import QtWidgets
import qdarkstyle

from scipy.stats import f as fdist

import chisurf.decorators
import chisurf.models
import chisurf.fitting.fit
import chisurf.math.statistics


class FTestWidget(QtWidgets.QWidget):

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
        menu = QtWidgets.QMenu()
        for f in chisurf.fits:
            for fs in f:
                Action = menu.addAction(fs.name)
                Action.triggered.connect(
                    self.read_values(
                        (fs, self.spinBox_4, self.spinBox_3, self.doubleSpinBox_5)
                    )
                )
        self.toolButton_3.setMenu(menu)

    def read_n1(self):
        menu = QtWidgets.QMenu()
        for f in chisurf.fits:
            for fs in f:
                Action = menu.addAction(fs.name)
                Action.triggered.connect(
                    self.read_v((fs, self.spinBox, self.doubleSpinBox))
                )
        self.toolButton.setMenu(menu)

    def read_n2(self):
        menu = QtWidgets.QMenu()
        for f in chisurf.fits:
            for fs in f:
                Action = menu.addAction(fs.name)
                Action.triggered.connect(
                    self.read_v((fs, self.spinBox_2, self.doubleSpinBox_3))
                )
        self.toolButton_2.setMenu(menu)

    @chisurf.decorators.init_with_ui(ui_filename="F-Calculator.ui")
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self._selected_fit = None

        # Upper part of F-Calculator
        self.actionN1Changed.triggered.connect(self.onN1Changed)
        self.actionN2Changed.triggered.connect(self.onN2Changed)
        self.actionConf_F_Changed.triggered.connect(self.onConfChanged)
        self.actionChi2_1_F_Changed.triggered.connect(self.onChi2_1_Changed)
        self.actionChi2_2_F_Changed.triggered.connect(self.onChi2_2_Changed)

        # Lower part of F-Calculator
        self.actionChi2MinChanged.triggered.connect(self.calculate_chi2_max)
        self.actionNParameterChanged.triggered.connect(self.calculate_chi2_max)
        self.actionDofChanged.triggered.connect(self.calculate_chi2_max)
        self.actionOnConf_2_Changed.triggered.connect(self.calculate_chi2_max)

        self.toolButton.clicked.connect(self.read_n1)
        self.toolButton_2.clicked.connect(self.read_n2)
        self.toolButton_3.clicked.connect(self.read_n)

    def calculate_chi2_max(self):
        if isinstance(
                self._selected_fit,
                chisurf.fitting.fit.Fit
        ):
            self.chi2_min = self._selected_fit.chi2r
        dof = max(1, self.dof)
        number_of_parameters = max(1, self.npars)
        self.chi2_max = chisurf.math.statistics.chi2_max(
            conf_level=self.conf_level_2,
            number_of_parameters=number_of_parameters,
            nu=dof,
            chi2_value=self.chi2_min
        )

    def onChi2_2_Changed(self):
        # recalculate confidence level
        conf_level = fdist.cdf(
            self.chi2_2 / self.chi2_1,
            self.n1,
            self.n2
        )
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
    def n1(
            self
    ) -> int:
        return int(self.spinBox.value())

    @n1.setter
    def n1(
            self,
            v: int
    ):
        self.spinBox.setValue(v)

    @property
    def n2(
            self
    ) -> int:
        return int(self.spinBox_2.value())

    @n2.setter
    def n2(
            self,
            v: int
    ):
        self.spinBox_2.setValue(v)

    @property
    def conf_level(
            self
    ) -> float:
        return float(self.doubleSpinBox_2.value())

    @conf_level.setter
    def conf_level(
            self,
            c: float
    ):
        self.doubleSpinBox_2.setValue(c)

    @property
    def chi2_1(
            self
    ) -> float:
        return float(self.doubleSpinBox.value())

    @chi2_1.setter
    def chi2_1(
            self,
            v: float
    ):
        self.doubleSpinBox.setValue(v)

    @property
    def chi2_2(
            self
    ) -> float:
        return float(self.doubleSpinBox_3.value())

    @chi2_2.setter
    def chi2_2(
            self,
            v: float
    ):
        self.doubleSpinBox_3.setValue(v)

    @property
    def npars(
            self
    ) -> int:
        return int(self.spinBox_4.value())

    @npars.setter
    def npars(
            self,
            v: int
    ):
        self.spinBox_4.setValue(v)

    @property
    def dof(
            self
    ) -> int:
        return int(self.spinBox_3.value())

    @dof.setter
    def dof(
            self,
            v: int
    ):
        self.spinBox_3.setValue(v)

    @property
    def conf_level_2(
            self
    ) -> float:
        return float(self.doubleSpinBox_4.value())

    @conf_level_2.setter
    def conf_level_2(
            self,
            c: float
    ):
        self.doubleSpinBox_4.setValue(c)

    @property
    def chi2_max(
            self
    ) -> float:
        return float(self.lineEdit.text())

    @chi2_max.setter
    def chi2_max(
            self,
            c: float
    ):
        self.lineEdit.setText(str(c))

    @property
    def chi2_min(
            self
    ) -> float:
        return float(self.doubleSpinBox_5.value())

    @chi2_min.setter
    def chi2_min(
            self,
            c: float
    ):
        self.doubleSpinBox_5.setValue(c)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FTestWidget()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
