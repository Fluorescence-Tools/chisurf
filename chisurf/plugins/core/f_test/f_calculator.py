from __future__ import annotations

import sys

from chisurf.gui import QtWidgets

from scipy.stats import f as fdist

import chisurf.core.decorators
import chisurf.gui.decorators
import chisurf.core.models
import chisurf.core.fitting.fit
import chisurf.core.math.statistics


class FTestWidget(QtWidgets.QWidget):

    def read_values(self, target):
        """Create a slot that copies a fit's n_points, n_free and chi2r into target spin boxes.

        Parameters
        ----------
        target : tuple
            ``(fit, spinbox_n_free, spinbox_n_points, spinbox_chi2r)``.

        Returns
        -------
        callable
            A slot suitable for ``QAction.triggered.connect``.
        """

        def linkcall():
            """Copy a fit's n_points, n_free and chi2r into the configured spin boxes."""
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
        """Create a slot that copies a fit's degrees of freedom and chi2r into target spin boxes.

        Parameters
        ----------
        target : tuple
            ``(fit, spinbox_dof, spinbox_chi2r)``.

        Returns
        -------
        callable
            A slot suitable for ``QAction.triggered.connect``.
        """
        def linkcall():
            """Copy a fit's dof and chi2r into the configured spin boxes."""
            fit = target[0]
            t = target[1]
            t.setValue(fit.model.n_points - fit.model.n_free)
            target[2].setValue(fit.chi2r)
        return linkcall

    def read_n(self):
        """Build a toolButton menu that copies ``n_points``, ``n_free`` and ``chi2r`` from any loaded fit."""
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
        """Build a toolButton menu that copies ``dof`` and ``chi2r`` for model 1 from any loaded fit."""
        menu = QtWidgets.QMenu()
        for f in chisurf.fits:
            for fs in f:
                Action = menu.addAction(fs.name)
                Action.triggered.connect(
                    self.read_v((fs, self.spinBox, self.doubleSpinBox))
                )
        self.toolButton.setMenu(menu)

    def read_n2(self):
        """Build a toolButton menu that copies ``dof`` and ``chi2r`` for model 2 from any loaded fit."""
        menu = QtWidgets.QMenu()
        for f in chisurf.fits:
            for fs in f:
                Action = menu.addAction(fs.name)
                Action.triggered.connect(
                    self.read_v((fs, self.spinBox_2, self.doubleSpinBox_3))
                )
        self.toolButton_2.setMenu(menu)

    @chisurf.gui.decorators.init_with_ui(
        ui_filename="F-Calculator.ui"
    )
    def __init__(
            self,
            *args,
            **kwargs
    ):
        """Initialize the F-Calculator widget and connect UI actions."""
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
        """Compute the upper chi2 limit from the selected fit and current parameters."""
        if isinstance(
                self._selected_fit,
                chisurf.core.fitting.fit.Fit
        ):
            self.chi2_min = self._selected_fit.chi2r
        dof = max(1, self.dof)
        number_of_parameters = max(1, self.npars)
        self.chi2_max = chisurf.core.math.statistics.chi2_max(
            conf_level=self.conf_level_2,
            number_of_parameters=number_of_parameters,
            nu=dof,
            chi2_value=self.chi2_min
        )

    def onChi2_2_Changed(self):
        """Recompute the confidence level when chi2_2 changes."""
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
        """Recompute chi2_2 from the current confidence level."""
        # recalculate chi2_max
        chi2_2 = self.chi2_1 * self.n2 / self.n1 * fdist.isf(1. - self.conf_level, self.n1, self.n2)
        self.doubleSpinBox_3.blockSignals(True)
        self.doubleSpinBox_3.setValue(chi2_2)
        self.doubleSpinBox_3.blockSignals(False)

    def onChi2_1_Changed(self):
        """Recompute chi2_2 from the updated chi2_1 value."""
        # recalculate chi2_2
        chi2_2 = self.chi2_1 * self.n2 / self.n1 * fdist.isf(1. - self.conf_level, self.n1, self.n2)
        self.doubleSpinBox_3.blockSignals(True)
        self.doubleSpinBox_3.setValue(chi2_2)
        self.doubleSpinBox_3.blockSignals(False)

    def onN1Changed(self):
        """Recompute the confidence level when n1 changes."""
        # recalculate confidence level
        conf_level = fdist.cdf(self.chi2_2 / self.chi2_1, self.n1, self.n2)
        self.doubleSpinBox_2.blockSignals(True)
        self.doubleSpinBox_2.setValue(conf_level)
        self.doubleSpinBox_2.blockSignals(False)

    def onN2Changed(self):
        """Recompute the confidence level when n2 changes."""
        # recalculate confidence level
        conf_level = fdist.cdf(self.chi2_2 / self.chi2_1, self.n1, self.n2)
        self.doubleSpinBox_2.blockSignals(True)
        self.doubleSpinBox_2.setValue(conf_level)
        self.doubleSpinBox_2.blockSignals(False)

    @property
    def n1(self) -> int:
        """Degrees of freedom of the first model (int)."""
        return int(self.spinBox.value())

    @n1.setter
    def n1(
            self,
            v: int
    ):
        """Set the value of the ``n1`` spin box.

        Parameters
        ----------
        v : int
            New value for ``n1``.
        """
        self.spinBox.setValue(v)

    @property
    def n2(self) -> int:
        """Degrees of freedom of the second model (int)."""
        return int(self.spinBox_2.value())

    @n2.setter
    def n2(
            self,
            v: int
    ):
        """Set the value of the ``n2`` spin box.

        Parameters
        ----------
        v : int
            New value for ``n2``.
        """
        self.spinBox_2.setValue(v)

    @property
    def conf_level(self) -> float:
        """Confidence level for the F-test (float)."""
        return float(self.doubleSpinBox_2.value())

    @conf_level.setter
    def conf_level(
            self,
            c: float
    ):
        """Set the confidence level spin box value.

        Parameters
        ----------
        c : float
            New confidence level.
        """
        self.doubleSpinBox_2.setValue(c)

    @property
    def chi2_1(self) -> float:
        """Reduced chi-squared of the first model (float)."""
        return float(self.doubleSpinBox.value())

    @chi2_1.setter
    def chi2_1(
            self,
            v: float
    ):
        """Set the value of the ``chi2_1`` spin box.

        Parameters
        ----------
        v : float
            New value for ``chi2_1``.
        """
        self.doubleSpinBox.setValue(v)

    @property
    def chi2_2(self) -> float:
        """Reduced chi-squared of the second model (float)."""
        return float(self.doubleSpinBox_3.value())

    @chi2_2.setter
    def chi2_2(
            self,
            v: float
    ):
        """Set the value of the ``chi2_2`` spin box.

        Parameters
        ----------
        v : float
            New value for ``chi2_2``.
        """
        self.doubleSpinBox_3.setValue(v)

    @property
    def npars(self) -> int:
        """Number of fitting parameters (int)."""
        return int(self.spinBox_4.value())

    @npars.setter
    def npars(
            self,
            v: int
    ):
        """Set the value of the ``npars`` spin box.

        Parameters
        ----------
        v : int
            New value for ``npars``.
        """
        self.spinBox_4.setValue(v)

    @property
    def dof(self) -> int:
        """Degrees of freedom used for the chi2_max calculation (int)."""
        return int(self.spinBox_3.value())

    @dof.setter
    def dof(
            self,
            v: int
    ):
        """Set the value of the ``dof`` spin box.

        Parameters
        ----------
        v : int
            New value for ``dof``.
        """
        self.spinBox_3.setValue(v)

    @property
    def conf_level_2(self) -> float:
        """Confidence level for the chi2_max calculation (float)."""
        return float(self.doubleSpinBox_4.value())

    @conf_level_2.setter
    def conf_level_2(
            self,
            c: float
    ):
        """Set the value of the ``conf_level_2`` spin box.

        Parameters
        ----------
        c : float
            New confidence level.
        """
        self.doubleSpinBox_4.setValue(c)

    @property
    def chi2_max(self) -> float:
        """Upper chi2 limit computed from the current parameters (float)."""
        return float(self.lineEdit.text())

    @chi2_max.setter
    def chi2_max(
            self,
            c: float
    ):
        """Display ``chi2_max`` in the read-only line edit.

        Parameters
        ----------
        c : float
            New chi2_max value.
        """
        self.lineEdit.setText(str(c))

    @property
    def chi2_min(self) -> float:
        """Reduced chi-squared of the selected fit (float)."""
        return float(self.doubleSpinBox_5.value())

    @chi2_min.setter
    def chi2_min(
            self,
            c: float
    ):
        """Set the value of the ``chi2_min`` spin box.

        Parameters
        ----------
        c : float
            New value for ``chi2_min``.
        """
        self.doubleSpinBox_5.setValue(c)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FTestWidget()
    win.show()
    sys.exit(app.exec_())
