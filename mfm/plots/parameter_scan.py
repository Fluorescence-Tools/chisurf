from __future__ import annotations
from typing import Tuple

import os

import numpy as np
import pyqtgraph as pg
from qtpy import  QtWidgets, uic
from pyqtgraph.dockarea import DockArea, Dock

import mfm
from mfm.plots import plotbase

plot_settings = mfm.settings.cs_settings['gui']['plot']
pyqtgraph_settings = mfm.settings.pyqtgraph_settings
colors = plot_settings['colors']
color_scheme = mfm.settings.colors
lw = plot_settings['line_width']


class ParameterScanWidget(QtWidgets.QWidget):
    def __init__(
            self,
            model: mfm.models.model.Model,
            parent
    ):
        super(ParameterScanWidget, self).__init__()
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "parameter_scan.ui"
            ),
            self
        )

        self.model = model
        self.parent = parent

        self.actionScanParameter.triggered.connect(self.scan_parameter)
        self.actionParameterChanged.triggered.connect(self.onParameterChanged)
        self.actionUpdateParameterList.triggered.connect(self.update)

        self.update()

    def onParameterChanged(self):
        self.parent.update_all()

    def update(
            self
    ) -> None:
        QtWidgets.QWidget.update(self)
        self.comboBox.blockSignals(True)

        pn = self.model.parameter_names
        self.comboBox.clear()
        self.comboBox.addItems(pn)

        self.comboBox.blockSignals(False)
        self.model.update_plots()

    def scan_parameter(
            self
    ) -> None:
        p_min = float(self.doubleSpinBox.value())
        p_max = float(self.doubleSpinBox_2.value())
        n_steps = int(self.spinBox.value())
        s = "cs.current_fit.model.parameters_all_dict['%s'].scan(cs.current_fit, rel_range=(%s, %s), n_steps=%s)" % (
            self.parameter.name,
            p_min,
            p_max,
            n_steps
        )
        mfm.run(s)
        self.parent.update_all()

    @property
    def selected_parameter(
            self
    ) -> Tuple[int, str]:
        idx = self.comboBox.currentIndex()
        name = self.comboBox.currentText()
        return idx, str(name)

    @property
    def parameter(
            self
    ) -> mfm.parameter.Parameter:
        idx, name = self.selected_parameter
        try:
            return self.model.parameters_all_dict[name]
        except AttributeError:
            return None


class ParameterScanPlot(plotbase.Plot):
    """
    Started off as a plotting class to display TCSPC-data displaying the IRF, the experimental data, the residuals
    and the autocorrelation of the residuals. Now it is also used also for FCS-data.

    In case the model is a :py:class:`~experiment.model.tcspc.LifetimeModel` it takes the irf and displays it:

        irf = fit.model.convolve.irf
        irf_y = irf.y

    The model data and the weighted residuals are taken directly from the fit:

        model_x, model_y = fit[:]
        wres_y = fit.weighted_residuals

    """

    name = "Parameter scan"

    def __init__(
            self,
            fit: mfm.fitting.fit.FitGroup,
            **kwargs
    ):
        super(ParameterScanPlot, self).__init__(fit)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.data_x, self.data_y = None, None
        self.pltControl = ParameterScanWidget(
            fit.model,
            self
        )

        area = DockArea()
        self.layout.addWidget(area)
        hide_title = plot_settings['hideTitle']
        d2 = Dock("Chi2-Surface", size=(500, 400), hideTitle=hide_title)

        self.p1 = QtWidgets.QPlainTextEdit()
        p2 = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])

        d2.addWidget(p2)

        area.addDock(d2, 'top')

        distribution_plot = p2.getPlotItem()

        self.distribution_plot = distribution_plot
        self.distribution_curve = distribution_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['data'], width=lw), name='Data')

    def update_all(
            self,
            *args,
            **kwargs
    ) -> None:
        pass
        try:
            p = self.pltControl.parameter
            x, y = p.parameter_scan
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                self.distribution_curve.setData(x=x, y=y)
        except:
            pass


