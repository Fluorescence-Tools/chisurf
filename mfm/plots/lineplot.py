from __future__ import annotations

import os
import numpy as np
import pyqtgraph as pg
from qtpy import  QtWidgets, uic
import matplotlib.colors as mpl_colors

import mfm
import mfm.math
import mfm.math.statistics
from mfm.plots import plotbase
from pyqtgraph.dockarea import *

pyqtgraph_settings = mfm.settings.pyqtgraph_settings
color_scheme = mfm.settings.colors


class LinePlotControl(QtWidgets.QWidget):

    def __init__(
            self,
            parent=None,
            scale_x: str = 'lin',
            d_scaley: str = 'log',
            r_scaley: str = 'lin',
            reference_curve: bool = False,
            xmin: float = 0.0,
            ymin: float = 1.0
    ):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "linePlotWidget.ui"
            ),
            self
        )
        self.parent = parent
        self.use_reference = reference_curve

        self.data_logy = d_scaley
        self.scale_x = scale_x
        self.res_logy = r_scaley
        self.xmin = xmin
        self.ymin = ymin

        self.actionUpdate_Plot.triggered.connect(parent.update_all)
        self.checkBox.stateChanged [int].connect(self.SetLog)
        self.checkBox_2.stateChanged [int].connect(self.SetLog)
        self.checkBox_4.stateChanged [int].connect(self.SetLog)
        self.checkBox_5.stateChanged [int].connect(self.SetReference)
        self.checkBox_3.stateChanged [int].connect(self.SetDensity)

    @property
    def data_logy(
            self
    ) -> str:
        """
        y-data is plotted logarithmically
        """
        return 'log' if self.checkBox.isChecked() else 'linear'

    @data_logy.setter
    def data_logy(
            self,
            v: str
    ):
        if v == 'lin':
            self.checkBox.setCheckState(0)
        else:
            self.checkBox.setCheckState(2)

    @property
    def scale_x(
            self
    ) -> str:
        """
        x-data is plotted logarithmically
        """
        return 'log' if self.checkBox_2.isChecked() else 'linear'

    @scale_x.setter
    def scale_x(
            self,
            v: str
    ):
        if v == 'lin':
            self.checkBox_2.setCheckState(0)
        else:
            self.checkBox_2.setCheckState(2)

    @property
    def data_is_log_x(
            self
    ) -> bool:
        return self.scale_x == 'log'

    @property
    def data_is_log_y(
            self
    ) -> bool:
        return self.data_logy == 'log'

    @property
    def res_is_log_y(
            self
    ) -> bool:
        return self.res_logy == 'log'

    @property
    def res_logy(
            self
    ) -> str:
        """
        y-residuals is plotted logarithmically
        """
        return 'log' if self.checkBox_4.isChecked() else 'lin'

    @res_logy.setter
    def res_logy(
            self,
            v: str
    ):
        if v == 'lin':
            self.checkBox_4.setCheckState(0)
        else:
            self.checkBox_4.setCheckState(2)

    @property
    def use_reference(
            self
    ) -> bool:
        """
        If true use a reference curve for plotting
        """
        return bool(self.checkBox_5.isChecked())

    @use_reference.setter
    def use_reference(
            self,
            v: bool
    ):
        if v is True:
            self.checkBox_5.setCheckState(2)
        else:
            self.checkBox_5.setCheckState(0)

    @property
    def ymin(
            self
    ) -> float:
        return self.doubleSpinBox_2.value()

    @ymin.setter
    def ymin(
            self,
            v: float
    ):
        return self.doubleSpinBox_2.setValue(v)

    @property
    def xmin(
            self
    ) -> float:
        return self.doubleSpinBox.value()

    @xmin.setter
    def xmin(
            self,
            v: float
    ):
        return self.doubleSpinBox.setValue(v)

    @property
    def x_shift(
            self
    ) -> float:
        return self.doubleSpinBox_6.value()

    @x_shift.setter
    def x_shift(
            self,
            v: float
    ):
        self.doubleSpinBox_6.setValue(v)

    @property
    def y_shift(
            self
    ) -> float:
        return self.doubleSpinBox_5.value()

    @y_shift.setter
    def y_shift(
            self,
            v: float
    ):
        self.doubleSpinBox_5.setValue(v)

    @property
    def is_density(
            self
    ) -> bool:
        return bool(self.checkBox_3.isChecked())

    @is_density.setter
    def is_density(
            self,
            v: bool
    ):
        if v is True:
            self.checkBox_3.setCheckState(2)
        else:
            self.checkBox_3.setCheckState(0)

    def SetReference(self):
        self.parent.update_all()

    def SetLog(self):
        self.parent.update_all()

    def SetDensity(self):
        self.parent.update_all()


class LinePlot(plotbase.Plot):
    """
    Started off as a plotting class to display TCSPC-data displaying the IRF, the experimental data, the residuals
    and the autocorrelation of the residuals. Now it is also used also for FCS-data.

    In case the models is a :py:class:`~experiment.models.tcspc.LifetimeModel` it takes the irf and displays it:

        irf = fit.models.convolve.irf
        irf_y = irf.y

    The models data and the weighted residuals are taken directly from the fit:

        model_x, model_y = fit[:]
        wres_y = fit.weighted_residuals

    """

    name = "Fit"

    def __init__(
            self,
            fit: mfm.fitting.fit.FitGroup,
            scale_x: str = 'lin',
            d_scaley: str = 'lin',
            r_scaley: str = 'lin',
            reference_curve: bool = False,
            x_label: str = 'x',
            y_label: str = 'y',
            plot_irf: bool = False,
            data_x: np.array = None,
            data_y: np.array = None,
            **kwargs
    ):
        super(LinePlot, self).__init__(fit)

        # plot control dialog
        self.pltControl = LinePlotControl(
            self,
            scale_x,
            d_scaley,
            r_scaley,
            reference_curve,
            **kwargs
        )

        self.layout = QtWidgets.QVBoxLayout(self)
        self.data_x = data_x
        self.data_y = data_y
        self.plot_irf = plot_irf

        area = DockArea()
        self.layout.addWidget(area)
        hide_title = mfm.settings.cs_settings['gui']['plot']['hideTitle']
        d1 = Dock("residuals", size=(300, 80), hideTitle=hide_title)
        d2 = Dock("a.corr.", size=(300, 80), hideTitle=hide_title)
        d3 = Dock("Fit", size=(300, 300), hideTitle=hide_title)

        p1 = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])
        p2 = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])
        p3 = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])

        d1.addWidget(p1)
        d2.addWidget(p2)
        d3.addWidget(p3)

        area.addDock(d1, 'top')
        area.addDock(d2, 'top', d1)
        area.addDock(d3, 'bottom', d1)

        residuals_plot = p1.getPlotItem()
        auto_corr_plot = p2.getPlotItem()
        data_plot = p3.getPlotItem()

        self.data_plot = data_plot
        self.res_plot = residuals_plot
        self.auto_corr_plot = auto_corr_plot

        # Labels
        self.text = pg.TextItem(text='', border='w', fill=(0, 0, 255, 100), anchor=(0, 0))
        self.data_plot.addItem(self.text)
        colors = mfm.settings.cs_settings['gui']['plot']['colors']

        # Fitting-region selector
        if mfm.settings.cs_settings['gui']['plot']['enable_region_selector']:
            ca = list(mpl_colors.hex2color(colors["region_selector"]))
            co = [ca[0] * 255, ca[1] * 255, ca[2] * 255, colors["region_selector_alpha"]]
            region = pg.LinearRegionItem(brush=co)
            data_plot.addItem(region)
            self.region = region

            def update_region(evt):
                lb, ub = region.getRegion()
                data_x = fit.data.x
                if self.pltControl.data_is_log_x:
                    lb, ub = 10**lb, 10**ub
                lb -= self.pltControl.x_shift
                ub -= self.pltControl.x_shift

                lb_i = np.searchsorted(data_x, lb, side='right')
                ub_i = np.searchsorted(data_x, ub, side='left')
                mfm.run("cs.current_fit.fit_range = (%s, %s)" % (lb_i - 1, ub_i))
                self.update_all(only_fit_range=True)

            region.sigRegionChangeFinished.connect(update_region)

        # Grid
        if mfm.settings.cs_settings['gui']['plot']['enable_grid']:
            if mfm.settings.cs_settings['gui']['plot']['show_data_grid']:
                data_plot.showGrid(True, True, 0.5)
            if mfm.settings.cs_settings['gui']['plot']['show_residual_grid']:
                residuals_plot.showGrid(True, True, 1.0)
            if mfm.settings.cs_settings['gui']['plot']['show_acorr_grid']:
                auto_corr_plot.showGrid(True, True, 1.0)

        # Labels
        self.residuals_plot = residuals_plot
        self.auto_corr_plot = auto_corr_plot

        if mfm.settings.cs_settings['gui']['plot']['label_axis']:
            residuals_plot.setLabel('left', "w.res.")
            auto_corr_plot.setLabel('left', "a.corr.")
            data_plot.setLabel('left', y_label)
            data_plot.setLabel('bottom', x_label)

        # Plotted lines
        lw = mfm.settings.cs_settings['gui']['plot']['line_width']
        if self.plot_irf:
            self.irf_curve = data_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['irf'], width=lw), name='IRF')
        self.data_curve = data_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['data'], width=lw), name='Data')
        self.fit_curve = data_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['model'], width=lw), name='Model')
        p1.setXLink(p3)
        p2.setXLink(p3)

    def update_all(
            self,
            only_fit_range: bool = False,
            *args,
            **kwargs
    ):
        fit = self.fit
        # Get parameters from plot-control
        plt_ctrl = self.pltControl

        data_log_y = plt_ctrl.data_is_log_y
        data_log_x = plt_ctrl.data_is_log_x
        res_log_y = plt_ctrl.res_is_log_y
        use_reference = self.pltControl.use_reference
        y_shift = plt_ctrl.y_shift
        x_shift = plt_ctrl.x_shift

        # Model function
        model_x, model_y = fit.model[fit.xmin:fit.xmax]
        model_x = np.copy(model_x) + x_shift
        model_y = np.copy(model_y) + y_shift
        # Update fitting-region
        data_x = np.copy(self.fit.data.x) + x_shift
        data_y = np.copy(self.fit.data.y) + y_shift
        # Weighted residuals + Autocorrelation
        wres_y = fit.weighted_residuals
        if self.pltControl.is_density:
            data_y = data_y[1:]
            data_y /= np.diff(data_x)

            model_y = model_y[1:]
            model_y /= np.diff(model_x)

        # IRF
        try:
            norm = fit.data.y.max() / fit.model.convolve.irf.y.max()
            irf_y = fit.model.convolve.irf.y * norm
            irf_x = data_x
        except AttributeError:
            irf_y = np.ones(10)
            irf_x = np.ones(10)
            if mfm.verbose:
                print("No instrument response to plot.")

        xmin = max(np.searchsorted(data_x, 1e-12, side='right'), self.fit.xmin) if data_log_x else self.fit.xmin
        lb_min, ub_max = data_x[0], data_x[-1]
        lb, ub = data_x[xmin], data_x[self.fit.xmax]
        if data_log_x:
            lb = np.log10(lb)
            ub = np.log10(ub)
            lb_min = np.log10(lb_min)
            ub_max = np.log10(ub_max)

        # Label
        y_max = data_y.max()
        if data_log_y:
            y_max = np.log10(y_max)
        self.text.setPos(ub_max * .7, y_max * .9)
        #self.legend.setPos(ub_max * .7, y_max * .3)
        self.text.setHtml('<div style="text-align: center">'
                          '     <span style="color: #FF0; font-size: 10pt;">'
                          '         Fit-range: %s, %s <br />'
                          '         &Chi;<sup>2</sup>=%.4f <br />'
                          '         DW=%.4f'
                          '     </span>'
                          '</div>' % (fit.xmin, fit.xmax,
                                      fit.chi2r,
                                      mfm.math.statistics.durbin_watson(fit.weighted_residuals[0]))
                          )

        # Reference-function
        if use_reference:
            reference = fit.model.reference
            model_y /= reference[fit.xmin:fit.xmax]
            data_y /= reference
            mm = max(model_y)
            irf_y *= mm / max(irf_y)

        if not only_fit_range:

            idx = np.where(data_y > plt_ctrl.ymin)[0] if data_log_y else list(range(len(data_x)))
            self.data_curve.setData(x=data_x[idx], y=data_y[idx])
            if self.plot_irf:
                idx = np.where(irf_y > plt_ctrl.ymin)[0] if data_log_y else list(range(len(irf_x)))
                self.irf_curve.setData(x=irf_x[idx], y=irf_y[idx])

            # Set log-scales
            self.res_plot.setLogMode(x=data_log_x, y=res_log_y)
            self.auto_corr_plot.setLogMode(x=data_log_x, y=res_log_y)
            self.data_plot.setLogMode(x=data_log_x, y=data_log_y)

            self.region.setBounds((lb_min, ub_max))
            self.region.setRegion((lb, ub))

        # Update the Model-lines (Model, wres, acorr
        idx = np.where(model_y > 0.0)[0] if data_log_y else list(range(len(model_x)))
        self.fit_curve.setData(x=model_x[idx], y=model_y[idx])

        self.residuals_plot.clear()
        self.auto_corr_plot.clear()

        colors = mfm.settings.cs_settings['gui']['plot']['colors']
        lw = mfm.settings.cs_settings['gui']['plot']['line_width']

        for i, w in enumerate(wres_y):
            self.residuals_plot.plot(x=model_x,
                                     y=w,
                                     pen=pg.mkPen(colors['residuals'], width=lw),
                                     name='residues')
            ac_y = mfm.math.signal.autocorr(w)
            self.auto_corr_plot.plot(x=model_x[1:],
                                     y=ac_y[1:],
                                     pen=pg.mkPen(colors['auto_corr'], width=lw),
                                     name='residues')
        self.data_x = data_x
        self.data_y = data_y

