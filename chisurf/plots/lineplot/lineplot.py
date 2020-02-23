from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.dockarea
import matplotlib.colors
from qtpy import QtWidgets

import chisurf.decorators
import chisurf.math
import chisurf.fitting
import chisurf.settings
import chisurf.math.statistics
from chisurf.plots import plotbase


class LinePlotControl(
    QtWidgets.QWidget
):

    @chisurf.decorators.init_with_ui(
        ui_filename="linePlotWidget.ui"
    )
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
        self.parent = parent
        self.use_reference = reference_curve

        self.data_logy = d_scaley
        self.scale_x = scale_x
        self.res_logy = r_scaley
        self.xmin = xmin
        self.ymin = ymin

        self.actionUpdate_Plot.triggered.connect(parent.update)
        self.checkBox.stateChanged.connect(self.SetLog)
        self.checkBox_2.stateChanged.connect(self.SetLog)
        self.checkBox_4.stateChanged.connect(self.SetLog)
        self.checkBox_5.stateChanged.connect(self.SetReference)
        self.checkBox_3.stateChanged.connect(self.SetDensity)

    @property
    def plot_ftt(
            self
    ) -> bool:
        return bool(
            self.checkBox_plot_ftt.isChecked()
        )

    @plot_ftt.setter
    def plot_ftt(
            self,
            v: bool
    ) -> None:
        if v:
            self.checkBox_plot_ftt.setCheckState(2)
        else:
            self.checkBox_plot_ftt.setCheckState(0)

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
        self.doubleSpinBox_2.setValue(v)

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
        self.doubleSpinBox.setValue(v)

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
        self.parent.update()

    def SetLog(self):
        self.parent.update()

    def SetDensity(self):
        self.parent.update()


class LinePlot(plotbase.Plot):
    """
    Started off as a plotting class to display TCSPC-data displaying the IRF,
    the experimental data, the residuals and the autocorrelation of the
    residuals. Now it is also used also for fcs-data.

    In case the models is a :py:class:`~experiment.models.tcspc.LifetimeModel`
    it takes the irf and displays it:

        irf = fit.models.convolve.irf
        irf_y = irf.y

    The models data and the weighted residuals are taken directly from the fit:

        model_x, model_y = fit[:]
        wres_y = fit.weighted_residuals

    """

    name = "Fit"

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            scale_x: str = 'lin',
            d_scaley: str = 'lin',
            r_scaley: str = 'lin',
            reference_curve: bool = False,
            x_label: str = 'x',
            y_label: str = 'y',
            plot_irf: bool = False
    ):
        super().__init__(fit)

        # plot control dialog
        self.pltControl = LinePlotControl(
            parent=self,
            scale_x=scale_x,
            d_scaley=d_scaley,
            r_scaley=r_scaley,
            reference_curve=reference_curve
        )

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.plot_irf = plot_irf

        area = pyqtgraph.dockarea.DockArea()
        self.layout.addWidget(area)
        hide_title = chisurf.settings.gui['plot']['hideTitle']
        d1 = pyqtgraph.dockarea.Dock("residuals", size=(250, 80), hideTitle=hide_title)
        d2 = pyqtgraph.dockarea.Dock("a.corr.", size=(250, 80), hideTitle=hide_title)
        d3 = pyqtgraph.dockarea.Dock("Fit", size=(250, 250), hideTitle=hide_title)

        p1 = pg.PlotWidget()
        p2 = pg.PlotWidget()
        p3 = pg.PlotWidget()

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
        self.text = pg.TextItem(
            text='',
            border='w',
            fill=(0, 0, 255, 100),
            anchor=(0, 1)
        )

        # self.text.setPos(
        #     min(data.x * 0.5),
        #     max(data.y * 0.5)
        # )

        self.data_plot.addItem(self.text)
        colors = chisurf.settings.gui['plot']['colors']

        # Fitting-region selector
        if chisurf.settings.gui['plot']['enable_region_selector']:
            ca = list(matplotlib.colors.hex2color(colors["region_selector"]))
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
                chisurf.run("cs.current_fit.fit_range = (%s, %s)" % (lb_i - 1, ub_i))
                self.update(only_fit_range=True)

            region.sigRegionChangeFinished.connect(update_region)

        # Grid
        if chisurf.settings.gui['plot']['enable_grid']:
            if chisurf.settings.gui['plot']['show_data_grid']:
                data_plot.showGrid(True, True, 0.5)
            if chisurf.settings.gui['plot']['show_residual_grid']:
                residuals_plot.showGrid(True, True, 1.0)
            if chisurf.settings.gui['plot']['show_acorr_grid']:
                auto_corr_plot.showGrid(True, True, 1.0)

        # Labels
        self.residuals_plot = residuals_plot
        self.auto_corr_plot = auto_corr_plot

        if chisurf.settings.gui['plot']['label_axis']:
            residuals_plot.setLabel('left', "w.res.")
            auto_corr_plot.setLabel('left', "a.corr.")
            data_plot.setLabel('left', y_label)
            data_plot.setLabel('bottom', x_label)

        curves = self.fit.get_curves()

        data_string = 'data'
        wres_string = 'weighted residuals'
        acorr_string = 'autocorrelation'

        lines = dict()
        for i, curve_key in enumerate(curves):
            color = chisurf.settings.colors[i % len(chisurf.settings.colors)]['hex']
            lw = chisurf.settings.gui['plot']['line_width']
            if wres_string == curve_key:
                lw = 2 * lw
            if wres_string in curve_key:
                lines[curve_key] = residuals_plot.plot(
                    x=[0.0], y=[0.0],
                    pen=pg.mkPen(colors['residuals'], width=lw),
                    name='w.res.'
                )
            elif acorr_string in curve_key:
                lines[curve_key] = auto_corr_plot.plot(
                    x=[0.0], y=[0.0],
                    pen=pg.mkPen(colors['auto_corr'], width=lw),
                    name='a.corr.'
                )
            else:
                lines[curve_key] = data_plot.plot(
                    x=[0.0], y=[0.0],
                    pen=pg.mkPen(color, width=lw),
                    name=curve_key
                )

        self.lines = lines
        p1.setXLink(p3)
        p2.setXLink(p3)

    def update(
            self,
            only_fit_range: bool = False,
            *args,
            **kwargs
    ) -> None:
        super().update(*args, **kwargs)
        # Get parameters from plot-control
        plt_ctrl = self.pltControl
        data_log_y = plt_ctrl.data_is_log_y
        data_log_x = plt_ctrl.data_is_log_x
        res_log_y = plt_ctrl.res_is_log_y
        use_reference = self.pltControl.use_reference
        # # Reference-function
        # if use_reference:
        #     reference = self.fit.model.reference
        #     if reference is None:
        #         reference = np.ones_like(data_y)
        #         print("WARNING: no reference curve provided by the model.")
        #     model_y /= reference[self.fit.xmin:self.fit.xmax]
        #     data_y /= reference
        #     mm = max(model_y)
        #     irf_y *= mm / max(irf_y)

        y_shift = plt_ctrl.y_shift
        x_shift = plt_ctrl.x_shift

        curves = self.fit.get_curves()
        for curve_key in curves:
            curve = curves[curve_key]
            y = curve.y + y_shift
            x = curve.x + x_shift
            if self.pltControl.is_density:
                y = curve.y / np.diff(curve.x)
            self.lines[curve_key].setData(
                x=x,
                y=y
            )

        #
        # model_y = np.copy(model.y) + y_shift
        # data_y = np.copy(data_y) + y_shift
        #

        # Set log-scales
        self.res_plot.setLogMode(
            x=data_log_x,
            y=res_log_y
        )
        self.auto_corr_plot.setLogMode(
            x=data_log_x,
            y=res_log_y
        )
        self.data_plot.setLogMode(
            x=data_log_x,
            y=data_log_y
        )

        # update region selector
        data = curves['data']
        lb_min, ub_max = data.x[0], data.x[-1]
        lb, ub = data.x[self.fit.xmin], data.x[self.fit.xmax]
        if data_log_x:
            lb = np.log10(lb)
            ub = np.log10(ub)
            lb_min = np.log10(lb_min)
            ub_max = np.log10(ub_max)
        self.region.setBounds((lb_min, ub_max))
        self.region.setRegion((lb, ub))

        self.text.updateTextPos()
        self.text.setHtml(
            '<div style="name-align: center">'
            '     <span style="color: #FF0; font-size: 10pt;">'
            '         Fit-range: %s, %s <br />'
            '         &Chi;<sup>2</sup>=%.4f <br />'
            '         DW=%.4f'
            '     </span>'
            '</div>' % (
                self.fit.xmin, self.fit.xmax,
                self.fit.chi2r,
                self.fit.durbin_watson
            )
        )
