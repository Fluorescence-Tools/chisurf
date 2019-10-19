from __future__ import annotations

import numpy as np
from qtpy import QtCore, QtWidgets
from guiqwt.builder import make
from guiqwt.plot import CurveDialog

import chisurf.mfm as mfm
import chisurf.decorators
from chisurf.math.signal import autocorr
from chisurf.plots.plotbase import Plot


class LinePlotWidget(QtWidgets.QWidget):

    @chisurf.decorators.init_with_ui(ui_filename="linePlotWidget.ui")
    def __init__(
            self,
            d_scalex='lin',
            d_scaley='log',
            r_scalex='lin',
            r_scaley='lin',
            *args,
            **kwargs
    ):
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.checkBox.stateChanged [int].connect(self.SetLog)
        self.checkBox_2.stateChanged [int].connect(self.SetLog)
        self.checkBox_3.stateChanged [int].connect(self.SetLog)
        self.checkBox_4.stateChanged [int].connect(self.SetLog)

        self.data_logy = d_scaley
        self.data_logx = d_scalex
        self.res_logx = r_scalex
        self.res_logy = r_scaley

    @property
    def data_logy(self):
        return 'log' if self.checkBox.isChecked() else 'lin'

    @data_logy.setter
    def data_logy(self, v):
        if v == 'lin':
            self.checkBox.setCheckState(0)
        else:
            self.checkBox.setCheckState(2)

    @property
    def data_logx(self):
        return 'log' if self.checkBox_2.isChecked() else 'lin'

    @data_logx.setter
    def data_logx(self, v):
        if v == 'lin':
            self.checkBox_2.setCheckState(0)
        else:
            self.checkBox_2.setCheckState(2)

    @property
    def res_logx(self):
        return 'log' if self.checkBox_3.isChecked() else 'lin'

    @res_logx.setter
    def res_logx(self, v):
        if v == 'lin':
            self.checkBox_3.setCheckState(0)
        else:
            self.checkBox_3.setCheckState(2)

    @property
    def res_logy(self):
        return 'log' if self.checkBox_4.isChecked() else 'lin'

    @res_logy.setter
    def res_logy(self, v):
        if v == 'lin':
            self.checkBox_4.setCheckState(0)
        else:
            self.checkBox_4.setCheckState(2)

    def SetLog(self):
        print("SetLog")
        self.parent.residualPlot.set_scales(self.res_logx, self.res_logy)
        self.parent.autoCorrPlot.set_scales(self.res_logx, self.res_logy)
        self.parent.dataPlot.set_scales(self.data_logx, self.data_logy)


class GlobalAnisotropy(Plot):

    name = "Fit"

    def __init__(
            self,
            fit: fitting.fit.FitGroup,
            d_scalex: str = 'lin',
            d_scaley: str = 'lin',
            r_scalex: str = 'lin',
            r_scaley: str = 'lin',
            **kwargs
    ):
        super(GlobalAnisotropy, self).__init__(
            fit=fit,
            **kwargs
        )
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.fit = fit

        bottom = QtWidgets.QFrame(self)
        bottom.setFrameShape(QtWidgets.QFrame.StyledPanel)
        botl = QtWidgets.QVBoxLayout(bottom)

        top = QtWidgets.QFrame(self)
        top.setMaximumHeight(140)
        top.setFrameShape(QtWidgets.QFrame.StyledPanel)
        topl = QtWidgets.QVBoxLayout(top)

        splitter1 = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter1.addWidget(top)
        splitter1.addWidget(bottom)
        self.layout.addWidget(splitter1)

        # Data-Fit dialog
        fd = CurveDialog(edit=False, toolbar=True)
        #self.get_itemlist_panel().show()
        plot = fd.get_plot()
        self.data_curve_vv = make.curve([],  [], color="b", linewidth=1)
        self.irf_curve_vv = make.curve([],  [], color="r", linewidth=1)
        self.model_curve_vv = make.curve([],  [], color="g", linewidth=4)

        self.data_curve_vh = make.curve([],  [], color="c", linewidth=1)
        self.irf_curve_vh = make.curve([],  [], color="m", linewidth=1)
        self.model_curve_vh = make.curve([],  [], color="k", linewidth=4)


        plot.add_item(self.data_curve_vv)
        plot.add_item(self.irf_curve_vv)
        plot.add_item(self.model_curve_vv)

        plot.add_item(self.data_curve_vh)
        plot.add_item(self.irf_curve_vh)
        plot.add_item(self.model_curve_vh)

        self.dataPlot = plot
        botl.addWidget(fd)

        splitter1 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        topl.addWidget(splitter1)

        # Residual dialog
        win = CurveDialog(edit=False, toolbar=True)
        plot = win.get_plot()
        plot.do_autoscale(True)

        self.residual_curve_vv = make.curve([],  [], color="r", linewidth=2)
        self.residual_curve_vh = make.curve([],  [], color="b", linewidth=2)
        plot.add_item(self.residual_curve_vv)
        plot.add_item(self.residual_curve_vh)

        self.chi2_label = make.label("", "R", (-10, 27), "R")
        plot.add_item(self.chi2_label)
        title = make.label("w.res.", "R", (0, -40), "R")
        plot.add_item(title)
        self.residualPlot = plot
        splitter1.addWidget(plot)

        win = CurveDialog(edit=False, toolbar=True)
        plot = win.get_plot()
        plot.do_autoscale(True)
        self.autocorr_curve_vv = make.curve([],  [], color="r", linewidth=2)
        self.autocorr_curve_vh = make.curve([],  [], color="b", linewidth=2)

        plot.add_item(self.autocorr_curve_vv)
        plot.add_item(self.autocorr_curve_vh)

        title = make.label("auto.cor.", "R", (0, -40), "R")
        plot.add_item(title)
        self.autoCorrPlot = plot
        splitter1.addWidget(plot)

        self.pltControl = LinePlotWidget(
            self,
            d_scalex,
            d_scaley,
            r_scalex,
            r_scaley
        )

    def update_all(self):
        if self.verbose:
            print("GlobalAnisotropy:update")
        fit = self.fit
        self.chi2_label.set_text("<b>&Chi;<sup>2</sup>=%.4f</b>" % fit.chi2r)
        try:
            data_x, data_y = fit.vv.data.data_x, fit.vv.data.data_y
            idx = np.where(data_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
            self.data_curve_vv.set_data(data_x[idx],  data_y[idx])

            try:
                irf = fit.vv.model.convolve.irf
                irf_y = irf.data_y
                idx = np.where(irf_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
                self.irf_curve_vv.set_data(irf.data_x[idx],  irf_y[idx])
            except AttributeError:
                print("No instrument response to plot.")
            try:
                model_x, model_y = fit.vv[:]
                idx = np.where(model_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
                if len(idx) > 0:
                    self.model_curve_vv.set_data(model_x[idx],  model_y[idx])
            except ValueError:
                print("No models/no fitted models to plot")
            try:
                wres_y = fit.vv.weighted_residuals
                self.residual_curve_vv.set_data(model_x, wres_y)
                if len(wres_y) > 0:
                    ac = autocorr(wres_y)
                    self.autocorr_curve_vv.set_data(model_x[1::], ac[1:])
            except (TypeError, AttributeError):
                pass
            self.residualPlot.do_autoscale()
            self.dataPlot.do_autoscale()
            self.autoCorrPlot.do_autoscale()

            data_x, data_y = fit.vh.data.data_x, fit.vh.data.data_y
            idx = np.where(data_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
            self.data_curve_vh.set_data(data_x[idx],  data_y[idx])

            try:
                irf = fit.vh.model.convolve.irf
                irf_y = irf.data_y
                idx = np.where(irf_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
                self.irf_curve_vh.set_data(irf.data_x[idx],  irf_y[idx])
            except AttributeError:
                print("No instrument response to plot.")
            try:
                model_x, model_y = fit.vh[:]
                idx = np.where(model_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
                if len(idx) > 0:
                    self.model_curve_vh.set_data(model_x[idx],  model_y[idx])
            except ValueError:
                print("No models/no fitted models to plot")
            try:
                wres_y = fit.vv.weighted_residuals
                self.residual_curve_vh.set_data(model_x, wres_y)
                if len(wres_y) > 0:
                    ac = autocorr(wres_y)
                    self.autocorr_curve_vh.set_data(model_x[1::], ac[1:])
            except (TypeError, AttributeError):
                pass
        except (TypeError, AttributeError):
            pass
        self.residualPlot.do_autoscale()
        self.dataPlot.do_autoscale()
        self.autoCorrPlot.do_autoscale()
        print("END:TCSPCPlot:update")


class GlobalEt(Plot):

    name = "GlobalEt"

    @chisurf.decorators.init_with_ui(ui_filename="et_plot_layout.ui")
    def __init__(
            self,
            fit: fitting.fit.FitGroup,
            f_scalex: str = 'log',
            f_scaley: str = 'lin',
            e_scalex: str = 'log',
            e_scaley: str = 'lin',
            *args,
            **kwargs
    ):
        self.fit = fit

        ## Distance distribution plot
        fd = CurveDialog(edit=False, toolbar=False)
        plot = fd.get_plot()
        self.p_rda_plot = plot
        self.verticalLayout_12.addWidget(fd)
        self.p_rda_curve = make.curve([1],  [1], color="r", linewidth=2)
        plot.add_item(self.p_rda_curve)

        ## Fluorescence intensity plot
        win = CurveDialog(edit=False, toolbar=False)
        plot = win.get_plot()
        plot.do_autoscale(True)

        title = make.label("FDA,FD0", "R", (0, 10), "R")
        plot.add_item(title)
        self.fd0_curve = make.curve([1],  [1], color="g", linewidth=2)
        self.fda_curve = make.curve([1],  [1], color="r", linewidth=2)
        plot.add_item(self.fd0_curve)
        plot.add_item(self.fda_curve)
        self.fd_plot = plot
        self.fd_plot.set_scales(f_scalex, f_scaley)
        self.verticalLayout.addWidget(plot)

        ## Calculated E(t) plot
        win = CurveDialog(edit=False, toolbar=False)
        plot = win.get_plot()
        plot.do_autoscale(True)
        self.et_curve = make.curve([1],  [1], color="b", linewidth=2)
        plot.add_item(self.et_curve)
        title = make.label("E(t)", "R", (0, 10), "R")
        plot.add_item(title)
        self.et_plot = plot
        self.et_plot.set_scales(e_scalex, e_scaley)
        self.verticalLayout_2.addWidget(plot)

        ## weighted-residuals of inversion
        win = CurveDialog(edit=False, toolbar=True)
        plot = win.get_plot()
        plot.do_autoscale(True)
        self.wres_curve = make.curve([1],  [1], color="m", linewidth=2)
        plot.add_item(self.wres_curve)
        title = make.label("w.res", "R", (0, 10), "R")
        plot.add_item(title)
        self.wres_plot = plot
        self.wres_plot.set_scales('lin', 'lin')
        self.verticalLayout_3.addWidget(plot)

        ## L-Curve plot
        win = CurveDialog(edit=False, toolbar=True)
        plot = win.get_plot()
        plot.do_autoscale(True)
        self.l_curve_1 = make.curve([1],  [1], color="k", linewidth=2)
        plot.add_item(self.l_curve_1)
        title = make.label("Reg.", "R", (0, 10), "R")
        plot.set_titles(ylabel='reg. par', xlabel='Chi2r')
        plot.add_item(title)
        self.l_curve_plot_1 = plot
        self.l_curve_plot_1.set_scales('lin', 'lin')
        self.verticalLayout_4.addWidget(self.l_curve_plot_1)

        win = CurveDialog(edit=False, toolbar=True)
        plot = win.get_plot()
        plot.do_autoscale(True)
        self.l_curve_2 = make.curve([1],  [1], color="k", linewidth=2)
        plot.add_item(self.l_curve_2)
        title = make.label("L-Curve", "R", (0, 10), "R")
        plot.set_titles(ylabel='|x| (sol. norm)', xlabel='Chi2r')
        plot.add_item(title)
        self.l_curve_plot_2 = plot
        self.l_curve_plot_2.set_scales('log', 'log')
        self.verticalLayout_7.addWidget(self.l_curve_plot_2)

    def update_all(self, *args, **kwargs):
        model = self.fit.model
        self.p_rda_curve.set_data(model.r_DA, model.p_rDA)
        self.p_rda_plot.do_autoscale()

        self.fda_curve.set_data(model.times,  model.fda)
        self.fd0_curve.set_data(model.times,  model.fd0)
        self.fd_plot.do_autoscale()

        self.et_curve.set_data(model.times,  model.et)
        self.et_plot.do_autoscale()

        self.l_curve_1.set_data(model.l_curve_chi2, model.l_curve_reg)
        self.l_curve_plot_1.do_autoscale()

        self.l_curve_2.set_data(model.l_curve_chi2, model.l_curve_solution_norm)
        self.l_curve_plot_2.do_autoscale()

        y = model.weighted_residuals()
        x = np.arange(y.shape[0])
        self.wres_curve.set_data(x, y)
        self.wres_plot.do_autoscale()
