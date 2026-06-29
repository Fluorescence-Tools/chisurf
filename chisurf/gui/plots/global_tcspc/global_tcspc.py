from __future__ import annotations

import numpy as np
from chisurf.gui.plots._qwt_compat import make, CurveDialog

import chisurf.gui.decorators
from chisurf.gui.plots.plotbase import Plot


class GlobalEt(Plot):

    name = "GlobalEt"

    @chisurf.gui.decorators.init_with_ui(ui_filename="et_plot_layout.ui")
    def __init__(
            self,
            fit: chisurf.core.fitting.fit.FitGroup,
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
