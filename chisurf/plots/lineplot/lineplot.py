from __future__ import annotations

from collections import OrderedDict
import numpy as np

from chisurf import typing

from chisurf.gui import QtWidgets, QtCore

import pyqtgraph as pg
import pyqtgraph.dockarea
import matplotlib.colors

import chisurf.data
import chisurf.experiments
import chisurf.decorators
import chisurf.gui.decorators
import chisurf.math
import chisurf.fitting
import chisurf.settings
import chisurf.math.statistics
from chisurf.plots import plotbase

colors = chisurf.settings.gui['plot']['colors']


class LinePlotControl(QtWidgets.QWidget):

    director = {
        'data': {
            'lw': 1.0,
            'color': colors['data'],
            'target': 'main_plot',
            'allow_reference_curve': True,
            'allow_shift': True,
            'allow_density': True,
            'plot_only_region': False
        },
        'IRF': {
            'lw': 2.0,
            'color': colors['irf'],
            'target': 'main_plot',
            'allow_reference_curve': False,
            'allow_shift': True,
            'allow_density': True,
            'plot_only_region': False
        },
        'model': {
            'lw': 2.0,
            'target': 'main_plot',
            'color': colors['model'],
            'allow_reference_curve': True,
            'allow_shift': True,
            'allow_density': True,
            'plot_only_region': True
        },
        'weighted residuals': {
            'lw': 2.0,
            'target': 'top_left_plot',
            'label': 'w.res.',
            'color': colors['residuals'],
            'allow_reference_curve': False,
            'allow_shift': True,
            'allow_density': False,
            'plot_only_region': False
        },
        'autocorrelation': {
            'lw': 2.0,
            'target': 'top_right_plot',
            'color': colors['auto_corr'],
            'label': 'a.cor.',
            'allow_reference_curve': False,
            'allow_shift': True,
            'allow_density': False,
            'plot_only_region': False
        },
        'default': {
            'lw': 2.0,
            'color': colors['data'],
            'target': 'main_plot',
            'allow_reference_curve': False,
            'allow_shift': True,
            'allow_density': False,
            'allow_clipping': False,
            'plot_only_region': False
        }
    }

    def getCheckState(self, name):
        for i in range(self.treeWidget.topLevelItemCount()):
            item = self.treeWidget.topLevelItem(i)
            if item.text(2) == name:
                return item.checkState(1)
        return True

    def fill_line_widget(self):
        self.treeWidget.blockSignals(True)
        for nbr, key in enumerate(self.parent.lines):
            item = QtWidgets.QTreeWidgetItem(self.treeWidget, [str(nbr), '', key])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(1, QtCore.Qt.Checked)
        self.treeWidget.blockSignals(False)

    @chisurf.gui.decorators.init_with_ui("linePlotWidget.ui")
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
        self.checkBox_3.stateChanged.connect(self.SetDensity)
        self.checkBox_4.stateChanged.connect(self.SetLog)
        self.checkBox_5.stateChanged.connect(self.SetReference)

    @property
    def plot_ftt(self) -> bool:
        return bool(self.checkBox_plot_ftt.isChecked())

    @plot_ftt.setter
    def plot_ftt(self, v: bool) -> None:
        if v:
            self.checkBox_plot_ftt.setCheckState(2)
        else:
            self.checkBox_plot_ftt.setCheckState(0)

    @property
    def data_logy(self) -> str:
        """
        y-data is plotted logarithmically
        """
        return 'log' if self.checkBox.isChecked() else 'linear'

    @data_logy.setter
    def data_logy(self, v: str):
        if v == 'lin':
            self.checkBox.setCheckState(0)
        else:
            self.checkBox.setCheckState(2)

    @property
    def scale_x(self) -> str:
        return 'log' if self.checkBox_2.isChecked() else 'linear'

    @scale_x.setter
    def scale_x(self, v: str):
        if v == 'lin':
            self.checkBox_2.setCheckState(0)
        else:
            self.checkBox_2.setCheckState(2)

    @property
    def data_is_log_x(self) -> bool:
        return self.scale_x == 'log'

    @property
    def data_is_log_y(self) -> bool:
        return self.data_logy == 'log'

    @property
    def use_reference(self) -> bool:
        """
        If true use a reference curve for plotting
        """
        return bool(self.checkBox_5.isChecked())

    @use_reference.setter
    def use_reference(self, v: bool):
        if v is True:
            self.checkBox_5.setCheckState(2)
        else:
            self.checkBox_5.setCheckState(0)

    @property
    def ymin(self) -> float:
        if self.checkBox_7.isChecked():
            return self.doubleSpinBox_2.value()
        else:
            return None

    @ymin.setter
    def ymin(self, v: float):
        self.doubleSpinBox_2.setValue(v)

    @property
    def ymax(self) -> float:
        if self.checkBox_8.isChecked():
            return self.doubleSpinBox_4.value()
        else:
            return None

    @ymax.setter
    def ymax(self, v: float):
        self.doubleSpinBox_4.setValue(v)

    @property
    def xmin(self) -> float:
        if self.checkBox_4.isChecked():
            return self.doubleSpinBox.value()
        else:
            return None

    @xmin.setter
    def xmin(self, v: float):
        self.doubleSpinBox.setValue(v)

    @property
    def xmax(self) -> float:
        if self.checkBox_6.isChecked():
            return self.doubleSpinBox_3.value()
        else:
            return None

    @xmax.setter
    def xmax(self, v: float):
        self.doubleSpinBox_3.setValue(v)

    @property
    def x_shift(self) -> float:
        return self.doubleSpinBox_6.value()

    @x_shift.setter
    def x_shift(self, v: float):
        self.doubleSpinBox_6.setValue(v)

    @property
    def y_shift(self) -> float:
        return self.doubleSpinBox_5.value()

    @y_shift.setter
    def y_shift(self, v: float):
        self.doubleSpinBox_5.setValue(v)

    @property
    def is_density(self) -> bool:
        return bool(self.checkBox_3.isChecked())

    @is_density.setter
    def is_density(self, v: bool):
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

    name = "Fit"

    def get_bounds(
            self,
            fit: chisurf.fitting.fit.Fit,
            region_selector: pg.LinearRegionItem
    ) -> typing.Tuple[int, int]:
        lb, ub = region_selector.getRegion()

        x_shift = self.plot_controller.x_shift
        lb -= x_shift
        ub -= x_shift

        data_x = fit.data.x
        x_len = len(data_x) - 1

        if self.plot_controller.data_is_log_x:
            lb, ub = 10.0 ** lb, 10.0 ** ub

        lb_i: int = np.searchsorted(data_x, lb, side='right')
        ub_i: int = np.searchsorted(data_x, ub, side='left')

        return np.clip(lb_i - 1, 0, x_len), np.clip(ub_i, 0, x_len)

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            scale_x: str = 'lin',
            d_scaley: str = 'lin',
            r_scaley: str = 'lin',
            reference_curve: bool = False,
            x_label: str = 'x',
            y_label: str = 'y',
            **kwargs
    ):
        # Internal state of region selector
        self.lb_i: int = 0
        self.ub_i: int = 0

        kwargs['fit'] = fit
        super().__init__(**kwargs)
        self.plot_controller = LinePlotControl(
                parent=self,
                scale_x=scale_x,
                d_scaley=d_scaley,
                r_scaley=r_scaley,
                reference_curve=reference_curve
        )
        p1 = pg.PlotWidget()
        p2 = pg.PlotWidget()
        p3 = pg.PlotWidget()
        p1.setXLink(p3)
        p2.setXLink(p3)

        plots = {
            'top_left_plot': p1.getPlotItem(),
            'top_right_plot': p2.getPlotItem(),
            'main_plot': p3.getPlotItem()
        }
        plots['top_left_plot'].hideAxis('bottom')
        plots['top_right_plot'].hideAxis('bottom')

        hide_dock_title = chisurf.settings.gui['plot']['hideTitle']
        d1 = pyqtgraph.dockarea.Dock(
            "Residuals",
            size=(250, 80),
            hideTitle=hide_dock_title
        )
        d2 = pyqtgraph.dockarea.Dock(
            "A.corr. residuals",
            size=(250, 80),
            hideTitle=hide_dock_title
        )
        d3 = pyqtgraph.dockarea.Dock(
            "Data",
            size=(250, 250),
            hideTitle=hide_dock_title
        )
        d1.addWidget(p1)
        d2.addWidget(p2)
        d3.addWidget(p3)

        area = pyqtgraph.dockarea.DockArea()
        area.addDock(d1, 'top')
        area.addDock(d2, 'top', d1)
        area.addDock(d3, 'bottom', d1)
        self.layout.addWidget(area)

        # Labels
        self.text = pg.TextItem(
            text='',
            border='w',
            fill=(0, 0, 255, 100),
            anchor=(0, 0)
        )
        self.text.setParentItem(plots['main_plot'])
        self.text.setPos(100, 0)

        # Fitting-region selector
        if chisurf.settings.gui['plot']['enable_region_selector']:
            ca = list(matplotlib.colors.hex2color(colors["region_selector"]))
            co = [ca[0] * 255, ca[1] * 255, ca[2] * 255, colors["region_selector_alpha"]]
            region = pg.LinearRegionItem(brush=co)
            plots['main_plot'].addItem(region)
            self.region = region

            def onRegionUpdate(evt):
                self.lb_i, self.ub_i = self.get_bounds(fit, region)
                lb, ub = fit.data.x[self.lb_i], fit.data.x[self.ub_i]
                x_shift = self.plot_controller.x_shift
                lb += x_shift
                ub += x_shift
                if self.plot_controller.data_is_log_x:
                    lb = np.log10(lb)
                    ub = np.log10(ub)
                self.region.setRegion((lb, ub))
                chisurf.run(f"cs.current_fit.fit_range = {self.lb_i}, {self.ub_i}")
                self.update(only_fit_range=True)

            region.sigRegionChangeFinished.connect(onRegionUpdate)

        # Grid
        if chisurf.settings.gui['plot']['enable_grid']:
            if chisurf.settings.gui['plot']['show_data_grid']:
                plots['main_plot'].showGrid(True, True, 0.5)
            if chisurf.settings.gui['plot']['show_residual_grid']:
                plots['top_left_plot'].showGrid(True, True, 1.0)
            if chisurf.settings.gui['plot']['show_acorr_grid']:
                plots['top_right_plot'].showGrid(True, True, 1.0)
        # Labels
        if chisurf.settings.gui['plot']['label_axis']:
            plots['top_left_plot'].setLabel('left', "w.res.")
            plots['top_right_plot'].setLabel('left', "a.corr.")
            plots['main_plot'].setLabel('left', y_label)
            plots['main_plot'].setLabel('bottom', x_label)

        lines = OrderedDict()
        curves = self.fit.get_curves()
        curves_keys = list(curves.keys())[::-1]
        for i, curve_key in enumerate(curves_keys):
            lines[curve_key] = self.add_plot(
                curves=curves,
                curve_key=curve_key,
                plot_dict=plots,
                index=i
            )
        self.lines = lines
        self.plots = plots
        self.plot_controller.fill_line_widget()

    def add_plot(
            self,
            curves: typing.Dict,
            curve_key: str,
            plot_dict: typing.Dict,
            index: int = 1
    ):
        color_idx = index % len(chisurf.settings.colors)
        pen_color = chisurf.settings.colors[color_idx]['hex']
        lw = chisurf.settings.gui['plot']['line_width']

        director = self.plot_controller.director
        if curve_key in director.keys():
            for ik in director.keys():
                # if the curve name matches the template
                if ik in curve_key:
                    curve_options = director[ik]
                    target_plot = plot_dict[
                        curve_options.get('target', 'main_plot')
                    ]
                    lw = curve_options.get('lw', lw)
                    pen_color = curve_options.get('color', pen_color)
                    label = curve_options.get('label', curve_key)
                    if curve_key != ik:
                        # make the line half as wide, and transparent (30%)
                        lw *= 0.5
                        pen_color = '#4D' + pen_color.split('#')[1]
                    return target_plot.plot(
                        x=[0.0], y=[0.0],
                        pen=pg.mkPen(pen_color, width=lw),
                        name=label
                    )
        else:
            curve = curves[curve_key]
            if isinstance(curve, chisurf.data.DataCurve):
                curve_options = director['default']
                target_plot = plot_dict[
                    curve_options.get('target', 'main_plot')
                ]
                return target_plot.plot(
                    x=[0.0], y=[0.0],
                    pen=pg.mkPen(pen_color, width=lw),
                    name=curve_key
                )

        return None

    def update(self, only_fit_range: bool = False, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

        fit = self.fit
        data_log_y = self.plot_controller.data_is_log_y
        data_log_x = self.plot_controller.data_is_log_x
        director = self.plot_controller.director

        curves = fit.get_curves()
        data = curves['data']

        y_shift = self.plot_controller.y_shift
        x_shift = self.plot_controller.x_shift

        # update region selector
        self.region.blockSignals(True)

        lb_min, ub_max = data.x[0], data.x[-1]
        lb, ub = data.x[self.fit.xmin], data.x[self.fit.xmax]

        lb_min += x_shift
        ub_max += x_shift
        lb += x_shift
        ub += x_shift

        if data_log_x:
            lb_min = np.log10(lb_min)
            ub_max = np.log10(ub_max)
            lb = np.log10(lb)
            ub = np.log10(ub)

        self.region.setBounds((lb_min, ub_max))
        self.region.setRegion((lb, ub))

        self.region.blockSignals(False)

        curves_keys = list(curves.keys())[::-1]
        for i, curve_key in enumerate(curves_keys):
            curve_settings = director.get(curve_key, director['default'])
            curve = curves[curve_key]

            y = np.copy(curve.y)
            x = np.copy(curve.x)

            if curve_settings['allow_shift']:
                y += y_shift
                x += x_shift

            # Reference-function
            if self.plot_controller.use_reference and curve_settings['allow_reference_curve']:
                reference = fit.model.reference
                if reference is None:
                    reference = np.ones_like(y)
                    chisurf.logging.warning("No reference curve provided by the model.")
                y /= reference

            if self.plot_controller.is_density and curve_settings['allow_density']:
                y[1:] = y[1:] / np.diff(x)

            line: pg.PlotDataItem = self.lines[curve_key]
            if curve_settings['plot_only_region']:
                line.setData(x=x[fit.xmin:fit.xmax], y=y[fit.xmin:fit.xmax])
            else:
                line.setData(x=x, y=y)
            if not self.plot_controller.getCheckState(curve_key):
                line.hide()
            else:
                line.show()

        # Set log-scales
        self.plots['main_plot'].setLogMode(x=data_log_x, y=data_log_y)
        self.plots['top_left_plot'].setLogMode(x=data_log_x)
        self.plots['top_right_plot'].setLogMode(x=data_log_x)

        # Set manual scale
        xRange, yRange = None, None
        a_min = self.plot_controller.xmin
        a_max = self.plot_controller.xmax
        c = data.x
        lm = data_log_x
        if a_min or a_max:
            a_min = c[0] if not a_min else a_min
            a_max = c[-1] if not a_max else a_max
            if lm:
                a_min = np.log10(a_min)
                a_max = np.log10(a_max)
            xRange = [a_min, a_max]

        a_min = self.plot_controller.ymin
        a_max = self.plot_controller.ymax
        c = data.y
        lm = data_log_y
        if a_min or a_max:
            a_min = c[0] if not a_min else a_min
            a_max = c[-1] if not a_max else a_max
            if lm:
                a_min = np.log10(a_min)
                a_max = np.log10(a_max)
            yRange = [a_min, a_max]
        if xRange or yRange:
            self.plots['main_plot'].setRange(xRange=xRange, yRange=yRange)

        self.text.updateTextPos()
        self.text.setHtml(
            f'<div style="name-align: center">'
            f'     <span style="color: #FF0; font-size: 10pt;">'
            f'         Fit-range {fit.xmin}, {fit.xmax} <br />'
            f'         &Chi;<sup>2</sup>={fit.chi2r:.4f} <br />'
            f'         DW={fit.durbin_watson: .4f}'
            f'     </span>'
            f'</div>'
        )
