from __future__ import annotations

from collections import OrderedDict
import numpy as np

from chisurf import typing

from chisurf.gui import QtWidgets, QtCore, QtGui

import pyqtgraph as pg
import pyqtgraph.dockarea
import matplotlib.colors
try:
    from qtpy import sip
except ImportError:
    try:
        import sip
    except ImportError:
        sip = None

import chisurf.core.data
import chisurf.core.experiments
import chisurf.core.decorators
import chisurf.gui.decorators
import chisurf.core.math
import chisurf.core.fitting
import chisurf.core.settings
import chisurf.core.math.statistics
from chisurf.gui.plots import plotbase
from chisurf.core.actions import record_action


class DraggableTextItem(pg.TextItem):
    """A TextItem that can be dragged with the mouse."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self._dragging = False
        self._dragOffset = QtCore.QPointF(0, 0)

    def hoverEnterEvent(self, event):
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self._dragOffset = event.pos()
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self._dragging and event.buttons() & QtCore.Qt.LeftButton:
            new_pos = self.mapToParent(event.pos() - self._dragOffset)
            self.setPos(new_pos)
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = False
            self.setCursor(QtCore.Qt.OpenHandCursor)
            event.accept()
        else:
            event.ignore()

colors = chisurf.core.settings.gui['plot']['colors']


class LinePlotControl(QtWidgets.QWidget):

    director = {
        'data': {
            'lw': 1.0,
            'color': colors['data'],
            'target': 'main_plot',
            'allow_reference_curve': True,
            'allow_shift': True,
            'allow_density': True,
            'plot_only_region': False,
            'auto_downsample': True,
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
            'plot_only_region': False,
            'auto_downsample': True,
        },
        'autocorrelation': {
            'lw': 2.0,
            'target': 'top_right_plot',
            'color': colors['auto_corr'],
            'label': 'a.cor.',
            'allow_reference_curve': False,
            'allow_shift': True,
            'allow_density': False,
            'plot_only_region': False,
            'auto_downsample': True,
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
        self.checkBox_9.stateChanged.connect(self.SetDisplayGroup)

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

    @property
    def display_group(self) -> bool:
        """
        If true, display all fits in the group with current fit highlighted
        """
        return bool(self.checkBox_9.isChecked())

    @display_group.setter
    def display_group(self, v: bool):
        if v is True:
            self.checkBox_9.setCheckState(2)
        else:
            self.checkBox_9.setCheckState(0)

    def SetReference(self):
        self.parent.update()

    def SetLog(self):
        self.parent.update()

    def SetDensity(self):
        self.parent.update()

    def SetDisplayGroup(self):
        self.parent.update()


class LinePlot(plotbase.Plot):

    name = "Fit"
    regionChanged = QtCore.Signal(int, int)

    def get_bounds(
            self,
            fit: chisurf.core.fitting.fit.Fit,
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
            fit: chisurf.core.fitting.fit.FitGroup,
            scale_x: str = 'lin',
            d_scaley: str = 'lin',
            r_scaley: str = 'lin',
            reference_curve: bool = False,
            x_label: str = 'x',
            y_label: str = 'y',
            curve_styles: typing.Dict | None = None,
            **kwargs
    ):
        # Internal state of region selector
        self.lb_i: int = 0
        self.ub_i: int = 0

        self.curve_styles = curve_styles or {}

        kwargs['fit'] = fit
        super().__init__(**kwargs)
        self.plot_controller = LinePlotControl(
                parent=self,
                scale_x=scale_x,
                d_scaley=d_scaley,
                r_scaley=r_scaley,
                reference_curve=reference_curve
        )

        # If the plot is associated with a FitGroup containing multiple local fits,
        # default to displaying the full group. Do this once and avoid overriding
        # user intent later.
        self._auto_display_group_applied = False
        try:
            self._auto_enable_display_group_if_grouped()
        except Exception:
            pass
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

        hide_dock_title = chisurf.core.settings.gui['plot']['hideTitle']
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

        # Labels - using draggable text item for quality parameters
        self.text = DraggableTextItem(
            text='',
            border='w',
            fill=(0, 0, 255, 100),
            anchor=(0, 0)
        )
        self.text.setParentItem(plots['main_plot'])
        self.text.setPos(100, 0)

        # Fitting-region selector
        if chisurf.core.settings.gui['plot']['enable_region_selector']:
            ca = list(matplotlib.colors.hex2color(colors["region_selector"]))
            co = [ca[0] * 255, ca[1] * 255, ca[2] * 255, colors["region_selector_alpha"]]
            region = pg.LinearRegionItem(brush=co)
            plots['main_plot'].addItem(region)
            self.region = region

            def onRegionUpdate(evt):
                # Get the currently selected fit for region update
                if hasattr(fit, 'selected_fit'):
                    current_fit = fit.selected_fit
                else:
                    current_fit = fit
                    
                self.lb_i, self.ub_i = self.get_bounds(current_fit, region)
                lb, ub = current_fit.data.x[self.lb_i], current_fit.data.x[self.ub_i]
                x_shift = self.plot_controller.x_shift
                lb += x_shift
                ub += x_shift
                if self.plot_controller.data_is_log_x:
                    lb = np.log10(lb)
                    ub = np.log10(ub)
                self.region.setRegion((lb, ub))
                import chisurf
                chisurf.core.actions.dispatch(
                    name="fit.range.set",
                    payload={
                        "xmin": int(self.lb_i),
                        "xmax": int(self.ub_i),
                        "fit_index": getattr(self.fit, "fit_idx", 0),
                    },
                )
                try:
                    self.regionChanged.emit(self.lb_i, self.ub_i)
                except Exception:
                    pass
                self.update(only_fit_range=True)

            region.sigRegionChangeFinished.connect(onRegionUpdate)

        # Grid
        if chisurf.core.settings.gui['plot']['enable_grid']:
            if chisurf.core.settings.gui['plot']['show_data_grid']:
                plots['main_plot'].showGrid(True, True, 0.5)
            if chisurf.core.settings.gui['plot']['show_residual_grid']:
                plots['top_left_plot'].showGrid(True, True, 1.0)
            if chisurf.core.settings.gui['plot']['show_acorr_grid']:
                plots['top_right_plot'].showGrid(True, True, 1.0)
        # Axis labels: always show for clarity
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

    def _auto_enable_display_group_if_grouped(self) -> None:
        """Enable "display group" by default for multi-fit FitGroups.

        This keeps the plot readable for grouped fits (e.g. VV/VH, grouped FCS)
        without forcing the user to manually toggle the checkbox each time.
        """

        if bool(getattr(self, "_auto_display_group_applied", False)):
            return
        grouped_fits = getattr(self.fit, "grouped_fits", None)
        if not isinstance(grouped_fits, (list, tuple)):
            return
        if len(grouped_fits) <= 1:
            return
        if self.plot_controller.display_group:
            self._auto_display_group_applied = True
            return

        cb = getattr(self.plot_controller, "checkBox_9", None)
        try:
            if cb is not None:
                cb.blockSignals(True)
            self.plot_controller.display_group = True
        finally:
            try:
                if cb is not None:
                    cb.blockSignals(False)
            except Exception:
                pass
        self._auto_display_group_applied = True

    def add_plot(
            self,
            curves: typing.Dict,
            curve_key: str,
            plot_dict: typing.Dict,
            index: int = 1
    ):
        color_idx = index % len(chisurf.core.settings.colors)
        pen_color = chisurf.core.settings.colors[color_idx]['hex']
        lw = chisurf.core.settings.gui['plot']['line_width']

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
                    auto_downsample = curve_options.get('auto_downsample', False)
                    clip_to_view = curve_options.get('clip_to_view', auto_downsample)
                    if curve_key != ik:
                        # make the line half as wide, and transparent (30%)
                        lw *= 0.5
                        pen_color = '#4D' + pen_color.split('#')[1]
                    pen = pg.mkPen(pen_color, width=lw)
                    line = target_plot.plot(
                        x=[0.0], y=[0.0],
                        pen=pen,
                        name=label,
                        autoDownsample=auto_downsample,
                        clipToView=clip_to_view,
                    )
                    self._apply_curve_style(curve_key, line)
                    return line
        else:
            curve = curves[curve_key]
            if isinstance(curve, chisurf.core.data.DataCurve):
                curve_options = director['default']
                target_plot = plot_dict[
                    curve_options.get('target', 'main_plot')
                ]
                auto_downsample = curve_options.get('auto_downsample', False)
                clip_to_view = curve_options.get('clip_to_view', auto_downsample)
                pen = pg.mkPen(pen_color, width=lw)
                line = target_plot.plot(
                    x=[0.0], y=[0.0],
                    pen=pen,
                    name=curve_key,
                    autoDownsample=auto_downsample,
                    clipToView=clip_to_view,
                )
                self._apply_curve_style(curve_key, line)
                return line

        return None

    def _get_curve_style(self, curve_key: str) -> typing.Optional[typing.Dict]:
        styles = getattr(self, "curve_styles", None)
        if not styles:
            return None
        if curve_key in styles:
            return styles[curve_key]
        if "_" in curve_key:
            base = curve_key.split("_", 1)[0]
            if base in styles:
                return styles[base]
        return None

    def _apply_curve_style(self, curve_key: str, line: pg.PlotDataItem) -> None:
        style = self._get_curve_style(curve_key)
        if not style:
            return
        try:
            if "pen" in style:
                line.setPen(style.get("pen"))
        except Exception:
            pass
        try:
            symbol = style.get("symbol")
            if symbol is not None:
                line.setSymbol(symbol)
        except Exception:
            pass
        try:
            size = style.get("symbol_size")
            if size is not None:
                line.setSymbolSize(size)
        except Exception:
            pass
        try:
            brush = style.get("symbol_brush")
            if brush is not None:
                line.setSymbolBrush(brush)
        except Exception:
            pass
        try:
            if style.get("no_line"):
                line.setPen(None)
        except Exception:
            pass

    def _update_reference_checkbox(self):
        """
        Check if the model has a reference attribute and update the checkbox state accordingly.
        If the model doesn't have a reference attribute, disable the checkbox.
        """
        has_reference = False
        try:
            # Check if model has reference attribute
            if hasattr(self.fit.model, 'reference'):
                has_reference = True
        except Exception:
            pass
            
        # Update the checkbox state
        self.plot_controller.checkBox_5.setEnabled(has_reference)
        if not has_reference and self.plot_controller.use_reference:
            # If reference is not available but checkbox is checked, uncheck it
            self.plot_controller.use_reference = False

    def _metrics_text_alive(self) -> bool:
        """Return True when the overlay TextItem and backing Qt objects are alive."""

        text_item = getattr(self, "text", None)
        if text_item is None:
            return False

        try:
            if sip.isdeleted(text_item):  # type: ignore[arg-type]
                return False
        except Exception:
            pass

        backing_text_item = getattr(text_item, "textItem", None)
        if backing_text_item is None:
            return False
        try:
            if sip.isdeleted(backing_text_item):  # type: ignore[arg-type]
                return False
        except Exception:
            pass

        return True

    def _build_metrics_overlay_text(self, current_fit) -> str:
        """Build a plain-text variant of the metrics overlay.

        Using plain text avoids Qt rich-text parsing paths that have shown
        instability in some Windows save/switch workflows.
        """

        grouped_fits = getattr(self.fit, "grouped_fits", None)
        show_group = bool(self.plot_controller.display_group) and isinstance(grouped_fits, (list, tuple)) and len(grouped_fits) > 1
        current_idx = getattr(self.fit, "selected_fit_index", None)
        if not isinstance(current_idx, int):
            try:
                current_idx = grouped_fits.index(current_fit) if show_group else 0
            except Exception:
                current_idx = 0

        def _fmt_float(v, nd=4):
            try:
                return f"{float(v):.{int(nd)}f}"
            except Exception:
                return "?"

        header = f"Range {int(getattr(current_fit, 'xmin', 0))}, {int(getattr(current_fit, 'xmax', 0))}"
        if show_group:
            lines = [header, "chi2r\tDW"]
            for idx, f in enumerate(grouped_fits):
                marker = "*" if idx == current_idx else " "
                chi2r = _fmt_float(getattr(f, "chi2r", None), nd=4)
                dw = _fmt_float(getattr(f, "durbin_watson", None), nd=4)
                lines.append(f"{marker}{chi2r}\t{dw}")
            return "\n".join(lines)

        return "\n".join([
            header,
            f"chi2r={_fmt_float(getattr(current_fit, 'chi2r', None), nd=4)}",
            f"DW={_fmt_float(getattr(current_fit, 'durbin_watson', None), nd=4)}",
        ])
            
    def update(self, only_fit_range: bool = False, *args, **kwargs) -> None:
        super().update(*args, **kwargs)

        # Auto-enable group display once for multi-fit groups (do not override
        # user toggles after initial application).
        try:
            self._auto_enable_display_group_if_grouped()
        except Exception:
            pass

        fit = self.fit
        data_log_y = self.plot_controller.data_is_log_y
        data_log_x = self.plot_controller.data_is_log_x
        director = self.plot_controller.director
        
        # Check if model has reference attribute and update checkbox state
        self._update_reference_checkbox()

        curves = fit.get_curves()
        data = curves['data']

        y_shift = self.plot_controller.y_shift
        x_shift = self.plot_controller.x_shift

        # update region selector
        self.region.blockSignals(True)

        # Get the currently selected fit for region selector bounds
        if hasattr(self.fit, 'selected_fit'):
            current_fit = self.fit.selected_fit
            current_data = current_fit.data
        else:
            current_fit = self.fit
            current_data = data
            
        lb_min, ub_max = current_data.x[0], current_data.x[-1]
        lb, ub = current_data.x[current_fit.xmin], current_data.x[current_fit.xmax]

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

        # Handle group display mode
        if self.plot_controller.display_group and hasattr(self.fit, 'grouped_fits'):
            # Display all fits in the group
            self._plot_group_curves(fit, data_log_x, data_log_y, director, x_shift, y_shift)
        elif hasattr(self.fit, 'grouped_fits'):
            # FitGroup but display group disabled: only show active fit
            self._plot_active_fit_only(fit, data_log_x, data_log_y, director, x_shift, y_shift)
        else:
            # Normal single fit display
            self._plot_single_fit_curves(fit, curves, data_log_x, data_log_y, director, x_shift, y_shift)

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

        if self._metrics_text_alive() and not bool(getattr(chisurf, "_suspend_plot_metrics_overlay", False)):
            try:
                self.text.updateTextPos()
                metrics_text = self._build_metrics_overlay_text(current_fit=current_fit)
                try:
                    self.text.setText(metrics_text, color="#FF0")
                except TypeError:
                    self.text.setText(metrics_text)
            except Exception:
                pass

    def _build_metrics_overlay_html(self, current_fit) -> str:
        """Build the metrics overlay HTML.

        For FitGroups with multiple local fits shown, present a compact table of
        chi2r and Durbin-Watson, with the current fit highlighted.
        """

        font_pt = 8
        try:
            # Keep it compact; allow user override if present.
            font_pt = int(chisurf.core.settings.gui.get("plot", {}).get("metrics_font_pt", font_pt))
        except Exception:
            font_pt = 8

        # Determine whether this is a multi-fit group display.
        grouped_fits = getattr(self.fit, "grouped_fits", None)
        show_group = bool(self.plot_controller.display_group) and isinstance(grouped_fits, (list, tuple)) and len(grouped_fits) > 1
        current_idx = getattr(self.fit, "selected_fit_index", None)
        if not isinstance(current_idx, int):
            try:
                current_idx = grouped_fits.index(current_fit) if show_group else 0
            except Exception:
                current_idx = 0

        def _fmt_float(v, nd=4):
            try:
                return f"{float(v):.{int(nd)}f}"
            except Exception:
                return "?"

        if show_group:
            rows = []
            for idx, f in enumerate(grouped_fits):
                chi2r = _fmt_float(getattr(f, "chi2r", None), nd=4)
                dw = _fmt_float(getattr(f, "durbin_watson", None), nd=4)
                if idx == current_idx:
                    row_style = "font-weight: 700; background-color: #2a2a2a;"
                else:
                    row_style = "color: #cccccc;"
                rows.append(
                    f"<tr style='{row_style}'>"
                    f"<td style='padding: 0px 6px 0px 0px; text-align:right;'>" + chi2r + "</td>"
                    f"<td style='padding: 0px 0px 0px 6px; text-align:right;'>" + dw + "</td>"
                    f"</tr>"
                )

            return (
                "<div style='name-align:center;'>"
                f"<div style='color:#FF0; font-size:{font_pt}pt;'>"
                + f"<div>Range {int(getattr(current_fit, 'xmin', 0))}, {int(getattr(current_fit, 'xmax', 0))}</div>"
                + "<table style='margin-top:3px; border-collapse:collapse;'>"
                + f"<tr style='color:#aaaaaa; font-size:{max(7, font_pt-1)}pt;'>"
                + "<th style='text-align:right; padding: 0px 6px 1px 0px;'>chi2r</th>"
                + "<th style='text-align:right; padding: 0px 0px 1px 6px;'>DW</th>"
                + "</tr>"
                + "".join(rows)
                + "</table>"
                + "</div></div>"
            )

        # Single-fit overlay (compact)
        return (
            "<div style='name-align: center'>"
            f"<span style='color: #FF0; font-size: {font_pt}pt;'>"
            f"Range {int(getattr(current_fit, 'xmin', 0))}, {int(getattr(current_fit, 'xmax', 0))}<br/>"
            f"&Chi;<sup>2</sup>={_fmt_float(getattr(current_fit, 'chi2r', None), nd=4)}<br/>"
            f"DW={_fmt_float(getattr(current_fit, 'durbin_watson', None), nd=4)}"
            "</span></div>"
        )

    def _plot_single_fit_curves(self, fit, curves, data_log_x, data_log_y, director, x_shift, y_shift):
        """Plot curves for a single fit (original behavior)"""
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
                try:
                    reference = fit.model.reference
                    if reference is None:
                        reference = np.ones_like(y)
                        chisurf.logging.warning("No reference curve provided by the model.")
                    y /= reference
                except AttributeError:
                    chisurf.logging.warning("Model does not have a reference attribute.")

            if self.plot_controller.is_density and curve_settings['allow_density']:
                y[1:] = y[1:] / np.diff(x)

            line: pg.PlotDataItem = self.lines[curve_key]
            # Base data for plotting: either full curve or fit-range only
            if curve_settings['plot_only_region']:
                x_plot = x[fit.xmin:fit.xmax]
                y_plot = y[fit.xmin:fit.xmax]
            else:
                x_plot = x
                y_plot = y

            line.setData(x=x_plot, y=y_plot)
            if not self.plot_controller.getCheckState(curve_key):
                line.hide()
            else:
                line.show()

    def _plot_group_curves(self, fit, data_log_x, data_log_y, director, x_shift, y_shift):
        """Plot curves for all fits in the group with highlighting"""
        grouped_fits = getattr(fit, 'grouped_fits', [])
        if not grouped_fits:
            # Fallback to single fit if no group
            self._plot_single_fit_curves(fit, fit.get_curves(), data_log_x, data_log_y, director, x_shift, y_shift)
            return

        current_fit_index = getattr(fit, 'selected_fit_index', 0)
        wres_offsets = self._compute_group_curve_offsets(
            grouped_fits,
            current_fit_index,
            curve_name='weighted residuals'
        )
        acor_offsets = self._compute_group_curve_offsets(
            grouped_fits,
            current_fit_index,
            curve_name='autocorrelation'
        )
        
        # Create line names for group fits
        group_line_names = {}
        for i, group_fit in enumerate(grouped_fits):
            group_curves = group_fit.get_curves()
            for curve_key in group_curves:
                group_line_names[f"{curve_key}_{i}"] = (group_fit, curve_key)

        # Plot all curves from all fits
        for line_name, (group_fit, curve_key) in group_line_names.items():
            group_curves = group_fit.get_curves()
            curve = group_curves[curve_key]
            curve_settings = director.get(curve_key, director['default'])

            y = np.copy(curve.y)
            x = np.copy(curve.x)

            if curve_settings['allow_shift']:
                y += y_shift
                x += x_shift

            # Reference-function
            if self.plot_controller.use_reference and curve_settings['allow_reference_curve']:
                try:
                    reference = group_fit.model.reference
                    if reference is None:
                        reference = np.ones_like(y)
                        chisurf.logging.warning("No reference curve provided by the model.")
                    y /= reference
                except AttributeError:
                    chisurf.logging.warning("Model does not have a reference attribute.")

            if self.plot_controller.is_density and curve_settings['allow_density']:
                y[1:] = y[1:] / np.diff(x)

            # In grouped display mode, vertically separate weighted residuals so
            # each local-fit residual trace remains readable.
            if 'weighted residuals' in curve_key:
                fit_index = grouped_fits.index(group_fit)
                y = y + wres_offsets.get(fit_index, 0.0)
            elif 'autocorrelation' in curve_key:
                fit_index = grouped_fits.index(group_fit)
                y = y + acor_offsets.get(fit_index, 0.0)

            # Get or create the line for this group curve
            if line_name not in self.lines:
                # Create a new line for this group curve using the same logic as original lines
                curve_options = director.get(curve_key, director['default'])
                target_plot = self.plots[curve_settings['target']]
                
                # Get color and width using the same logic as original line creation
                lw = curve_options.get('lw', 2)
                pen_color = curve_options.get('color', '#FFFFFF')
                label = curve_options.get('label', curve_key)
                auto_downsample = curve_options.get('auto_downsample', False)
                clip_to_view = curve_options.get('clip_to_view', auto_downsample)
                
                # Apply the same transparency logic for non-primary curves
                if curve_key != curve_options.get('name', curve_key):
                    # make the line half as wide, and transparent (30%)
                    lw *= 0.5
                    pen_color = '#4D' + pen_color.split('#')[1]
                
                pen = pg.mkPen(pen_color, width=lw)
                line = target_plot.plot(
                    x=[0.0], y=[0.0],
                    pen=pen,
                    name=label,
                    autoDownsample=auto_downsample,
                    clipToView=clip_to_view,
                )
                self._apply_curve_style(curve_key, line)
                self.lines[line_name] = line
            else:
                line = self.lines[line_name]

            # Base data for plotting: either full curve or fit-range only
            if curve_settings['plot_only_region']:
                x_plot = x[group_fit.xmin:group_fit.xmax]
                y_plot = y[group_fit.xmin:group_fit.xmax]
            else:
                x_plot = x
                y_plot = y

            # Apply transparency and highlighting
            fit_index = grouped_fits.index(group_fit)
            if fit_index == current_fit_index:
                # Current fit: solid (full opacity)
                alpha = 1.0  # Full opacity
            else:
                # Other fits: transparent (reduced alpha)
                alpha = 0.4  # Semi-transparent
            
            # Try multiple approaches for transparency
            try:
                # Method 1: Try Qt graphics opacity effect
                from qtpy import QtCore, QtGui
                if hasattr(line, 'setGraphicsEffect'):
                    effect = QtGui.QGraphicsOpacityEffect()
                    effect.setOpacity(alpha)
                    line.setGraphicsEffect(effect)
                else:
                    raise AttributeError("No graphics effect support")
            except:
                try:
                    # Method 2: Set opacity on the line item
                    line.setOpacity(alpha)
                except:
                    try:
                        # Method 3: Use setAlpha
                        line.setAlpha(int(alpha * 255), auto=False)
                    except:
                        try:
                            # Method 4: Modify pen color with alpha
                            current_pen = line.opts['pen']
                            if hasattr(current_pen, 'color'):
                                color = current_pen.color()
                                if isinstance(color, str) and color.startswith('#'):
                                    # Convert hex color to RGB and add alpha
                                    rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    # Use RGBA format instead of ARGB
                                    new_color = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}{int(alpha*255):02x}'
                                    new_pen = pg.mkPen(new_color, width=current_pen.width())
                                    line.setPen(new_pen)
                                else:
                                    # Create new pen with alpha using QColor
                                    qcolor = QtGui.QColor(color)
                                    qcolor.setAlphaF(alpha)
                                    new_pen = pg.mkPen(qcolor, width=current_pen.width())
                                    line.setPen(new_pen)
                            else:
                                # Direct pen color with QColor
                                pen_color = current_pen
                                if isinstance(pen_color, str) and pen_color.startswith('#'):
                                    rgb = tuple(int(pen_color[i:i+2], 16) for i in (1, 3, 5))
                                    qcolor = QtGui.QColor(*rgb)
                                    qcolor.setAlphaF(alpha)
                                    new_pen = pg.mkPen(qcolor, width=2)
                                    line.setPen(new_pen)
                        except:
                            pass  # If all methods fail, continue without transparency
                
            line.setData(x=x_plot, y=y_plot)

            # Show/hide based on checkbox state
            if not self.plot_controller.getCheckState(curve_key):
                line.hide()
            else:
                line.show()

        # Hide original single-fit lines when in group mode
        for curve_key in fit.get_curves():
            if curve_key in self.lines:
                self.lines[curve_key].hide()

    def _plot_active_fit_only(self, fit, data_log_x, data_log_y, director, x_shift, y_shift):
        """Plot only the currently selected fit from a FitGroup"""
        # Get the currently selected fit
        current_fit = getattr(fit, 'selected_fit', fit)
        current_curves = current_fit.get_curves()
        
        # Hide all group lines first
        grouped_fits = getattr(fit, 'grouped_fits', [])
        for i, group_fit in enumerate(grouped_fits):
            group_curves = group_fit.get_curves()
            for curve_key in group_curves:
                line_name = f"{curve_key}_{i}"
                if line_name in self.lines:
                    self.lines[line_name].hide()
        
        # Plot only the current fit using the single fit logic
        self._plot_single_fit_curves(current_fit, current_curves, data_log_x, data_log_y, director, x_shift, y_shift)

    def _compute_group_curve_offsets(
            self,
            grouped_fits,
            current_fit_index: int,
            curve_name: str
    ) -> typing.Dict[int, float]:
        """Return per-fit y-offsets for grouped residual-like curve display.

        The spacing is derived from residual amplitudes across the group and the
        active fit is centered at zero.
        """

        if len(grouped_fits) <= 1:
            return {}

        amplitudes = []
        for group_fit in grouped_fits:
            try:
                curves = group_fit.get_curves()
                y = None
                if curve_name in curves:
                    y = np.asarray(curves[curve_name].y, dtype=float)
                else:
                    for key, c in curves.items():
                        if curve_name in str(key):
                            y = np.asarray(c.y, dtype=float)
                            break
                if y is None:
                    continue
            except Exception:
                continue
            if y.size == 0:
                continue
            finite = np.isfinite(y)
            if not np.any(finite):
                continue
            yv = y[finite]
            amp = float(np.nanpercentile(np.abs(yv), 95.0))
            if np.isfinite(amp) and amp > 0.0:
                amplitudes.append(amp)

        if amplitudes:
            typical_amp = float(np.nanmedian(amplitudes))
        else:
            typical_amp = 1.0

        # Keep residual traces separated, similar to pyqtgraph's multi-curve
        # demonstration where each curve is shifted by a fixed y-step.
        spacing = max(1.0, 2.5 * typical_amp)

        return {
            idx: (idx - int(current_fit_index)) * spacing
            for idx in range(len(grouped_fits))
        }
