from __future__ import annotations

import pyqtgraph as pg
import copy
import numpy as np

from chisurf.gui import QtWidgets
from chisurf.gui.tools.parameter_editor import ParameterEditor

import chisurf.fitting
import chisurf.fluorescence
import chisurf.math.datatools
from chisurf.plots import plotbase
from chisurf.plots.lineplot.lineplot import DraggableTextItem

plot_settings = chisurf.settings.gui['plot']
colors = plot_settings['colors']
color_scheme = chisurf.settings.colors
lw = plot_settings['line_width']

"""
For plotting
"""
def d21(x, **kwargs):
    return x[0][0], x[0][1]

# Define some distribution options: If an attribute in a model is there it will be plotted.
# accessor: a function that accesses the attribute and returns a pair (y, x)
# that is plotted
# accessor_kwargs: are kwargs that are passed to the accessor function
# plot_options: default options used for plotting
distribution_options = {
    'Distance': {
        'attribute': 'distance_distribution',
        'accessor': lambda x, **kwargs: (x[0][0], x[0][1]),
        'accessor_kwargs': {'sort': False},
        'curve_options': {
            'stepMode': False,  # 'right'
            'connect': False,   # 'all'
            'symbol': "t",
            'multi_curve': False
        }
    },
    'FRET-rate': {
        'attribute': 'fret_rate_spectrum',
        'accessor': chisurf.math.datatools.interleaved_to_two_columns,
        'accessor_kwargs': {'sort': True},
        'curve_options': {
            'stepMode': False, #'right',
            'connect': False, # 'all',
            'symbol': "x",
            'multi_curve': False
        }
    },
    'Lifetime': {
        'attribute': 'lifetime_spectrum',
        'accessor': chisurf.math.datatools.interleaved_to_two_columns,
        'accessor_kwargs': {
            'sort': True
        },
        'curve_options': {
            'stepMode': False,
            'connect': False,
            'symbol': "o",
            'multi_curve': False
        }
    }
}


class DistributionPlotControl(QtWidgets.QWidget):

    @property
    def distribution_type(self):
        return str(self.selector.currentText())

    @property
    def show_gaussians(self) -> bool:
        """Return whether individual Gaussian component curves should be shown.

        For non-Gaussian PDA models this has no effect, but for
        PdaGaussianDistanceModel the extra component curves are drawn only
        when this checkbox is enabled.
        """
        try:
            return bool(self.gauss_checkbox.isChecked())
        except AttributeError:
            return True

    def add_distribution_choices(self, options: dict = None) -> None:
        if options is None:
            options = distribution_options

        model = self.parent.fit.model
        items = list()

        for distribution_type in options.keys():
            d = options[distribution_type]
            try:
                attr = model.__getattribute__(d['attribute'])
                items.append(distribution_type)
            except AttributeError:
                pass
        self.selector.addItems(items)

    def update_parameter(self):
        self.parameter_editor._dict = self.parent.distribution_options[self.distribution_type]
        self.parameter_editor.update()
        self.parent.update()

    def __init__(
            self,
            *args,
            parent: QtWidgets.QWidget = None,
            distribution_options: dict = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.selector = QtWidgets.QComboBox(None)
        self.selector.blockSignals(True)
        self.layout.addWidget(self.selector)
        self.add_distribution_choices(distribution_options)
        self.selector.currentIndexChanged[int].connect(self.update_parameter)
        self.selector.blockSignals(False)
        d = copy.deepcopy(parent.distribution_options[self.distribution_type])
        self.parameter_editor = ParameterEditor(json_file='', target=d, callback=parent.update)
        self.layout.addWidget(self.parameter_editor)

        # Optional checkbox to show/hide individual component curves in
        # distribution plots (e.g., Gaussian components in PDA
        # Gaussian-distance models). For other models this has no effect but
        # is harmless.
        self.gauss_checkbox = QtWidgets.QCheckBox("Show components")
        self.gauss_checkbox.setChecked(True)
        self.gauss_checkbox.stateChanged.connect(parent.update)
        self.layout.addWidget(self.gauss_checkbox)


class DistributionPlot(plotbase.Plot):

    name = "Distribution"

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            parent: QtWidgets.QWidget = None,
            distribution_options: dict = None,
            **kwargs
    ):
        super().__init__(fit=fit, parent=parent)
        self.data_x, self.data_y = None, None
        self.distribution_options = distribution_options

        # Optional axis scaling and an optional residual panel (for PDA 1D
        # histograms), where weighted residuals are shown on a separate top
        # plot, similar to the TCSPC LinePlot layout.
        self._scale_x = kwargs.pop('scale_x', 'lin')
        self._scale_y = kwargs.pop('scale_y', 'lin')
        self._with_residual_panel = kwargs.pop('with_residual_panel', False)
        self.residual_plot = None

        self.plot_controller = DistributionPlotControl(
            self,
            parent=self,
            distribution_options=distribution_options
        )

        if self._with_residual_panel:
            pw_res = pg.PlotWidget()
            pw_main = pg.PlotWidget()
            try:
                pw_res.setXLink(pw_main)
            except Exception:
                pass
            # Give the residuals panel ~1/3 of the total height and the main
            # histogram ~2/3 using stretch factors 1 and 2.
            self.layout.addWidget(pw_res, 1)
            self.layout.addWidget(pw_main, 2)
            self.residual_plot = pw_res.getPlotItem()
            self.distribution_plot = pw_main.getPlotItem()
            try:
                self.residual_plot.hideAxis('bottom')
                self.residual_plot.setLabel('left', "w.res.")
            except Exception:
                pass
        else:
            pw = pg.PlotWidget()
            self.layout.addWidget(pw)
            self.distribution_plot = pw.getPlotItem()

        # Match LinePlot's grid settings where applicable: show a grid on the
        # main distribution plot and, when present, on the residuals.
        try:
            if plot_settings.get('enable_grid', False):
                if plot_settings.get('show_data_grid', False):
                    self.distribution_plot.showGrid(True, True, 0.5)
                if self.residual_plot is not None and plot_settings.get('show_residual_grid', False):
                    self.residual_plot.showGrid(True, True, 1.0)
        except Exception:
            pass

        # Apply requested axis scaling (x) to main plot and residuals.
        try:
            log_x = str(self._scale_x).lower() == 'log'
            log_y = str(self._scale_y).lower() == 'log'
            self.distribution_plot.setLogMode(x=log_x, y=log_y)
            if self.residual_plot is not None:
                self.residual_plot.setLogMode(x=log_x, y=False)
        except Exception:
            pass
        pen = pg.mkPen(colors['data'], width=lw)
        self.distribution_curve = self.distribution_plot.plot(
            x=[0.0],
            y=[0.0],
            pen=pen,
            fillLevel=0,
            fillBrush=colors['data']
        )

        # Draggable fit-quality box (χ²_red and DW), similar to LinePlot.
        self._quality_text = None
        try:
            self._quality_text = DraggableTextItem(
                text='',
                border='w',
                fill=(0, 0, 255, 100),
                anchor=(0, 0)
            )
            self._quality_text.setParentItem(self.distribution_plot)
            # Initial position; user can drag to taste.
            self._quality_text.setPos(0, 0)
        except Exception:
            self._quality_text = None

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        # Clear curve and recreate plots
        self.distribution_plot.clear()
        if self.residual_plot is not None:
            self.residual_plot.clear()

        # Re-attach the draggable statistics box after clear so it remains
        # visible, mirroring the behavior of the TCSPC LinePlot.
        try:
            if self._quality_text is not None:
                self._quality_text.setParentItem(self.distribution_plot)
        except Exception:
            pass

        # Get distribution
        ds = self.plot_controller.parameter_editor.dict

        # Update x-axis label to reflect the currently used histogram axis /
        # function. Prefer an explicit axis label or kw_hist['_axis_type']
        # when configured; otherwise fall back to the distribution selector
        # text (e.g. 'S1/(S0+S1)', 'S0/S1', 'Distance').
        try:
            axis_label = ds.get('axis_label')
        except Exception:
            axis_label = None
        if not axis_label:
            try:
                kw_hist = ds.get('accessor_kwargs', {}).get('kw_hist', {})
                axis_label = kw_hist.get('_axis_type')
            except Exception:
                axis_label = None
        if not axis_label:
            try:
                axis_label = str(self.plot_controller.distribution_type)
            except Exception:
                axis_label = ''
        try:
            if axis_label:
                self.distribution_plot.setLabel('bottom', axis_label)
        except Exception:
            pass

        # Update axis scaling based on the currently selected distribution
        try:
            scale_x = ds.get('scale_x', self._scale_x)
            scale_y = ds.get('scale_y', self._scale_y)
            log_x = str(scale_x).lower() == 'log'
            log_y = str(scale_y).lower() == 'log'
            self.distribution_plot.setLogMode(x=log_x, y=log_y)
            if self.residual_plot is not None:
                self.residual_plot.setLogMode(x=log_x, y=False)
        except Exception:
            pass
        r = ds['accessor'](
            self.fit.model.__getattribute__(ds['attribute']),
            **ds['accessor_kwargs']
        )

        # Helper to drop curves with no finite support. This prevents
        # feeding all-NaN or empty arrays into pyqtgraph's ScatterPlotItem,
        # which otherwise emits RuntimeWarnings.
        def _sanitize_curve(y, x):
            try:
                x_arr = np.asarray(x, dtype=float).ravel()
                y_arr = np.asarray(y, dtype=float).ravel()
            except Exception:
                return None
            if x_arr.size == 0 or y_arr.size == 0:
                return None
            if not np.any(np.isfinite(x_arr)) or not np.any(np.isfinite(y_arr)):
                return None
            return y_arr, x_arr

        # Optionally derive weighted residuals and basic fit statistics from the
        # first two curves (data, model) using counting shot noise
        # sigma = sqrt(max(data, 1)). For PDA 1D histograms this matches the
        # definition used in chisurf.models.pda.widgets.get_distribution and
        # allows us to define DW directly from the currently shown histogram
        # rather than only from the global Fit object. Bins with zero
        # experimental counts do not contribute to DW or the effective
        # fit-range; we only use bins with at least one photon.
        wres_curve = None
        chi2r = None
        dw = None
        hist_i_min = None
        hist_i_max = None
        try:
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                data_y, data_x = r[0]
                model_y, model_x = r[1]
                dy = np.asarray(data_y, dtype=float)
                my = np.asarray(model_y, dtype=float)
                if dy.shape == my.shape and dy.size > 0:
                    # Consider only bins with at least one photon in the data
                    mask = dy > 0.0
                    nz_idx = np.nonzero(mask)[0]
                    if nz_idx.size > 0:
                        hist_i_min = int(nz_idx[0])
                        hist_i_max = int(nz_idx[-1])
                        dy_nz = dy[mask]
                        my_nz = my[mask]
                        sigma_nz = np.sqrt(np.maximum(dy_nz, 1.0))
                        resid_nz = (dy_nz - my_nz) / sigma_nz
                        # Durbin–Watson statistic for these residuals
                        if resid_nz.size > 1:
                            num = float(np.sum(np.diff(resid_nz) ** 2))
                            den = float(np.sum(resid_nz ** 2))
                            if den > 0.0:
                                dw = num / den
                        # Build a residual curve that is zero outside the
                        # non-empty bins so the w.res. panel visually matches
                        # the effective fit-range.
                        resid_full = np.zeros_like(dy, dtype=float)
                        resid_full[mask] = resid_nz
                        has_explicit_residual = isinstance(r, (list, tuple)) and len(r) >= 3
                        if self.residual_plot is not None and not has_explicit_residual:
                            wres_curve = (resid_full, data_x)
        except Exception:
            wres_curve = None
            chi2r = None
            dw = None
            hist_i_min = None
            hist_i_max = None

        p = dict(ds.get('curve_options', {}))

        # Normalize optional fill/line colors for single-curve distributions.
        # If a fillBrush is provided, use the same RGB values for the line and
        # make the fill about 50% transparent so that the histogram area is
        # softly shaded but the outline remains fully opaque.
        try:
            if not p.get('multi_curve', False) and 'fillBrush' in p:
                base = p['fillBrush']
                col = pg.mkColor(base)
                r_c, g_c, b_c, _ = col.getRgb()
                alpha_fill = int(0.5 * 255)
                p['fillBrush'] = (r_c, g_c, b_c, alpha_fill)
                if 'pen' not in p:
                    p['pen'] = (r_c, g_c, b_c, 255)
        except Exception:
            pass

        bar_mode = p.pop('bar_mode', None)
        multi_curve = p.get('multi_curve', False)

        # If we have a residual panel and a synthesized wres curve, append it so
        # that it is plotted in the residual axis (index >= 2).
        if multi_curve and self.residual_plot is not None and wres_curve is not None:
            try:
                r = list(r)
                r.append(wres_curve)
            except Exception:
                pass

        if multi_curve:
            n_curves = len(r)
            pens = p.pop('pen', ['r', 'b', 'g', 'y', 'c', 'm', 'k'])
            symbols = p.pop('symbol', ['o', 'x', 'v', '^', '<'])

            # Normalize pens/symbols so they are lists of at least n_curves
            # elements. This avoids IndexError when more curves are returned
            # than there are explicit colors/symbols configured.
            if isinstance(pens, str):
                pens = [pens]
            if isinstance(symbols, str):
                symbols = [symbols]

            if len(pens) < n_curves:
                base = pens if pens else ['w']
                pens = [base[i % len(base)] for i in range(n_curves)]
            if len(symbols) < n_curves:
                base = symbols if symbols else ['o']
                symbols = [base[i % len(base)] for i in range(n_curves)]

            # Remove any global stepMode/connect from p; we choose them per
            # curve index below so that data/model remain stepped histograms
            # while residuals and Gaussian components are smooth lines or
            # discrete "sticks" when requested via bar_mode.
            p.pop('stepMode', None)
            p.pop('connect', None)

            # Optional filled-under-curve styling for multi-curve plots. When
            # a fillBrush/fillLevel is provided in curve_options, apply it to
            # the first curve only (typically the experimental PDA histogram).
            # The fill uses the same RGB values as the data line but with
            # ~50% transparency so that the line remains clearly visible.
            fill_brush = None
            fill_level = None
            try:
                if 'fillBrush' in p:
                    fill_brush = p.pop('fillBrush')
                    fill_level = p.pop('fillLevel', 0.0)
                    if pens:
                        base_col = pg.mkColor(pens[0])
                        r_c, g_c, b_c, _ = base_col.getRgb()
                        alpha_fill = int(0.5 * 255)
                        fill_brush = (r_c, g_c, b_c, alpha_fill)
                        pens[0] = (r_c, g_c, b_c, 255)
            except Exception:
                fill_brush = None
                fill_level = None

            for i in range(n_curves):
                y_raw, x_raw = r[i]
                cur = _sanitize_curve(y_raw, x_raw)
                if cur is None:
                    continue
                y, x = cur
                c, s = pens[i], symbols[i]

                # Allow the plot controller to hide individual component
                # curves (indices >= 3 in PDA Gaussian-distance plots) while
                # keeping data, model and residuals visible.
                try:
                    if i >= 3 and not self.plot_controller.show_gaussians:
                        continue
                except Exception:
                    pass

                # For discrete PDA histograms (PDA-discrete), we want data,
                # model, and residuals as vertical lines (sticks) when the
                # accessor requested bar_mode == 'sticks'.
                use_sticks = bar_mode == 'sticks' and i in (0, 1, 2)
                if use_sticks:
                    x_arr = np.asarray(x, dtype=float)
                    y_arr = np.asarray(y, dtype=float)
                    if x_arr.size == 0 or not np.any(np.isfinite(x_arr)) or not np.any(np.isfinite(y_arr)):
                        continue
                    xs = np.empty(3 * x_arr.size, dtype=float)
                    ys = np.empty_like(xs)
                    xs[0::3] = x_arr
                    xs[1::3] = x_arr
                    xs[2::3] = np.nan
                    ys[0::3] = 0.0
                    ys[1::3] = y_arr
                    ys[2::3] = np.nan
                    x_plot, y_plot = xs, ys
                else:
                    x_plot, y_plot = x, y

                # By convention for PDA multi-curve plots:
                #   0: data      (stepped or sticks, main plot)
                #   1: model     (stepped or sticks, main plot)
                #   2: w.res.    (stepped or sticks, residual plot if present)
                #   3+: extra curves (e.g. Gaussian components) -> smooth, main
                if self.residual_plot is not None and i == 2:
                    target_plot = self.residual_plot
                    if use_sticks:
                        curve_step = False
                    else:
                        # Weighted residuals: stepped style for visual
                        # consistency with data/model.
                        curve_step = 'right'
                    curve_connect = 'all'
                    # Use the dedicated residuals color for the w.res. curve.
                    try:
                        c = colors.get('residuals', c)
                    except Exception:
                        pass
                else:
                    target_plot = self.distribution_plot
                    if i in (0, 1):
                        # Data and model: histogram-style stepped plot or
                        # sticks, depending on bar_mode.
                        curve_step = False if use_sticks else 'right'
                    else:
                        # Gaussian components and any other extra curves:
                        # smooth lines.
                        curve_step = False
                    curve_connect = 'all'

                if fill_brush is not None and i == 0:
                    target_plot.plot(
                        x_plot,
                        y_plot,
                        stepMode=curve_step,
                        connect=curve_connect,
                        **p,
                        pen=pg.mkPen(c, width=lw),
                        symbol=s,
                        fillLevel=fill_level,
                        fillBrush=fill_brush,
                    )
                else:
                    target_plot.plot(
                        x_plot,
                        y_plot,
                        stepMode=curve_step,
                        connect=curve_connect,
                        **p,
                        pen=pg.mkPen(c, width=lw),
                        symbol=s,
                    )
        else:
            y_raw, x_raw = r
            cur = _sanitize_curve(y_raw, x_raw)
            if cur is None:
                return
            y, x = cur
            c = p.pop('pen', 'b')
            s = p.pop('symbol', 'o')
            if bar_mode == 'sticks':
                # Single-curve discrete distributions (e.g., lifetime
                # distributions) are rendered as vertical lines from 0 to y at
                # each x when bar_mode == 'sticks'.
                x_arr = np.asarray(x, dtype=float)
                y_arr = np.asarray(y, dtype=float)
                if x_arr.size > 0:
                    xs = np.empty(3 * x_arr.size, dtype=float)
                    ys = np.empty_like(xs)
                    xs[0::3] = x_arr
                    xs[1::3] = x_arr
                    xs[2::3] = np.nan
                    ys[0::3] = 0.0
                    ys[1::3] = y_arr
                    ys[2::3] = np.nan
                    p.pop('stepMode', None)
                    p.pop('connect', None)
                    self.distribution_plot.plot(
                        xs,
                        ys,
                        stepMode=False,
                        connect='all',
                        **p,
                        pen=pg.mkPen(c, width=lw),
                        symbol=s,
                    )
            else:
                self.distribution_plot.plot(x, y, **p, pen=pg.mkPen(c, width=lw), symbol=s)

        # Show basic fit quality metrics in a draggable box. Prefer the
        # histogram-based chi²/DW computed above and fall back to the Fit
        # object's statistics only if necessary, while keeping the HTML
        # formatting identical to the TCSPC LinePlot.
        try:
            if self._quality_text is not None:
                fit = self.fit
                try:
                    self._quality_text.updateTextPos()
                except Exception:
                    pass
                chi2_display = getattr(fit, 'chi2r', None)
                dw_display = dw if dw is not None else getattr(fit, 'durbin_watson', None)
                # Prefer the effective histogram fit-range (first/last non-empty
                # bin) and fall back to the Fit object's xmin/xmax otherwise.
                xmin = hist_i_min if hist_i_min is not None else getattr(fit, 'xmin', None)
                xmax = hist_i_max if hist_i_max is not None else getattr(fit, 'xmax', None)
                if chi2_display is not None and dw_display is not None:
                    if xmin is None or xmax is None:
                        fit_range_line = ''
                    else:
                        fit_range_line = f'         Fit-range {xmin}, {xmax} <br />'
                    self._quality_text.setHtml(
                        f'<div style="name-align: center">'
                        f'     <span style="color: #FF0; font-size: 10pt;">'
                        f'{fit_range_line}'
                        f'         &Chi;<sub>r</sub><sup>2</sup>={chi2_display:.4f} <br />'
                        f'         DW={dw_display: .4f}'
                        f'     </span>'
                        f'</div>'
                    )
        except Exception:
            pass

        # Keep the main plot title free of statistics, like the LinePlot.
        try:
            self.distribution_plot.setTitle("")
        except Exception:
            pass
