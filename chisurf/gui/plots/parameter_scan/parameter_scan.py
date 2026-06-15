from __future__ import annotations
from chisurf import typing

import time
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
from pyqtgraph.dockarea import DockArea, Dock

import chisurf as cs
import chisurf.gui.decorators
import chisurf.core.settings
import chisurf.core.fitting
import chisurf.core.parameter
import chisurf.core.decorators
import chisurf.core.models
from chisurf.gui.plots import plotbase
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client

plot_settings = cs.core.settings.gui['plot']
colors = plot_settings['colors']
color_scheme = cs.core.settings.colors
lw = plot_settings['line_width']

OVERLAY_PEN = pg.mkPen((255, 128, 0), width=1.5, style=QtCore.Qt.DashLine)
CROSSING_PEN = pg.mkPen((0, 180, 0), width=1.5, style=QtCore.Qt.DashDotDotLine)


class ParameterScanWidget(
    QtWidgets.QWidget
):

    @cs.gui.decorators.init_with_ui(
        ui_filename="parameter_scan.ui"
    )
    def __init__(
            self,
            model: cs.core.models.Model = None,
            parent: QtWidgets.QWidget = None,
            *args,
            **kwargs
    ):

        self.model = model
        self.parent = parent

        self.actionScanParameter.triggered.connect(self.scan_parameter)
        self.actionSmartScanParameter.triggered.connect(self.smart_scan_parameter)
        self.actionParameterChanged.triggered.connect(self.onParameterChanged)
        self.actionUpdateParameterList.triggered.connect(self.update)

        self.update()
        self._update_range_spinboxes()

    def _update_range_spinboxes(self):
        """Set the relative range spinboxes based on the selected parameter."""
        p = self.parameter
        if p is None:
            return
        v = p.value
        if abs(v) < 1e-15:
            v = 1.0
        err = p.error_estimate
        if isinstance(err, float) and not np.isnan(err) and err > 0:
            rel_range = min(max(3.0 * err / abs(v), 0.05), 2.0)
        else:
            rel_range = 0.1
        self.doubleSpinBox.setValue(rel_range)
        self.doubleSpinBox_2.setValue(rel_range)

    def onParameterChanged(self):
        self._update_range_spinboxes()
        self.parent.update()

    def update(self) -> None:
        super().update()
        self.comboBox.blockSignals(True)

        pn = self.model.parameter_names
        self.comboBox.clear()
        self.comboBox.addItems(pn)

        self.comboBox.blockSignals(False)
        self.model.update_plots()

    def _poll_scan_result(self, job_id: str, param) -> None:
        """Poll server for scan results and store locally.

        Called in a loop from scan_parameter / smart_scan_parameter
        when using the RPC path.
        """
        fc = get_fitting_client()
        if fc is None:
            return
        deadline = time.monotonic() + 300.0  # 5 min timeout
        while time.monotonic() < deadline:
            try:
                result = fc.parameter_scan_result(job_id)
                status = result.get("status", "")
                if status == "completed":
                    values = result.get("values", [])
                    chi2 = result.get("chi2", [])
                    if values and chi2:
                        param.parameter_scan = (values, chi2)
                    break
                if status in ("failed", "cancelled"):
                    break
            except Exception:
                break
            # Process Qt events while polling so the UI stays responsive
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.05)
        self.parent.update()

    def scan_parameter(self) -> None:
        p_min = float(self.doubleSpinBox.value())
        p_max = float(self.doubleSpinBox_2.value())
        _, name = self.selected_parameter
        v = self.model.parameter_dict[name].value
        v_min = (1. - p_min) * v
        v_max = (1. + p_max) * v
        n_steps = int(self.spinBox.value())

        fc = get_fitting_client()
        if fc is not None:
            try:
                fit_uid = str(getattr(self.parent.fit, "unique_identifier", "") or "")
                result = fc.start_parameter_scan(
                    parameter_name=self.parameter.name,
                    n_steps=n_steps,
                    range_factor=2.0,
                    fit_uid=fit_uid,
                )
                job_id = result.get("job_id")
                if job_id:
                    self._poll_scan_result(job_id, self.parameter)
                    return
            except Exception:
                pass
        self.parent.update()

    def smart_scan_parameter(self) -> None:
        _, name = self.selected_parameter
        p_value = float(self.doubleSpinBox_3.value())
        max_points = int(self.spinBox.value())

        # Build optional scan_range from the relative spinboxes + checkbox
        use_range = self.checkBox.isChecked()
        if use_range:
            p_min = float(self.doubleSpinBox.value())
            p_max = float(self.doubleSpinBox_2.value())
            v = self.model.parameter_dict[name].value
            v_min = (1. - p_min) * v
            v_max = (1. + p_max) * v
            scan_range = (v_min, v_max)
        else:
            scan_range = (None, None)

        fc = get_fitting_client()
        if fc is not None:
            try:
                fit_uid = str(getattr(self.parent.fit, "unique_identifier", "") or "")
                result = fc.start_parameter_scan(
                    parameter_name=self.parameter.name,
                    n_steps=max_points,
                    range_factor=2.0,
                    fit_uid=fit_uid,
                )
                job_id = result.get("job_id")
                if job_id:
                    self._poll_scan_result(job_id, self.parameter)
                    return
            except Exception:
                pass
        self.parent.update()

    @property
    def selected_parameter(self) -> typing.Tuple[int, str]:
        idx = self.comboBox.currentIndex()
        name = self.comboBox.currentText()
        return idx, str(name)

    @property
    def parameter(self) -> cs.core.parameter.Parameter:
        idx, name = self.selected_parameter
        try:
            return self.model.parameters_all_dict[name]
        except AttributeError:
            return None


class ParameterScanPlot(
    plotbase.Plot
):

    name = "Parameter scan"

    def __init__(
            self,
            fit: cs.core.fitting.fit.FitGroup,
            *args,
            **kwargs
    ):
        super(ParameterScanPlot, self).__init__(fit)

        self.data_x, self.data_y = None, None

        self.plot_controller = ParameterScanWidget(
           model=fit.model,
           parent=self
        )

        area = DockArea()
        self.layout.addWidget(area)
        hide_title = plot_settings['hideTitle']
        d2 = Dock("Chi2-Surface", hideTitle=hide_title)

        self.p1 = QtWidgets.QPlainTextEdit()
        p2 = pg.PlotWidget()

        d2.addWidget(p2)

        area.addDock(d2, 'top')

        distribution_plot = p2.getPlotItem()

        self.distribution_plot = distribution_plot
        self.distribution_curve = distribution_plot.plot(
            x=[0.0],
            y=[0.0],
            pen=pg.mkPen(colors['data'], width=lw),
            name='Data'
        )

        self._overlay_items = []

    def _clear_overlays(self):
        for item in self._overlay_items:
            self.distribution_plot.removeItem(item)
        self._overlay_items = []

    def _add_overlay(self, item):
        self.distribution_plot.addItem(item)
        self._overlay_items.append(item)

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        try:
            p = self.plot_controller.parameter
            if p is None:
                return

            x, y = p.parameter_scan
            if x is None or y is None:
                return

            x = np.asarray(x)
            y = np.asarray(y)
            if x.size == 0 or y.size == 0:
                return

            # Avoid feeding all-NaN arrays into pyqtgraph, which leads to
            # RuntimeWarnings about NaN slices.
            if not np.any(np.isfinite(x)) or not np.any(np.isfinite(y)):
                return

            self.distribution_curve.setData(x=x, y=y)

            # Draw overlays from smart-scan result
            self._clear_overlays()
            result = getattr(p, 'scan_result', None)
            if result is not None:
                threshold = result.get('threshold')
                if threshold is not None:
                    thr_line = pg.InfiniteLine(
                        pos=threshold, angle=0,
                        pen=OVERLAY_PEN,
                        label='p={:.2f}'.format(result.get('p_value', 0.99)),
                    )
                    self._add_overlay(thr_line)

                crossings = result.get('crossings', (None, None))
                for cr in crossings:
                    if cr is not None:
                        vline = pg.InfiniteLine(
                            pos=cr, angle=90,
                            pen=CROSSING_PEN,
                            label='{:.4g}'.format(cr),
                        )
                        self._add_overlay(vline)
        except Exception as e:
            cs.logging.warning(f"ParameterScanPlot: update failed: {e}")
