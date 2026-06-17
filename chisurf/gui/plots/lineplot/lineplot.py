from __future__ import annotations
import chisurf as cs

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
import chisurf.core.plot_transforms as plot_transforms
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

colors = cs.core.settings.gui['plot']['colors']


class LinePlotControl(QtWidgets.QWidget):

    director = {
        'data': {
            'lw': 1.0,
            'color': colors['data'],
            'target': 'main_plot',
            'allow_reference_transform': True,
            'allow_shift': True,
            'allow_density': True,
            'plot_only_region': False,
            'auto_downsample': True,
        },
        'IRF': {
            'lw': 2.0,
            'color': colors['irf'],
            'target': 'main_plot',
            'allow_reference_transform': False,
            'allow_shift': True,
            'allow_density': True,
            'plot_only_region': False
        },
        'model': {
            'lw': 2.0,
            'target': 'main_plot',
            'color': colors['model'],
            'allow_reference_transform': True,
            'allow_shift': True,
            'allow_density': True,
            'plot_only_region': True
        },
        'weighted residuals': {
            'lw': 2.0,
            'target': 'top_left_plot',
            'label': 'w.res.',
            'color': colors['residuals'],
            'allow_reference_transform': False,
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
            'allow_reference_transform': False,
            'allow_shift': True,
            'allow_density': False,
            'plot_only_region': False,
            'auto_downsample': True,
        },
        'default': {
            'lw': 2.0,
            'color': colors['data'],
            'target': 'main_plot',
            'allow_reference_transform': False,
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

    @cs.gui.decorators.init_with_ui("linePlotWidget.ui")
    def __init__(
            self,
            parent=None,
            scale_x: str = 'lin',
            d_scaley: str = 'log',
            r_scaley: str = 'lin',
            xmin: float = 0.0,
            ymin: float = 1.0
    ):
        self.parent = parent
        self._reference_modes: typing.OrderedDict[str, plot_transforms.PlotReferenceMode] = OrderedDict()
        self._reference_parameter_widgets: typing.Dict[str, QtWidgets.QWidget] = {}
        self._reference_parameter_specs: typing.Dict[str, plot_transforms.PlotReferenceParameter] = {}
        self._pending_reference_mode: str | None = None
        self._pending_reference_parameters: typing.Dict[str, typing.Any] = {}
        self._install_reference_controls()

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
        self.comboBox_reference.currentIndexChanged.connect(self.SetReference)
        self.toolButton_reference_reset.clicked.connect(self.reset_reference_parameters)
        self.checkBox_9.stateChanged.connect(self.SetDisplayGroup)

    def _install_reference_controls(self) -> None:
        """Install the reference-mode selector and dynamic parameter area."""
        try:
            placeholder = getattr(self, "referencePlaceholder", None)
            if placeholder is not None:
                self.gridLayout_2.removeWidget(placeholder)
                placeholder.hide()
        except Exception:
            pass

        ref_row = QtWidgets.QWidget(self)
        ref_layout = QtWidgets.QHBoxLayout(ref_row)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        ref_layout.setSpacing(2)
        ref_layout.addWidget(QtWidgets.QLabel("Reference", ref_row))

        self.comboBox_reference = QtWidgets.QComboBox(ref_row)
        self.comboBox_reference.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        ref_layout.addWidget(self.comboBox_reference, 1)

        self.toolButton_reference_reset = QtWidgets.QToolButton(ref_row)
        self.toolButton_reference_reset.setText("Reset")
        self.toolButton_reference_reset.setToolTip("Reset reference-mode parameters")
        ref_layout.addWidget(self.toolButton_reference_reset)

        self.gridLayout_2.addWidget(ref_row, 2, 1)

        self.reference_parameter_widget = QtWidgets.QWidget(self)
        self.reference_parameter_layout = QtWidgets.QGridLayout(self.reference_parameter_widget)
        self.reference_parameter_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_parameter_layout.setHorizontalSpacing(4)
        self.reference_parameter_layout.setVerticalSpacing(1)
        self.reference_parameter_widget.hide()
        self.verticalLayout.insertWidget(1, self.reference_parameter_widget)

        self.set_reference_modes([])

    @property
    def reference_mode(self) -> str:
        """Current reference mode key.

        Returns
        -------
        str
            Selected mode key or ``"raw"``.
        """
        data = self.comboBox_reference.currentData()
        return str(data) if data else "raw"

    @reference_mode.setter
    def reference_mode(self, key: str) -> None:
        """Select a reference mode by key.

        Parameters
        ----------
        key : str
            Mode key.
        """
        key = str(key or "raw")
        idx = self.comboBox_reference.findData(key)
        if idx < 0:
            self._pending_reference_mode = key
            idx = self.comboBox_reference.findData("raw")
        if idx >= 0:
            self.comboBox_reference.setCurrentIndex(idx)
            self._rebuild_reference_parameter_controls(self._pending_reference_parameters)

    @property
    def reference_parameters(self) -> typing.Dict[str, typing.Any]:
        """Return current reference-mode parameter values.

        Returns
        -------
        dict
            Parameter values keyed by parameter id.
        """
        values = {}
        for key, widget in self._reference_parameter_widgets.items():
            spec = self._reference_parameter_specs.get(key)
            if spec is None:
                continue
            values[key] = self._reference_widget_value(widget, spec)
        return values

    @reference_parameters.setter
    def reference_parameters(self, values: typing.Mapping[str, typing.Any]) -> None:
        """Set reference parameter widgets from a mapping.

        Parameters
        ----------
        values : mapping
            Parameter values keyed by parameter id.
        """
        if not isinstance(values, dict):
            return
        if not self._reference_parameter_widgets:
            self._pending_reference_parameters = dict(values)
            return
        for key, value in values.items():
            widget = self._reference_parameter_widgets.get(key)
            spec = self._reference_parameter_specs.get(key)
            if widget is not None and spec is not None:
                self._set_reference_widget_value(widget, spec, value)

    def selected_reference_mode(self) -> plot_transforms.PlotReferenceMode | None:
        """Return the selected reference mode object.

        Returns
        -------
        PlotReferenceMode or None
            Selected mode, or None for raw plotting.
        """
        key = self.reference_mode
        return self._reference_modes.get(key)

    def set_reference_modes(
            self,
            modes: typing.Iterable[plot_transforms.PlotReferenceMode]
    ) -> None:
        """Populate the reference-mode selector.

        Parameters
        ----------
        modes : iterable
            Available reference modes.
        """
        old_mode = self._pending_reference_mode or self.reference_mode
        old_parameters = dict(self._pending_reference_parameters)
        old_parameters.update(self.reference_parameters)

        valid_modes = OrderedDict()
        for mode in modes or []:
            if isinstance(mode, plot_transforms.PlotReferenceMode):
                valid_modes[str(mode.key)] = mode

        signature = tuple(
            (key, mode.label, tuple((p.key, p.label, p.kind, p.default) for p in mode.parameters))
            for key, mode in valid_modes.items()
        )
        if getattr(self, "_reference_mode_signature", None) == signature:
            self._rebuild_reference_parameter_controls(old_parameters)
            return

        self._reference_mode_signature = signature
        self._reference_modes = valid_modes

        self.comboBox_reference.blockSignals(True)
        try:
            self.comboBox_reference.clear()
            self.comboBox_reference.addItem("Raw", "raw")
            for key, mode in valid_modes.items():
                self.comboBox_reference.addItem(str(mode.label), key)
            idx = self.comboBox_reference.findData(old_mode)
            if idx < 0:
                idx = self.comboBox_reference.findData("raw")
            self.comboBox_reference.setCurrentIndex(max(0, idx))
        finally:
            self.comboBox_reference.blockSignals(False)

        self._rebuild_reference_parameter_controls(old_parameters)

    def _clear_reference_parameter_controls(self) -> None:
        """Remove all dynamic reference parameter controls."""
        while self.reference_parameter_layout.count():
            item = self.reference_parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._reference_parameter_widgets = {}
        self._reference_parameter_specs = {}

    def _rebuild_reference_parameter_controls(
            self,
            values: typing.Mapping[str, typing.Any] | None = None
    ) -> None:
        """Recreate controls for the selected mode's parameter specs.

        Parameters
        ----------
        values : mapping, optional
            Values to preserve where possible.
        """
        values = dict(values or {})
        self._clear_reference_parameter_controls()
        mode = self.selected_reference_mode()
        if mode is None or not mode.parameters:
            self.reference_parameter_widget.hide()
            return

        for row, spec in enumerate(mode.parameters):
            label = QtWidgets.QLabel(str(spec.label), self.reference_parameter_widget)
            widget = self._make_reference_parameter_widget(spec)
            self.reference_parameter_layout.addWidget(label, row, 0)
            self.reference_parameter_layout.addWidget(widget, row, 1)
            self._reference_parameter_widgets[spec.key] = widget
            self._reference_parameter_specs[spec.key] = spec
            self._set_reference_widget_value(widget, spec, values.get(spec.key, spec.default))

        self.reference_parameter_widget.show()
        self._pending_reference_mode = None
        self._pending_reference_parameters = {}

    def _make_reference_parameter_widget(
            self,
            spec: plot_transforms.PlotReferenceParameter
    ) -> QtWidgets.QWidget:
        """Create a Qt widget for a reference parameter.

        Parameters
        ----------
        spec : PlotReferenceParameter
            Parameter declaration.

        Returns
        -------
        QWidget
            New editor widget.
        """
        kind = str(spec.kind).lower()
        if kind == "bool":
            widget = QtWidgets.QCheckBox(self.reference_parameter_widget)
            widget.stateChanged.connect(self.SetReference)
            return widget
        if kind == "int":
            widget = QtWidgets.QSpinBox(self.reference_parameter_widget)
            widget.setRange(
                int(spec.minimum if spec.minimum is not None else -999999999),
                int(spec.maximum if spec.maximum is not None else 999999999),
            )
            widget.setSingleStep(int(spec.step if spec.step is not None else 1))
            widget.valueChanged.connect(self.SetReference)
            return widget
        if kind == "choice":
            widget = QtWidgets.QComboBox(self.reference_parameter_widget)
            for choice in spec.choices:
                if isinstance(choice, (tuple, list)) and len(choice) >= 2:
                    widget.addItem(str(choice[1]), choice[0])
                else:
                    widget.addItem(str(choice), choice)
            widget.currentIndexChanged.connect(self.SetReference)
            return widget

        widget = QtWidgets.QDoubleSpinBox(self.reference_parameter_widget)
        widget.setRange(
            float(spec.minimum if spec.minimum is not None else -999999999.0),
            float(spec.maximum if spec.maximum is not None else 999999999.0),
        )
        widget.setDecimals(6)
        widget.setSingleStep(float(spec.step if spec.step is not None else 0.1))
        widget.valueChanged.connect(self.SetReference)
        return widget

    def _reference_widget_value(
            self,
            widget: QtWidgets.QWidget,
            spec: plot_transforms.PlotReferenceParameter
    ) -> typing.Any:
        """Return a reference parameter value from a widget.

        Parameters
        ----------
        widget : QWidget
            Editor widget.
        spec : PlotReferenceParameter
            Parameter declaration.

        Returns
        -------
        object
            Current widget value.
        """
        kind = str(spec.kind).lower()
        if kind == "bool":
            return bool(widget.isChecked())
        if kind == "choice":
            return widget.currentData()
        if kind == "int":
            return int(widget.value())
        return float(widget.value())

    def _set_reference_widget_value(
            self,
            widget: QtWidgets.QWidget,
            spec: plot_transforms.PlotReferenceParameter,
            value: typing.Any
    ) -> None:
        """Set a reference parameter widget value.

        Parameters
        ----------
        widget : QWidget
            Editor widget.
        spec : PlotReferenceParameter
            Parameter declaration.
        value : object
            New value.
        """
        widget.blockSignals(True)
        try:
            kind = str(spec.kind).lower()
            if kind == "bool":
                widget.setChecked(bool(value))
            elif kind == "choice":
                idx = widget.findData(value)
                if idx < 0:
                    idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif kind == "int":
                widget.setValue(int(value))
            else:
                widget.setValue(float(value))
        except Exception:
            pass
        finally:
            widget.blockSignals(False)

    def reset_reference_parameters(self) -> None:
        """Reset selected reference-mode parameters to their defaults."""
        mode = self.selected_reference_mode()
        if mode is None:
            return
        for spec in mode.parameters:
            widget = self._reference_parameter_widgets.get(spec.key)
            if widget is not None:
                self._set_reference_widget_value(widget, spec, spec.default)
        self.SetReference()

    @property
    def plot_ftt(self) -> bool:
        widget = getattr(self, "checkBox_plot_ftt", None)
        return bool(widget.isChecked()) if widget is not None else False

    @plot_ftt.setter
    def plot_ftt(self, v: bool) -> None:
        widget = getattr(self, "checkBox_plot_ftt", None)
        if widget is None:
            return
        if v:
            widget.setCheckState(2)
        else:
            widget.setCheckState(0)

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

    def get_state(self) -> dict:
        """Return project-serializable line plot controller state.

        Returns
        -------
        dict
            State of the visible plot controls and curve checkboxes.
        """
        curve_visibility = {}
        for i in range(self.treeWidget.topLevelItemCount()):
            item = self.treeWidget.topLevelItem(i)
            curve_visibility[item.text(2)] = bool(item.checkState(1))
        return {
            "data_logy": self.data_logy,
            "scale_x": self.scale_x,
            "res_logy": self.res_logy,
            "reference_mode": self.reference_mode,
            "reference_parameters": self.reference_parameters,
            "is_density": self.is_density,
            "display_group": self.display_group,
            "plot_ftt": self.plot_ftt,
            "xmin_enabled": bool(self.checkBox_4.isChecked()),
            "xmax_enabled": bool(self.checkBox_6.isChecked()),
            "ymin_enabled": bool(self.checkBox_7.isChecked()),
            "ymax_enabled": bool(self.checkBox_8.isChecked()),
            "xmin": float(self.doubleSpinBox.value()),
            "xmax": float(self.doubleSpinBox_3.value()),
            "ymin": float(self.doubleSpinBox_2.value()),
            "ymax": float(self.doubleSpinBox_4.value()),
            "x_shift": float(self.x_shift),
            "y_shift": float(self.y_shift),
            "curve_visibility": curve_visibility,
        }

    def set_state(self, state: dict) -> None:
        """Restore line plot controller state from a project.

        Parameters
        ----------
        state : dict
            State produced by :meth:`get_state`.
        """
        if not isinstance(state, dict):
            return
        self.treeWidget.blockSignals(True)
        try:
            if "data_logy" in state:
                self.data_logy = str(state["data_logy"])
            if "scale_x" in state:
                self.scale_x = str(state["scale_x"])
            if "res_logy" in state:
                self.res_logy = str(state["res_logy"])
            for key, widget in (
                ("xmin_enabled", self.checkBox_4),
                ("xmax_enabled", self.checkBox_6),
                ("ymin_enabled", self.checkBox_7),
                ("ymax_enabled", self.checkBox_8),
            ):
                if key in state:
                    widget.setChecked(bool(state[key]))
            for key, widget in (
                ("xmin", self.doubleSpinBox),
                ("xmax", self.doubleSpinBox_3),
                ("ymin", self.doubleSpinBox_2),
                ("ymax", self.doubleSpinBox_4),
                ("x_shift", self.doubleSpinBox_6),
                ("y_shift", self.doubleSpinBox_5),
            ):
                if key in state:
                    try:
                        widget.setValue(float(state[key]))
                    except Exception:
                        pass
            if "reference_mode" in state:
                self.reference_mode = str(state["reference_mode"])
            if "reference_parameters" in state:
                self.reference_parameters = state.get("reference_parameters", {})
            if "is_density" in state:
                self.is_density = bool(state["is_density"])
            if "display_group" in state:
                self.display_group = bool(state["display_group"])
            if "plot_ftt" in state:
                self.plot_ftt = bool(state["plot_ftt"])
            visibility = state.get("curve_visibility")
            if isinstance(visibility, dict):
                for i in range(self.treeWidget.topLevelItemCount()):
                    item = self.treeWidget.topLevelItem(i)
                    key = item.text(2)
                    if key in visibility:
                        item.setCheckState(1, QtCore.Qt.Checked if visibility[key] else QtCore.Qt.Unchecked)
        finally:
            self.treeWidget.blockSignals(False)
        try:
            self.parent.update()
        except Exception:
            pass

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
            fit: cs.core.fitting.fit.Fit,
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
            fit: cs.core.fitting.fit.FitGroup,
            scale_x: str = 'lin',
            d_scaley: str = 'lin',
            r_scaley: str = 'lin',
            x_label: str = 'x',
            y_label: str = 'y',
            curve_styles: typing.Dict | None = None,
            **kwargs
    ):
        # Internal state of region selector
        self.lb_i: int = 0
        self.ub_i: int = 0

        self.curve_styles = curve_styles or {}
        self._base_y_label = y_label

        kwargs['fit'] = fit
        super().__init__(**kwargs)
        self.plot_controller = LinePlotControl(
                parent=self,
                scale_x=scale_x,
                d_scaley=d_scaley,
                r_scaley=r_scaley
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
        p1.getViewBox().setXLink(p3.getViewBox())
        p2.getViewBox().setXLink(p3.getViewBox())

        plots = {
            'top_left_plot': p1.getPlotItem(),
            'top_right_plot': p2.getPlotItem(),
            'main_plot': p3.getPlotItem()
        }
        plots['top_left_plot'].hideAxis('bottom')
        plots['top_right_plot'].hideAxis('bottom')

        hide_dock_title = cs.core.settings.gui['plot']['hideTitle']
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
        if cs.core.settings.gui['plot']['enable_region_selector']:
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
                cs.core.actions.dispatch(
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
        if cs.core.settings.gui['plot']['enable_grid']:
            if cs.core.settings.gui['plot']['show_data_grid']:
                plots['main_plot'].showGrid(True, True, 0.5)
            if cs.core.settings.gui['plot']['show_residual_grid']:
                plots['top_left_plot'].showGrid(True, True, 1.0)
            if cs.core.settings.gui['plot']['show_acorr_grid']:
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
        color_idx = index % len(cs.core.settings.colors)
        pen_color = cs.core.settings.colors[color_idx]['hex']
        lw = cs.core.settings.gui['plot']['line_width']

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
            if isinstance(curve, cs.core.data.DataCurve):
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

    def _reference_modes_for_model(
            self,
            model
    ) -> typing.List[plot_transforms.PlotReferenceMode]:
        """Return model-provided plot reference modes.

        Parameters
        ----------
        model : object
            Model object to query.

        Returns
        -------
        list
            Valid reference modes.
        """
        getter = getattr(model, "get_plot_reference_modes", None)
        if not callable(getter):
            return []
        try:
            modes = getter()
        except Exception as exc:
            cs.logging.warning("Could not query plot reference modes: %s", exc)
            return []
        return [
            mode for mode in modes or []
            if isinstance(mode, plot_transforms.PlotReferenceMode)
        ]

    def _update_reference_modes(self, current_fit) -> None:
        """Refresh the reference-mode selector for the active model.

        Parameters
        ----------
        current_fit : cs.core.fitting.fit.Fit
            Fit whose model controls the available modes.
        """
        model = getattr(current_fit, "model", None)
        modes = self._reference_modes_for_model(model)
        self.plot_controller.set_reference_modes(modes)

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

    @staticmethod
    def _axis_range(min_value, max_value, values, log_mode: bool = False):
        """Return a finite pyqtgraph axis range, or ``None`` if invalid."""
        if min_value is None and max_value is None:
            return None

        try:
            values = np.asarray(values)
            if values.size == 0:
                return None
            a_min = values[0] if min_value is None else min_value
            a_max = values[-1] if max_value is None else max_value
            a_min = float(a_min)
            a_max = float(a_max)
        except (TypeError, ValueError, IndexError):
            return None

        if not np.isfinite(a_min) or not np.isfinite(a_max):
            return None
        if log_mode:
            if a_min <= 0.0 or a_max <= 0.0:
                return None
            a_min = np.log10(a_min)
            a_max = np.log10(a_max)
        if a_min > a_max:
            return None
        return [a_min, a_max]

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

        self._reference_y_label_override = None
        self._update_reference_modes(current_fit)
            
        x_last = max(0, len(current_data.x) - 1)
        xmin_i = int(np.clip(getattr(current_fit, "xmin", 0), 0, x_last))
        xmax_i = int(np.clip(getattr(current_fit, "xmax", x_last), 0, x_last))
        lb_min, ub_max = current_data.x[0], current_data.x[-1]
        lb, ub = current_data.x[xmin_i], current_data.x[xmax_i]

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
        self.plots['main_plot'].setLabel(
            'left',
            self._reference_y_label_override or self._base_y_label
        )

        # Set manual scale
        xRange = self._axis_range(
            self.plot_controller.xmin,
            self.plot_controller.xmax,
            data.x,
            data_log_x,
        )
        yRange = self._axis_range(
            self.plot_controller.ymin,
            self.plot_controller.ymax,
            data.y,
            data_log_y,
        )
        if xRange is not None or yRange is not None:
            self.plots['main_plot'].setRange(xRange=xRange, yRange=yRange)

        if self._metrics_text_alive() and not bool(getattr(cs, "_suspend_plot_metrics_overlay", False)):
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
            font_pt = int(cs.core.settings.gui.get("plot", {}).get("metrics_font_pt", font_pt))
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

    def _selected_reference_mode_for_model(
            self,
            model
    ) -> plot_transforms.PlotReferenceMode | None:
        """Return the selected reference mode for ``model``.

        Parameters
        ----------
        model : object
            Model that may provide reference modes.

        Returns
        -------
        PlotReferenceMode or None
            Selected mode for this model, or None for raw plotting.
        """
        key = self.plot_controller.reference_mode
        if key == "raw":
            return None
        for mode in self._reference_modes_for_model(model):
            if mode.key == key:
                return mode
        return None

    def _apply_reference_mode_to_curve(
            self,
            fit,
            model,
            curve_key: str,
            x: np.ndarray,
            y: np.ndarray,
            curves: typing.Mapping[str, typing.Any],
            group_fits: typing.Sequence | None = None,
            group_index: int | None = None,
            selected_group_index: int | None = None
    ) -> plot_transforms.PlotReferenceResult:
        """Apply the selected reference mode to one curve.

        Parameters
        ----------
        fit : object
            Fit owning the curve.
        model : object
            Model attached to ``fit``.
        curve_key : str
            Curve name being plotted.
        x : numpy.ndarray
            Curve x-values.
        y : numpy.ndarray
            Curve y-values.
        curves : mapping
            Curves available for ``fit``.
        group_fits : sequence, optional
            Grouped fits.
        group_index : int, optional
            Index of ``fit`` in ``group_fits``.
        selected_group_index : int, optional
            Active grouped fit index.

        Returns
        -------
        PlotReferenceResult
            Transformed curve result.
        """
        mode = self._selected_reference_mode_for_model(model)
        if mode is None:
            return plot_transforms.PlotReferenceResult(x=x, y=y)

        context = plot_transforms.PlotReferenceContext(
            fit=fit,
            model=model,
            curve_key=curve_key,
            x=x,
            y=y,
            curves=curves,
            group_fits=tuple(group_fits or ()),
            group_index=group_index,
            selected_group_index=selected_group_index,
            parameters=self.plot_controller.reference_parameters,
        )
        if not mode.applies(context):
            return plot_transforms.PlotReferenceResult(x=x, y=y)

        try:
            result = mode.callback(context)
        except Exception as exc:
            cs.logging.warning("Plot reference mode '%s' failed for %s: %s", mode.key, curve_key, exc)
            return plot_transforms.PlotReferenceResult(x=x, y=y)

        if result is None:
            return plot_transforms.PlotReferenceResult(x=x, y=y)
        if isinstance(result, plot_transforms.PlotReferenceResult):
            self._reference_y_label_override = result.y_label or mode.y_label or self._reference_y_label_override
            return result
        if isinstance(result, tuple) and len(result) >= 2:
            transformed = plot_transforms.PlotReferenceResult(
                x=np.asarray(result[0], dtype=float),
                y=np.asarray(result[1], dtype=float),
            )
        else:
            try:
                transformed = plot_transforms.PlotReferenceResult(
                    x=x,
                    y=np.asarray(result, dtype=float),
                )
            except Exception:
                transformed = plot_transforms.PlotReferenceResult(x=x, y=y)
        if mode.y_label:
            self._reference_y_label_override = mode.y_label
        return transformed

    def _plot_single_fit_curves(self, fit, curves, data_log_x, data_log_y, director, x_shift, y_shift):
        """Plot curves for a single fit (original behavior)"""
        curves_keys = list(curves.keys())[::-1]
        for i, curve_key in enumerate(curves_keys):
            curve_settings = director.get(curve_key, director['default'])
            curve = curves[curve_key]

            y = np.copy(curve.y)
            x = np.copy(curve.x)

            line: pg.PlotDataItem = self.lines[curve_key]

            if curve_settings.get('allow_reference_transform', False):
                result = self._apply_reference_mode_to_curve(
                    fit=fit,
                    model=getattr(fit, "model", None),
                    curve_key=curve_key,
                    x=x,
                    y=y,
                    curves=curves,
                )
                if not result.visible:
                    line.setData(x=[], y=[])
                    line.hide()
                    continue
                x = np.asarray(result.x, dtype=float)
                y = np.asarray(result.y, dtype=float)

            if curve_settings['allow_shift']:
                y += y_shift
                x += x_shift

            if self.plot_controller.is_density and curve_settings['allow_density']:
                y[1:] = y[1:] / np.diff(x)

            # Base data for plotting: either full curve or fit-range only
            if curve_settings['plot_only_region'] and len(x) == len(curve.x):
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

            if curve_settings.get('allow_reference_transform', False):
                result = self._apply_reference_mode_to_curve(
                    fit=group_fit,
                    model=getattr(group_fit, "model", None),
                    curve_key=curve_key,
                    x=x,
                    y=y,
                    curves=group_curves,
                    group_fits=grouped_fits,
                    group_index=grouped_fits.index(group_fit),
                    selected_group_index=current_fit_index,
                )
                if not result.visible:
                    if line_name in self.lines:
                        self.lines[line_name].setData(x=[], y=[])
                        self.lines[line_name].hide()
                    continue
                x = np.asarray(result.x, dtype=float)
                y = np.asarray(result.y, dtype=float)

            if curve_settings['allow_shift']:
                y += y_shift
                x += x_shift

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
            if curve_settings['plot_only_region'] and len(x) == len(curve.x):
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
