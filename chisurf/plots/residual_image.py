from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

import chisurf.fitting
from chisurf.plots import plotbase


class Residual2DPlotControl(QtWidgets.QWidget):
    def __init__(self, parent: "Residual2DPlot" | None = None) -> None:
        super().__init__(parent)
        self._plot = parent

        form = QtWidgets.QFormLayout(self)

        self.sb_vmin = QtWidgets.QDoubleSpinBox(self)
        self.sb_vmax = QtWidgets.QDoubleSpinBox(self)
        for sb in (self.sb_vmin, self.sb_vmax):
            sb.setDecimals(6)
            sb.setRange(-1e12, 1e12)
        form.addRow("vmin", self.sb_vmin)
        form.addRow("vmax", self.sb_vmax)

        self.sb_xmin = QtWidgets.QDoubleSpinBox(self)
        self.sb_xmax = QtWidgets.QDoubleSpinBox(self)
        self.sb_ymin = QtWidgets.QDoubleSpinBox(self)
        self.sb_ymax = QtWidgets.QDoubleSpinBox(self)
        for sb in (self.sb_xmin, self.sb_xmax, self.sb_ymin, self.sb_ymax):
            sb.setDecimals(6)
            sb.setRange(-1e12, 1e12)
        xy_row = QtWidgets.QGridLayout()
        xy_row.addWidget(QtWidgets.QLabel("x min"), 0, 0)
        xy_row.addWidget(self.sb_xmin, 0, 1)
        xy_row.addWidget(QtWidgets.QLabel("x max"), 0, 2)
        xy_row.addWidget(self.sb_xmax, 0, 3)
        xy_row.addWidget(QtWidgets.QLabel("y min"), 1, 0)
        xy_row.addWidget(self.sb_ymin, 1, 1)
        xy_row.addWidget(QtWidgets.QLabel("y max"), 1, 2)
        xy_row.addWidget(self.sb_ymax, 1, 3)
        form.addRow(xy_row)

        self.cb_cmap = QtWidgets.QComboBox(self)
        # Prefer a diverging colormap as default for signed residuals
        self.cb_cmap.addItems(["RdBu", "bwr", "viridis", "plasma", "inferno", "magma", "cividis"])
        form.addRow("Colormap", self.cb_cmap)

        self.sb_vmin.editingFinished.connect(self._on_levels_changed)
        self.sb_vmax.editingFinished.connect(self._on_levels_changed)
        self.sb_xmin.editingFinished.connect(self._on_ranges_changed)
        self.sb_xmax.editingFinished.connect(self._on_ranges_changed)
        self.sb_ymin.editingFinished.connect(self._on_ranges_changed)
        self.sb_ymax.editingFinished.connect(self._on_ranges_changed)
        self.cb_cmap.currentTextChanged.connect(self._on_cmap_changed)

    def set_initial_ranges(self, x: np.ndarray, y: np.ndarray, vmin: float, vmax: float) -> None:
        try:
            self.blockSignals(True)
            if x is not None and x.size > 0:
                self.sb_xmin.setValue(float(np.nanmin(x)))
                self.sb_xmax.setValue(float(np.nanmax(x)))
            if y is not None and y.size > 0:
                self.sb_ymin.setValue(float(np.nanmin(y)))
                self.sb_ymax.setValue(float(np.nanmax(y)))
            self.sb_vmin.setValue(float(vmin))
            self.sb_vmax.setValue(float(vmax))
        finally:
            self.blockSignals(False)

    def levels(self) -> tuple[float, float]:
        return float(self.sb_vmin.value()), float(self.sb_vmax.value())

    def ranges(self) -> tuple[float, float, float, float]:
        return (
            float(self.sb_xmin.value()),
            float(self.sb_xmax.value()),
            float(self.sb_ymin.value()),
            float(self.sb_ymax.value()),
        )

    def cmap_name(self) -> str:
        return str(self.cb_cmap.currentText())

    def _on_levels_changed(self) -> None:
        if self._plot is not None:
            self._plot.apply_levels_from_controller()

    def _on_ranges_changed(self) -> None:
        if self._plot is not None:
            self._plot.apply_ranges_from_controller()

    def _on_cmap_changed(self, _text: str) -> None:
        if self._plot is not None:
            self._plot.apply_cmap_from_controller()


class Residual2DPlot(plotbase.Plot):
    """Generic 2D residual image plot using a model-provided accessor.

    The accessor is responsible for computing the 2D residual matrix and the
    corresponding x/y axes so this plot remains model-agnostic.

    Expected signature:

        accessor(fit_group: chisurf.fitting.fit.FitGroup, **kwargs)
            -> (image_2d, x_axis, y_axis)
    """

    name = "Residuals 2D"

    def __init__(
        self,
        fit: chisurf.fitting.fit.FitGroup,
        *args,
        accessor=None,
        accessor_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(fit=fit, *args, **kwargs)

        self._accessor = accessor
        self._accessor_kwargs = {} if accessor_kwargs is None else dict(accessor_kwargs)

        self._image: np.ndarray | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

        self._plot_widget = pg.PlotWidget()
        self.layout.addWidget(self._plot_widget)
        self._view_box = self._plot_widget.getPlotItem().getViewBox()
        self._view_box.setAspectLocked(False)

        self._image_item = pg.ImageItem()
        self._view_box.addItem(self._image_item)

        self.plot_controller = Residual2DPlotControl(self)
        self.widgets.append(self.plot_controller)

    def _compute_image(self) -> None:
        if self._accessor is None:
            return
        try:
            img, x, y = self._accessor(self.fit, **self._accessor_kwargs)
        except Exception:
            return

        if img is None:
            return
        arr = np.asarray(img, dtype=float)
        if arr.ndim != 2:
            return

        self._image = arr
        self._x = np.asarray(x) if x is not None else np.arange(arr.shape[1], dtype=float)
        self._y = np.asarray(y) if y is not None else np.arange(arr.shape[0], dtype=float)

        self._image_item.setImage(self._image, autoLevels=False)

        finite = np.isfinite(self._image)
        if np.any(finite):
            # Choose a symmetric range around zero based on |residuals| to
            # visualize positive and negative deviations in a balanced way.
            abs_vals = np.abs(self._image[finite])
            try:
                level = float(np.percentile(abs_vals, 99.0))
            except Exception:
                level = float(abs_vals.max()) if abs_vals.size > 0 else 1.0
            if not np.isfinite(level) or level <= 0.0:
                level = 1.0
            vmin = -level
            vmax = level
        else:
            vmin, vmax = -1.0, 1.0

        self.plot_controller.set_initial_ranges(self._x, self._y, vmin, vmax)
        self.apply_levels_from_controller()
        self.apply_ranges_from_controller()
        self.apply_cmap_from_controller()

    def apply_levels_from_controller(self) -> None:
        if self._image is None:
            return
        vmin, vmax = self.plot_controller.levels()
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            finite = np.isfinite(self._image)
            if not np.any(finite):
                return
            vmin = float(self._image[finite].min())
            vmax = float(self._image[finite].max())
        self._image_item.setLevels((vmin, vmax))

    def apply_ranges_from_controller(self) -> None:
        if self._x is None or self._y is None:
            return
        xmin, xmax, ymin, ymax = self.plot_controller.ranges()
        if xmax <= xmin:
            xmin, xmax = float(self._x.min()), float(self._x.max())
        if ymax <= ymin:
            ymin, ymax = float(self._y.min()), float(self._y.max())
        self._view_box.setXRange(xmin, xmax, padding=0.0)
        self._view_box.setYRange(ymin, ymax, padding=0.0)

    def apply_cmap_from_controller(self) -> None:
        name = self.plot_controller.cmap_name()
        try:
            cm = pg.colormap.get(name)  # type: ignore[attr-defined]
            lut = cm.getLookupTable(alpha=False)
            self._image_item.setLookupTable(lut)
        except Exception:
            self._image_item.setLookupTable(None)

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._compute_image()

    def update_all(self, *args, **kwargs) -> None:
        self.update(*args, **kwargs)
