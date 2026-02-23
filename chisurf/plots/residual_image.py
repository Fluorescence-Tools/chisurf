from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore

import chisurf.fitting
from chisurf.plots import plotbase
from chisurf.runtime.actions import record_action


class _DraggableTextItem(pg.TextItem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self._dragging = False
        self._drag_offset = QtCore.QPointF(0, 0)

    def hoverEnterEvent(self, event):
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self._drag_offset = event.pos()
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self._dragging and event.buttons() & QtCore.Qt.LeftButton:
            new_pos = self.mapToParent(event.pos() - self._drag_offset)
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


class Residual2DPlotControl(QtWidgets.QWidget):
    def __init__(self, parent: "Residual2DPlot" | None = None) -> None:
        super().__init__(parent)
        self._plot = parent

        form = QtWidgets.QFormLayout(self)
        # Compact controller layout: remove margins and extra spacing so
        # multiple plot controllers can sit closely without wasting space.
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(0)

        self.sb_vmin = QtWidgets.QDoubleSpinBox(self)
        self.sb_vmax = QtWidgets.QDoubleSpinBox(self)
        for sb in (self.sb_vmin, self.sb_vmax):
            sb.setDecimals(6)
            sb.setRange(-1e12, 1e12)

        # Single compact row for vmin / vmax and the Auto contrast toolbutton.
        v_row = QtWidgets.QGridLayout()
        v_row.setContentsMargins(0, 0, 0, 0)
        v_row.setSpacing(0)
        v_row.addWidget(QtWidgets.QLabel("vmin"), 0, 0)
        v_row.addWidget(self.sb_vmin, 0, 1)
        v_row.addWidget(QtWidgets.QLabel("vmax"), 0, 2)
        v_row.addWidget(self.sb_vmax, 0, 3)

        # Optional helper to restore reasonable contrast based on the
        # underlying image data. This is especially handy after manual edits
        # of vmin/vmax or when switching image sources.
        self.btn_auto = QtWidgets.QToolButton(self)
        self.btn_auto.setText("Auto contrast")
        v_row.addWidget(self.btn_auto, 0, 4)

        form.addRow(v_row)

        self.sb_xmin = QtWidgets.QSpinBox(self)
        self.sb_xmax = QtWidgets.QSpinBox(self)
        self.sb_ymin = QtWidgets.QSpinBox(self)
        self.sb_ymax = QtWidgets.QSpinBox(self)
        for sb in (self.sb_xmin, self.sb_xmax, self.sb_ymin, self.sb_ymax):
            sb.setRange(-int(1e9), int(1e9))
        xy_row = QtWidgets.QGridLayout()
        xy_row.setContentsMargins(0, 0, 0, 0)
        xy_row.setSpacing(0)
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

        # Optional image source selector (e.g. residual, data, model, intensity).
        # Hidden by default and only populated when the parent plot provides
        # multiple sources. When unused, both the label and the combo box are
        # hidden to avoid clutter.
        self.cb_source = QtWidgets.QComboBox(self)
        self.cb_source.setVisible(False)
        try:
            self._lbl_source = QtWidgets.QLabel("Image source", self)
        except Exception:
            self._lbl_source = None
        if self._lbl_source is not None:
            self._lbl_source.setVisible(False)
            form.addRow(self._lbl_source, self.cb_source)
        else:
            form.addRow("Image source", self.cb_source)

        # Optional frame index selector for image stacks. Hidden by default
        # and enabled only when the parent configures a frame range. When not
        # used, hide its label as well.
        self.sb_frame = QtWidgets.QSpinBox(self)
        self.sb_frame.setRange(0, 0)
        self.sb_frame.setVisible(False)
        try:
            self._lbl_frame = QtWidgets.QLabel("Frame index", self)
        except Exception:
            self._lbl_frame = None
        if self._lbl_frame is not None:
            self._lbl_frame.setVisible(False)
            form.addRow(self._lbl_frame, self.sb_frame)
        else:
            form.addRow("Frame index", self.sb_frame)

        self.sb_vmin.editingFinished.connect(self._on_levels_changed)
        self.sb_vmax.editingFinished.connect(self._on_levels_changed)
        self.sb_xmin.editingFinished.connect(self._on_ranges_changed)
        self.sb_xmax.editingFinished.connect(self._on_ranges_changed)
        self.sb_ymin.editingFinished.connect(self._on_ranges_changed)
        self.sb_ymax.editingFinished.connect(self._on_ranges_changed)
        self.cb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        self.cb_source.currentIndexChanged.connect(self._on_source_changed)
        self.sb_frame.valueChanged.connect(self._on_frame_changed)
        self.btn_auto.clicked.connect(self._on_auto_clicked)

    def set_initial_ranges(self, x: np.ndarray, y: np.ndarray, vmin: float, vmax: float) -> None:
        try:
            self.blockSignals(True)
            if x is not None and x.size > 0:
                x_min = int(np.nanmin(x))
                x_max = int(np.nanmax(x))
                if x_max < x_min:
                    x_min, x_max = x_max, x_min
                self.sb_xmin.setRange(x_min, x_max)
                self.sb_xmax.setRange(x_min, x_max)
                self.sb_xmin.setValue(x_min)
                self.sb_xmax.setValue(x_max)
            if y is not None and y.size > 0:
                y_min = int(np.nanmin(y))
                y_max = int(np.nanmax(y))
                if y_max < y_min:
                    y_min, y_max = y_max, y_min
                self.sb_ymin.setRange(y_min, y_max)
                self.sb_ymax.setRange(y_min, y_max)
                self.sb_ymin.setValue(y_min)
                self.sb_ymax.setValue(y_max)
            self.sb_vmin.setValue(float(vmin))
            self.sb_vmax.setValue(float(vmax))
        finally:
            self.blockSignals(False)

    def levels(self) -> tuple[float, float]:
        return float(self.sb_vmin.value()), float(self.sb_vmax.value())

    def ranges(self) -> tuple[int, int, int, int]:
        return (
            int(self.sb_xmin.value()),
            int(self.sb_xmax.value()),
            int(self.sb_ymin.value()),
            int(self.sb_ymax.value()),
        )

    def cmap_name(self) -> str:
        return str(self.cb_cmap.currentText())

    # --- Optional multi-source / stack controls ---------------------------------

    def set_sources(self, names: list[str]) -> None:
        """Populate the image-source selector.

        When *names* is empty the selector is hidden and the plot behaves like
        a single-source residual view (backwards compatible behaviour).
        """

        try:
            self.cb_source.blockSignals(True)
            self.cb_source.clear()
            for name in names:
                self.cb_source.addItem(str(name))
            has_sources = bool(names)
            self.cb_source.setVisible(has_sources)
            lbl = getattr(self, "_lbl_source", None)
            if lbl is not None:
                lbl.setVisible(has_sources)
        finally:
            self.cb_source.blockSignals(False)

    def current_source_name(self) -> str:
        return str(self.cb_source.currentText())

    def set_frame_range(self, n_frames: int) -> None:
        """Configure the frame slider range for image stacks.

        If ``n_frames <= 1`` the slider is hidden and effectively disabled.
        """

        try:
            self.sb_frame.blockSignals(True)
            has_frames = not (n_frames is None or n_frames <= 1)
            if not has_frames:
                self.sb_frame.setVisible(False)
                self.sb_frame.setRange(0, 0)
                self.sb_frame.setValue(0)
            else:
                n = int(n_frames)
                if n < 1:
                    n = 1
                self.sb_frame.setVisible(True)
                self.sb_frame.setRange(0, n - 1)
                # Clamp current value into range
                v = self.sb_frame.value()
                if v < 0:
                    v = 0
                if v > n - 1:
                    v = n - 1
                self.sb_frame.setValue(v)
            lbl = getattr(self, "_lbl_frame", None)
            if lbl is not None:
                lbl.setVisible(has_frames)
        finally:
            self.sb_frame.blockSignals(False)

    def frame_index(self) -> int:
        return int(self.sb_frame.value())

    def _on_levels_changed(self) -> None:
        if self._plot is not None:
            self._plot.apply_levels_from_controller()

    def _on_ranges_changed(self) -> None:
        if self._plot is not None:
            self._plot.apply_ranges_from_controller()

    def _on_cmap_changed(self, _text: str) -> None:
        if self._plot is not None:
            self._plot.apply_cmap_from_controller()

    def _on_auto_clicked(self) -> None:
        if self._plot is not None and hasattr(self._plot, "auto_contrast"):
            try:
                self._plot.auto_contrast()
            except Exception:
                pass

    def _on_source_changed(self, _idx: int) -> None:
        if self._plot is not None:
            self._plot.on_source_changed()

    def _on_frame_changed(self, _val: int) -> None:
        if self._plot is not None:
            self._plot.on_frame_changed()


class Residual2DPlot(plotbase.Plot):
    """Generic 2D residual image plot using a model-provided accessor.

    The accessor is responsible for computing the 2D residual matrix and the
    corresponding x/y axes so this plot remains model-agnostic.

    Expected signature:

        accessor(fit_group: chisurf.fitting.fit.FitGroup, **kwargs)
            -> (image_2d, x_axis, y_axis)
    """

    name = "Residuals 2D"
    # Optional signal used by some fit widgets (e.g. RICS) to synchronize
    # the 1D fit range with a 2D selection. FitSubWindow already connects
    # LinePlot.regionChanged to the fit controller; by exposing the same
    # signal here we can reuse that wiring for 2D ROI-based range updates.
    regionChanged = QtCore.Signal(int, int)

    def __init__(
        self,
        fit: chisurf.fitting.fit.FitGroup,
        *args,
        accessor=None,
        accessor_kwargs: dict | None = None,
        sources: dict | None = None,
        frame_kw: str | None = None,
        max_frames_accessor=None,
        **kwargs,
    ) -> None:
        super().__init__(fit=fit, *args, **kwargs)

        self._accessor = accessor
        self._accessor_kwargs = {} if accessor_kwargs is None else dict(accessor_kwargs)

        # Optional support for multiple named image sources (e.g. residual,
        # data, model, intensity). When provided, the plot controller exposes
        # a selector and this plot switches between the configured accessors.
        self._sources: dict[str, tuple[callable, dict]] | None = None
        self._current_source_key: str | None = None

        if sources:
            src_map: dict[str, tuple[callable, dict]] = {}
            for key, spec in sources.items():
                if isinstance(spec, tuple) and len(spec) == 2:
                    fn, kw = spec
                elif isinstance(spec, dict):
                    fn = spec.get("accessor")
                    kw = spec.get("accessor_kwargs", {})
                else:
                    continue
                if fn is None:
                    continue
                src_map[str(key)] = (fn, dict(kw or {}))
            if src_map:
                self._sources = src_map
                self._current_source_key = next(iter(src_map.keys()))
                fn, kw = src_map[self._current_source_key]
                self._accessor = fn
                self._accessor_kwargs = dict(kw or {})

        # Optional configuration for frame-aware accessors (image stacks).
        # If *frame_kw* is not None, the current frame index from the
        # controller will be injected into accessor_kwargs[frame_kw] before
        # calling the accessor, but only if that key already exists in the
        # kwargs for the current source.
        self._frame_kw = frame_kw
        self._max_frames_accessor = max_frames_accessor

        self._image: np.ndarray | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

        self._plot_widget = pg.PlotWidget()
        self.layout.addWidget(self._plot_widget)
        self._view_box = self._plot_widget.getPlotItem().getViewBox()
        self._view_box.setAspectLocked(False)

        self._image_item = pg.ImageItem()
        self._view_box.addItem(self._image_item)

        try:
            self._quality_text = _DraggableTextItem(
                text="",
                border="w",
                fill=(0, 0, 255, 100),
                anchor=(0, 0),
            )
            plot_item = self._plot_widget.getPlotItem()
            self._quality_text.setParentItem(plot_item)
            self._quality_text.setPos(0, 0)
        except Exception:
            self._quality_text = None

        self._quality_text_initialized = False

        # Optional rectangular ROI used to define a 2D selection that can be
        # mapped back to a 1D fit-range (flattened lag index), analogous to
        # the LinearRegionItem in LinePlot. Created lazily on first image.
        self._roi = None
        self._roi_sync_in_progress = False
        self._roi_initialized = False

        self.plot_controller = Residual2DPlotControl(self)
        self.widgets.append(self.plot_controller)

        # Expose optional multi-source selector and frame slider in the
        # controller if configured.
        if self._sources is not None:
            try:
                self.plot_controller.set_sources(list(self._sources.keys()))
            except Exception:
                pass
        if self._max_frames_accessor is not None:
            try:
                n_frames = int(self._max_frames_accessor(self.fit))
            except Exception:
                n_frames = 0
            try:
                self.plot_controller.set_frame_range(n_frames)
            except Exception:
                pass

    def _compute_image(self) -> None:
        if self._accessor is None:
            return
        # Refresh accessor / kwargs from currently selected source, if any.
        if self._sources is not None and self._current_source_key in self._sources:
            fn, kw = self._sources[self._current_source_key]
            self._accessor = fn
            self._accessor_kwargs = dict(kw or {})

        # Inject current frame index for frame-aware sources when requested.
        if self._frame_kw and isinstance(self._accessor_kwargs, dict) and self._frame_kw in self._accessor_kwargs:
            try:
                frame_idx = int(self.plot_controller.frame_index())
                self._accessor_kwargs[self._frame_kw] = frame_idx
            except Exception:
                pass

        # Update frame range dynamically from the accessor if configured.
        if self._max_frames_accessor is not None:
            try:
                n_frames = int(self._max_frames_accessor(self.fit))
            except Exception:
                n_frames = 0
            try:
                self.plot_controller.set_frame_range(n_frames)
            except Exception:
                pass

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

        # Initialize ROI once to cover the full image in axis coordinates.
        if not getattr(self, "_roi_initialized", False):
            roi = self._ensure_roi()
            if roi is not None:
                try:
                    self._roi_sync_in_progress = True
                    xmin = float(self._x.min()) if self._x is not None and self._x.size > 0 else 0.0
                    xmax = float(self._x.max()) if self._x is not None and self._x.size > 0 else float(self._image.shape[1] - 1)
                    ymin = float(self._y.min()) if self._y is not None and self._y.size > 0 else 0.0
                    ymax = float(self._y.max()) if self._y is not None and self._y.size > 0 else float(self._image.shape[0] - 1)
                    roi.setPos((xmin, ymin))
                    roi.setSize((xmax - xmin, ymax - ymin))
                finally:
                    self._roi_sync_in_progress = False
                self._roi_initialized = True

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

        try:
            fit_obj = getattr(self.fit, "selected_fit", self.fit)
            chi2r = float(getattr(fit_obj, "chi2r", float("nan")))
            dw = float(getattr(fit_obj, "durbin_watson", float("nan")))
            if self._quality_text is not None and np.isfinite(chi2r) and np.isfinite(dw):
                if not getattr(self, "_quality_text_initialized", False):
                    try:
                        if self._x is not None and self._x.size > 0:
                            xmin = float(self._x.min())
                            xmax = float(self._x.max())
                        else:
                            xmin = 0.0
                            xmax = float(self._image.shape[1] - 1)
                        if self._y is not None and self._y.size > 0:
                            ymin = float(self._y.min())
                            ymax = float(self._y.max())
                        else:
                            ymin = 0.0
                            ymax = float(self._image.shape[0] - 1)
                        x_pos = xmin + 0.7 * (xmax - xmin)
                        y_pos = ymin + 0.9 * (ymax - ymin)
                        self._quality_text.setPos(x_pos, y_pos)
                    except Exception:
                        pass
                    else:
                        self._quality_text_initialized = True
                html = (
                    "<div style=\"name-align: center\">"
                    "<span style=\"color: #FF0; font-size: 10pt;\">"
                    f"&Chi;<sup>2</sup>={chi2r:.4f}<br />DW={dw:.4f}"
                    "</span></div>"
                )
                self._quality_text.setHtml(html)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Hooks used by Residual2DPlotControl
    # ------------------------------------------------------------------

    def on_source_changed(self) -> None:
        """Switch to a different image source and recompute the plot."""

        if self._sources is None:
            return
        try:
            key = self.plot_controller.current_source_name()
        except Exception:
            key = None
        if not key or key not in self._sources:
            return
        self._current_source_key = key
        self.update()

    def on_frame_changed(self) -> None:
        """Recompute the image for the newly selected frame index."""

        if self._frame_kw is None:
            return
        self.update()

    def auto_contrast(self) -> None:
        """Reset vmin/vmax to data-driven levels and apply them.

        For residual-like images (with both positive and negative values) a
        symmetric range around zero is chosen based on a high percentile of
        ``|image|``. For non-negative images the full [min, max] range is
        used. Spin boxes in the controller are updated accordingly.
        """

        if self._image is None:
            return

        import numpy as np

        finite = np.isfinite(self._image)
        if not np.any(finite):
            return

        img = self._image[finite]

        has_neg = bool(np.any(img < 0))
        has_pos = bool(np.any(img > 0))

        if has_neg and has_pos:
            # Likely a residual image: choose a symmetric window around zero
            # based on the 99th percentile of absolute values.
            try:
                level = float(np.percentile(np.abs(img), 99.0))
            except Exception:
                level = float(np.max(np.abs(img))) if img.size > 0 else 1.0
            if not np.isfinite(level) or level <= 0.0:
                level = 1.0
            vmin, vmax = -level, level
        else:
            # Purely non-negative or non-positive image: use full data range.
            try:
                vmin = float(img.min())
                vmax = float(img.max())
            except Exception:
                return
            if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
                return

        # Update the controller spin boxes without emitting change signals
        try:
            self.plot_controller.blockSignals(True)
            self.plot_controller.sb_vmin.setValue(vmin)
            self.plot_controller.sb_vmax.setValue(vmax)
        except Exception:
            pass
        finally:
            try:
                self.plot_controller.blockSignals(False)
            except Exception:
                pass

        # Apply the new levels to the image item
        self.apply_levels_from_controller()

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

    # ------------------------------------------------------------------
    # ROI → 1D fit-range mapping
    # ------------------------------------------------------------------

    def _ensure_roi(self):
        """Create the rectangular ROI on first use and attach callbacks."""

        if self._roi is not None:
            return self._roi
        try:
            roi = pg.RectROI(
                [0, 0],
                [1, 1],
                pen={"color": 'y', "width": 1},
                rotatable=False,
            )
        except Exception:
            return None

        try:
            self._view_box.addItem(roi)
            roi.setZValue(10)
            roi.sigRegionChanged.connect(self._on_roi_changed)
        except Exception:
            pass

        self._roi = roi
        return self._roi

    def _on_roi_changed(self) -> None:
        """Update 1D fit range when the 2D ROI is moved or resized.

        The 2D residual image is defined on a regular grid; for RICS this
        grid corresponds to the lag indices used to flatten the ICS map into
        the 1D data vector (row-major order). We therefore map the ROI's
        integer (y, x) bounds to a contiguous [xmin, xmax] index interval
        covering the selected rectangle and propagate it to cs.current_fit.
        """

        if getattr(self, "_roi_sync_in_progress", False):
            return
        if self._image is None:
            return

        roi = getattr(self, "_roi", None)
        if roi is None:
            return

        try:
            pos = roi.pos()
            size = roi.size()
        except Exception:
            return

        try:
            x0 = float(pos.x())
            y0 = float(pos.y())
            w = float(size.x())
            h = float(size.y())
        except Exception:
            return

        ny, nx = int(self._image.shape[0]), int(self._image.shape[1])
        if ny <= 0 or nx <= 0:
            return

        # Enforce symmetry of the ROI around the image center so that the
        # 2D selection corresponds to a symmetric lag window. This mimics
        # two line selectors acting symmetrically about the central pixel.
        cx = 0.5 * float(nx - 1)
        cy = 0.5 * float(ny - 1)

        left = x0
        right = x0 + w
        top = y0
        bottom = y0 + h

        # Compute half-width/height as the maximum distance from center to
        # either ROI edge along each axis.
        dx = max(abs(cx - left), abs(right - cx), 0.5)
        dy = max(abs(cy - top), abs(bottom - cy), 0.5)

        x0_sym = cx - dx
        x1_sym = cx + dx
        y0_sym = cy - dy
        y1_sym = cy + dy

        # Clamp symmetric bounds to valid pixel coordinates.
        x0_sym = max(0.0, min(x0_sym, float(nx - 1)))
        x1_sym = max(0.0, min(x1_sym, float(nx - 1)))
        y0_sym = max(0.0, min(y0_sym, float(ny - 1)))
        y1_sym = max(0.0, min(y1_sym, float(ny - 1)))

        # Ensure at least one pixel in each direction.
        if x1_sym <= x0_sym:
            x1_sym = min(float(nx - 1), x0_sym + 1.0)
        if y1_sym <= y0_sym:
            y1_sym = min(float(ny - 1), y0_sym + 1.0)

        # Snap ROI geometry back to the symmetric bounds.
        try:
            self._roi_sync_in_progress = True
            roi.setPos((x0_sym, y0_sym))
            roi.setSize((x1_sym - x0_sym, y1_sym - y0_sym))
        finally:
            self._roi_sync_in_progress = False

        try:
            ix0 = int(np.floor(x0_sym))
            iy0 = int(np.floor(y0_sym))
            ix1 = int(np.ceil(x1_sym))
            iy1 = int(np.ceil(y1_sym))
        except Exception:
            return

        # Clamp ROI bounds to valid pixel indices
        ix0 = max(0, min(ix0, nx - 1))
        iy0 = max(0, min(iy0, ny - 1))
        ix1 = max(ix0 + 1, min(ix1, nx))
        iy1 = max(iy0 + 1, min(iy1, ny))

        # Map symmetric 2D rectangle to a contiguous 1D index range in
        # row-major order.
        xmin_idx = iy0 * nx + ix0
        xmax_idx = (iy1 - 1) * nx + (ix1 - 1)
        if xmax_idx < xmin_idx:
            xmin_idx, xmax_idx = xmax_idx, xmin_idx

        # Propagate to the current fit's range using the same mechanism as
        # LinePlot so downstream widgets and macros stay in sync.
        try:
            import chisurf

            chisurf.run(f"cs.current_fit.fit_range = {int(xmin_idx)}, {int(xmax_idx)}")
            try:
                fit_group_name = str(getattr(self.fit, "name", ""))
                local_fit_name = ""
                local_fit = getattr(self.fit, "selected_fit", None)
                if local_fit is not None:
                    local_fit_name = str(getattr(local_fit, "name", ""))
                record_action(
                    action_type="fit_range_set",
                    summary=(
                        f"set fit range for '{fit_group_name}' to [{int(xmin_idx)}, {int(xmax_idx)}) "
                        "from residual image"
                    ),
                    payload={
                        "fit_group": fit_group_name,
                        "local_fit": local_fit_name,
                        "xmin": int(xmin_idx),
                        "xmax": int(xmax_idx),
                        "source": "residual_image_roi",
                    },
                )
            except Exception:
                pass
        except Exception:
            pass

        # Notify listeners (e.g. FittingControllerWidget) so their range
        # spinboxes can mirror the ROI-selected fit range.
        try:
            self.regionChanged.emit(int(xmin_idx), int(xmax_idx))
        except Exception:
            pass

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._compute_image()

    def update_all(self, *args, **kwargs) -> None:
        self.update(*args, **kwargs)
