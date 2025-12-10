from __future__ import annotations

from typing import Sequence

from qtpy import QtCore, QtWidgets

try:  # optional dependency
    import pyqtgraph as pg  # type: ignore
except Exception:  # pragma: no cover - fallback when pyqtgraph is missing
    pg = None  # type: ignore

from ..theme import color as theme_color, metric as theme_metric


class PtPlotWidget(QtWidgets.QWidget):
    """Small 2D plot widget for PT graphs using pyqtgraph when available.

    The surrounding node is expected to call :meth:`set_data` with matching
    x/y sequences. When pyqtgraph is not installed, a simple text placeholder
    is shown instead so the node editor still functions.
    """

    def __init__(self, title: str = "Damped sine", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._title_label = QtWidgets.QLabel(title, self)
        self._title_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        layout.addWidget(self._title_label)

        self._plot_widget = None
        self._curve = None

        if pg is not None:
            w = pg.PlotWidget(self)

            # Make the widget itself transparent so only the plot area draws.
            try:
                w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
                w.setAutoFillBackground(False)
            except Exception:
                pass
            try:
                w.setFrameStyle(0)
            except Exception:
                pass
            try:
                w.setStyleSheet("background: transparent; border: 0px;")
            except Exception:
                pass

            # Background and line/axis colors driven by theme so the plot
            # integrates with the node editor's dark UI.
            fg = theme_color("plot_foreground", (235, 235, 235))
            bg_base = theme_color("plot_background", (0, 0, 0))
            try:
                opacity = float(theme_metric("plot_background_opacity", 0.0))
            except Exception:
                opacity = 0.0

            # Use an explicit RGBA color with configurable alpha so the
            # QGraphicsView background can be fully transparent when desired.
            try:
                alpha_bg = int(max(0.0, min(1.0, opacity)) * 255.0)
                bg_color_view = pg.mkColor(bg_base)
                bg_color_view.setAlpha(alpha_bg)
                w.setBackground(bg_color_view)
            except Exception:
                pass

            # Thin, themed plot line and axes.
            try:
                line_width = float(theme_metric("plot_line_width", 1.5))
            except Exception:
                line_width = 1.5
            if line_width <= 0.0:
                line_width = 1.0

            try:
                plot_item = w.getPlotItem()
            except Exception:
                plot_item = None

            if plot_item is not None:
                # Configure the inner ViewBox background using the theme,
                # keeping it fully transparent by default.
                try:
                    vb = plot_item.getViewBox()
                    alpha = int(max(0.0, min(1.0, opacity)) * 255.0)
                    bg_color = pg.mkColor(bg_base)
                    bg_color.setAlpha(alpha)
                    vb.setBackgroundColor(bg_color)
                except Exception:
                    pass

                try:
                    # Grid with subtle alpha over the node background.
                    plot_item.showGrid(x=True, y=True, alpha=0.2)
                except Exception:
                    pass

                pen = pg.mkPen(fg, width=line_width)
                self._curve = plot_item.plot([], [], pen=pen)

                # Axis lines and tick labels in the same foreground color.
                axis_pen = pg.mkPen(fg)
                for name in ("bottom", "left"):
                    try:
                        ax = plot_item.getAxis(name)
                        ax.setPen(axis_pen)
                        ax.setTextPen(axis_pen)
                    except Exception:
                        continue
            else:
                # Fallback if getPlotItem() is unavailable; style at
                # PlotWidget level and use a simple curve.
                try:
                    w.showGrid(x=True, y=True, alpha=0.2)
                except Exception:
                    pass
                try:
                    pen = pg.mkPen(fg, width=line_width)
                    self._curve = w.plot([], [], pen=pen)
                except Exception:
                    self._curve = w.plot([], [])

            self._plot_widget = w
            layout.addWidget(w, 1)
        else:
            placeholder = QtWidgets.QLabel("pyqtgraph not available", self)
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(placeholder, 1)

    # ----- API -----------------------------------------------------------
    def set_title(self, title: str) -> None:
        self._title_label.setText(title)

    def set_data(self, x: Sequence[float] | None, y: Sequence[float] | None) -> None:
        if self._curve is None or x is None or y is None:
            return
        try:
            self._curve.setData(list(x), list(y))
        except Exception:
            pass

    def clear(self) -> None:
        if self._curve is None:
            return
        try:
            self._curve.setData([], [])
        except Exception:
            pass
