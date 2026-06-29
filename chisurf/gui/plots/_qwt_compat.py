"""Minimal pyqtgraph-backed shims for the small slice of the guiqwt API that a
couple of legacy chisurf plot widgets still used.

ChiSurf standardizes on pyqtgraph; this module lets ``global_tcspc`` and
``surfaceplot`` drop their ``guiqwt`` / ``guidata`` dependency (and the heavy
PythonQwt/Qwt C++ stack) without a full rewrite. It reproduces only the methods
those widgets actually call:

- ``CurveDialog`` / ``ImageDialog`` -> a ``pg.PlotWidget`` that also answers
  ``get_plot()`` (returning itself) plus ``add_item`` / ``do_autoscale`` /
  ``set_scales`` / ``set_titles`` / ``set_aspect_ratio``.
- ``make.curve`` / ``make.label`` / ``make.histogram`` / ``make.histogram2D`` /
  ``make.range`` -> lightweight adapter items with the ``set_data`` /
  ``set_hist_data`` / ``set_logscale`` / ``set_range`` / ``get_range`` methods
  the callers use.

It is intentionally not a general guiqwt emulation.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg


class _PgPlot(pg.PlotWidget):
    """A ``pg.PlotWidget`` exposing the handful of guiqwt plot methods used."""

    def __init__(self, *, aspect_locked: bool = False, **kwargs):
        super().__init__(**kwargs)
        if aspect_locked:
            self.setAspectLocked(True)

    # guiqwt: ``win.get_plot()`` returned the plot; here the widget is the plot.
    def get_plot(self) -> "_PgPlot":
        return self

    def add_item(self, item) -> None:
        if item is None:
            return
        attach = getattr(item, "attach", None)
        if callable(attach):
            attach(self)
        else:
            self.addItem(item)

    def do_autoscale(self, *args, **kwargs) -> None:
        self.enableAutoRange()
        self.autoRange()

    def set_scales(self, xscale: str = "lin", yscale: str = "lin") -> None:
        self.setLogMode(x=(xscale == "log"), y=(yscale == "log"))

    def set_titles(self, ylabel: str | None = None, xlabel: str | None = None, **kwargs) -> None:
        if ylabel:
            self.setLabel("left", str(ylabel))
        if xlabel:
            self.setLabel("bottom", str(xlabel))

    def set_aspect_ratio(self, lock: bool = False, **kwargs) -> None:
        self.setAspectLocked(bool(lock))


class _PgCurve:
    def __init__(self, color="w", linewidth: int = 2):
        self._item = pg.PlotDataItem(pen=pg.mkPen(color, width=linewidth))

    def attach(self, plot: _PgPlot) -> None:
        plot.addItem(self._item)

    def set_data(self, x, y) -> None:
        self._item.setData(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


class _PgLabel:
    """guiqwt legend-style label; rendered as the plot title (best effort)."""

    def __init__(self, text: str):
        self.text = str(text)

    def attach(self, plot: _PgPlot) -> None:
        try:
            if not plot.plotItem.titleLabel.text:
                plot.setTitle(self.text)
        except Exception:
            pass


class _PgHistogram:
    def __init__(self, color="w", bins: int = 50):
        self._pen = pg.mkPen(color)
        self._brush = pg.mkBrush(color)
        self._item = pg.PlotDataItem()
        self._log = False
        self._bins = bins

    def attach(self, plot: _PgPlot) -> None:
        plot.addItem(self._item)

    def set_logscale(self, log: bool) -> None:
        self._log = bool(log)

    def set_hist_data(self, data) -> None:
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            self._item.clear()
            return
        counts, edges = np.histogram(data, bins=self._bins)
        if self._log:
            counts = np.log10(counts.astype(float) + 1.0)
        # stepMode=True requires len(x) == len(y) + 1, which np.histogram gives.
        self._item.setData(edges, counts, stepMode=True, fillLevel=0,
                           pen=self._pen, brush=self._brush)


class _PgHistogram2D:
    """2D histogram backed by a ``pg.ImageItem``.

    The legacy caller currently leaves ``set_data`` commented out, so this is a
    structural placeholder that still honours the methods that are invoked.
    """

    def __init__(self, logscale: bool = True, bins=(50, 50)):
        self._item = pg.ImageItem()
        self._log = logscale
        self._bins = bins

    def attach(self, plot: _PgPlot) -> None:
        plot.addItem(self._item)

    def set_color_map(self, name: str) -> None:
        try:
            self._item.setColorMap(pg.colormap.get(name, source="matplotlib"))
        except Exception:
            pass

    def set_bins(self, nx, ny) -> None:
        self._bins = (int(nx), int(ny))

    def set_interpolation(self, *args, **kwargs) -> None:
        pass

    def set_data(self, x, y) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            return
        hist, _xe, _ye = np.histogram2d(x, y, bins=self._bins)
        if self._log:
            hist = np.log10(hist + 1.0)
        self._item.setImage(hist)


class _PgRange:
    def __init__(self, lower: float, upper: float):
        self._item = pg.LinearRegionItem(values=(lower, upper))

    def attach(self, plot: _PgPlot) -> None:
        plot.addItem(self._item)

    def set_range(self, lower: float, upper: float) -> None:
        self._item.setRegion((lower, upper))

    def get_range(self):
        return self._item.getRegion()


class _Make:
    """Drop-in for ``guiqwt.builder.make`` (only the used factory methods)."""

    @staticmethod
    def curve(x, y, color="w", linewidth: int = 2, **kwargs) -> _PgCurve:
        curve = _PgCurve(color=color, linewidth=linewidth)
        curve.set_data(x, y)
        return curve

    @staticmethod
    def label(text, *args, **kwargs) -> _PgLabel:
        return _PgLabel(text)

    @staticmethod
    def histogram(data, color="w", **kwargs) -> _PgHistogram:
        hist = _PgHistogram(color=color)
        data = np.asarray(data)
        if data.size > 0:
            hist.set_hist_data(data)
        return hist

    @staticmethod
    def histogram2D(x, y, logscale: bool = True, **kwargs) -> _PgHistogram2D:
        return _PgHistogram2D(logscale=logscale)

    @staticmethod
    def range(lower: float, upper: float) -> _PgRange:
        return _PgRange(lower, upper)


make = _Make()


def CurveDialog(*args, **kwargs) -> _PgPlot:  # noqa: N802 (guiqwt name)
    return _PgPlot()


def ImageDialog(*args, **kwargs) -> _PgPlot:  # noqa: N802 (guiqwt name)
    return _PgPlot(aspect_locked=False)
