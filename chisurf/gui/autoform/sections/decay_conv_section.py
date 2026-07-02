"""AutoForm ``decay_conv`` section: live decay + IRF plot with a draggable range.

Plots the selected detector's data decay (semilog) and, if set, its IRF, with a
draggable convolution-range region. Dragging the region writes the range back to
the model (``set_conv_range``); the model's value fields stay in sync. Reads
``model.<target>()`` → ``{"data", "irf", "conv": (start, stop), "bg", "n"}``.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

from .registry import register_section

logger = logging.getLogger(__name__)


@register_section("decay_conv")
class DecayConvWidget(QtWidgets.QWidget):
    """Decay/IRF plot with a draggable convolution-range region + BG line."""

    AUTOFORM_REFRESH = True
    _autoform_expanding = True

    def __init__(self, model, target: str, **options):
        super().__init__()
        self._model = model
        self._target = target
        self._updating = False

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plot = pg.PlotWidget()
        self._plot.setLogMode(y=True)
        self._plot.setLabel("bottom", "micro-time channel")
        self._plot.setLabel("left", "counts")
        self._plot.addLegend(offset=(-10, 10))
        if options.get("title"):
            self._plot.setTitle(str(options["title"]))
        self._data_curve = self._plot.plot([], [], pen=pg.mkPen((0, 200, 255), width=1), name="data")
        self._irf_vv_curve = self._plot.plot([], [], pen=pg.mkPen((255, 80, 200), width=1), name="IRF VV")
        self._irf_vh_curve = self._plot.plot([], [], pen=pg.mkPen((255, 170, 60), width=1), name="IRF VH")
        # Convolution/fit window (blue) and the separate IRF window (green).
        self._region = pg.LinearRegionItem(brush=(80, 160, 255, 40))
        self._region.sigRegionChangeFinished.connect(self._on_region)
        self._plot.addItem(self._region)
        self._irf_region = pg.LinearRegionItem(
            brush=(80, 255, 140, 30), pen=pg.mkPen((80, 255, 140), width=1)
        )
        self._irf_region.sigRegionChangeFinished.connect(self._on_irf_region)
        self._plot.addItem(self._irf_region)
        # Background-estimation region (grey): mean data counts here → bg_vv/bg_vh.
        self._bg_region = pg.LinearRegionItem(
            brush=(180, 180, 180, 40), pen=pg.mkPen((180, 180, 180), width=1)
        )
        self._bg_region.sigRegionChangeFinished.connect(self._on_bg_region)
        self._plot.addItem(self._bg_region)
        layout.addWidget(self._plot, 1)
        self.refresh()

    def _payload(self):
        source = getattr(self._model, self._target, None) if self._target else None
        if not callable(source):
            return None
        try:
            return source()
        except Exception:
            logger.debug("decay_conv: source %r failed", self._target, exc_info=True)
            return None

    def refresh(self) -> None:
        """Redraw data + IRF and move the range/BG markers to the model's values."""
        payload = self._payload()
        if not payload:
            return
        self._updating = True
        try:
            data = np.asarray(payload["data"], dtype=float)
            x = np.arange(data.size)
            self._data_curve.setData(x, np.clip(data, 0.1, None))
            data_peak = max(float(data.max()), 1.0)
            for curve, key in ((self._irf_vv_curve, "irf_vv"), (self._irf_vh_curve, "irf_vh")):
                irf = payload.get(key)
                if irf is not None and len(irf):
                    irf = np.asarray(irf, dtype=float)
                    # normalised IRF (unit area) → scale its peak to the data for overlay
                    scale = data_peak / max(float(irf.max()), 1e-12)
                    curve.setData(np.arange(irf.size), np.clip(irf * scale, 0.1, None))
                else:
                    curve.setData([], [])
            start, stop = payload.get("conv", (0, data.size))
            self._region.setRegion((float(start), float(stop)))
            irf_start, irf_stop = payload.get("irf_range", (0, data.size))
            self._irf_region.setRegion((float(irf_start), float(irf_stop)))
            bg_start, bg_stop = payload.get("bg_range", (0, 0))
            self._bg_region.setRegion((float(bg_start), float(bg_stop)))
        finally:
            self._updating = False

    def _on_region(self) -> None:
        if self._updating:
            return
        start, stop = self._region.getRegion()
        setter = getattr(self._model, "set_conv_range", None)
        if callable(setter):
            setter(start, stop)

    def _on_irf_region(self) -> None:
        if self._updating:
            return
        start, stop = self._irf_region.getRegion()
        setter = getattr(self._model, "set_irf_range", None)
        if callable(setter):
            setter(start, stop)

    def _on_bg_region(self) -> None:
        if self._updating:
            return
        start, stop = self._bg_region.getRegion()
        setter = getattr(self._model, "set_bg_range", None)
        if callable(setter):
            setter(start, stop)
