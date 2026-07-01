"""General ``waterfall`` AutoForm section — a reusable waterfall image for any plugin.

A time-vs-(micro-time/lifetime) waterfall usable from any ``.view.json``.

Register a waterfall in a ``.view.json`` with::

    {"type": "custom", "key": "waterfall", "target": "waterfall_payload",
     "options": {"position_source": "waterfall_position"}}

``target`` names a zero-arg model method returning a *payload* dict (or ``None``
when there is nothing to show)::

    {
      "rgb_data": np.ndarray,        # (n_micro, n_macro, 3)
      "macro_t_s": np.ndarray,       # x axis (seconds)
      "micro_centers": np.ndarray,   # y axis (bins or lifetime)
      "n_macro_bins": int,
      "n_micro_bins": int,
      "title": str,                  # optional
      "x_label": str | None,         # optional axis relabel
      "log_x": bool,                 # optional
    }

The section opts into :meth:`AutoForm.refresh_plots` via ``AUTOFORM_REFRESH`` and
only re-uploads the image when the payload identity changes, so an optional
``position_source`` (a model attribute holding the current macro-bin position)
can drive a moving indicator cheaply during playback.
"""

from __future__ import annotations

import logging

from qtpy import QtWidgets

from chisurf.gui.widgets.waterfall_plot import WaterfallPlotWidget

from .registry import register_section

logger = logging.getLogger(__name__)


class WaterfallSectionWidget(QtWidgets.QWidget):
    """AutoForm section wrapping :class:`WaterfallPlotWidget`, driven by a source."""

    #: marker so :meth:`AutoForm.refresh_plots` re-reads this widget.
    AUTOFORM_REFRESH = True
    #: marker so a hosting panel gives this section the spare vertical space.
    _autoform_expanding = True

    def __init__(self, model, target: str, **options):
        super().__init__()
        self._model = model
        self._target = target
        self._position_source = options.get("position_source")
        self._show_position = bool(options.get("show_position", True))
        self._last_payload_id: int | None = None

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plot = WaterfallPlotWidget(self)
        self._plot.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        if options.get("title"):
            self._plot.set_title(str(options["title"]))
        layout.addWidget(self._plot, 1)
        self.refresh()

    @property
    def plot(self) -> WaterfallPlotWidget:
        """The underlying :class:`WaterfallPlotWidget` (for advanced use)."""
        return self._plot

    def _payload(self):
        source = getattr(self._model, self._target, None) if self._target else None
        if not callable(source):
            return None
        try:
            return source()
        except Exception as exc:  # pragma: no cover - source is model-defined
            logger.warning("WaterfallSection: source %r failed: %s", self._target, exc)
            return None

    def refresh(self) -> None:
        """Re-read the payload (only re-uploads on change) and update position."""
        payload = self._payload()
        if payload is None:
            return
        # Only re-upload the (expensive) image when the payload object changed.
        if id(payload) != self._last_payload_id:
            self._last_payload_id = id(payload)
            self._plot.set_waterfall_data(
                rgb_data=payload["rgb_data"],
                macro_t_s=payload["macro_t_s"],
                micro_centers=payload["micro_centers"],
                n_macro_bins=payload["n_macro_bins"],
                n_micro_bins=payload["n_micro_bins"],
            )
            if payload.get("title"):
                self._plot.set_title(payload["title"])
            if payload.get("x_label"):
                pw = self._plot.get_plot_widget()
                pw.setLabel("bottom", payload["x_label"])
                pw.setLogMode(x=bool(payload.get("log_x", False)), y=False)
        # Cheap per-refresh position update (e.g. during audio playback).
        if self._position_source:
            pos = getattr(self._model, self._position_source, None)
            if pos is None:
                self._plot.show_position_indicator(False)
            else:
                self._plot.show_position_indicator(self._show_position)
                self._plot.set_position(float(pos))


@register_section("waterfall")
def _waterfall_section_factory(model, target: str, **options):
    """Custom-section factory for the general waterfall image dock."""
    return WaterfallSectionWidget(model, target, **options)


__all__ = ["WaterfallSectionWidget"]
