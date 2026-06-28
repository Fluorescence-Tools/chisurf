"""Reusable spectrum viewer widget.

Works standalone (imperative API) or driven by an AutoForm model.

Generic trace format
--------------------
The primary plotting method :meth:`plot_series` accepts a list of trace dicts::

    trace = {
        "name": "Cy3B [Absorption]",   # legend label
        "x": [400, 500, 600],          # x-axis values (e.g. wavelengths)
        "y": [0.0, 1.0, 0.0],          # y-axis values (e.g. intensities)
        "color": (230, 25, 75),        # optional -- RGB tuple
        "style": "solid",              # optional -- solid | dash | dot | dashdot
        "width": 2,                    # optional -- pen width
    }

    view.plot_series([trace, ...])

Convenience methods :meth:`display` and :meth:`display_multiple` accept the
legacy mfdb ``fluorophores.get`` response format (probe dicts with a ``"spectra"``
key) and internally convert to the generic trace format.

AutoForm declaration
--------------------
Declare in a view JSON::

    {"type": "custom", "key": "spectrum_view", "target": "spectra_data"}

The model attribute ``spectra_data`` must return a list of trace dicts, a
single ``fluorophores.get`` response dict, or a list of such dicts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from chisurf.gui.autoform.sections.registry import register_section

#: Default colour palette for traces that do not specify a colour.
_PROBE_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
]

#: Default colour per spectrum type (used by the convenience display methods).
_SPECTRUM_COLORS = {
    "absorption": (0, 100, 200),
    "emission": (200, 0, 0),
    "transmission": (0, 150, 0),
    "excitation": (0, 100, 200),
    "quantum_efficiency": (150, 0, 150),
    "responsivity": (0, 150, 150),
    "reflectance": (150, 150, 0),
}

#: Default line style per spectrum type (used by the convenience display methods).
_SPECTRUM_LABELS = {
    "absorption": "Absorption",
    "emission": "Emission",
    "transmission": "Transmission",
    "excitation": "Excitation",
    "quantum_efficiency": "Quantum Efficiency",
    "responsivity": "Responsivity",
    "reflectance": "Reflectance",
}

_LINE_STYLES = {
    "solid": QtCore.Qt.SolidLine,
    "dash": QtCore.Qt.DashLine,
    "dot": QtCore.Qt.DotLine,
    "dashdot": QtCore.Qt.DashDotLine,
}

_LINE_STYLE_CYCLE = [
    QtCore.Qt.SolidLine,
    QtCore.Qt.DashLine,
    QtCore.Qt.DotLine,
    QtCore.Qt.DashDotLine,
]


@register_section("spectrum_view")
class SpectrumView(QtWidgets.QWidget):
    """Displays one or more spectral traces (absorption, emission, …).

    Accepts data in a generic trace format via :meth:`plot_series`, or in the
    legacy mfdb ``fluorophores.get`` response format via :meth:`display` /
    :meth:`display_multiple`.

    Multi-trace mode
    ----------------
    When multiple traces are plotted without explicit colours, each trace
    receives a colour from a fixed palette.  The ``style`` field controls
    the line style (solid / dash / dot / dashdot).
    """

    AUTOFORM_REFRESH = True
    is_form_field = False

    def __init__(
        self,
        model: Any = None,
        target: str = "",
        parent: QtWidgets.QWidget | None = None,
        **options: Any,
    ):
        super().__init__(parent)
        self._model = model
        self._target = target

        self.setLayout(QtWidgets.QVBoxLayout(self))
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        _axis = "#b0b0b0"
        _text = "#d0d0d0"
        for _ax in ("left", "bottom"):
            axis = self.plot.getAxis(_ax)
            axis.setPen(pg.mkPen(_axis))
            axis.setTextPen(pg.mkPen(_text))
            axis.enableAutoSIPrefix(False)
        legend = self.plot.addLegend()
        try:
            legend.setLabelTextColor(_text)
        except Exception:
            pass
        self.plot.setLabel("bottom", "Wavelength", units="nm", color=_text)
        self.plot.setLabel("left", "Normalized Value", units="a.u.", color=_text)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.layout().addWidget(self.plot)

        if self._model is not None and self._target:
            self.refresh()

    # ------------------------------------------------------------------
    # AutoForm protocol
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read data from ``getattr(model, target)`` and redraw."""
        if self._model is None or not self._target:
            return
        data = getattr(self._model, self._target, None)
        if data is None:
            self.clear()
            return
        if isinstance(data, list):
            if data and isinstance(data[0], dict) and "spectra" in data[0]:
                self.display_multiple(data)
            elif data and isinstance(data[0], dict) and "x" in data[0]:
                self.plot_series(data)
            else:
                self.display_multiple(data)
        elif isinstance(data, dict):
            if "probes" in data:
                self.display_multiple(data["probes"])
            elif "spectra" in data:
                self.display(data)
            else:
                self.display(data)
        else:
            self.clear()

    # ------------------------------------------------------------------
    # Generic API
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all plotted curves."""
        self.plot.clear()

    def plot_series(self, traces: list[dict[str, Any]]) -> None:
        """Plot a list of trace dicts.

        Each trace supports: ``name``, ``x``, ``y``, ``color`` (optional RGB
        tuple), ``style`` (optional: solid/dash/dot/dashdot), ``width``
        (optional int, default 2).
        """
        self.clear()
        if not traces:
            self._show_empty()
            return

        for idx, tr in enumerate(traces):
            x = np.asarray(tr.get("x", []), dtype=float)
            y = np.asarray(tr.get("y", []), dtype=float)
            if len(x) == 0 or len(y) == 0:
                continue

            name = tr.get("name", f"Trace {idx + 1}")
            color = tr.get("color") or _PROBE_COLORS[idx % len(_PROBE_COLORS)]
            style_name = tr.get("style", "solid")
            style = _LINE_STYLES.get(style_name, QtCore.Qt.SolidLine)
            width = int(tr.get("width", 2))

            pen = pg.mkPen(color, width=width, style=style)
            self.plot.plot(x, y, pen=pen, name=name)

    def _show_empty(self) -> None:
        """Show a placeholder when no spectra are available."""
        self.plot.addItem(pg.TextItem(
            "No spectra data found.",
            color=(150, 150, 150),
            anchor=(0.5, 0.5),
        ))

    # ------------------------------------------------------------------
    # Legacy mfdb convenience API
    # ------------------------------------------------------------------

    def display(self, data: dict[str, Any]) -> None:
        """Display spectra for a single probe (mfdb ``fluorophores.get`` format).

        Parameters
        ----------
        data : dict
            Dict with ``"probe"``, ``"spectra"``, ``"optical_properties"`` keys.
        """
        self.clear()
        spectra = data.get("spectra", [])
        traces = self._probe_to_traces(data, spectra)
        if traces:
            self.plot_series(traces)
        else:
            self._show_empty()

    def display_multiple(self, probes_data: list[dict[str, Any]]) -> None:
        """Overlay spectra for multiple probes.

        Each probe gets a colour from the palette; spectrum types use
        different line styles.

        Parameters
        ----------
        probes_data : list[dict]
            List of ``fluorophores.get`` response dicts.
        """
        self.clear()
        all_traces: list[dict[str, Any]] = []
        for p_idx, data in enumerate(probes_data):
            probe = data.get("probe", {})
            name = probe.get("chromophore_name", probe.get("name", f"Probe {p_idx + 1}"))
            color = _PROBE_COLORS[p_idx % len(_PROBE_COLORS)]
            spectra = data.get("spectra", [])
            for s_idx, spec in enumerate(spectra):
                stype = spec.get("spectrum_type", "")
                wl = np.array(spec.get("wavelengths", []), dtype=float)
                iv = np.array(spec.get("intensity", []), dtype=float)
                if len(wl) == 0 or len(iv) == 0:
                    continue
                if np.nanmax(iv) > 0:
                    iv = iv / np.nanmax(iv)
                label = _SPECTRUM_LABELS.get(stype, stype.capitalize())
                all_traces.append({
                    "name": f"{name} [{label}]",
                    "x": wl,
                    "y": iv,
                    "color": color,
                    "style": ["solid", "dash", "dot", "dashdot"][s_idx % 4],
                    "width": 2,
                })
        if all_traces:
            self.plot_series(all_traces)
        else:
            self._show_empty()

    @staticmethod
    def _probe_to_traces(
        data: dict[str, Any],
        spectra: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert a single probe's spectra to a list of trace dicts."""
        traces: list[dict[str, Any]] = []
        for spec in spectra:
            stype = spec.get("spectrum_type", "")
            wl = np.array(spec.get("wavelengths", []), dtype=float)
            iv = np.array(spec.get("intensity", []), dtype=float)
            if len(wl) == 0 or len(iv) == 0:
                continue
            if np.nanmax(iv) > 0:
                iv = iv / np.nanmax(iv)
            color = _SPECTRUM_COLORS.get(stype, (100, 100, 100))
            label = _SPECTRUM_LABELS.get(stype, stype.capitalize())
            traces.append({
                "name": label,
                "x": wl,
                "y": iv,
                "color": color,
                "style": "solid",
                "width": 2,
            })
        return traces
