"""Interactive phasor-plot calculator embeddable in the Calculators hub (PRD-56).

A data-free phasor plot: the model holds the frequency, reference lifetimes and the
FRET / two-component overlay settings; ``phasor.view.json`` (AutoForm) renders the
controls beside the phasor section, which draws the universal semicircle plus the
selected reference geometry. Overlay geometry is assembled by the shared toolkit
(:func:`chisurf.plugins.microscopy.img_pixel_phasor.analysis.build_overlays`), so the
calculator, the imaging plugin and the ``phasor.overlays`` RPC method stay in sync.
"""

from __future__ import annotations

import pathlib
from typing import Any

from qtpy import QtWidgets

_GUI_DIR = pathlib.Path(__file__).parent


class _PhasorCalcModel:
    """Backing model for the phasor calculator; fields declared in phasor.view.json."""

    #: Plot extent (slightly padded around the universal semicircle).
    PHASOR_G_RANGE = (-0.05, 1.05)
    PHASOR_S_RANGE = (-0.02, 0.62)

    def __init__(self) -> None:
        self.frequency = 80.0
        self.harmonic = 1
        self.taus = "0.5, 1, 2, 4, 8"
        self.show_grid = True
        self.show_ticks = True
        self.show_polar_grid = False
        self.show_fret = False
        self.tau_d0 = 4.0
        self.show_component = False
        self.g1, self.s1 = 0.80, 0.35
        self.g2, self.s2 = 0.30, 0.45
        # N-component mixing region / fraction-weighted mixture (uses c1/c2 above)
        self.show_mixing = False
        self.frac1 = 0.5
        # gating cursor outline
        self.show_cursor = False
        self.cursor_g, self.cursor_s = 0.55, 0.30
        self.cursor_radius = 0.05

    def view_spec(self):
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_GUI_DIR / "phasor.view.json")

    # -- helpers ----------------------------------------------------------------------
    def _tau_list(self) -> list[float]:
        out: list[float] = []
        for tok in str(self.taus).replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(float(tok))
            except ValueError:
                pass
        return out or [1.0]

    def _sets(self) -> list[str]:
        sets: list[str] = []
        if self.show_grid:
            sets.append("lifetime_grid")
        if self.show_ticks:
            sets.append("lifetime_ticks")
        if self.show_polar_grid:
            sets.append("polar_grid")
        if self.show_fret:
            sets.append("fret")
        if self.show_component:
            sets.append("component_line")
        if self.show_mixing:
            sets.append("components")
        if self.show_cursor:
            sets.append("cursor")
        return sets

    # -- AutoForm sources -------------------------------------------------------------
    def phasor_overlays(self) -> list[dict]:
        from chisurf.plugins.microscopy.img_pixel_phasor import analysis

        f1 = min(max(float(self.frac1), 0.0), 1.0)
        return analysis.build_overlays(
            frequency_mhz=float(self.frequency),
            harmonic=int(self.harmonic),
            sets=self._sets(),
            taus=self._tau_list(),
            c1=(self.g1, self.s1),
            c2=(self.g2, self.s2),
            tau_d0=float(self.tau_d0),
            components=[[self.g1, self.s1], [self.g2, self.s2]],
            fractions=[f1, 1.0 - f1],
            cursors=[
                {"center": [self.cursor_g, self.cursor_s],
                 "radius": float(self.cursor_radius), "name": "cursor"}
            ],
        )

    def results_html(self) -> str:
        from chisurf.plugins.microscopy.img_pixel_phasor import analysis

        freq = float(self.frequency) * int(self.harmonic)
        rows = "".join(
            f"<tr><td>{t:g}</td><td>{float(g):.3f}</td><td>{float(s):.3f}</td></tr>"
            for t in self._tau_list()
            for g, s in [analysis.lifetime_to_phasor(t, freq)]
        )
        return (
            f"<b>Effective f = {freq:g} MHz</b>"
            "<table><tr><th>&tau; (ns)</th><th>g</th><th>s</th></tr>"
            f"{rows}</table>"
        )


class PhasorCalculatorTool(QtWidgets.QMainWindow):
    """Interactive phasor plot; constructs with no required arguments (hub-embeddable)."""

    name = "Phasor calculator"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Phasor calculator")
        self.resize(780, 540)

        from chisurf.gui.autoform import AutoForm

        self._model = _PhasorCalcModel()
        self._form = AutoForm(self._model, self)
        self.setCentralWidget(self._form)
        self._build_toolbar()
        self._wire_refresh()
        self._form.refresh_plots()
        self._apply_statefulness()

    def _build_toolbar(self) -> None:
        """Add a compact toolbar with a Help button opening the modal help window."""
        toolbar = self.addToolBar("Phasor")
        toolbar.setObjectName("phasorCalcToolbar")
        toolbar.setMovable(False)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        self._help_btn = QtWidgets.QToolButton()
        self._help_btn.setText("❓")
        self._help_btn.setToolTip("What is a phasor plot? (help)")
        self._help_btn.clicked.connect(self._show_help)
        toolbar.addWidget(self._help_btn)

    def _show_help(self) -> None:
        """Open the modal phasor-plot help dialog."""
        from .help import PhasorHelpDialog

        PhasorHelpDialog(self).exec_()

    def _wire_refresh(self) -> None:
        """Redraw the phasor plot whenever any control commits a new value."""
        from chisurf.gui.autoform.sections.builtin import ToggleWidget, ValueWidget

        for vw in self._form.findChildren(ValueWidget):
            editor = getattr(vw, "editor", None)
            if editor is None:
                continue
            signal = getattr(editor, "editingFinished", None) or getattr(editor, "valueChanged", None)
            if signal is not None:
                signal.connect(self._on_change)
        for tw in self._form.findChildren(ToggleWidget):
            checkbox = getattr(tw, "checkbox", None)
            if checkbox is not None:
                checkbox.toggled.connect(self._on_change)

    def _on_change(self, *_args: Any) -> None:
        self._form.refresh_plots()

    def _apply_statefulness(self) -> None:
        try:
            from chisurf.core.plugin import load_manifest
            from chisurf.core.plugin.registry import apply_manifest_statefulness

            manifest = load_manifest(pathlib.Path(__file__).parents[1] / "manifest.json")
            if manifest is not None:
                apply_manifest_statefulness(self, manifest)
        except Exception:
            pass
