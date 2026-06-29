"""FRET Line Generator — embeds the native ChiSurf model editors in docks.

Each mixture component is a real :class:`FitGroup` built with one of the TCSPC
widget-model classes (``GaussianModelWidget``, ``WormLikeChainModelWidget`` …).
The component's editor is the *same* rich widget used in the main fitting GUI —
donor ``LifetimeWidget`` (add/remove lifetimes, donor references), the FRET
parameter group (R0, τ_D0, κ², donor-only) and the distance editors with their
add/remove buttons.

The panels (components, editor, sweep, plots) live in a ChiSurf ``DockArea`` —
the same draggable/splittable dock system used by the fitting windows.

The user assembles a mixture of these live models, then sweeps either a model
parameter or a mixing fraction (linear or log spacing) to produce the FRET line.
"""

from __future__ import annotations

import importlib
import typing

import numpy as np
from qtpy import QtCore, QtWidgets

import chisurf as cs
import chisurf.core.data as _data
import chisurf.core.fitting.fit as _fit
from chisurf.gui.widgets.dock_area import DockArea
from chisurf.gui.widgets.models.model_editor import build_model_editor

from ..core.algorithms import compute_fret_line_for_models, sweep_targets_for_models

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c


# display name → "module:ClassName" of the TCSPC widget-model
WIDGET_MODELS: dict[str, str] = {
    "FRET: FD (Gaussian)": "chisurf.gui.widgets.models.tcspc:GaussianModelWidget",
    "FRET: FD (Worm-like chain)": "chisurf.gui.widgets.models.tcspc:WormLikeChainModelWidget",
    "FRET: FD (Discrete)": "chisurf.gui.widgets.models.tcspc:FRETrateModelWidget",
    "Lifetime": "chisurf.gui.widgets.models.tcspc:LifetimeModelWidget",
}

# dock page titles
_DOCK_COMPONENTS = "Components"
_DOCK_EDITOR = "Editor"
_DOCK_SWEEP = "Sweep"
_DOCK_PLOT_E = "FRET line"
_DOCK_PLOT_TAUX = "τ_X(τ_F)"


def _resolve_class(path: str):
    mod_name, cls_name = path.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)


def _new_fit_group(model_class) -> _fit.FitGroup:
    """Build a standalone FitGroup carrying a fresh widget-model."""
    x = np.linspace(0.0, 50.0, 200)
    dg = _data.ExperimentDataCurveGroup([_data.DataCurve(x=x, y=np.zeros_like(x))])
    return _fit.FitGroup(data=dg, model_class=model_class)


def _make_plot_widget(parent=None):
    try:
        import pyqtgraph as pg

        pw = pg.PlotWidget(parent=parent)
        pw.setBackground("w")
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setMinimumHeight(140)
        return pw, True
    except ImportError:
        lbl = QtWidgets.QLabel("(install pyqtgraph for inline plots)", parent=parent)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("color:#888; font-style:italic;")
        lbl.setMinimumHeight(50)
        return lbl, False


def _default_dock_layout() -> dict:
    """Sensible initial split: components/sweep | editor on top, plots below."""
    return {
        "version": 1,
        "root": {
            "type": "splitter",
            "orientation": "vertical",
            "sizes": [520, 300],
            "children": [
                {
                    "type": "splitter",
                    "orientation": "horizontal",
                    "sizes": [260, 520],
                    "children": [
                        {
                            "type": "tab",
                            "tabs": [
                                {"tab_name": _DOCK_COMPONENTS},
                                {"tab_name": _DOCK_SWEEP},
                            ],
                        },
                        {"type": "tab", "tabs": [{"tab_name": _DOCK_EDITOR}]},
                    ],
                },
                {
                    "type": "tab",
                    "tabs": [
                        {"tab_name": _DOCK_PLOT_E},
                        {"tab_name": _DOCK_PLOT_TAUX},
                    ],
                },
            ],
        },
    }


@persist_plugin_state("fret_line")
class FRETLineTool(QtWidgets.QWidget):
    """FRET line generator that embeds the native model editors in docks."""

    name = "FRETLineTool"

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        # Each component: {"label", "fit", "model", "editor", "weight"}
        self._components: list[dict] = []
        self._cur = -1
        self._result: dict | None = None
        self._loading = False
        self._build_ui()
        self._add_component()

    # ── construction ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self._dock = DockArea(self)
        self._dock.setNewTabButtonVisible(False)
        self._dock.addTab(self._make_components_panel(), _DOCK_COMPONENTS)
        self._dock.addTab(self._make_editor_panel(), _DOCK_EDITOR)
        self._dock.addTab(self._make_sweep_panel(), _DOCK_SWEEP)
        self._dock.addTab(self._make_plots()[0], _DOCK_PLOT_E)
        self._dock.addTab(self._plot_taux_container, _DOCK_PLOT_TAUX)
        root.addWidget(self._dock, 1)
        try:
            self._dock.set_layout_state(_default_dock_layout(), emit_change=False)
        except Exception:
            pass

        # persistent action bar (always visible regardless of dock arrangement)
        bar = QtWidgets.QHBoxLayout()
        self._compute_btn = QtWidgets.QPushButton("Compute")
        self._compute_btn.setDefault(True)
        self._save_btn = QtWidgets.QPushButton("Save CSV")
        self._save_btn.setEnabled(False)
        self._push_btn = QtWidgets.QPushButton("Push to ndxplorer")
        self._push_btn.setEnabled(False)
        bar.addWidget(self._compute_btn)
        bar.addWidget(self._save_btn)
        bar.addStretch()
        bar.addWidget(self._push_btn)
        root.addLayout(bar)

        self._compute_btn.clicked.connect(self._do_compute)
        self._save_btn.clicked.connect(self._on_save)
        self._push_btn.clicked.connect(self._on_push)

    def _make_components_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        self._comp_list = QtWidgets.QListWidget()
        self._comp_list.setToolTip("Models combined into the mixture")
        v.addWidget(self._comp_list, 1)

        cbtn = QtWidgets.QHBoxLayout()
        self._add_btn = QtWidgets.QPushButton("+ Add")
        self._rm_btn = QtWidgets.QPushButton("− Remove")
        cbtn.addWidget(self._add_btn)
        cbtn.addWidget(self._rm_btn)
        cbtn.addStretch()
        v.addLayout(cbtn)

        form = QtWidgets.QFormLayout()
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.addItems(list(WIDGET_MODELS.keys()))
        form.addRow("Model:", self._model_combo)
        self._weight_spin = QtWidgets.QDoubleSpinBox()
        self._weight_spin.setRange(0.0, 1e6)
        self._weight_spin.setDecimals(4)
        self._weight_spin.setValue(1.0)
        self._weight_spin.setToolTip("Initial mixing weight (normalized across components)")
        form.addRow("Weight:", self._weight_spin)
        v.addLayout(form)

        self._comp_list.currentRowChanged.connect(self._on_comp_selected)
        self._add_btn.clicked.connect(self._add_component)
        self._rm_btn.clicked.connect(self._remove_component)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        self._weight_spin.valueChanged.connect(self._on_weight_changed)
        return w

    def _make_editor_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        self._editor_scroll = QtWidgets.QScrollArea()
        self._editor_scroll.setWidgetResizable(True)
        self._editor_stack = QtWidgets.QStackedWidget()
        self._editor_scroll.setWidget(self._editor_stack)
        v.addWidget(self._editor_scroll)
        return w

    def _make_sweep_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)

        # searchable sweep-target selector
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Vary:"))
        self._sweep_combo = QtWidgets.QComboBox()
        self._sweep_combo.setEditable(True)
        self._sweep_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self._sweep_combo.setToolTip(
            "Parameter or mixing fraction to vary.\nType to filter (substring match)."
        )
        comp = self._sweep_combo.completer()
        comp.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        comp.setFilterMode(QtCore.Qt.MatchContains)
        comp.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        row.addWidget(self._sweep_combo, 1)
        self._show_all_check = QtWidgets.QCheckBox("all params")
        self._show_all_check.setToolTip(
            "Also list timing / IRF / background / anisotropy nuisance parameters"
        )
        row.addWidget(self._show_all_check)
        v.addLayout(row)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Min:"), 0, 0)
        self._min_spin = QtWidgets.QDoubleSpinBox()
        self._min_spin.setRange(-1e9, 1e9)
        self._min_spin.setDecimals(4)
        grid.addWidget(self._min_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Max:"), 0, 2)
        self._max_spin = QtWidgets.QDoubleSpinBox()
        self._max_spin.setRange(-1e9, 1e9)
        self._max_spin.setDecimals(4)
        self._max_spin.setValue(100.0)
        grid.addWidget(self._max_spin, 0, 3)
        self._log_check = QtWidgets.QCheckBox("log")
        self._log_check.setToolTip("Logarithmic spacing (endpoints must be > 0)")
        grid.addWidget(self._log_check, 0, 4)
        grid.addWidget(QtWidgets.QLabel("Points:"), 1, 0)
        self._n_pts_spin = QtWidgets.QSpinBox()
        self._n_pts_spin.setRange(2, 10000)
        self._n_pts_spin.setValue(100)
        grid.addWidget(self._n_pts_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel("τ_D0 (ns):"), 1, 2)
        self._tau_d0_spin = QtWidgets.QDoubleSpinBox()
        self._tau_d0_spin.setRange(0.0, 1000.0)
        self._tau_d0_spin.setDecimals(3)
        self._tau_d0_spin.setValue(4.0)
        self._tau_d0_spin.setToolTip(
            "Reference donor lifetime for E_FRET = 1 − τ_X / τ_D0.\n"
            "Set 0 to auto-detect from the first FRET component."
        )
        grid.addWidget(self._tau_d0_spin, 1, 3)
        v.addLayout(grid)
        v.addStretch()

        self._show_all_check.toggled.connect(lambda _=None: self._refresh_sweep_targets())
        return w

    def _make_plots(self):
        self._plot_e, self._has_pg = _make_plot_widget()
        self._plot_taux, _ = _make_plot_widget()
        self._plot_taux_container = self._plot_taux
        if self._has_pg:
            self._plot_e.setLabel("bottom", "τ_F (ns)")
            self._plot_e.setLabel("left", "E_FRET")
            self._plot_e.addLegend()
            self._plot_taux.setLabel("bottom", "τ_F (ns)")
            self._plot_taux.setLabel("left", "τ_X (ns)")
        return self._plot_e, self._plot_taux

    # ── component lifecycle ───────────────────────────────────────────

    def _build_component(self, label: str, weight: float = 1.0) -> dict:
        cls = _resolve_class(WIDGET_MODELS[label])
        fg = _new_fit_group(cls)
        model = fg.model
        editor = build_model_editor(model)  # the widget-model is its own editor
        return {"label": label, "fit": fg, "model": model, "editor": editor, "weight": weight}

    def _add_component(self) -> None:
        label = self._model_combo.currentText() or next(iter(WIDGET_MODELS))
        comp = self._build_component(label)
        self._components.append(comp)
        self._editor_stack.addWidget(comp["editor"])
        self._refresh_comp_list()
        self._comp_list.setCurrentRow(len(self._components) - 1)
        self._invalidate()

    def _remove_component(self) -> None:
        if len(self._components) <= 1:
            return
        idx = self._cur
        if not (0 <= idx < len(self._components)):
            return
        comp = self._components.pop(idx)
        self._editor_stack.removeWidget(comp["editor"])
        comp["editor"].deleteLater()
        self._refresh_comp_list()
        self._comp_list.setCurrentRow(min(idx, len(self._components) - 1))
        self._invalidate()

    def _refresh_comp_list(self) -> None:
        self._loading = True
        self._comp_list.clear()
        for i, c in enumerate(self._components):
            self._comp_list.addItem(f"C{i}: {c['label']}  (w={c['weight']:g})")
        self._loading = False
        self._rm_btn.setEnabled(len(self._components) > 1)
        self._refresh_sweep_targets()

    def _on_comp_selected(self, row: int) -> None:
        if self._loading or not (0 <= row < len(self._components)):
            return
        self._cur = row
        comp = self._components[row]
        self._editor_stack.setCurrentWidget(comp["editor"])
        self._loading = True
        self._model_combo.setCurrentText(comp["label"])
        self._weight_spin.setValue(comp["weight"])
        self._loading = False
        # Point the global current_fit at this component so the editor's
        # add/remove-component buttons (which dispatch to current_fit) act here.
        if getattr(cs, "cs", None) is not None:
            try:
                cs.cs.current_fit = comp["fit"]
            except Exception:
                pass

    def _on_model_changed(self, label: str) -> None:
        if self._loading or not (0 <= self._cur < len(self._components)):
            return
        old = self._components[self._cur]
        comp = self._build_component(label, old["weight"])
        self._editor_stack.removeWidget(old["editor"])
        old["editor"].deleteLater()
        self._components[self._cur] = comp
        self._editor_stack.insertWidget(self._cur, comp["editor"])
        self._editor_stack.setCurrentWidget(comp["editor"])
        self._refresh_comp_list()
        self._comp_list.setCurrentRow(self._cur)
        self._invalidate()

    def _on_weight_changed(self, val: float) -> None:
        if self._loading or not (0 <= self._cur < len(self._components)):
            return
        self._components[self._cur]["weight"] = val
        self._refresh_comp_list()
        self._comp_list.setCurrentRow(self._cur)
        self._invalidate()

    # ── sweep targets ─────────────────────────────────────────────────

    def _live_models(self) -> list:
        return [c["model"] for c in self._components]

    def _refresh_sweep_targets(self) -> None:
        prev = self._sweep_combo.currentData()
        try:
            targets = sweep_targets_for_models(
                self._live_models(),
                [c["label"] for c in self._components],
                relevant_only=not self._show_all_check.isChecked(),
            )
        except Exception:
            targets = []
        self._sweep_combo.blockSignals(True)
        self._sweep_combo.clear()
        for t in targets:
            self._sweep_combo.addItem(t["label"], t)
        if prev is not None:
            for i in range(self._sweep_combo.count()):
                d = self._sweep_combo.itemData(i)
                if (
                    d
                    and d.get("kind") == prev.get("kind")
                    and d.get("component") == prev.get("component")
                    and d.get("name") == prev.get("name")
                ):
                    self._sweep_combo.setCurrentIndex(i)
                    break
        self._sweep_combo.blockSignals(False)

    def _invalidate(self) -> None:
        self._result = None
        self._save_btn.setEnabled(False)
        self._push_btn.setEnabled(False)

    # ── compute ───────────────────────────────────────────────────────

    def _do_compute(self) -> None:
        self._refresh_sweep_targets()
        sweep = self._sweep_combo.currentData()
        if not sweep:
            QtWidgets.QMessageBox.information(self, "Sweep", "No sweep target selected.")
            return
        fractions = [c["weight"] for c in self._components]
        tau_d0 = self._tau_d0_spin.value() or None

        result = compute_fret_line_for_models(
            self._live_models(),
            sweep=sweep,
            param_min=self._min_spin.value(),
            param_max=self._max_spin.value(),
            n_points=self._n_pts_spin.value(),
            fractions=fractions,
            tau_d0=tau_d0,
            log_scale=self._log_check.isChecked(),
        )
        if not result.get("ok"):
            QtWidgets.QMessageBox.warning(self, "Compute error", result.get("error", "?"))
            return
        self._result = result["result"]
        self._update_plots(sweep["label"])
        self._save_btn.setEnabled(True)
        self._push_btn.setEnabled(True)

    def _update_plots(self, label: str) -> None:
        if self._result is None or not self._has_pg:
            return
        import pyqtgraph as pg

        r = self._result
        tau_f = np.asarray(r["tau_f"])
        tau_x = np.asarray(r["tau_x"])
        e_fret = np.asarray(r["e_fret"])
        for pw in (self._plot_e, self._plot_taux):
            pw.clear()
            try:
                pw.getPlotItem().legend.clear()
            except Exception:
                pass
        self._plot_e.plot(tau_f, e_fret, pen=pg.mkPen("#e05c00", width=2), name=label)
        self._plot_taux.plot(tau_f, tau_x, pen=pg.mkPen("#1f77b4", width=2), name=label)
        tau_max = float(self._tau_d0_spin.value()) or float(tau_f.max() if tau_f.size else 1.0)
        self._plot_taux.plot(
            [0.0, tau_max],
            [0.0, tau_max],
            pen=pg.mkPen("#aaaaaa", width=1, style=QtCore.Qt.DashLine),
        )

    # ── save / push ───────────────────────────────────────────────────

    def _on_save(self) -> None:
        if self._result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save FRET line", "fret_line.csv", "CSV (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        r = self._result
        sweep = self._sweep_combo.currentText()
        comps = "; ".join(
            f"C{i}={c['label']}(w={c['weight']:g})" for i, c in enumerate(self._components)
        )
        try:
            with open(path, "w") as fh:
                fh.write(f"# FRET Line  sweep={sweep}  log={self._log_check.isChecked()}\n")
                fh.write(f"# components: {comps}\n")
                fh.write("# parameter,tau_F_ns,tau_X_ns,E_FRET\n")
                for p, tf, tx, e in zip(r["parameter_values"], r["tau_f"], r["tau_x"], r["e_fret"]):
                    fh.write(f"{p:.8g},{tf:.8g},{tx:.8g},{e:.8g}\n")
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save error", str(exc))

    def _on_push(self) -> None:
        if self._result is None:
            return
        try:
            from ndxplorer.plotting.curve_overlay import CurveOverlayWidget
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self,
                "ndxplorer not found",
                "The ndxplorer package is not installed or not importable.",
            )
            return
        overlays = [
            w
            for w in QtWidgets.QApplication.allWidgets()
            if isinstance(w, CurveOverlayWidget) and w.isVisible()
        ]
        if not overlays:
            QtWidgets.QMessageBox.information(
                self, "Push to ndxplorer", "No visible ndxplorer Overlay panel found."
            )
            return
        tau_f = np.asarray(self._result["tau_f"]).copy()
        e_fret = np.asarray(self._result["e_fret"]).copy()
        label = f"FRET line — {self._sweep_combo.currentText()}"

        def _curve():
            return tau_f, e_fret

        for ov in overlays:
            ov.add_curve(_curve, is_function=True, base_name=label)
        QtWidgets.QMessageBox.information(
            self, "Pushed", f"Added '{label}' to {len(overlays)} ndxplorer panel(s)."
        )
