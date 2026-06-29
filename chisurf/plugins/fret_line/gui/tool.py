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
parameter or a mixing fraction (linear or log spacing) to produce a FRET line.

Computation is *additive*: each press of "+ Add FRET line" snapshots the current
mixture/sweep and appends it to the collection, overlaying it on the plots with
its own colour. The ``FRET lines`` panel lists every computed line and lets the
user remove individual lines or clear them all. Save-CSV and push-to-ndxplorer
operate on the whole collection.
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
_DOCK_LINES = "FRET lines"
_DOCK_PLOT_E = "FRET line"
_DOCK_PLOT_TAUX = "τ_X(τ_F)"

# colour cycle for overlaid FRET lines (matplotlib-ish, distinct on white)
_PALETTE: tuple[str, ...] = (
    "#e05c00",
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
)


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
                                {"tab_name": _DOCK_LINES},
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
        # Accumulated, computed FRET lines (snapshots) — additive across
        # successive "Compute" presses. Each entry:
        #   {"name", "sweep_label", "result", "color", "log", "tau_d0",
        #    "components"}
        self._lines: list[dict] = []
        self._line_seq = 0
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
        self._dock.addTab(self._make_lines_panel(), _DOCK_LINES)
        self._dock.addTab(self._make_plots()[0], _DOCK_PLOT_E)
        self._dock.addTab(self._plot_taux_container, _DOCK_PLOT_TAUX)
        root.addWidget(self._dock, 1)
        try:
            self._dock.set_layout_state(_default_dock_layout(), emit_change=False)
        except Exception:
            pass

        # persistent action bar (always visible regardless of dock arrangement)
        bar = QtWidgets.QHBoxLayout()
        self._compute_btn = QtWidgets.QPushButton("+ Add FRET line")
        self._compute_btn.setDefault(True)
        self._compute_btn.setToolTip(
            "Compute the current mixture/sweep and add it as a new FRET line.\n"
            "Press repeatedly to overlay several lines."
        )
        self._save_btn = QtWidgets.QPushButton("Save CSV")
        self._save_btn.setToolTip("Save all computed FRET lines to a single CSV.")
        self._save_btn.setEnabled(False)
        self._push_btn = QtWidgets.QPushButton("Push to ndxplorer")
        self._push_btn.setToolTip("Overlay all computed FRET lines on ndxplorer panels.")
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

    def _make_lines_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(QtWidgets.QLabel("Computed FRET lines (overlaid on the plots):"))
        self._lines_list = QtWidgets.QListWidget()
        self._lines_list.setToolTip(
            "Each press of '+ Add FRET line' appends a snapshot here.\n"
            "Tick / untick the checkbox to show / hide a line on the plots.\n"
            "The text colour matches the plotted curve."
        )
        v.addWidget(self._lines_list, 1)

        vis = QtWidgets.QHBoxLayout()
        self._line_show_all_btn = QtWidgets.QPushButton("Show all")
        self._line_show_all_btn.setToolTip("Make every FRET line visible")
        self._line_hide_all_btn = QtWidgets.QPushButton("Hide all")
        self._line_hide_all_btn.setToolTip("Hide every FRET line (kept in the list)")
        vis.addWidget(self._line_show_all_btn)
        vis.addWidget(self._line_hide_all_btn)
        vis.addStretch()
        v.addLayout(vis)

        btns = QtWidgets.QHBoxLayout()
        self._line_rm_btn = QtWidgets.QPushButton("− Remove")
        self._line_rm_btn.setToolTip("Remove the selected FRET line")
        self._line_clear_btn = QtWidgets.QPushButton("Clear all")
        self._line_clear_btn.setToolTip("Remove all computed FRET lines")
        btns.addWidget(self._line_rm_btn)
        btns.addWidget(self._line_clear_btn)
        btns.addStretch()
        v.addLayout(btns)

        self._lines_list.itemChanged.connect(self._on_line_item_changed)
        self._line_show_all_btn.clicked.connect(lambda: self._set_all_visible(True))
        self._line_hide_all_btn.clicked.connect(lambda: self._set_all_visible(False))
        self._line_rm_btn.clicked.connect(self._remove_line)
        self._line_clear_btn.clicked.connect(self._clear_lines)
        self._refresh_lines_list()
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
        """Editing the mixture does NOT discard already-computed lines.

        Computed lines are independent snapshots, so the only thing to refresh
        when the editor changes is the enabled-state of the export buttons.
        """
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        has = bool(self._lines)
        self._save_btn.setEnabled(has)
        self._push_btn.setEnabled(has)

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

        self._line_seq += 1
        line = {
            "name": f"Line {self._line_seq}",
            "sweep_label": sweep["label"],
            "result": result["result"],
            "color": _PALETTE[(self._line_seq - 1) % len(_PALETTE)],
            "visible": True,
            "log": self._log_check.isChecked(),
            "tau_d0": tau_d0,
            "components": "; ".join(
                f"C{i}={c['label']}(w={c['weight']:g})" for i, c in enumerate(self._components)
            ),
        }
        self._lines.append(line)
        self._refresh_lines_list()
        self._replot_all()
        self._update_action_buttons()

    # ── computed-line management ───────────────────────────────────────

    def _refresh_lines_list(self) -> None:
        lst = getattr(self, "_lines_list", None)
        if lst is None:
            return
        from qtpy import QtGui

        lst.blockSignals(True)  # programmatic check-state changes must not replot
        lst.clear()
        for ln in self._lines:
            item = QtWidgets.QListWidgetItem(f"{ln['name']} · {ln['sweep_label']}")
            item.setForeground(QtGui.QColor(ln["color"]))
            item.setToolTip(ln["components"])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.Checked if ln.get("visible", True) else QtCore.Qt.Unchecked
            )
            lst.addItem(item)
        lst.blockSignals(False)
        has = bool(self._lines)
        self._line_rm_btn.setEnabled(has)
        self._line_clear_btn.setEnabled(has)
        self._line_show_all_btn.setEnabled(has)
        self._line_hide_all_btn.setEnabled(has)

    def _on_line_item_changed(self, item: "QtWidgets.QListWidgetItem") -> None:
        row = self._lines_list.row(item)
        if not (0 <= row < len(self._lines)):
            return
        self._lines[row]["visible"] = item.checkState() == QtCore.Qt.Checked
        self._replot_all()

    def _set_all_visible(self, visible: bool) -> None:
        if not self._lines:
            return
        for ln in self._lines:
            ln["visible"] = visible
        self._refresh_lines_list()
        self._replot_all()

    def _remove_line(self) -> None:
        if not self._lines:
            return
        idx = self._lines_list.currentRow()
        if not (0 <= idx < len(self._lines)):
            idx = len(self._lines) - 1
        self._lines.pop(idx)
        self._refresh_lines_list()
        self._replot_all()
        self._update_action_buttons()

    def _clear_lines(self) -> None:
        if not self._lines:
            return
        self._lines.clear()
        self._refresh_lines_list()
        self._replot_all()
        self._update_action_buttons()

    def _replot_all(self) -> None:
        """Redraw both plots from scratch, overlaying every computed line."""
        if not self._has_pg:
            return
        import pyqtgraph as pg

        for pw in (self._plot_e, self._plot_taux):
            pw.clear()
            try:
                pw.getPlotItem().legend.clear()
            except Exception:
                pass

        tau_max = 1.0
        for ln in self._lines:
            if not ln.get("visible", True):
                continue
            r = ln["result"]
            tau_f = np.asarray(r["tau_f"])
            tau_x = np.asarray(r["tau_x"])
            e_fret = np.asarray(r["e_fret"])
            pen = pg.mkPen(ln["color"], width=2)
            self._plot_e.plot(tau_f, e_fret, pen=pen, name=ln["name"])
            self._plot_taux.plot(tau_f, tau_x, pen=pen, name=ln["name"])
            if tau_f.size:
                tau_max = max(tau_max, float(tau_f.max()))

        # diagonal reference (τ_X = τ_F) on the τ_X plot
        ref = float(self._tau_d0_spin.value()) or tau_max
        self._plot_taux.plot(
            [0.0, ref],
            [0.0, ref],
            pen=pg.mkPen("#aaaaaa", width=1, style=QtCore.Qt.DashLine),
        )

    # ── save / push ───────────────────────────────────────────────────

    def _on_save(self) -> None:
        if not self._lines:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save FRET lines", "fret_lines.csv", "CSV (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            with open(path, "w") as fh:
                fh.write(f"# {len(self._lines)} FRET line(s)\n")
                fh.write("# line,sweep,log,components,parameter,tau_F_ns,tau_X_ns,E_FRET\n")
                for ln in self._lines:
                    r = ln["result"]
                    sweep = ln["sweep_label"]
                    comps = ln["components"].replace(",", ";")
                    for p, tf, tx, e in zip(
                        r["parameter_values"], r["tau_f"], r["tau_x"], r["e_fret"]
                    ):
                        fh.write(
                            f"{ln['name']},{sweep},{ln['log']},{comps},"
                            f"{p:.8g},{tf:.8g},{tx:.8g},{e:.8g}\n"
                        )
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Saved {len(self._lines)} line(s) to:\n{path}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save error", str(exc))

    def _on_push(self) -> None:
        if not self._lines:
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

        def _make_curve(line):
            tau_f = np.asarray(line["result"]["tau_f"]).copy()
            e_fret = np.asarray(line["result"]["e_fret"]).copy()

            def _curve():
                return tau_f, e_fret

            return _curve

        for ln in self._lines:
            label = f"FRET line — {ln['name']} · {ln['sweep_label']}"
            curve = _make_curve(ln)
            for ov in overlays:
                ov.add_curve(curve, is_function=True, base_name=label)
        QtWidgets.QMessageBox.information(
            self,
            "Pushed",
            f"Added {len(self._lines)} line(s) to {len(overlays)} ndxplorer panel(s).",
        )
