"""K² Distribution Calculator widget.

Uses AutoForm (PRD-40) for a declarative, layout-driven GUI backed by
``Kappa2DistClient`` (which dispatches to core via in-process RPC).
Backward-compat attributes are maintained for
``kappa2_helpers.open_experimental_k2_dialog``.
"""

from __future__ import annotations

import pathlib
import typing

import numpy as np
from qtpy import QtCore, QtWidgets

from chisurf.core.dataspec import load_view_spec
from chisurf.core.fluorescence.anisotropy.kappa2 import s2delta

from .client import Kappa2DistClient
from .help_dialog import Kappa2DistHelpDialog

_GUI_DIR = pathlib.Path(__file__).parent


try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c


class _Kappa2DistModel:
    """Backing model for the k² distribution calculator.

    Input fields are declared in ``k2dist.view.json`` and bound directly via
    AutoForm's ``ValueSection`` / ``ChoiceSection`` / ``ToggleSection``.
    Call :meth:`compute` to run the selected model, then read the output fields.
    """

    def __init__(self) -> None:
        self.model_type = "cone"
        self.r_0 = 0.380
        self.r_Dinf = 0.050
        self.r_Ainf = 0.100
        self.r_ADinf = 0.005
        self.kappa2_true = 0.667
        self.fret_efficiency = 0.001
        self.step = 1.5
        self.n_bins = 131
        self.rAD_known = False

        self.k2_mean = 0.0
        self.k2_sd = 0.0
        self.Rapp_mean = 0.0
        self.RappSD = 0.0
        self.delta_deg = 0.0
        self._k2scale: np.ndarray | None = None
        self._k2hist: np.ndarray | None = None
        self._k2_values: np.ndarray | None = None

    def view_spec(self):
        return load_view_spec(_GUI_DIR / ".." / "k2dist.view.json")

    def kappa2_plot_series(self) -> list[dict]:
        if self._k2scale is None or self._k2hist is None:
            return []
        x = np.asarray(self._k2scale[1:], dtype=float)
        y = np.asarray(self._k2hist, dtype=float)
        return [{"x": x.tolist(), "y": y.tolist(), "name": "kappa2",
                 "color": "#1f77b4", "width": 2}]

    @property
    def SD2(self) -> float:
        ratio = self.r_Dinf / max(self.r_0, 1e-10)
        return -np.sqrt(max(ratio, 0.0))

    @property
    def SA2(self) -> float:
        ratio = self.r_Ainf / max(self.r_0, 1e-10)
        return np.sqrt(max(ratio, 0.0))

    @property
    def delta(self) -> float:
        _, d = s2delta(
            s2_donor=self.SD2,
            s2_acceptor=self.SA2,
            r_inf_AD=self.r_ADinf,
            r_0=self.r_0,
        )
        return d

    def compute(self, client: Kappa2DistClient) -> None:
        """Execute the computation via the RPC client."""
        result = client.compute(
            model_type=self.model_type,
            r_0=self.r_0,
            r_Dinf=self.r_Dinf,
            r_Ainf=self.r_Ainf,
            r_ADinf=self.r_ADinf,
            kappa2_true=self.kappa2_true,
            fret_efficiency=self.fret_efficiency,
            step=self.step,
            n_bins=self.n_bins,
            rAD_known=self.rAD_known,
        )
        if not result.get("ok"):
            self._k2scale = None
            self._k2hist = None
            self._k2_values = None
            return

        r = result["result"]
        self._k2scale = np.asarray(r["k2_scale"], dtype=float)
        self._k2hist = np.asarray(r["k2_hist"], dtype=float)
        self._k2_values = np.asarray(r["k2_values"], dtype=float)
        self.k2_mean = r["k2_mean"]
        self.k2_sd = r["k2_sd"]
        self.Rapp_mean = r["Rapp_mean"]
        self.RappSD = r["RappSD"]
        self.delta_deg = r["delta_deg"]


@persist_plugin_state("kappa2_dist")
class Kappa2Dist(QtWidgets.QWidget):
    """K² orientation-factor distribution calculator.

    Uses AutoForm (PRD-40) for a declarative, layout-driven GUI backed by
    a client-server architecture.

    Backward-compat attributes (``radioButton_2``, ``radioButton``,
    ``k2scale``, ``k2hist``, ``k2_mean``, ``onUpdateHist``) are
    maintained for ``kappa2_helpers.open_experimental_k2_dialog``.
    """

    name = "Kappa2Dist"

    def __init__(
            self,
            kappa2: float = 0.667,
            *args: typing.Any,
            **kwargs: typing.Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kappa2 = kappa2
        self._client = Kappa2DistClient()
        self._model = _Kappa2DistModel()
        self._model.kappa2_true = kappa2
        self._compute_timer = QtCore.QTimer(self)
        self._compute_timer.setSingleShot(True)
        self._compute_timer.timeout.connect(self._do_compute)
        self._build_ui()
        self._wire_backward_compat()
        self._connect_signals()
        self._do_compute()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        from chisurf.gui.autoform import AutoForm

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._form = AutoForm(self._model, parent=self)

        button_layout = QtWidgets.QHBoxLayout()
        self.helpButton = QtWidgets.QPushButton("?")
        self.helpButton.setFixedWidth(28)
        self.helpButton.setToolTip("Theory and background on κ²")
        self.pushButton = QtWidgets.QPushButton("Compute")
        self.saveButton = QtWidgets.QPushButton("Save")
        self.saveButton.setEnabled(False)
        button_layout.addWidget(self.helpButton)
        button_layout.addStretch()
        button_layout.addWidget(self.pushButton)
        button_layout.addWidget(self.saveButton)

        layout.addWidget(self._form)
        layout.addLayout(button_layout)

    def _wire_backward_compat(self) -> None:
        """Find the radio buttons inside the ChoiceWidget for old-style access."""
        from chisurf.gui.autoform.sections.builtin import ChoiceWidget

        self.radioButton_2 = None
        self.radioButton = None
        self.radioButton_iso = None
        for cw in self._form.findChildren(ChoiceWidget):
            sec = getattr(cw, "_section", None)
            if sec is None or getattr(sec, "attr", None) != "model_type":
                continue
            group = vars(cw).get("_group", None)
            if group is None or not hasattr(group, "buttons"):
                continue
            for btn in group.buttons():
                idx = group.id(btn)
                opt = sec.options[idx] if 0 <= idx < len(sec.options) else ""
                if opt == "cone":
                    self.radioButton_2 = btn
                elif opt == "diffusion":
                    self.radioButton = btn
                elif opt == "isotropic":
                    self.radioButton_iso = btn

    # ── signal wiring ────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        from chisurf.gui.autoform.sections.builtin import (
            ChoiceWidget,
            ToggleWidget,
            ValueWidget,
        )

        for vw in self._form.findChildren(ValueWidget):
            editor = getattr(vw, "editor", None)
            if editor is not None:
                editor.editingFinished.connect(self._schedule_compute)

        for tw in self._form.findChildren(ToggleWidget):
            cb = getattr(tw, "checkbox", None)
            if cb is not None:
                cb.toggled.connect(self._schedule_compute)

        for cw in self._form.findChildren(ChoiceWidget):
            group = vars(cw).get("_group")
            if group is not None and hasattr(group, "buttonClicked"):
                group.buttonClicked.connect(self._schedule_compute)

        self.pushButton.clicked.connect(self._do_compute)
        self.saveButton.clicked.connect(self._on_save)
        self.helpButton.clicked.connect(self._on_help)

    def _schedule_compute(self) -> None:
        self._compute_timer.start(50)

    def _do_compute(self) -> None:
        self._compute_timer.stop()
        self._model.compute(self._client)
        self._form.sync_fields()
        self._form.refresh_plots()
        if self._model._k2hist is not None and np.sum(self._model._k2hist) > 0:
            self.saveButton.setEnabled(True)
        else:
            self.saveButton.setEnabled(False)

    def onUpdateHist(self) -> None:
        """Backward-compat: called by ``open_experimental_k2_dialog``."""
        self._do_compute()

    # ── backward-compat properties ───────────────────────────────────

    @property
    def k2scale(self) -> np.ndarray | None:
        return self._model._k2scale

    @property
    def k2hist(self) -> np.ndarray | None:
        return self._model._k2hist

    @property
    def k2_mean(self) -> float:
        return self._model.k2_mean

    @k2_mean.setter
    def k2_mean(self, v: float) -> None:
        self._model.k2_mean = v

    # ── help ─────────────────────────────────────────────────────────

    def _on_help(self) -> None:
        dlg = Kappa2DistHelpDialog(self)
        dlg.exec_()

    # ── save ─────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Kappa2 Distribution",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        k2scale = self._model._k2scale
        k2hist = self._model._k2hist
        if k2scale is None or k2hist is None:
            QtWidgets.QMessageBox.warning(self, "Save", "Nothing to save.")
            return

        bins = k2scale[1:]
        header = [
            f"# Kappa2 Distribution",
            f"# Model: {self._model.model_type}",
            f"# SD2: {self._model.SD2:.6f}",
            f"# SA2: {self._model.SA2:.6f}",
            f"# Mean kappa2: {self._model.k2_mean:.6f}",
            f"# SD kappa2: {self._model.k2_sd:.6f}",
            f"# Assumed kappa2: {self._model.kappa2_true:.6f}",
            f"# Mean Rapp: {self._model.Rapp_mean:.6f}",
            f"# SD Rapp: {self._model.RappSD:.6f}",
            f"#",
            f"# kappa2,probability",
        ]
        try:
            with open(file_path, "w") as f:
                f.write("\n".join(header) + "\n")
                for x, y in zip(bins, k2hist):
                    f.write(f"{x:.6f},{y:.6f}\n")
            QtWidgets.QMessageBox.information(
                self, "Save Successful", f"Saved to:\n{file_path}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save Error", f"An error occurred:\n{exc}"
            )


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = Kappa2Dist()
    win.show()
    app.exec_()
