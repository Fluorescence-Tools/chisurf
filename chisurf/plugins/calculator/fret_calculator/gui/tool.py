"""Combined FRET / HomoFRET Calculator GUI.

Provides a tabbed QMainWindow with two calculator tabs, both backed by the
new backend services via :class:`FretCalculatorClient`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy import QtWidgets

from .client import FretCalculatorClient


class _FretTab(QtWidgets.QWidget):
    """HeteroFRET calculator tab."""

    def __init__(self, client: FretCalculatorClient, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._building = False
        self._init_ui()
        self._connect_signals()
        self._compute()

    def _init_ui(self) -> None:
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.spin_tau0 = QtWidgets.QDoubleSpinBox(self)
        self.spin_tau0.setRange(0.001, 9999.0)
        self.spin_tau0.setDecimals(4)
        self.spin_tau0.setSuffix(" ns")
        self.spin_tau0.setValue(4.0)
        layout.addRow("Lifetime (D0)", self.spin_tau0)

        self.spin_R0 = QtWidgets.QDoubleSpinBox(self)
        self.spin_R0.setRange(0.1, 999.0)
        self.spin_R0.setDecimals(2)
        self.spin_R0.setSuffix(" A")
        self.spin_R0.setValue(52.0)
        layout.addRow("Forster Radius", self.spin_R0)

        self.spin_tau = QtWidgets.QDoubleSpinBox(self)
        self.spin_tau.setRange(0.0, 9999.0)
        self.spin_tau.setDecimals(4)
        self.spin_tau.setSuffix(" ns")
        self.spin_tau.setValue(3.0)
        layout.addRow("Lifetime (DA)", self.spin_tau)

        self.spin_R = QtWidgets.QDoubleSpinBox(self)
        self.spin_R.setRange(0.1, 9999.0)
        self.spin_R.setDecimals(2)
        self.spin_R.setSuffix(" A")
        self.spin_R.setValue(50.0)
        layout.addRow("Distance (DA)", self.spin_R)

        self.spin_sigma = QtWidgets.QDoubleSpinBox(self)
        self.spin_sigma.setRange(0.0, 999.0)
        self.spin_sigma.setDecimals(2)
        self.spin_sigma.setSuffix(" A")
        self.spin_sigma.setValue(0.0)
        layout.addRow("Sigma", self.spin_sigma)

        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addRow(sep)

        self.spin_E = QtWidgets.QDoubleSpinBox(self)
        self.spin_E.setRange(0.0, 1.0)
        self.spin_E.setDecimals(6)
        self.spin_E.setValue(0.5)
        layout.addRow("Efficiency", self.spin_E)

        self.spin_kFRET = QtWidgets.QDoubleSpinBox(self)
        self.spin_kFRET.setRange(0.0, 9999.0)
        self.spin_kFRET.setDecimals(6)
        self.spin_kFRET.setValue(0.25)
        layout.addRow("kFRET", self.spin_kFRET)

    def _connect_signals(self) -> None:
        self.spin_tau0.editingFinished.connect(self._on_tau0_changed)
        self.spin_R0.editingFinished.connect(self._on_R0_changed)
        self.spin_tau.editingFinished.connect(self._on_tau_changed)
        self.spin_R.editingFinished.connect(self._on_R_changed)
        self.spin_sigma.editingFinished.connect(self._on_sigma_changed)
        self.spin_E.editingFinished.connect(self._on_E_changed)
        self.spin_kFRET.editingFinished.connect(self._on_kFRET_changed)

    # ── signal handlers ────────────────────────────────────────────

    def _block(self, flag: bool) -> None:
        for w in (
            self.spin_tau0, self.spin_R0, self.spin_tau,
            self.spin_R, self.spin_sigma, self.spin_E, self.spin_kFRET,
        ):
            w.blockSignals(flag)

    def _set_from_result(self, r: dict[str, float]) -> None:
        self._block(True)
        self.spin_R.setValue(r["R"])
        self.spin_E.setValue(r["E"])
        self.spin_tau.setValue(r["tau_DA"])
        self.spin_kFRET.setValue(r["kFRET"])
        self._block(False)

    def _compute(self) -> None:
        r = self._client.compute_fret(
            R=self.spin_R.value(),
            R0=self.spin_R0.value(),
            tau0=self.spin_tau0.value(),
            kappa2=0.667,
            sigma=self.spin_sigma.value(),
        )
        if r.get("ok"):
            self._set_from_result(r["result"])

    def _on_tau0_changed(self) -> None:
        self._compute()

    def _on_R0_changed(self) -> None:
        self._compute()

    def _on_tau_changed(self) -> None:
        if self._block:
            return
        self._block(True)
        r = self._client.compute_fret_from_lifetime(
            tau_DA=self.spin_tau.value(),
            R0=self.spin_R0.value(),
            tau0=self.spin_tau0.value(),
        )
        if r.get("ok"):
            self._set_from_result(r["result"])
        self._block(False)

    def _on_R_changed(self) -> None:
        self._compute()

    def _on_sigma_changed(self) -> None:
        self._compute()

    def _on_E_changed(self) -> None:
        self._block(True)
        r = self._client.compute_fret_from_efficiency(
            E=self.spin_E.value(),
            R0=self.spin_R0.value(),
            tau0=self.spin_tau0.value(),
        )
        if r.get("ok"):
            self._set_from_result(r["result"])
        self._block(False)

    def _on_kFRET_changed(self) -> None:
        self._block(True)
        r = self._client.compute_fret_from_rate(
            kFRET=self.spin_kFRET.value(),
            R0=self.spin_R0.value(),
            tau0=self.spin_tau0.value(),
        )
        if r.get("ok"):
            self._set_from_result(r["result"])
        self._block(False)


class _HomoFretTab(QtWidgets.QWidget):
    """HomoFRET calculator tab."""

    def __init__(self, client: FretCalculatorClient, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._building = False
        self._init_ui()
        self._connect_signals()
        self._compute()

    def _init_ui(self) -> None:
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.spin_tau0 = QtWidgets.QDoubleSpinBox(self)
        self.spin_tau0.setRange(0.001, 9999.0)
        self.spin_tau0.setDecimals(4)
        self.spin_tau0.setSuffix(" ns")
        self.spin_tau0.setValue(4.0)
        layout.addRow("tau0", self.spin_tau0)

        self.spin_R0 = QtWidgets.QDoubleSpinBox(self)
        self.spin_R0.setRange(0.1, 999.0)
        self.spin_R0.setDecimals(2)
        self.spin_R0.setSuffix(" A")
        self.spin_R0.setValue(52.0)
        layout.addRow("R0", self.spin_R0)

        self.spin_tRM = QtWidgets.QDoubleSpinBox(self)
        self.spin_tRM.setRange(0.001, 9999.0)
        self.spin_tRM.setDecimals(4)
        self.spin_tRM.setSuffix(" ns")
        self.spin_tRM.setValue(1.0)
        layout.addRow("t_RM", self.spin_tRM)

        self.spin_rho = QtWidgets.QDoubleSpinBox(self)
        self.spin_rho.setRange(0.001, 9999.0)
        self.spin_rho.setDecimals(4)
        self.spin_rho.setSuffix(" ns")
        self.spin_rho.setValue(2.0)
        layout.addRow("rho", self.spin_rho)

        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addRow(sep)

        self.spin_kHomo = QtWidgets.QDoubleSpinBox(self)
        self.spin_kHomo.setRange(0.0, 9999.0)
        self.spin_kHomo.setDecimals(6)
        self.spin_kHomo.setReadOnly(True)
        self.spin_kHomo.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        layout.addRow("k_homo", self.spin_kHomo)

        self.spin_Rhomo = QtWidgets.QDoubleSpinBox(self)
        self.spin_Rhomo.setRange(0.0, 9999.0)
        self.spin_Rhomo.setDecimals(2)
        self.spin_Rhomo.setSuffix(" A")
        self.spin_Rhomo.setValue(50.0)
        layout.addRow("R_DA", self.spin_Rhomo)

    def _connect_signals(self) -> None:
        self.spin_tau0.editingFinished.connect(self._compute)
        self.spin_R0.editingFinished.connect(self._compute)
        self.spin_tRM.editingFinished.connect(self._compute_forward)
        self.spin_rho.editingFinished.connect(self._compute)
        self.spin_Rhomo.editingFinished.connect(self._compute_backmap)

    def _compute(self) -> None:
        r = self._client.compute_homo_fret(
            t_RM=self.spin_tRM.value(),
            rho=self.spin_rho.value(),
            tau0=self.spin_tau0.value(),
            R0=self.spin_R0.value(),
        )
        if r.get("ok"):
            res = r["result"]
            self.spin_kHomo.setValue(res["k_homo"])
            rda = res["R_DA"]
            if np.isfinite(rda) and rda > 0:
                self.spin_Rhomo.setValue(rda)

    def _compute_forward(self) -> None:
        self.spin_kHomo.blockSignals(True)
        self.spin_Rhomo.blockSignals(True)
        self._compute()
        self.spin_kHomo.blockSignals(False)
        self.spin_Rhomo.blockSignals(False)

    def _compute_backmap(self) -> None:
        self.spin_kHomo.blockSignals(True)
        self.spin_tRM.blockSignals(True)
        r = self._client.homo_backmap(
            R_DA=self.spin_Rhomo.value(),
            R0=self.spin_R0.value(),
            tau0=self.spin_tau0.value(),
            rho=self.spin_rho.value(),
        )
        if r.get("ok"):
            res = r["result"]
            if np.isfinite(res["k_homo"]):
                self.spin_kHomo.setValue(res["k_homo"])
            if np.isfinite(res["t_RM"]) and res["t_RM"] > 0:
                self.spin_tRM.setValue(res["t_RM"])
        self.spin_kHomo.blockSignals(False)
        self.spin_tRM.blockSignals(False)


class FretCalculatorTool(QtWidgets.QMainWindow):
    """Combined FRET / HomoFRET Calculator.

    Appears in the Plugins menu as ``Main:Tools:FRET-Calculator``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle("FRET Calculator")
        self.resize(380, 320)
        self._client = FretCalculatorClient()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(_FretTab(self._client), "HeteroFRET")
        self.tabs.addTab(_HomoFretTab(self._client), "HomoFRET")
        main_layout.addWidget(self.tabs)
