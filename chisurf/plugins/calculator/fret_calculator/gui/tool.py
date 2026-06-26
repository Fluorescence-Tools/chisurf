"""Combined FRET / HomoFRET Calculator GUI.

Provides a tabbed QMainWindow with two calculator tabs, both backed by the
new backend services via :class:`FretCalculatorClient`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy import QtCore, QtWidgets

from .client import FretCalculatorClient


import pathlib

_GUI_DIR = pathlib.Path(__file__).parent


def _distribution_series(mean, sigma, chi_active, xform=None, wide=False, trim=False):
    """Build Gaussian + chi distribution series for a declarative ``plot`` section.

    ``xform`` maps the sampled distances to the plotted x-axis (identity for the
    distance distribution, a rate / time transform for the derived plots). The
    active distribution is drawn solid, the other dashed. ``wide`` evaluates the
    distance distribution on a broad common grid (so a sharp peak at large R is
    still shown in context); ``trim`` drops the extreme weight tails (used for
    the rate/time transforms whose tails extend over many decades).
    """
    import numpy as np

    from chisurf.plugins.calculator.fret_calculator.core.algorithms import (
        distance_distribution,
    )

    sigma = max(float(sigma), 0.1)
    bins = None
    if wide:
        # Cover 0 well past the peak, independent of sigma, so a narrow
        # distribution centred at large R is still framed with context.
        r_max = max(2.0 * float(mean), float(mean) + 5.0 * sigma, 80.0)
        bins = np.linspace(0.0, r_max, 400)

    out = []
    for kind, color in (("gaussian", "#1f77b4"), ("chi", "#d62728")):
        r, w = distance_distribution(mean, sigma, kind, bins=bins)
        x = np.asarray(xform(r) if xform is not None else r, dtype=float)
        w = np.asarray(w, dtype=float)
        order = np.argsort(x)
        x, w = x[order], w[order]
        if trim and w.sum() > 0:
            cw = np.cumsum(w) / w.sum()
            keep = (cw >= 0.005) & (cw <= 0.995)
            if keep.any():
                x, w = x[keep], w[keep]
        active = chi_active if kind == "chi" else not chi_active
        out.append({
            "x": x, "y": w, "name": "chi" if kind == "chi" else "Gaussian",
            "color": color, "width": 2 if active else 1,
            "style": "solid" if active else "dash",
        })
    return out


class _FretModel:
    """Backing model for the DA-FRET tab; fields are declared in fret.view.json."""

    def __init__(self) -> None:
        self.tau0 = 4.0
        self.R0 = 52.0
        self.tau = 3.0
        self.R = 50.0
        self.sigma = 6.0
        self.use_chi = False
        self.E = 0.5
        self.kFRET = 0.25

    def view_spec(self):
        from chisurf.core.dataspec import load_view_spec
        return load_view_spec(_GUI_DIR / "fret.view.json")

    def _rate(self, r):
        from chisurf.plugins.calculator.fret_calculator.core.algorithms import (
            distance_to_fret_rate_constant,
        )
        return distance_to_fret_rate_constant(r, self.R0, self.tau0, 0.667)

    def distance_plot_series(self) -> list[dict]:
        return _distribution_series(self.R, self.sigma, bool(self.use_chi), wide=True)

    def rate_plot_series(self) -> list[dict]:
        # FRET-rate-constant distribution induced by the distance distribution.
        return _distribution_series(
            self.R, self.sigma, bool(self.use_chi), xform=self._rate, trim=True,
        )


class _HomoFretModel:
    """Backing model for the homo-FRET tab; fields declared in homofret.view.json."""

    def __init__(self) -> None:
        self.tau0 = 2.3
        self.R0 = 52.0
        self.t_RM = 1.0
        self.rho = 16.0
        self.sigma = 6.0
        self.use_chi = False
        self.k_homo = 0.0
        self.R_DA = 50.0

    def view_spec(self):
        from chisurf.core.dataspec import load_view_spec
        return load_view_spec(_GUI_DIR / "homofret.view.json")

    def _rate(self, r):
        from chisurf.plugins.calculator.fret_calculator.core.algorithms import (
            distance_to_fret_rate_constant,
        )
        return distance_to_fret_rate_constant(r, self.R0, self.tau0, 0.667)

    def distance_plot_series(self) -> list[dict]:
        return _distribution_series(self.R_DA, self.sigma, bool(self.use_chi), wide=True)

    def aniso_time_plot_series(self) -> list[dict]:
        # Characteristic anisotropy-decay time: rotational depolarisation (rho)
        # in parallel with energy-transfer depolarisation (2·k_FRET).
        import numpy as np

        rho = max(float(self.rho), 1e-9)

        def _tau(r):
            return 1.0 / (1.0 / rho + 2.0 * np.asarray(self._rate(r), dtype=float))

        return _distribution_series(
            self.R_DA, self.sigma, bool(self.use_chi),
            xform=_tau, trim=True,
        )


def _build_autoform(model, parent):
    """Render ``model`` via AutoForm and return ``(form, editors, toggles)``.

    ``editors`` maps attr → live QSpinBox/QDoubleSpinBox/QLineEdit and ``toggles``
    maps attr → QCheckBox, so existing compute code can keep using the widgets
    directly (``.value()`` / ``.isChecked()``).
    """
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.builtin import ValueWidget, ToggleWidget

    form = AutoForm(model, parent=parent)
    editors: dict[str, QtWidgets.QWidget] = {}
    for vw in form.findChildren(ValueWidget):
        attr = getattr(getattr(vw, "_section", None), "attr", None)
        if attr:
            editors[attr] = vw.editor
    toggles: dict[str, QtWidgets.QWidget] = {}
    for tw in form.findChildren(ToggleWidget):
        attr = getattr(getattr(tw, "_section", None), "attr", None)
        if attr:
            toggles[attr] = tw.checkbox
    return form, editors, toggles


def _grid_label(text: str, tip: str = "") -> QtWidgets.QLabel:
    """Right-aligned compact label whose detail lives in the tooltip."""
    lbl = QtWidgets.QLabel(text)
    lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    if tip:
        lbl.setToolTip(tip)
    return lbl


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
        # Fields, layout AND the distance-distribution plot are declared in
        # fret.view.json and rendered by AutoForm; we grab the live editor
        # widgets so the compute code below can keep using ``self.spin_*``.
        self._model = _FretModel()
        form, ed, tg = _build_autoform(self._model, self)
        self._form = form
        self.spin_tau0 = ed["tau0"]
        self.spin_R0 = ed["R0"]
        self.spin_tau = ed["tau"]
        self.spin_R = ed["R"]
        self.spin_sigma = ed["sigma"]
        self.spin_E = ed["E"]
        self.spin_kFRET = ed["kFRET"]
        self.check_chi = tg["use_chi"]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(form)

    def _connect_signals(self) -> None:
        self.spin_tau0.editingFinished.connect(self._on_tau0_changed)
        self.spin_R0.editingFinished.connect(self._on_R0_changed)
        self.spin_tau.editingFinished.connect(self._on_tau_changed)
        self.spin_R.editingFinished.connect(self._on_R_changed)
        self.spin_sigma.editingFinished.connect(self._on_sigma_changed)
        self.spin_E.editingFinished.connect(self._on_E_changed)
        self.spin_kFRET.editingFinished.connect(self._on_kFRET_changed)
        self.check_chi.toggled.connect(self._on_distribution_changed)
        # Any parameter edit refreshes the distribution plots (the specific
        # handler runs first and updates the fields, then this syncs+redraws).
        for w in (
            self.spin_tau0, self.spin_R0, self.spin_tau,
            self.spin_R, self.spin_sigma, self.spin_E, self.spin_kFRET,
        ):
            w.editingFinished.connect(self._update_distribution_plot)

    def _distribution(self) -> str:
        return "chi" if self.check_chi.isChecked() else "gaussian"

    def _on_distribution_changed(self) -> None:
        self._compute()

    def _sync_model(self) -> None:
        """Copy the displayed values into the model so the plots read them.

        ``_set_from_result`` blocks the editors' signals while updating them, so
        AutoForm's own commit does not fire; this keeps the model authoritative.
        """
        self._model.tau0 = self.spin_tau0.value()
        self._model.R0 = self.spin_R0.value()
        self._model.tau = self.spin_tau.value()
        self._model.R = self.spin_R.value()
        self._model.sigma = self.spin_sigma.value()
        self._model.E = self.spin_E.value()
        self._model.kFRET = self.spin_kFRET.value()
        self._model.use_chi = self.check_chi.isChecked()

    def _update_distribution_plot(self) -> None:
        try:
            self._sync_model()
            self._form.refresh_plots()
        except Exception:
            pass

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
            distribution=self._distribution(),
        )
        if r.get("ok"):
            self._set_from_result(r["result"])
        self._update_distribution_plot()

    def _on_tau0_changed(self) -> None:
        self._compute()

    def _on_R0_changed(self) -> None:
        self._compute()

    def _on_tau_changed(self) -> None:
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
        # Fields/layout AND the distance-distribution plot are declared in
        # homofret.view.json and rendered by AutoForm.
        self._model = _HomoFretModel()
        form, ed, tg = _build_autoform(self._model, self)
        self._form = form
        self.spin_tau0 = ed["tau0"]
        self.spin_R0 = ed["R0"]
        self.spin_tRM = ed["t_RM"]
        self.spin_rho = ed["rho"]
        self.spin_sigma = ed["sigma"]
        self.spin_kHomo = ed["k_homo"]
        self.spin_Rhomo = ed["R_DA"]
        self.check_chi = tg["use_chi"]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(form)

    def _connect_signals(self) -> None:
        self.spin_tau0.editingFinished.connect(self._compute)
        self.spin_R0.editingFinished.connect(self._compute)
        self.spin_tRM.editingFinished.connect(self._compute_forward)
        self.spin_rho.editingFinished.connect(self._compute)
        self.spin_Rhomo.editingFinished.connect(self._compute_backmap)
        self.check_chi.toggled.connect(self._update_distribution_plot)
        # Any parameter edit refreshes the distribution plots.
        for w in (
            self.spin_tau0, self.spin_R0, self.spin_tRM,
            self.spin_rho, self.spin_Rhomo, self.spin_sigma,
        ):
            w.editingFinished.connect(self._update_distribution_plot)

    def _sync_model(self) -> None:
        self._model.tau0 = self.spin_tau0.value()
        self._model.R0 = self.spin_R0.value()
        self._model.t_RM = self.spin_tRM.value()
        self._model.rho = self.spin_rho.value()
        self._model.sigma = self.spin_sigma.value()
        self._model.k_homo = self.spin_kHomo.value()
        self._model.R_DA = self.spin_Rhomo.value()
        self._model.use_chi = self.check_chi.isChecked()

    def _update_distribution_plot(self) -> None:
        try:
            self._sync_model()
            self._form.refresh_plots()
        except Exception:
            pass

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
        self._update_distribution_plot()

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

        # Remember window position/size across sessions. The manifest declares
        # window statefulness; wire it here too so it applies however the plugin
        # is launched (the helper is idempotent).
        try:
            import pathlib

            from chisurf.core.plugin import load_manifest
            from chisurf.core.plugin.registry import apply_manifest_statefulness

            _manifest = load_manifest(
                pathlib.Path(__file__).parents[1] / "manifest.json"
            )
            if _manifest is not None:
                apply_manifest_statefulness(self, _manifest)
        except Exception:
            pass
