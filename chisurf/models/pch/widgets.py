from __future__ import annotations

from typing import Any

import numpy as np
from qtpy import QtWidgets, QtGui

import chisurf
import chisurf.plots
from chisurf.models.model import ModelCurve, ModelWidget
from chisurf.fitting.parameter import FittingParameter


def _compute_p1(k_vals: np.ndarray, brightness: float, x_vals: np.ndarray, dx: float) -> np.ndarray:
    n = k_vals.shape[0]
    p1 = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        k = int(k_vals[i])
        if k <= 0:
            continue
        fact = 1.0
        for j in range(1, k + 1):
            fact *= j
        total = 0.0
        for xi in x_vals:
            exp_term = np.exp(-2.0 * xi * xi)
            lam = brightness * exp_term
            total += (lam**k / fact) * np.exp(-lam)
        p1[i] = total * dx
    s = p1[1:].sum()
    p1[0] = max(0.0, 1.0 - s)
    return p1


def _pch_single_species(k_vals: np.ndarray, brightness: float) -> np.ndarray:
    x_vals = np.linspace(0.0, 5.0, 1000)
    dx = x_vals[1] - x_vals[0]
    return _compute_p1(k_vals, float(brightness), x_vals, float(dx))


def _pch_open_system(k_vals: np.ndarray, brightness: float, avgN: float, maxN: int = 30) -> np.ndarray:
    from scipy.stats import poisson  # type: ignore[import]

    p1 = _pch_single_species(k_vals, brightness)
    length = k_vals.shape[0]
    pk_tot = np.zeros(length, dtype=float)
    avgN = float(max(avgN, 0.0))
    for N in range(maxN + 1):
        w = poisson.pmf(N, avgN)
        if w <= 0.0:
            continue
        if N == 0:
            base = np.zeros(length, dtype=float)
            base[0] = 1.0
        else:
            base = p1.copy()
            for _ in range(1, N):
                out = np.zeros(length, dtype=float)
                for i in range(length):
                    for j in range(length - i):
                        out[i + j] += base[i] * p1[j]
                base = out
        pk_tot += w * base
    return pk_tot


def _pch_mixture(k_vals: np.ndarray, epsilons: np.ndarray, avgNs: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve  # type: ignore[import]

    k_vals = np.asarray(k_vals, dtype=float)
    eps = np.asarray(epsilons, dtype=float)
    Ns = np.asarray(avgNs, dtype=float)
    if eps.size == 0 or Ns.size == 0:
        return np.ones_like(k_vals, dtype=float)

    pk = np.zeros_like(k_vals, dtype=float)
    pk[0] = 1.0
    for e, n in zip(eps, Ns):
        if n <= 0.0 or e <= 0.0:
            continue
        pj = _pch_open_system(k_vals, float(e), float(n))
        pj = np.asarray(pj, dtype=float)
        if pj.shape != pk.shape:
            m = min(pk.size, pj.size)
            pj = pj[:m]
            pk = pk[:m]
        pk = fftconvolve(pk, pj)[: pk.size]
    if not np.any(np.isfinite(pk)):
        pk = np.ones_like(k_vals, dtype=float)
    return pk


class PchMultiComponentModel(ModelCurve):

    name = "PCH multi-component"

    def __init__(self, fit: chisurf.fitting.fit.Fit, *args: Any, **kwargs: Any) -> None:  # type: ignore[name-defined]
        super().__init__(fit, *args, **kwargs)

        self._eps1 = FittingParameter(
            name="eps1",
            value=2.0,
            lb=0.0,
            ub=1.0e6,
            bounds_on=False,
            fixed=False,
            registry_id="pch.eps1",
        )
        self._N1 = FittingParameter(
            name="N1",
            value=3.0,
            lb=0.0,
            ub=1.0e6,
            bounds_on=False,
            fixed=False,
            registry_id="pch.N1",
        )
        self._eps2 = FittingParameter(
            name="eps2",
            value=0.0,
            lb=0.0,
            ub=1.0e6,
            bounds_on=False,
            fixed=False,
            registry_id="pch.eps2",
        )
        self._N2 = FittingParameter(
            name="N2",
            value=0.0,
            lb=0.0,
            ub=1.0e6,
            bounds_on=False,
            fixed=False,
            registry_id="pch.N2",
        )
        self._eps3 = FittingParameter(
            name="eps3",
            value=0.0,
            lb=0.0,
            ub=1.0e6,
            bounds_on=False,
            fixed=False,
            registry_id="pch.eps3",
        )
        self._N3 = FittingParameter(
            name="N3",
            value=0.0,
            lb=0.0,
            ub=1.0e6,
            bounds_on=False,
            fixed=False,
            registry_id="pch.N3",
        )

        try:
            self.find_parameters()
        except Exception:
            pass
        self.n_components = 1

    def update_model(self, **kwargs: Any) -> None:  # type: ignore[override]
        fit = getattr(self.fit, "selected_fit", self.fit)
        data = getattr(fit, "data", None)

        meta = getattr(data, "meta_data", {}) or {}
        pch_meta = meta.get("pch", {}) or {}

        k_vals = pch_meta.get("k_vals", getattr(data, "x", None))
        try:
            k = np.asarray(k_vals, dtype=float)
        except Exception:
            k = np.array([], dtype=float)
        if k.size == 0:
            y_model = np.zeros_like(getattr(data, "y", np.zeros(0, dtype=float)), dtype=float)
            try:
                self.x = np.arange(y_model.size, dtype=float)
                self.y = y_model
            except Exception:
                pass
            return

        eps = np.array([
            float(self._eps1.value),
            float(self._eps2.value),
            float(self._eps3.value),
        ], dtype=float)
        Ns = np.array([
            float(self._N1.value),
            float(self._N2.value),
            float(self._N3.value),
        ], dtype=float)

        mask = (eps > 0.0) & (Ns > 0.0)
        if not np.any(mask):
            y_model = np.zeros_like(k, dtype=float)
        else:
            try:
                pk = _pch_mixture(k, eps[mask], Ns[mask])
                pk = np.asarray(pk, dtype=float)
            except Exception:
                pk = np.zeros_like(k, dtype=float)
            if pk.size != k.size:
                m = min(pk.size, k.size)
                pk = pk[:m]
                k = k[:m]
            s = float(np.sum(pk))
            if s > 0.0 and np.isfinite(s):
                pk = pk / s
            y_model = pk

        data_x = getattr(data, "x", None)
        try:
            x_data = np.asarray(data_x, dtype=float)
        except Exception:
            x_data = None
        if x_data is not None and getattr(x_data, "size", 0) == y_model.size:
            self.x = x_data
        else:
            self.x = np.asarray(k, dtype=float)

        try:
            self.y = y_model
        except Exception:
            pass

    def add_component(self) -> None:
        try:
            n = int(getattr(self, "n_components", 1) or 1)
        except Exception:
            n = 1
        if n >= 3:
            return
        n += 1
        self.n_components = n
        try:
            self._set_component_defaults(n)
        except Exception:
            pass

    def remove_component(self) -> None:
        try:
            n = int(getattr(self, "n_components", 1) or 1)
        except Exception:
            n = 1
        if n <= 1:
            return
        try:
            self._clear_component(n)
        except Exception:
            pass
        n -= 1
        self.n_components = n

    def _set_component_defaults(self, idx: int) -> None:
        try:
            if idx == 1:
                p_eps, p_N = self._eps1, self._N1
            elif idx == 2:
                p_eps, p_N = self._eps2, self._N2
            elif idx == 3:
                p_eps, p_N = self._eps3, self._N3
            else:
                return
            if float(p_eps.value) <= 0.0:
                p_eps.value = 2.0
            if float(p_N.value) <= 0.0:
                p_N.value = 3.0
            p_eps.fixed = False
            p_N.fixed = False
        except Exception:
            pass

    def _clear_component(self, idx: int) -> None:
        try:
            if idx == 1:
                p_eps, p_N = self._eps1, self._N1
            elif idx == 2:
                p_eps, p_N = self._eps2, self._N2
            elif idx == 3:
                p_eps, p_N = self._eps3, self._N3
            else:
                return
            p_eps.value = 0.0
            p_N.value = 0.0
        except Exception:
            pass


class PchMultiComponentModelWidget(ModelWidget, PchMultiComponentModel):

    try:
        plot_classes = [
            (
                chisurf.plots.LinePlot,
                {
                    "scale_x": "lin",
                    "d_scaley": "log",
                    "r_scaley": "lin",
                    "x_label": "k",
                    "y_label": "P(k)",
                    "curve_styles": {
                        "data": {
                            "symbol": "o",
                            "symbol_size": 6,
                            "no_line": True,
                        },
                    },
                },
            ),
            (chisurf.plots.ResidualPlot, {}),
            (chisurf.plots.FitInfo, {}),
            (chisurf.plots.ParameterScanPlot, {}),
        ]
    except Exception:
        plot_classes = []

    name = "PCH multi-component"

    def __init__(
        self,
        fit: chisurf.fitting.fit.FitGroup,
        icon: QtGui.QIcon | None = None,
        **kwargs: Any,
    ) -> None:
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/fcs.png")
        super().__init__(fit=fit, icon=icon, **kwargs)

        try:
            from chisurf.gui.widgets.fitting.widgets import (
                make_fitting_parameter_widget,
            )
        except Exception:
            make_fitting_parameter_widget = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._param_widget = None
        self._gb = []
        self._make_fitting_parameter_widget = make_fitting_parameter_widget

        components_box = QtWidgets.QGroupBox()
        components_box.setTitle("PCH components")
        components_layout = QtWidgets.QVBoxLayout()
        components_layout.setContentsMargins(0, 0, 0, 0)
        components_layout.setSpacing(0)
        components_box.setLayout(components_layout)
        layout.addWidget(components_box)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(0)
        self.button_add_component = QtWidgets.QPushButton("add", self)
        self.button_remove_component = QtWidgets.QPushButton("del", self)
        btn_layout.addWidget(self.button_add_component)
        btn_layout.addWidget(self.button_remove_component)
        components_layout.addLayout(btn_layout)

        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        components_layout.addLayout(self.grid_layout)

        # Expose a generic "components" handle so that global model macros
        # can operate on PCH components in the same way as for lifetime,
        # PDA and TCSPC Gaussian distance components.
        self.components = self

        if self._make_fitting_parameter_widget is not None:
            try:
                self._append_groupbox(1)
            except Exception:
                pass

        try:
            self.button_add_component.clicked.connect(self.onAddComponent)
            self.button_remove_component.clicked.connect(self.onRemoveComponent)
        except Exception:
            pass

        self.setLayout(layout)
        self.layout = layout

    def _get_component_parameters(self, idx: int):
        if idx == 1:
            return self._eps1, self._N1
        elif idx == 2:
            return self._eps2, self._N2
        elif idx == 3:
            return self._eps3, self._N3
        return None, None

    def _append_groupbox(self, idx: int) -> None:
        if self._make_fitting_parameter_widget is None:
            return
        if idx < 1 or idx > 3:
            return

        p_eps, p_N = self._get_component_parameters(idx)
        if p_eps is None or p_N is None:
            return

        gb = QtWidgets.QGroupBox()
        gb.setTitle(f"C{idx}")

        g_layout = QtWidgets.QVBoxLayout()
        g_layout.setContentsMargins(0, 0, 0, 0)
        g_layout.setSpacing(0)

        try:
            self._make_fitting_parameter_widget(
                fitting_parameter=p_eps,
                layout=g_layout,
            )
            self._make_fitting_parameter_widget(
                fitting_parameter=p_N,
                layout=g_layout,
            )
        except Exception:
            gb.setLayout(g_layout)
            self._gb.append(gb)
            return

        gb.setLayout(g_layout)
        row = (idx - 1) // 2 + 1
        col = (idx - 1) % 2
        self.grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)

    def append(self, *args, **kwargs) -> None:
        try:
            n_before = int(getattr(self, "n_components", 1) or 1)
        except Exception:
            n_before = 1

        self.add_component()

        try:
            n_after = int(getattr(self, "n_components", n_before) or n_before)
        except Exception:
            n_after = n_before

        if self._make_fitting_parameter_widget is None:
            return
        if n_after > 3:
            return
        if n_after > len(self._gb):
            try:
                self._append_groupbox(n_after)
            except Exception:
                pass

    def pop(self) -> None:
        try:
            n = int(getattr(self, "n_components", 1) or 1)
        except Exception:
            n = 1
        if n <= 1:
            return
        if self._gb:
            try:
                gb = self._gb.pop()
                gb.close()
            except Exception:
                pass
        self.remove_component()

    def onAddComponent(self) -> None:
        # Add a new PCH component to all fits in the current fit group via
        # the shared model macros.
        try:
            chisurf.run("chisurf.macros.model.add_component('components')")
        except Exception:
            pass

    def onRemoveComponent(self) -> None:
        # Remove the last PCH component from all fits in the current fit
        # group.
        try:
            chisurf.run("chisurf.macros.model.remove_component('components')")
        except Exception:
            pass

    def update_widgets(self) -> None:  # type: ignore[override]
        try:
            if self._param_widget is not None and hasattr(self._param_widget, "finalize"):
                self._param_widget.finalize()
        except Exception:
            pass
