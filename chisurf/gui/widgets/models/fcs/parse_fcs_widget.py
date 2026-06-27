from __future__ import annotations

import pathlib
import numpy as np
from qtpy import QtGui, QtWidgets

import chisurf as cs
from chisurf import typing
import chisurf.core.fitting
import chisurf.core.plot_transforms as plot_transforms
from chisurf.gui import plots
from chisurf.gui.widgets.models.parse.widget import ParseModelWidget
from chisurf.core.fitting.parameter import FittingParameter
import chisurf.gui.widgets.fitting.widgets as fitting_widgets
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client
from chisurf.core.fluorescence.fcs import (
    background_factor_ac,
    resolve_total_mean_count_rate,
    diffusion_reference_component,
    fcs_diffusion_reference,
    normalize_fcs_curve,
    compute_cpm,
    compute_cpm_all,
)


class ParseFCSWidget(ParseModelWidget):

    def _parameter_value(self, name, default=None):
        """Return a finite fitted parameter value when available.

        Parameters
        ----------
        name : str
            Parameter name.
        default : float, optional
            Value returned when the parameter is missing or non-finite.

        Returns
        -------
        float or None
            Finite parameter value, or ``default``.
        """
        parameters = getattr(self, "parameters_all_dict", {}) or {}
        parameter = parameters.get(name)
        try:
            value = float(parameter.value)
        except Exception:
            return default
        if not np.isfinite(value):
            return default
        return value

    def _fcs_diffusion_reference_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Normalize an FCS curve by the fitted diffusion component.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Normalized curve.
        """
        b = float(context.parameters.get("b", self._parameter_value("b", 1.0) or 1.0))
        params = {
            "b": b,
            "N": self._parameter_value("N"),
            "td": self._parameter_value("td"),
            "s": self._parameter_value("s"),
        }
        reference = fcs_diffusion_reference(context.x, params)
        if reference is None:
            raise ValueError("FCS diffusion reference is unavailable")
        return plot_transforms.PlotReferenceResult(
            x=context.x,
            y=normalize_fcs_curve(context.y, reference, b),
            y_label="(G - b) / Gdiff",
        )

    def _fcs_molecule_reference_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Normalize an FCS curve by fitted molecule number.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Molecule-normalized curve.
        """
        b = float(context.parameters.get("b", self._parameter_value("b", 1.0) or 1.0))
        n = float(context.parameters.get("N", self._parameter_value("N", 1.0) or 1.0))
        return plot_transforms.PlotReferenceResult(
            x=context.x,
            y=n * (np.asarray(context.y, dtype=float) - b),
            y_label="N * (G - b)",
        )

    def get_plot_reference_modes(self) -> typing.List[plot_transforms.PlotReferenceMode]:
        """Return FCS reference modes for the line plot.

        Returns
        -------
        list
            Plot reference modes.
        """
        b = self._parameter_value("b", 1.0)
        n = self._parameter_value("N", 1.0)
        return [
            plot_transforms.PlotReferenceMode(
                key="fcs_diffusion",
                label="FCS diffusion",
                callback=self._fcs_diffusion_reference_mode,
                parameters=(
                    plot_transforms.PlotReferenceParameter(
                        key="b",
                        label="b",
                        kind="float",
                        default=float(b if b is not None else 1.0),
                        step=0.01,
                    ),
                ),
                applies_to=("data", "model"),
                y_label="(G - b) / Gdiff",
                y_range=(0, 1.0),
                y_padding=0.05,
            ),
            plot_transforms.PlotReferenceMode(
                key="fcs_molecules",
                label="FCS molecules",
                callback=self._fcs_molecule_reference_mode,
                parameters=(
                    plot_transforms.PlotReferenceParameter(
                        key="N",
                        label="N",
                        kind="float",
                        default=float(n if n is not None else 1.0),
                        minimum=1e-12,
                        step=0.1,
                    ),
                    plot_transforms.PlotReferenceParameter(
                        key="b",
                        label="b",
                        kind="float",
                        default=float(b if b is not None else 1.0),
                        step=0.01,
                    ),
                ),
                applies_to=("data", "model"),
                y_label="N * (G - b)",
                y_range=(0, 1.05),
                y_padding=0.05,
            ),
        ]

    try:
        plot_classes = [
            (
                plots.LinePlot, {
                    'scale_x': 'log',
                    'd_scaley': 'lin',
                    'r_scaley': 'lin',
                    'x_label': 't_c (ms)',
                    'y_label': 'G_c(t_c)'
                }
            ),
            (plots.FitTablePlot, {}),
            (plots.FitInfo, {}),
            (plots.ParameterScanPlot, {}),
            (cs.gui.plots.ResidualPlot, {})
        ]
    except Exception:
        plot_classes = []

    def __init__(
            self,
            fit: cs.core.fitting.fit.FitGroup,
            icon: QtGui.QIcon = None,
            **kwargs
    ):
        """Initialize the FCS parse-model widget.

        Parameters
        ----------
        fit : cs.core.fitting.fit.FitGroup
            The fit group this widget belongs to.
        icon : QtGui.QIcon, optional
            Icon for the model tab.
        **kwargs
            Additional keyword arguments forwarded to the base class.
        """
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/fcs.png")
        self.icon = icon
        fn = pathlib.Path(cs.__file__).parent / 'core' / 'models' / 'fcs' / 'models.yaml'
        super().__init__(fit=fit, model_file=fn, **kwargs)

        self._cpm = FittingParameter(
            name="cpm",
            value=float("nan"),
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
            is_output=True,
        )

        self._cpm_all = FittingParameter(
            name="cpm_all",
            label_text="cpm<sub>all</sub>",
            value=float("nan"),
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
            is_output=True,
        )

        self._S = FittingParameter(
            name="S",
            value=float("nan"),
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
            is_output=True,
        )
        self._B = FittingParameter(
            name="B",
            value=0.0,
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
        )

        self._bg_correction_enabled = False

        try:
            self.find_parameters()
        except Exception:
            pass

        default_model = kwargs.get('model_name', "3D Gauss, 1 bunching")
        try:
            if hasattr(self, 'parse') and default_model in self.parse.models:
                self.parse.model_name = default_model
                self.parse.onModelChanged()
        except Exception:
            pass

        try:
            cpm_widget = fitting_widgets.make_fitting_parameter_widget(self._cpm, suffix=" kHz")
            cpm_all_widget = fitting_widgets.make_fitting_parameter_widget(self._cpm_all, label_text="cpm<sub>all</sub>", suffix=" kHz")
            S_widget = fitting_widgets.make_fitting_parameter_widget(self._S, suffix=" kHz")
            B_widget = fitting_widgets.make_fitting_parameter_widget(self._B, suffix=" kHz")

            self._bg_checkbox = QtWidgets.QCheckBox("Apply background correction")
            self._bg_checkbox.setChecked(False)
            self._bg_checkbox.toggled.connect(self._on_bg_correction_toggled)

            parse = getattr(self, "parse", None)
            fcs_layout = None
            if parse is not None:
                try:
                    fcs_box = getattr(parse, "groupBox_fcs", None)
                except Exception:
                    fcs_box = None
                if fcs_box is not None:
                    try:
                        fcs_layout = fcs_box.layout()
                    except Exception:
                        fcs_layout = None
                if fcs_layout is None:
                    fcs_layout = getattr(parse, "gridLayout_fcs", None)

            param_layout = fcs_layout
            if (
                cpm_widget is not None
                and cpm_all_widget is not None
                and S_widget is not None
                and B_widget is not None
                and param_layout is not None
                and hasattr(param_layout, "addWidget")
            ):
                try:
                    base_row = param_layout.rowCount() if hasattr(param_layout, "rowCount") else 0
                except Exception:
                    base_row = 0
                row0 = base_row
                row1 = base_row + 1
                row2 = base_row + 2

                param_layout.addWidget(S_widget, row0, 0, 1, 1)
                param_layout.addWidget(B_widget, row0, 1, 1, 1)
                param_layout.addWidget(cpm_widget, row1, 0, 1, 1)
                param_layout.addWidget(cpm_all_widget, row1, 1, 1, 1)
                param_layout.addWidget(self._bg_checkbox, row2, 0, 1, 2)
            elif cpm_widget is not None:
                lay = self.layout()
                if lay is not None:
                    if S_widget is not None:
                        lay.addWidget(S_widget)
                    if B_widget is not None:
                        lay.addWidget(B_widget)
                    lay.addWidget(cpm_widget)
                    if cpm_all_widget is not None:
                        lay.addWidget(cpm_all_widget)
                    lay.addWidget(self._bg_checkbox)
        except Exception:
            pass

    def update_model(self, **kwargs):
        """Update FCS parse model and derived CPM parameter.

        CPM (counts per molecule) is computed from the mean count rate stored
        in the data's metadata (e.g. Kristine .cor reader) and the fitted
        particle number N.
        """
        super().update_model(**kwargs)

        try:
            y = np.asarray(self.y, dtype=float)
        except Exception:
            y = None

        if self._bg_correction_enabled and y is not None and y.size > 0:
            try:
                b_param = self.parameters_all_dict.get("b", None)
            except Exception:
                b_param = None
            try:
                baseline = float(b_param.value) if b_param is not None else float("nan")
            except Exception:
                baseline = float("nan")
            if not np.isfinite(baseline):
                try:
                    baseline = float(y[-1])
                except Exception:
                    baseline = 0.0

            try:
                S_val = float(self._S.value)
                B_val = float(self._B.value)
            except Exception:
                S_val = B_val = None

            if S_val is not None and B_val is not None:
                k_bg = background_factor_ac(S_val, B_val)
                if k_bg != 1.0:
                    y_corr = baseline + k_bg * (y - baseline)
                    self.y = y_corr
                    y = y_corr

        try:
            data = self.fit.data
        except Exception:
            return

        meta = getattr(data, "meta_data", {}) or {}
        mean_cr_total = resolve_total_mean_count_rate(meta)

        if mean_cr_total is not None:
            try:
                fc = get_fitting_client()
                if fc is not None:
                    fc.set_parameter_value(
                        parameter_name=str(self._S.name),
                        value=float(mean_cr_total),
                        fit_index=getattr(self.fit, "fit_idx", None),
                    )
                    fc.set_parameter_fixed(
                        parameter_name=str(self._S.name),
                        fixed=True,
                        fit_index=getattr(self.fit, "fit_idx", None),
                    )
            except Exception:
                pass

        if mean_cr_total is None:
            return

        try:
            N_param = self.parameters_all_dict["N"]
        except Exception:
            return

        try:
            N = float(N_param.value)
            cr = float(mean_cr_total)
        except Exception:
            return

        if not (N > 0.0):
            return

        cpm = compute_cpm(cr, N)

        try:
            fc = get_fitting_client()
            if fc is not None:
                fc.set_parameter_value(
                    parameter_name=str(self._cpm.name),
                    value=cpm,
                    fit_index=getattr(self.fit, "fit_idx", None),
                )
                fc.set_parameter_fixed(
                    parameter_name=str(self._cpm.name),
                    fixed=True,
                    fit_index=getattr(self.fit, "fit_idx", None),
                )
        except Exception:
            pass

        # Extract bunching parameters for cpm_all
        bunch_params = {}
        for name, p in self.parameters_all_dict.items():
            if name.startswith("ba"):
                try:
                    v = float(p.value)
                except Exception:
                    continue
                if not np.isfinite(v):
                    continue
                bunch_params[name] = v

        cpm_all = compute_cpm_all(cr, N, bunch_params)

        if cpm_all is not None:
            try:
                fc = get_fitting_client()
                if fc is not None:
                    fc.set_parameter_value(
                        parameter_name=str(self._cpm_all.name),
                        value=cpm_all,
                        fit_index=getattr(self.fit, "fit_idx", None),
                    )
                    fc.set_parameter_fixed(
                        parameter_name=str(self._cpm_all.name),
                        fixed=True,
                        fit_index=getattr(self.fit, "fit_idx", None),
                    )
            except Exception:
                pass

    def _on_bg_correction_toggled(self, checked: bool) -> None:
        """Qt slot: toggle model-based background correction on/off."""
        self._bg_correction_enabled = bool(checked)
        try:
            fc = get_fitting_client()
            if fc is not None:
                fc.update_fit(fit_index=getattr(self.fit, "fit_idx", None))
        except Exception:
            pass
