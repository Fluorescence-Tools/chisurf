from __future__ import annotations

import pathlib
import numpy as np
from qtpy import QtGui, QtWidgets

import chisurf as cs
import chisurf.core.fitting
from chisurf.gui import plots
from chisurf.gui.widgets.models.parse.widget import ParseModelWidget
from chisurf.core.fitting.parameter import FittingParameter
import chisurf.gui.widgets.fitting.widgets as fitting_widgets
from chisurf.core.fluorescence.fcs import background_factor_ac


class ParseFCSWidget(ParseModelWidget):

    @staticmethod
    def _to_float_or_none(value):
        """Convert *value* to float, returning ``None`` on failure.

        Parameters
        ----------
        value : any
            Value to convert.

        Returns
        -------
        float or None
            Numeric value, or ``None`` if conversion fails or is non-finite.
        """
        try:
            v = float(value)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return v

    def _resolve_total_mean_count_rate(self, meta):
        """Extract the total mean count rate from metadata.

        Handles both ``mean_count_rate_total`` and
        ``mean_count_rate`` with a ``per_detector`` semantic.

        Parameters
        ----------
        meta : dict
            Data metadata dictionary.

        Returns
        -------
        float or None
            Total mean count rate in kHz, or ``None`` if not available.
        """
        if not isinstance(meta, dict):
            return None

        total = self._to_float_or_none(meta.get("mean_count_rate_total"))
        if total is not None:
            return total

        mean_cr = self._to_float_or_none(meta.get("mean_count_rate"))
        if mean_cr is None:
            return None

        semantics = str(meta.get("mean_count_rate_semantics", "")).strip().lower()
        if "per_detector" in semantics:
            detector_count = self._to_float_or_none(meta.get("detector_count"))
            if detector_count is not None and detector_count > 1.0:
                return mean_cr * detector_count

        return mean_cr

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
            cpm_all_widget = fitting_widgets.make_fitting_parameter_widget(self._cpm_all, suffix=" kHz")
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
        mean_cr_total = self._resolve_total_mean_count_rate(meta)

        if mean_cr_total is not None:
            try:
                self._S.value = float(mean_cr_total)
                self._S.fixed = True
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

        cpm = cr / N

        try:
            self._cpm.value = cpm
            self._cpm.fixed = True
        except Exception:
            pass

        try:
            bunch_sum = 0.0
            for name, p in self.parameters_all_dict.items():
                if not name.startswith("ba"):
                    continue
                try:
                    v = float(p.value)
                except Exception:
                    continue
                if not np.isfinite(v):
                    continue
                v = abs(v)
                if v < 0.0:
                    v = 0.0
                if v > 1.0:
                    v = 1.0
                bunch_sum += v

            bright_fraction = 1.0 - bunch_sum
            if bright_fraction <= 0.0 or not np.isfinite(bright_fraction):
                return

            N_all = N / bright_fraction
            if not (N_all > 0.0 and np.isfinite(N_all)):
                return

            cpm_all = cr / N_all

            self._cpm_all.value = cpm_all
            self._cpm_all.fixed = True
        except Exception:
            pass

    def _on_bg_correction_toggled(self, checked: bool) -> None:
        """Qt slot: toggle model-based background correction on/off."""
        self._bg_correction_enabled = bool(checked)
        try:
            self.fit.update()
        except Exception:
            pass
