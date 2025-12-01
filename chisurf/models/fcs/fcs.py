from __future__ import annotations

import os
import pathlib
import numpy as np
from qtpy import QtGui, QtWidgets

import chisurf
import chisurf.fitting
from chisurf import plots
from chisurf.models.parse.widget import ParseModelWidget
from chisurf.fitting.parameter import FittingParameter
import chisurf.gui.widgets.fitting.widgets as fitting_widgets
from chisurf.fluorescence.fcs import background_factor_ac


class ParseFCSWidget(ParseModelWidget):

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
            (chisurf.plots.ResidualPlot, {})
        ]
    except Exception:
        plot_classes = []

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            icon: QtGui.QIcon = None,
            **kwargs
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/fcs.png")
        self.icon = icon
        fn = pathlib.Path(__file__).parent / 'models.yaml'
        super().__init__(fit=fit, model_file=fn, **kwargs)

        # Derived CPM parameter (counts per molecule). This is not used in the
        # analytical equation; it is populated from metadata (mean_count_rate)
        # and the fitted N parameter after each model update and exposed as a
        # fixed parameter in the parameter table.
        self._cpm = FittingParameter(
            name="cpm",
            value=float("nan"),
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
        )

        # Second derived CPM that accounts for dark/bunching states. This uses
        # N_all = N / (1 - sum(ba_i)) in the denominator, where ba_i are the
        # amplitudes of bunching terms (parameters whose names start with 'ba').
        self._cpm_all = FittingParameter(
            name="cpm_all",
            value=float("nan"),
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
        )

        # Signal and background countrates (kHz). These are fixed fitting
        # parameters that are populated from metadata (mean_count_rate) and
        # optionally edited by the user via the GUI. They are not varied
        # during fitting but are used for model-based background correction.
        self._S = FittingParameter(
            name="S",
            value=float("nan"),
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
        )
        self._B = FittingParameter(
            name="B",
            value=0.0,
            lb=float("-inf"),
            ub=float("inf"),
            bounds_on=False,
            fixed=True,
        )

        # Toggle for model-based background correction. When disabled, the
        # analytical FCS model is used unchanged and only CPM is updated.
        # Start with correction disabled; users can opt-in per curve.
        self._bg_correction_enabled = False

        # Ensure derived CPM parameters are registered with the parameter
        # machinery so they show up
        # in parameters_all_dict and the parameter editor.
        try:
            self.find_parameters()
        except Exception:
            pass

        # Ensure the default selected model is "Diffusion with one bunching"
        # which corresponds to the YAML entry "3D Gauss, 1 bunching".
        # Allow callers to override via kwargs['model_name'] if provided.
        default_model = kwargs.get('model_name', "3D Gauss, 1 bunching")
        try:
            if hasattr(self, 'parse') and default_model in self.parse.models:
                self.parse.model_name = default_model
                # Update equation, parameters, and UI to reflect the selection
                self.parse.onModelChanged()
        except Exception:
            # Be forgiving during initialization if UI is not fully wired yet
            pass

        # Add FCS-related parameter widgets (signal/background countrates,
        # derived CPM values, and the background-correction toggle). Prefer
        # inserting them into the dedicated FCS parameter groupbox
        # (gridLayout_fcs) on the parse widget; fall back to the main parse
        # parameter grid (gridLayout_1) or the outer layout if needed.
        try:
            cpm_widget = fitting_widgets.make_fitting_parameter_widget(self._cpm, suffix=" kHz")
            cpm_all_widget = fitting_widgets.make_fitting_parameter_widget(self._cpm_all, suffix=" kHz")
            S_widget = fitting_widgets.make_fitting_parameter_widget(self._S, suffix=" kHz")
            B_widget = fitting_widgets.make_fitting_parameter_widget(self._B, suffix=" kHz")

            self._bg_checkbox = QtWidgets.QCheckBox("Apply background correction")
            self._bg_checkbox.setChecked(False)
            self._bg_checkbox.toggled.connect(self._on_bg_correction_toggled)

            parse = getattr(self, "parse", None)
            # Prefer the dedicated FCS layout if present. Try the layout of
            # groupBox_fcs first (more robust against UI changes), then fall
            # back to a named gridLayout_fcs attribute, and finally to the
            # generic parse-parameter grid.
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

            # Only place FCS widgets into the dedicated FCS layout; if that is
            # unavailable, fall back to the outer widget layout (see below),
            # not into the generic parse-parameter grid.
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
                # Arrange FCS parameters in a compact 2-column grid to save
                # vertical space:
                #   [ S        | B        ]
                #   [ cpm      | cpm_all  ]
                #   [ Apply background correction ] (spans both columns)
                row0 = base_row
                row1 = base_row + 1
                row2 = base_row + 2

                param_layout.addWidget(S_widget, row0, 0, 1, 1)
                param_layout.addWidget(B_widget, row0, 1, 1, 1)
                param_layout.addWidget(cpm_widget, row1, 0, 1, 1)
                param_layout.addWidget(cpm_all_widget, row1, 1, 1, 1)
                # Checkbox spans both columns
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
        particle number N. This mirrors the QuickFit3 concept but ignores any
        explicit background correction at this stage.
        """

        # First update the underlying parse model (equation-based curve)
        super().update_model(**kwargs)

        # Optionally apply model-based background correction: only the
        # amplitude relative to the baseline is scaled; the raw data stays
        # untouched. This mirrors the PyCorrFit/Thompson formulas but is
        # implemented on the model side only.
        try:
            y = np.asarray(self.y, dtype=float)
        except Exception:
            y = None

        if self._bg_correction_enabled and y is not None and y.size > 0:
            # Estimate baseline from parameter "b" if present; otherwise from
            # the tail of the model curve.
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

            # Use S/B parameters (kHz) to compute the amplitude attenuation.
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

        # Derive CPM only if we have both metadata and an N parameter
        try:
            data = self.fit.data
        except Exception:
            return

        meta = getattr(data, "meta_data", {}) or {}
        mean_cr = meta.get("mean_count_rate")

        # Populate the signal countrate parameter from metadata if available.
        if mean_cr is not None:
            try:
                self._S.value = float(mean_cr)
                self._S.fixed = True
            except Exception:
                pass

        if mean_cr is None:
            return

        try:
            N_param = self.parameters_all_dict["N"]
        except Exception:
            return

        try:
            N = float(N_param.value)
            cr = float(mean_cr)
        except Exception:
            return

        if not (N > 0.0):
            return

        # Simple CPM definition: mean count rate per bright molecule. The
        # units follow whatever mean_count_rate carries (typically kHz in
        # Kristine format).
        cpm = cr / N

        try:
            self._cpm.value = cpm
            self._cpm.fixed = True
        except Exception:
            pass

        # Second CPM: account for all molecules by correcting N with the
        # amplitudes of bunching states. For models of the form
        # (1 - ba + ba * exp(-x/bt)), ba is the dark fraction. If there are
        # multiple bunching terms, we approximate the total dark fraction as
        # sum(ba_i) over parameters whose names start with 'ba'.
        try:
            bunch_sum = 0.0
            for name, p in self.parameters_all_dict.items():
                if not name.startswith("ba"):
                    continue
                try:
                    v = float(p.value)
                except Exception:
                    continue
                # Clip to [0, 1] to avoid pathological values.
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

