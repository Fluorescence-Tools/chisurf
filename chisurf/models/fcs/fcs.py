from __future__ import annotations

import os
import pathlib
import numpy as np
from qtpy import QtGui

import chisurf
import chisurf.fitting
from chisurf import plots
from chisurf.models.parse.widget import ParseModelWidget
from chisurf.fitting.parameter import FittingParameter
import chisurf.gui.widgets.fitting.widgets as fitting_widgets


class ParseFCSWidget(ParseModelWidget):

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

        # Add a read-only CPM parameter widget close to the other parameters.
        # Prefer inserting into the parse parameter grid (gridLayout_1) so the
        # alignment matches N, td, s, etc. Fall back to the outer layout only
        # if the grid is not available.
        try:
            cpm_widget = fitting_widgets.make_fitting_parameter_widget(self._cpm)
            cpm_all_widget = fitting_widgets.make_fitting_parameter_widget(self._cpm_all)
            parse = getattr(self, "parse", None)
            param_layout = getattr(parse, "gridLayout_1", None)
            if (
                cpm_widget is not None
                and cpm_all_widget is not None
                and param_layout is not None
                and hasattr(param_layout, "addWidget")
            ):
                try:
                    row = param_layout.rowCount() if hasattr(param_layout, "rowCount") else 1000
                    col_span = param_layout.columnCount() if hasattr(param_layout, "columnCount") else 1
                except Exception:
                    row, col_span = 1000, 1
                # Place CPM (bright molecules) and CPM_all (all molecules)
                param_layout.addWidget(cpm_widget, row, 0, 1, col_span)
                param_layout.addWidget(cpm_all_widget, row + 1, 0, 1, col_span)
            elif cpm_widget is not None:
                lay = self.layout()
                if lay is not None:
                    lay.addWidget(cpm_widget)
                    if cpm_all_widget is not None:
                        lay.addWidget(cpm_all_widget)
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

        # Derive CPM only if we have both metadata and an N parameter
        try:
            data = self.fit.data
        except Exception:
            return

        meta = getattr(data, "meta_data", {}) or {}
        mean_cr = meta.get("mean_count_rate")
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

