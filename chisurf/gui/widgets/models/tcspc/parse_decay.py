from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np
import chisurf as cs
from chisurf import typing
from qtpy import QtWidgets, QtCore, QtGui
import chisurf.gui.plots
import chisurf.core.curve
import chisurf.core.plot_transforms as plot_transforms

from chisurf.gui.widgets.models.model_widget import ModelWidget
from chisurf.core.models.tcspc.parse.tcspc_parse import ParseDecayModel
import chisurf.gui.widgets.models.parse.widget

# These will be imported from the new module structure
from chisurf.gui.widgets.models.tcspc.convolve import ConvolveWidget
from chisurf.gui.widgets.models.tcspc.generic import GenericWidget
from chisurf.gui.widgets.models.tcspc.corrections import CorrectionsWidget

if TYPE_CHECKING:
    from chisurf.core.fitting.fit import FitGroup


class ParseDecayModelWidget(ParseDecayModel, ModelWidget):

    plot_classes = [
        (
            cs.gui.plots.LinePlot,
            {
                'd_scalex': 'lin',
                'd_scaley': 'log',
                'r_scalex': 'lin',
                'r_scaley': 'lin',
                'x_label': 'time (ns)',
                'y_label': 'counts',
                'plot_irf': True
            }
         ),
        (cs.gui.plots.FitTablePlot, {}),
        (cs.gui.plots.FitInfo, {}),
        (cs.gui.plots.ParameterScanPlot, {}),
        (cs.gui.plots.ResidualPlot, {})
    ]

    # TODO: needs docstring
    def get_curves(self, copy_curves: bool = False) -> typing.Dict[str, cs.core.curve.Curve]:
        """Return a dictionary of curves for plotting."""
        d = super().get_curves(copy_curves)
        d['IRF'] = self.convolve.irf
        return d

    def _tcspc_reference_window(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> np.ndarray:
        """Return the y-window used for photon normalization.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        numpy.ndarray
            Finite y-values used for the denominator.
        """
        y = np.asarray(context.y, dtype=float)
        if bool(context.parameters.get("fit_range_only", False)):
            try:
                data_x = np.asarray(getattr(getattr(context.fit, "data", None), "x", []), dtype=float)
                if y.size == data_x.size:
                    xmin = int(getattr(context.fit, "xmin", 0))
                    xmax = int(getattr(context.fit, "xmax", y.size))
                    y = y[max(0, xmin):min(y.size, xmax)]
            except Exception:
                pass
        return y[np.isfinite(y)]

    def _tcspc_total_photons_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Normalize TCSPC counts by total photons.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Photon-normalized curve.
        """
        denominator = float(np.nansum(self._tcspc_reference_window(context)))
        if not np.isfinite(denominator) or denominator == 0.0:
            raise ValueError("total photon count is zero")
        return plot_transforms.PlotReferenceResult(
            x=context.x,
            y=np.asarray(context.y, dtype=float) / denominator,
            y_label="counts / total photons",
        )

    def _tcspc_peak_photons_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Normalize TCSPC counts by peak photons.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Peak-normalized curve.
        """
        window = self._tcspc_reference_window(context)
        if window.size == 0:
            raise ValueError("peak photon count is unavailable")
        denominator = float(np.nanmax(window))
        if not np.isfinite(denominator) or denominator == 0.0:
            raise ValueError("peak photon count is zero")
        return plot_transforms.PlotReferenceResult(
            x=context.x,
            y=np.asarray(context.y, dtype=float) / denominator,
            y_label="counts / peak photons",
        )

    def get_plot_reference_modes(self) -> typing.List[plot_transforms.PlotReferenceMode]:
        """Return TCSPC parse-decay reference modes.

        Returns
        -------
        list
            Plot reference modes.
        """
        fit_range_param = plot_transforms.PlotReferenceParameter(
            key="fit_range_only",
            label="fit range",
            kind="bool",
            default=False,
        )
        return [
            plot_transforms.PlotReferenceMode(
                key="tcspc_total_photons",
                label="Total photons",
                callback=self._tcspc_total_photons_mode,
                parameters=(fit_range_param,),
                applies_to=("data", "model"),
                y_label="counts / total photons",
                y_range=(0, 1.0),
                y_padding=0.05,
            ),
            plot_transforms.PlotReferenceMode(
                key="tcspc_peak_photons",
                label="Peak photons",
                callback=self._tcspc_peak_photons_mode,
                parameters=(fit_range_param,),
                applies_to=("data", "model"),
                y_label="counts / peak photons",
                y_range=(0, 1.0),
                y_padding=0.05,
            ),
        ]

    # TODO: needs docstring
    def __init__(
            self,
            fit: FitGroup,
            icon: QtGui.QIcon = None,
            **kwargs
    ):
        """Initialize the instance."""
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        super(ModelWidget, self).__init__(fit=fit, icon=icon)
        super(ParseDecayModel, self).__init__(fit=fit, icon=icon)

        self.convolve = ConvolveWidget(
            fit=fit,
            model=self,
            hide_curve_convolution=False,
            dt=fit.data.dx,
            **kwargs
        )
        # Parse models have no lifetime spectrum — only "full" (numpy)
        # convolution is valid.  Hide the per/exp mode radio buttons.
        self.convolve.radioButton.hide()
        self.convolve.radioButton_2.hide()

        generic = GenericWidget(
            fit=fit,
            parent=self,
            model=self,
            **kwargs
        )
        fn = pathlib.Path(__file__).parent.parent.parent.parent.parent / 'core' / 'models' / 'tcspc' / 'tcspc.models.json'
        pw = cs.gui.widgets.models.parse.widget.ParseFormulaWidget(
            model=self,
            model_file=fn
        )
        # Hide the FCS parameters group box — those are exclusive to FCS models
        try:
            pw.groupBox_fcs.hide()
        except Exception:
            pass

        corrections = CorrectionsWidget(
            fit=fit,
            model=self,
            **kwargs
        )

        self.fit = fit
        super().__init__(
            fit=fit,
            parse=pw,
            icon=icon,
            convolve=self.convolve,
            generic=generic,
            corrections=corrections
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(self.convolve)
        layout.addWidget(generic)
        layout.addWidget(corrections)
        layout.addWidget(pw)
        self.setLayout(layout)
