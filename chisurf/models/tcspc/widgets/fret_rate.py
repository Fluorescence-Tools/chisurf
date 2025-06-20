from __future__ import annotations

import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting
import chisurf.fitting.fit

import chisurf.models.tcspc.fret as fret

from chisurf.models.tcspc.widgets.lifetime import LifetimeWidget, LifetimeModelWidgetBase
from chisurf.models.tcspc.widgets.discrete_distance import DiscreteDistanceWidget

# Import plot_cls_dist_default from the original module
from chisurf.models.tcspc.widgets import plot_cls_dist_default


class FRETrateModelWidget(fret.FRETrateModel, LifetimeModelWidgetBase):

    plot_classes = plot_cls_dist_default

    def get_parameter_widgets(self):
        """
        Get all parameter widgets for this model.

        Returns
        -------
        list
            List of parameter widgets.
        """
        widgets = super().get_parameter_widgets() if hasattr(super(), 'get_parameter_widgets') else []
        widgets.append(self._orientation_widget)
        widgets.append(self._fret_parameters_widget)
        return widgets

    def __init__(self, fit: chisurf.fitting.fit.Fit, **kwargs):
        self.donor = LifetimeWidget(
            parent=self,
            model=self,
            title='Donor(0)',
            name='donor'
        )
        self.fret_rates = DiscreteDistanceWidget(
            donors=self.donor,
            parent=self,
            model=self,
            short='G',
            **kwargs
        )
        super().__init__(
            fit=fit,
            fret_rates=self.fret_rates,
            donor=self.donor
        )

        # Create parameter widgets
        self._fret_parameters_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
            self.fret_parameters
        )
        self.layout.addWidget(self._fret_parameters_widget)

        # self._orientation_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
        #     self.orientation_parameter
        # )
        # self.layout.addWidget(self._orientation_widget)

        self.layout.addWidget(self.donor)
        self.layout.addWidget(self.fret_rates)