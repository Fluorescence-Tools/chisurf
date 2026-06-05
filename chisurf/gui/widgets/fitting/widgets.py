from __future__ import annotations

import chisurf.core.settings
from .fit_list import ModelDataRepresentationSelector
from .fit_controller import FittingControllerWidget
from .fit_subwindow import FitSubWindow
from .parameter_widgets import (
    FittingParameterDetailPopup,
    FittingParameterGroupWidget,
    FittingParameterWidget,
    make_fitting_parameter_group_widget,
    make_fitting_parameter_widget,
)

parameter_settings = chisurf.core.settings.parameter

__all__ = [
    'parameter_settings',
    'ModelDataRepresentationSelector',
    'FittingControllerWidget',
    'FitSubWindow',
    'FittingParameterDetailPopup',
    'FittingParameterWidget',
    'FittingParameterGroupWidget',
    'make_fitting_parameter_widget',
    'make_fitting_parameter_group_widget',
]
