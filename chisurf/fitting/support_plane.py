"""

"""
from __future__ import annotations
from chisurf import typing

import numpy as np

import chisurf.fitting.fit


def scan_parameter(
        fit: chisurf.fitting.fit.Fit,
        parameter_name: str,
        scan_range=(None, None),
        rel_range: float = 0.2,
        n_steps: int = 30
) -> typing.Dict:
    """Performs a chi2-scan for the parameter

    :param fit: the fit of type 'fitting.Fit'
    :param parameter_name: the name of the parameter (in the parameter dictionary)
    :param scan_range: the range within the parameter is scanned if not provided 'rel_range' is used
    :param rel_range: defines +/- values for scanning
    :param n_steps: number of steps between +/-
    :return:
    """
    # Store initial values before varying the parameter
    initial_parameter_values = fit.model.parameter_values

    varied_parameter = fit.model.parameters_all_dict[parameter_name]
    is_fixed = varied_parameter.fixed

    varied_parameter.fixed = True
    chi2r_array = np.empty(n_steps, dtype=float)

    # Determine range within the parameter is varied
    parameter_value = varied_parameter.value
    p_min, p_max = scan_range
    if p_min is None or p_max is None:
        p_min = parameter_value * (1. - rel_range)
        p_max = parameter_value * (1. + rel_range)
    parameter_array = np.linspace(p_min, p_max, n_steps)

    for i, p in enumerate(parameter_array):
        varied_parameter.fixed = is_fixed
        fit.model.parameter_values = initial_parameter_values
        varied_parameter.fixed = True
        varied_parameter.value = p
        fit.run()
        chi2r_array[i] = fit.chi2r

    varied_parameter.fixed = is_fixed
    fit.model.parameter_values = initial_parameter_values
    fit.update()

    return {
        'chi2r': chi2r_array,
        'parameter_values': parameter_array,
        'parameter_names': [parameter_name]
    }
