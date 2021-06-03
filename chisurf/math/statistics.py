import numba as nb
import numpy as np
import scipy.stats

from scikit_fluorescence.math.statistics import durbin_watson


def chi2_max(
        chi2_value: float = 1.0,
        number_of_parameters: int = 1,
        nu: int = 1,
        conf_level: float = 0.95
) -> float:
    """Calculate the maximum chi2r of a fit given a certain confidence level

    :param chi2_value: the chi2 value
    :param number_of_parameters: the number of parameters of the models
    :param conf_level: the confidence level that is used to calculate the
    maximum chi2
    :param nu: the number of free degrees of freedom (number of observations
    - number of models parameters)
    """
    return chi2_value * (
            1.0 + float(
        number_of_parameters
        ) / nu *
            scipy.stats.f.isf(
                1. - conf_level, number_of_parameters, nu
            )
    )

