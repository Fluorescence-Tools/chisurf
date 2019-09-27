"""
This module is responsible for all experiments/fits

The :py:mod:`experiments` module contains the fitting models and the setups (assembled reading routines) for
different experimental setups. Furthermore, it contains a set of plotting libraries.


"""
from __future__ import annotations
from typing import List

from . import experiment
from . import data
from . import reader
from . import fcs
from . import tcspc
from . import globalfit
from . import modelling


def get_data(
        curve_type: str = 'experiment',
        data_set: List[data.ExperimentalData] = None
) -> List[data.ExperimentalData]:
    """Returns all curves `mfm.curve.DataCurve` except if the
    curve is names "Global-fit"

    :param curve_type: if this value is set to `experiment` only curves
    that are experimental curves, i.e., curves that inherit from
    `mfm.experiments.data.ExperimentalData` are returned.

    :param data_set: A list containing the

    :return:
    """
    if curve_type == 'experiment':
        return [
            d for d in data_set if (
                    (
                            isinstance(
                                d,
                                data.ExperimentalData
                            ) or
                            isinstance(
                                d,
                                data.ExperimentDataGroup
                            )
                    ) and
                    d.name != "Global-fit"
            )
        ]
    else: #elif curve_type == 'all':
        return [
            d for d in data_set if
            isinstance(
                d,
                data.ExperimentalData
            ) or isinstance(
                d, data.ExperimentDataGroup
            )
        ]
