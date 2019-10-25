"""
This module is responsible for all experiments/fits

The :py:mod:`experiments` module contains the fitting models and the setups (assembled reading routines) for
different experimental setups. Furthermore, it contains a set of plotting libraries.


"""
from __future__ import annotations
from typing import List

import chisurf.experiments.experiment
import chisurf.experiments.data
import chisurf.experiments.reader
import chisurf.experiments.fcs
import chisurf.experiments.tcspc
import chisurf.experiments.globalfit
import chisurf.experiments.modelling
from chisurf.experiments.experiment import Experiment


def get_data(
        curve_type: str = 'experiment',
        data_set: List[
            chisurf.experiments.data.ExperimentalData
        ] = None
) -> List[
    chisurf.experiments.data.ExperimentalData
]:
    """Returns all curves `chisurf.curve.DataCurve` except if the
    curve is names "Global-fit"

    :param curve_type: if this value is set to `experiment` only curves
    that are experimental curves, i.e., curves that inherit from
    `experiments.data.ExperimentalData` are returned.

    :param data_set: A list containing the

    :return:
    """
    if curve_type == 'experiment':
        return [
            d for d in data_set if (
                    (
                            isinstance(
                                d,
                                chisurf.experiments.data.ExperimentalData
                            ) or
                            isinstance(
                                d,
                                chisurf.experiments.data.ExperimentDataGroup
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
                d,
                chisurf.experiments.data.ExperimentDataGroup
            )
        ]
