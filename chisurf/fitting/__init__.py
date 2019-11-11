"""

"""
from __future__ import annotations

from typing import List

import numpy as np

import chisurf.fitting.parameter
import chisurf.models
import chisurf.curve
import chisurf.experiments
import chisurf.fitting.fit
import chisurf.fitting.sample
import chisurf.fitting.support_plane


# def find_fit_idx_of_model(
#         model: chisurf.models.Model,
#         fits: List[chisurf.fitting.fit.Fit]
# ) -> int:
#     """Returns index of the fit of a model in chisurf.fits array
#
#     :param model:
#     :param fits:
#     :return:
#     """
#     for idx, f in enumerate(fits):
#         if f.model is model:
#             return idx


def calculate_weighted_residuals(
        data: chisurf.experiments.data.DataCurve,
        model: chisurf.curve.Curve,
        xmin: int,
        xmax: int,
) -> np.ndarray:
    """Calculates the weighted residuals for a DataCurve and a
    model curve given the range as provided by xmin and xmax. The
    weighted residuals are given by (data - model) / weights. Here,
    the weights are the errors of the data.

    :param data: the experimental data
    :param model: the model
    :param xmin: minimum index
    :param xmax: maximum index
    :return: a numpy array containing the weighted residuals
    """
    model_x, model_y = model[xmin:xmax]
    data_x, data_y, _, data_y_error = data[xmin:xmax]
    ml = min([len(model_y), len(data_y)])
    wr = np.array(
        (data_y[:ml] - model_y[:ml]) / data_y_error[:ml],
        dtype=np.float64
    )
    return wr
