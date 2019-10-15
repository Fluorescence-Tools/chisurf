"""

"""
from __future__ import annotations

from typing import List

import numpy as np

import mfm.curve
import mfm.models
import fitting.parameter
import experiments
import fitting.fit
import fitting.sample
import fitting.widgets


def find_fit_idx_of_model(
        model: mfm.models.model.Model,
        fits: List[fitting.fit.Fit]
) -> int:
    """Returns index of the fit of a model in mfm.fits array

    :param model:
    :param fits:
    :return:
    """
    for idx, f in enumerate(fits):
        if f.model is model:
            return idx


def calculate_weighted_residuals(
        data: experiments.data.DataCurve,
        model: mfm.curve.Curve,
        xmin: int,
        xmax: int,
) -> np.array:
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
