"""

"""
from __future__ import annotations

from typing import List

import numpy as np

import mfm.fitting.parameter
import mfm.fitting.fit
import mfm.fitting.sample
import mfm.fitting.widgets


def find_fit_idx_of_model(
        model: mfm.models.model.Model,
        fits: List[mfm.fitting.fit.Fit]
) -> int:
    """Returns index of the fit of a model in mfm.fits array

    :param model:
    :param fits:
    :return:
    """
    for idx, f in enumerate(fits):
        if f.model == model:
            return idx


def calculate_weighted_residuals(
        data: mfm.experiments.data.DataCurve,
        model: mfm.curve.Curve,
        xmin: int,
        xmax: int,
) -> np.array:
    model_x, model_y = model[xmin:xmax]
    data_x, data_y, _, data_y_error = data[xmin:xmax]
    ml = min([len(model_y), len(data_y)])
    wr = np.array(
        (data_y[:ml] - model_y[:ml]) / data_y_error[:ml],
        dtype=np.float64
    )
    return wr
