"""

"""
from __future__ import annotations

from typing import List

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
