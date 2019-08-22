from __future__ import annotations

from typing import List

import mfm.fitting.parameter
import mfm.fitting.fit

#from . import fit
#import mfm.fitting.models
import mfm.fitting.widgets
#from mfm.fitting.parameter import FittingParameter, FittingParameterGroup
from mfm.models import model


def find_fit_idx_of_model(
        model: model.Model,
        fits: List[mfm.fitting.fit.Fit]
):
    """Returns index of the fit of a model in mfm.fits array

    :param model:
    :return:
    """
    for idx, f in enumerate(fits):
        if f.model == model:
            return idx