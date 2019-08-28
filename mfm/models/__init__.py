"""
This module is responsible contains all fitting modules for experimental data

The :py:mod:`.model`

1. :py:mod:`.model.tcspc`
2. :py:mod:`.model.fcs`
3. :py:mod:`.model.gloablfit`
4. :py:mod:`.model.parse`
5. :py:mod:`.model.proteinMC`
6. :py:mod:`.model.stopped_flow`


"""
from __future__ import annotations

from typing import List

import mfm.fitting
from mfm.models import model

from . import model
from . import parse
from . import fcs
from . import tcspc
from . import globalfit
#from . import parse
#from .globalfit import *


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