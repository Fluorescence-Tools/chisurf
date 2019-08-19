"""
This module is responsible contains all fitting modules for experimental data

The :py:mod:`.models`

1. :py:mod:`.models.tcspc`
2. :py:mod:`.models.fcs`
3. :py:mod:`.models.gloablfit`
4. :py:mod:`.models.parse`
5. :py:mod:`.models.proteinMC`
6. :py:mod:`.models.stopped_flow`


"""
from __future__ import annotations

import mfm.fitting.models.model
import mfm.fitting.parameter
import mfm.curve
import mfm.parameter
import mfm.fitting
from mfm.fitting.models import Model
from mfm.fitting.models.model import Model

from . import fcs
from . import tcspc
from . import parse
from .globalfit import *


def find_fit_idx_of_model(
        model: mfm.fitting.models.model.Model
):
    """Returns index of the fit of a model in mfm.fits array

    :param model:
    :return:
    """
    for idx, f in enumerate(mfm.fits):
        if f.model == model:
            return idx

