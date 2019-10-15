"""
This module is responsible contains all fitting modules for experimental data

The :py:mod:`.models`

1. :py:mod:`.models.tcspc`
2. :py:mod:`.models.fcs`
3. :py:mod:`.models.global_model`
4. :py:mod:`.models.parse`
5. :py:mod:`.models.proteinMC`
6. :py:mod:`.models.stopped_flow`


"""
from . import model
from . import parse
from . import fcs
from . import tcspc
from . import global_model
from . import stopped_flow
