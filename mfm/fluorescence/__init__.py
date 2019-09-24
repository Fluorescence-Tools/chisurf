"""
Fluorescence

"""
from . import anisotropy
from . import fcs
from . import intensity
from . import tcspc
from . import general
from . import fps
from . import pda

import mfm.settings
import numpy as np

rda_axis = np.linspace(
    mfm.settings.cs_settings['fret']['rda_min'],
    mfm.settings.cs_settings['fret']['rda_max'],
    mfm.settings.cs_settings['fret']['rda_resolution'], dtype=np.float64
)
