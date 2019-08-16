"""
Fluorescence

"""
import mfm
from . import anisotropy
from . import fcs
from . import intensity
from . import tcspc
from .general import *

#import pda

rda_axis = np.linspace(mfm.cs_settings['fret']['rda_min'],
                       mfm.cs_settings['fret']['rda_max'],
                       mfm.cs_settings['fret']['rda_resolution'], dtype=np.float64)