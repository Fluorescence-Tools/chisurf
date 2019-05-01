"""
Fluorescence

"""
import mfm
from .general import *
from . import fcs
from . import intensity
from . import anisotropy
from . import tcspc
#import pda

rda_axis = np.linspace(mfm.cs_settings['fret']['rda_min'],
                       mfm.cs_settings['fret']['rda_max'],
                       mfm.cs_settings['fret']['rda_resolution'], dtype=np.float64)