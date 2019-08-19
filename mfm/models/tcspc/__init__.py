"""
This module contains all time-resolved fluorescence models (TCSPC)

.. automodule:: models.tcspc
   :members:
"""
from __future__ import annotations

import mfm
# import mix_model
#from mfm.fitting.models.tcspc import Lifetime, LifetimeModel
#from mfm.fitting.models.tcspc.anisotropy import Anisotropy
#from mfm.fitting.models.tcspc.lifetime import Lifetime, LifetimeWidget, LifetimeModel, LifetimeModelWidgetBase, \
#    LifetimeModelWidget
#from mfm.fitting.models.tcspc.nusiance import Generic, Corrections, CorrectionsWidget, GenericWidget, Convolve, \
#    ConvolveWidget
from . import lifetime
from . import fret
# from . import nusiance
from . import parse
from . import pddem
# import dye_diffusion
# import et
# import membrane
# import fret_structure
from mfm.models import ModelCurve

models = [
    lifetime.LifetimeModelWidget,
    fret.FRETrateModelWidget,
    fret.GaussianModelWidget,
    pddem.PDDEMModelWidget,
    fret.WormLikeChainModelWidget,
    ModelCurve
    #mix_model.LifetimeMixModelWidget,
]

testing = [
    #fret_structure.FRETStructureWidget,
    #dye_diffusion.TransientDecayGenerator,
    #membrane.GridModelWidget,
    fret.SingleDistanceModelWidget,
    parse.ParseDecayModelWidget
]

models += testing if mfm.settings.cs_settings['experimental_models'] else []

