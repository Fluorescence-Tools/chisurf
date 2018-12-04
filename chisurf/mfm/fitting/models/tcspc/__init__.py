"""
This module contains all time-resolved fluorescence models (TCSPC)

.. automodule:: models.tcspc
   :members:
"""
import mfm
import nusiance
#import mix_model
import fret
#import dye_diffusion
import tcspc
import parse
import pddem
#import et
#import membrane
#import fret_structure
from .. import ModelCurve

models = [
    tcspc.LifetimeModelWidget,
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

models += testing if mfm.cs_settings['experimental_models'] else []
