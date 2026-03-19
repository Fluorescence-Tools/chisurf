import os
import sys

sys.path.insert(0, os.path.abspath('e:/dev/chisurf'))
import logging
logging.basicConfig(level=logging.INFO)
import chisurf
from chisurf.fitting.fit import Fit, FitGroup
from chisurf.models.tcspc.fret import GaussianModel
import traceback
import faulthandler

faulthandler.enable()

fit1 = Fit()
fit2 = Fit()
fit_group = FitGroup([fit1, fit2])

try:
    print('Creating model 1...')
    m1 = GaussianModel(fit=fit1)
    print('Creating model 2...')
    m2 = GaussianModel(fit=fit2)
    print('link')
    fit_group.add_model(m1)
    fit_group.add_model(m2)
    print('All good')
except Exception as e:
    traceback.print_exc()
