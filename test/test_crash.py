import os
import sys

sys.path.insert(0, os.path.abspath('e:/dev/cs'))
import logging
logging.basicConfig(level=logging.INFO)
import chisurf as cs
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.models.tcspc.fret import GaussianModel
import traceback
import faulthandler

faulthandler.enable()

def test_global_linking():
    fit1 = Fit()
    fit2 = Fit()
    fit_group = FitGroup([fit1, fit2])
    
    m1 = GaussianModel(fit=fit1)
    m2 = GaussianModel(fit=fit2)
    
    # Check if add_model exists or use append_fit on _model
    if hasattr(fit_group, 'add_model'):
        fit_group.add_model(m1)
        fit_group.add_model(m2)
    else:
        fit_group._model.append_fit(fit1)
        fit_group._model.append_fit(fit2)
        fit1.model = m1
        fit2.model = m2

if __name__ == '__main__':
    test_global_linking()
    print("Test passed!")
