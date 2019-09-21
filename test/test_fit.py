import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import numpy as np
import mfm


class Tests(unittest.TestCase):

    def test_fit_init(self):
        x = np.linspace(0, 32, 1024)
        tau = 4.1
        y = np.exp(-x / tau)
        data = mfm.experiments.data.DataCurve(
            x=x,
            y=y,
            ey=np.sqrt(y)
        )
        fit = mfm.fitting.fit.FitGroup(
            data=mfm.experiments.data.DataGroup(
                [data]
            ),
            model_class=mfm.models.parse.ParseModel
        )
