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
        c_value = 3.1
        a_value = 1.2
        y_data = c_value + a_value * x**2.0

        data = mfm.experiments.data.DataCurve(
            x=x,
            y=y_data,
            ey=np.ones_like(y_data)
        )
        fit = mfm.fitting.fit.FitGroup(
            data=mfm.experiments.data.DataGroup(
                [data]
            ),
            model_class=mfm.models.parse.ParseModel
        )
        model = fit.model
        model.parse.func = 'c+a*x**2'
        self.assertEqual(
            len(model.parameters),
            0
        )
        model.find_parameters()
        self.assertEqual(
            len(model.parameters),
            2
        )
        self.assertSetEqual(
            set(model.parameter_names),
            set(['a', 'c'])
        )
        self.assertEqual(
            np.allclose(
                model.y,
                np.zeros_like(model.y)
            ),
            True
        )
        model.update()
        y_model = model.parameter_dict['c'] + model.parameter_dict['a'] * x**2.0
        self.assertEqual(
            np.allclose(
                y_model.value,
                model.y
            ),
            True
        )
