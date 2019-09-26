import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm
import mfm.curve
import mfm.math.signal


class Tests(unittest.TestCase):

    def test_fwhm(self):
        import numpy as np
        import scipy.stats.distributions

        x = np.linspace(0, 10, 100)
        y = scipy.stats.distributions.norm.pdf(
            x,
            loc=5,
            scale=2
        )
        c2 = mfm.curve.Curve(x, y)
        fwhm, (i_lower, i_upper), (f_lower, f_upper) = mfm.math.signal.calculate_fwhm(
            c2,
            background=0.0,
            verbose=False
        )
        self.assertEqual(fwhm, 4.545454545454545)
        self.assertEqual(i_lower, 27)
        self.assertEqual(i_upper, 72)
        self.assertEqual(f_lower, 2.727272727272727)
        self.assertEqual(f_upper, 7.2727272727272725)
