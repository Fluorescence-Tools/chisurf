import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm


class Tests(unittest.TestCase):

    def test_curve_init(self):
        import numpy as np

        c1 = mfm.curve.Curve()
        self.assertEqual(c1.x, np.array([], dtype=np.float))
        self.assertEqual(c1.y, np.array([], dtype=np.float))
        self.assertEqual(c1.name, 'Curve')

        x = np.linspace(0, 2. * np.pi, 100)
        y = np.sin(x)
        c1.x = x
        c1.y = y

        name = 'C2'
        c2 = mfm.curve.Curve(x, y, name=name)
        self.assertEqual(c1.x, c2.x)
        self.assertEqual(c1.y, c2.y)
        self.assertEqual(c2.name, name)

    def test_attributes(self):
        import numpy as np
        import scipy.stats
        import mfm.math.signal

        x = np.linspace(0, 10, 100)
        y = scipy.stats.distributions.norm.pdf(
            x,
            loc=5,
            scale=2
        )
        c2 = mfm.curve.Curve(x, y)
        self.assertEqual(
            c2.fwhm,
            mfm.math.signal.calculate_fwhm(
                c2
            )[0]
        )

        cdf = c2.cdf
        self.assertEqual(
            cdf.y,
            np.cumsum(c2.y)
        )
        # STOP


if __name__ == '__main__':
    unittest.main()
