import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import numpy as np
import tempfile

import mfm


class Tests(unittest.TestCase):

    def test_curve_init(self):
        c1 = mfm.curve.Curve()
        self.assertEqual(
            np.array_equal(
                c1.x, np.array([], dtype=np.float64)
            ),
            True
        )
        self.assertEqual(
            np.array_equal(
                c1.y, np.array([], dtype=np.float64)
            ),
            True
        )
        self.assertEqual(c1.name, 'Curve')

        c1 = mfm.curve.Curve()
        x = np.linspace(0, 2. * np.pi, 100)
        y = np.sin(x)
        c1.x = x
        c1.y = y

        name = 'C2'
        c2 = mfm.curve.Curve(x, y, name=name)
        self.assertEqual(
            np.array_equal(c1.x, c2.x), True
        )
        self.assertEqual(
            np.array_equal(c1.y, c2.y),
            True
        )
        self.assertEqual(c2.name, name)

    def test_attributes(self):
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
            np.array_equal(cdf.y, np.cumsum(c2.y)),
            True
        )

        self.assertEqual(
            np.array_equal(cdf.dx, np.diff(c2.x)),
            True
        )

    def test_reading(self):
        file = tempfile.NamedTemporaryFile(suffix='.yaml')
        filename = file.name

        x = np.linspace(0, 2. * np.pi, 100)
        y = np.sin(x)
        c1 = mfm.curve.Curve(x, y)

        c1.save(
            filename=filename,
            file_type='yaml'
        )

        c2 = mfm.curve.Curve()
        c2.load(
            filename=filename,
            file_type='yaml'
        )
        self.assertEqual(
            np.array_equal(c2.x, c1.x),
            True
        )
        self.assertEqual(
            np.array_equal(c2.y, c1.y),
            True
        )
        self.assertEqual(
            c2.unique_identifier,
            c1.unique_identifier
        )

    def test_normalize(self):
        pass


if __name__ == '__main__':
    unittest.main()
