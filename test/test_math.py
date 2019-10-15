import utils
import os
import unittest

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../chisurf/')
)
utils.set_search_paths(TOPDIR)

import numpy as np
import scipy.stats.distributions

import chisurf.math
import chisurf.math.signal
import chisurf.curve


class Tests(unittest.TestCase):

    def test_angle(self):
        a = np.array([0, 0, 0], dtype=np.float64)
        b = np.array([1, 0, 0], dtype=np.float64)
        c = np.array([0, 1, 0], dtype=np.float64)
        angle = chisurf.math.linalg.angle(a, b, c) / np.pi * 360
        self.assertAlmostEqual(
            angle,
            90.0
        )

    def test_dihedral(self):
        a = np.array([-1, 1, 0], dtype=np.float64)
        b = np.array([-1, 0, 0], dtype=np.float64)
        c = np.array([0, 0, 0], dtype=np.float64)
        d = np.array([0, -1, 0], dtype=np.float64)
        dihedral = chisurf.math.linalg.dihedral(a, b, c, d) / np.pi * 360
        self.assertAlmostEqual(
            dihedral,
            -360.0
        )

    def test_vec3(self):
        a = np.array([0, 0, 0], dtype=np.float64)
        b = np.array([1, 0, 0], dtype=np.float64)
        dist = chisurf.math.linalg.dist3(a, b)
        self.assertEqual(
            dist,
            1.0
        )
        c = chisurf.math.linalg.sub3(a, b)
        self.assertEqual(
            np.allclose(
                c,
                a - b
            ),
            True
        )
        c = chisurf.math.linalg.add3(a, b)
        self.assertEqual(
            np.allclose(
                c,
                a + b
            ),
            True
        )
        c = chisurf.math.linalg.dot3(a, b)
        self.assertEqual(
            np.allclose(
                c,
                np.dot(a, b)
            ),
            True
        )

    def test_fwhm(self):
        x = np.linspace(0, 10, 100)
        y = scipy.stats.distributions.norm.pdf(
            x,
            loc=5,
            scale=2
        )
        c2 = chisurf.curve.Curve(x, y)
        fwhm, (i_lower, i_upper), (f_lower, f_upper) = chisurf.math.signal.calculate_fwhm(
            c2,
            background=0.0,
            verbose=False
        )
        self.assertEqual(fwhm, 4.545454545454545)
        self.assertEqual(i_lower, 27)
        self.assertEqual(i_upper, 72)
        self.assertEqual(f_lower, 2.727272727272727)
        self.assertEqual(f_upper, 7.2727272727272725)


