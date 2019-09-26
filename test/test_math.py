import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import numpy as np
import mfm.math


class Tests(unittest.TestCase):

    def test_angle(self):
        a = np.array([0, 0, 0], dtype=np.float64)
        b = np.array([1, 0, 0], dtype=np.float64)
        c = np.array([0, 1, 0], dtype=np.float64)
        angle = mfm.math.linalg.angle(a, b, c) / np.pi * 360
        self.assertAlmostEqual(
            angle,
            90.0
        )

    def test_dihedral(self):
        a = np.array([-1, 1, 0], dtype=np.float64)
        b = np.array([-1, 0, 0], dtype=np.float64)
        c = np.array([0, 0, 0], dtype=np.float64)
        d = np.array([0, -1, 0], dtype=np.float64)
        dihedral = mfm.math.linalg.dihedral(a, b, c, d) / np.pi * 360
        self.assertAlmostEqual(
            dihedral,
            -360.0
        )
