import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import numpy as np
import mfm.fluorescence


class Tests(unittest.TestCase):

    def test_fit_init(self):
        distance_reference, kappa_reference = 0.8660254037844386, 1.0000000000000002
        donor_dipole = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        acceptor_dipole = np.array([[0.0, 0.5, 0.0], [0.0, 0.5, 1.0]], dtype=np.float64)
        distance, kappa = mfm.fluorescence.anisotropy.kappa2.kappa(
            donor_dipole,
            acceptor_dipole
        )
        self.assertAlmostEqual(
            distance,
            distance_reference,
        )
        self.assertAlmostEqual(
            kappa,
            kappa_reference,
        )
