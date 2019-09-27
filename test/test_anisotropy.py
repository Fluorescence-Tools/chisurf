import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import numpy as np
import mfm.fluorescence


class Tests(unittest.TestCase):

    def test_kappa(self):
        distance_reference, kappa_reference = 0.8660254037844386, 1.0000000000000002
        donor_dipole = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ],
            dtype=np.float64
        )
        acceptor_dipole = np.array(
            [
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 1.0]
            ], dtype=np.float64
        )
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
        distance, kappa = mfm.fluorescence.anisotropy.kappa2.kappa_distance(
            donor_dipole[0],
            donor_dipole[1],
            acceptor_dipole[0],
            acceptor_dipole[1],
        )
        self.assertAlmostEqual(
            distance,
            distance_reference,
        )
        self.assertAlmostEqual(
            kappa,
            kappa_reference,
        )

    def test_vm_vv_vh(self):
        times = np.linspace(0, 50, 32)
        lifetime_spectrum = np.array([1., 4], dtype=np.float)
        times, vm = mfm.fluorescence.general.calculate_fluorescence_decay(
            lifetime_spectrum=lifetime_spectrum,
            time_axis=times
        )
        anisotropy_spectrum = np.array([0.1, 0.6, 0.38 - 0.1, 10.0])
        vv, vh = mfm.fluorescence.anisotropy.decay.vm_rt_to_vv_vh(
            times,
            vm,
            anisotropy_spectrum
        )
        vv_ref = np.array(
            [1.20000000e+00, 6.77248886e-01, 4.46852328e-01, 2.98312250e-01,
             1.99308989e-01, 1.33170004e-01, 8.89790065e-02, 5.94523193e-02,
             3.97237334e-02, 2.65418577e-02, 1.77342397e-02, 1.18493310e-02,
             7.91726329e-03, 5.29000820e-03, 3.53457826e-03, 2.36166808e-03,
             1.57797499e-03, 1.05434168e-03, 7.04470208e-04, 4.70699664e-04,
             3.14503256e-04, 2.10138875e-04, 1.40406645e-04, 9.38142731e-05,
             6.26830580e-05, 4.18823877e-05, 2.79841867e-05, 1.86979480e-05,
             1.24932435e-05, 8.34750065e-06, 5.57747611e-06, 3.72665317e-06]
        )
        vh_ref = np.array(
            [9.00000000e-01, 6.63617368e-01, 4.46232934e-01, 2.98284106e-01,
             1.99307711e-01, 1.33169946e-01, 8.89790039e-02, 5.94523192e-02,
             3.97237334e-02, 2.65418577e-02, 1.77342397e-02, 1.18493310e-02,
             7.91726329e-03, 5.29000820e-03, 3.53457826e-03, 2.36166808e-03,
             1.57797499e-03, 1.05434168e-03, 7.04470208e-04, 4.70699664e-04,
             3.14503256e-04, 2.10138875e-04, 1.40406645e-04, 9.38142731e-05,
             6.26830580e-05, 4.18823877e-05, 2.79841867e-05, 1.86979480e-05,
             1.24932435e-05, 8.34750065e-06, 5.57747611e-06, 3.72665317e-06]
        )
        self.assertEqual(
            np.allclose(
                vv_ref,
                vv
            ),
            True
        )
        self.assertEqual(
            np.allclose(
                vh_ref,
                vh
            ),
            True
        )
