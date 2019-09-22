import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import numpy as np
import mfm.fluorescence.fret


class Tests(unittest.TestCase):

    def test_acceptor(self):
        times = np.linspace(0, 50, 1024)
        tau_da = 3.5
        decay_da = np.exp(-times / tau_da)
        acceptor_lifetime_spectrum = np.array([0.1, 0.5, 0.9, 2.0])

        transfer_efficiency = 0.3
        decay_ad = mfm.fluorescence.fret.acceptor.da_a0_to_ad(
            times=times,
            decay_da=decay_da,
            acceptor_lifetime_spectrum=acceptor_lifetime_spectrum,
            transfer_efficiency=transfer_efficiency
        )
        eff = decay_ad.sum() / (decay_ad.sum() + decay_da.sum())

        self.assertAlmostEqual(
            eff,
            transfer_efficiency
        )

        for target_value in np.linspace(0.1, 0.9):
            scaled_acceptor = mfm.fluorescence.fret.acceptor.scale_acceptor(
                donor=decay_da,
                acceptor=decay_ad,
                transfer_efficiency=target_value
            )
            eff = sum(scaled_acceptor) / (sum(scaled_acceptor) + sum(decay_da))
            self.assertAlmostEqual(
                eff,
                target_value
            )
