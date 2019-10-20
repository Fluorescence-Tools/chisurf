import utils
import os
import unittest

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)

import numpy as np
import chisurf.fio
import chisurf.fluorescence
import chisurf.fluorescence.fret
import chisurf.fluorescence.anisotropy


class Tests(unittest.TestCase):

    def test_s2delta(self):
        r0 = 0.38
        s2donor = 0.3
        s2acceptor = 0.3
        r_inf_AD = 0.05
        v = chisurf.fluorescence.anisotropy.kappa2.s2delta(
            r_0=r0,
            s2donor=s2donor,
            s2acceptor=s2acceptor,
            r_inf_AD=r_inf_AD)
        self.assertAlmostEqual(
            v,
            1.4619883040935675
        )

    def test_p_isotropic_orientation_factor(self):
        k2 = np.linspace(0.1, 4, 32)
        p_k2 = chisurf.fluorescence.anisotropy.kappa2.p_isotropic_orientation_factor(
            k2=k2
        )
        p_k2_ref = np.array(
            [0.17922824, 0.11927194, 0.09558154, 0.08202693, 0.07297372,
               0.06637936, 0.06130055, 0.05723353, 0.04075886, 0.03302977,
               0.0276794, 0.02359627, 0.02032998, 0.01763876, 0.01537433,
               0.01343829, 0.01176177, 0.01029467, 0.00899941, 0.00784718,
               0.00681541, 0.00588615, 0.00504489, 0.0042798, 0.0035811,
               0.00294063, 0.00235153, 0.001808, 0.00130506, 0.00083845,
               0.0004045, 0.]
        )
        self.assertEqual(
            np.allclose(
                p_k2_ref, p_k2
            ),
            True
        )

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
        distance, kappa = chisurf.fluorescence.anisotropy.kappa2.kappa(
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
        distance, kappa = chisurf.fluorescence.anisotropy.kappa2.kappa_distance(
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
        times, vm = chisurf.fluorescence.general.calculate_fluorescence_decay(
            lifetime_spectrum=lifetime_spectrum,
            time_axis=times
        )
        anisotropy_spectrum = np.array([0.1, 0.6, 0.38 - 0.1, 10.0])
        vv, vh = chisurf.fluorescence.anisotropy.decay.vm_rt_to_vv_vh(
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

    def test_fcs(self):
        import numpy as np
        import glob
        import chisurf.fluorescence
        import chisurf.fio
        import pylab as p
        directory = './test/data/tttr/BH/132/'
        spc_files = glob.glob(directory + '/BH_SPC132.spc')
        photons = chisurf.fio.photons.Photons(spc_files, reading_routine="bh132")
        cr_filter = np.ones_like(photons.mt, dtype=np.float)
        w1 = np.ones_like(photons.mt, dtype=np.float)
        w2 = np.ones_like(photons.mt, dtype=np.float)
        points_per_decade = 5
        number_of_decades = 10
        results = chisurf.fluorescence.fcs.correlate.log_corr(
            macro_times=photons.mt,
            tac_channels=photons.tac,
            rout=photons.rout,
            cr_filter=cr_filter,
            weights_1=w1,
            weights_2=w2,
            B=points_per_decade,
            nc=number_of_decades,
            fine=False,
            number_of_tac_channels=photons.n_tac
        )
        np_1 = results['number_of_photons_ch1']
        np_2 = results['number_of_photons_ch2']
        dt_1 = results['measurement_time_ch1']
        dt_2 = results['measurement_time_ch2']
        tau = results['correlation_time_axis']
        corr = results['correlation_amplitude']
        cr = chisurf.fluorescence.fcs.correlate.normalize(
            np_1, np_2, dt_1, dt_2, tau, corr, points_per_decade
        )
        cr /= photons.dt
        dur = float(min(dt_1, dt_2)) * photons.dt / 1000.  # seconds
        tau = tau.astype(np.float64)
        tau *= photons.dt

    def test_acceptor(self):
        times = np.linspace(0, 50, 1024)
        tau_da = 3.5
        decay_da = np.exp(-times / tau_da)
        acceptor_lifetime_spectrum = np.array(
            [0.1, 0.5, 0.9, 2.0]
        )

        transfer_efficiency = 0.3
        decay_ad = chisurf.fluorescence.fret.acceptor.da_a0_to_ad(
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
            scaled_acceptor = chisurf.fluorescence.fret.acceptor.scale_acceptor(
                donor=decay_da,
                acceptor=decay_ad,
                transfer_efficiency=target_value
            )
            eff = sum(scaled_acceptor) / (sum(scaled_acceptor) + sum(decay_da))
            self.assertAlmostEqual(
                eff,
                target_value
            )
