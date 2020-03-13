import unittest
import numpy as np
import glob
import scipy.stats

import chisurf.fio
import chisurf.fluorescence
import chisurf.fluorescence.fret
import chisurf.fluorescence.fcs
import chisurf.fluorescence.tcspc
import chisurf.fluorescence.general
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
        directory = './test/data/tttr/BH/132/'
        spc_files = glob.glob(directory + '/BH_SPC132.spc')
        photons = chisurf.fio.photons.Photons(spc_files, reading_routine="bh132")
        cr_filter = np.ones_like(photons.macro_times, dtype=np.float)
        w1 = np.ones_like(photons.macro_times, dtype=np.float)
        w2 = np.ones_like(photons.macro_times, dtype=np.float)
        points_per_decade = 5
        number_of_decades = 10
        results = chisurf.fluorescence.fcs.correlate.log_corr(
            macro_times=photons.macro_times,
            tac_channels=photons.micro_times,
            rout=photons.routing_channels,
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

    def calculate_fcs_filter(self):
        lifetime_1 = 1.0
        lifetime_2 = 3.0
        times = np.linspace(0, 20, num=10)
        d1 = np.exp(-times / lifetime_1)
        d2 = np.exp(-times / lifetime_2)
        decays = [d1, d2]
        w1 = 0.8  # weight of first component
        experimental_decay = w1 * d1 + (1.0 - w1) * d2
        filters = chisurf.fluorescence.fcs.filtered.calc_lifetime_filter(decays, experimental_decay)
        self.assertEqual(
            np.allclose(
                np.array([[1.19397553, -0.42328685, -1.94651679, -2.57788423, -2.74922322,
                           -2.78989942, -2.79923872, -2.80136643, -2.80185031, -2.80196031],
                          [-0.19397553, 1.42328685, 2.94651679, 3.57788423, 3.74922322,
                           3.78989942, 3.79923872, 3.80136643, 3.80185031, 3.80196031]]),
                filters
            ),
            True
        )

    def convolve_lifetime_spectrum(self):
        reference_decay = np.array(
            [0.00000000e+00, 4.52643742e-06, 4.30136935e-05, 3.02142457e-04,
             1.65108038e-03, 7.06796476e-03, 2.38186826e-02, 6.35639959e-02,
             1.35361393e-01, 2.32352532e-01, 3.25883836e-01, 3.80450045e-01,
             3.79182520e-01, 3.33586262e-01, 2.69710789e-01, 2.08975297e-01,
             1.60624297e-01, 1.25068007e-01, 9.94465318e-02, 8.07767559e-02,
             6.68428191e-02, 5.61578412e-02, 4.77471111e-02, 4.09691949e-02,
             3.53963836e-02, 3.07383493e-02, 2.67937209e-02, 2.34193202e-02,
             2.05105461e-02, 1.79888012e-02, 1.57933761e-02, 1.38761613e-02,
             1.21981607e-02, 1.07271560e-02, 9.43611289e-03, 8.30206791e-03,
             7.30533192e-03, 6.42890327e-03, 5.65802363e-03, 4.97983239e-03,
             4.38309103e-03, 3.85795841e-03, 3.39580432e-03, 2.98905244e-03,
             2.63104654e-03, 2.31593553e-03, 2.03857411e-03, 1.79443629e-03,
             1.57954010e-03, 1.39038167e-03]
        )
        time_axis = np.linspace(0, 25, 50)
        irf_position = 5.0
        irf_width = 1.0
        irf = scipy.stats.norm.pdf(time_axis, loc=irf_position, scale=irf_width)
        lifetime_spectrum = np.array([0.8, 1.1, 0.2, 4.0])
        model_decay = np.zeros_like(time_axis)
        chisurf.fluorescence.tcspc.convolve.convolve_lifetime_spectrum(
            model_decay,
            lifetime_spectrum=lifetime_spectrum,
            instrument_response_function=irf,
            time_axis=time_axis
        )
        self.assertEqual(
            np.allclose(
                model_decay,
                reference_decay
            ),
            True
        )
