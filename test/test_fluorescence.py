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

from chisurf.fluorescence.anisotropy.decay import calculcate_spectrum
from chisurf.fluorescence.tcspc.corrections import compute_linearization_table


class Tests(unittest.TestCase):

    def test_kappasq_all(self):
        k2_scale, k2_hist, k2 = chisurf.fluorescence.anisotropy.kappa2.kappasq_all(
            sD2=0.3,
            sA2=0.5,
            n_bins=31,
            n_samples=10000
        )
        self.assertEqual(
            np.allclose(
                k2_scale,
                np.array(
                    [0., 0.13333333, 0.26666667, 0.4, 0.53333333,
                     0.66666667, 0.8, 0.93333333, 1.06666667, 1.2,
                     1.33333333, 1.46666667, 1.6, 1.73333333, 1.86666667,
                     2., 2.13333333, 2.26666667, 2.4, 2.53333333,
                     2.66666667, 2.8, 2.93333333, 3.06666667, 3.2,
                     3.33333333, 3.46666667, 3.6, 3.73333333, 3.86666667,
                     4.])
            ),
            True
        )

        reference = np.array(
            [0.000e+00, 0.000e+00, 0.000e+00, 3.156e+03, 4.381e+03, 1.453e+03,
             5.770e+02, 2.660e+02, 1.090e+02, 4.300e+01, 1.400e+01, 1.000e+00,
             0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
             0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
             0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00]
        )
        self.assertEqual(
            np.allclose(reference, k2_hist, rtol=0.35, atol=10.0),
            True
        )

    def test_s2delta(self):
        r0 = 0.38
        s2_donor = 0.2
        s2_acceptor = 0.3
        r_inf_AD = 0.01
        v = chisurf.fluorescence.anisotropy.kappa2.s2delta(
            s2_donor=s2_donor,
            s2_acceptor=s2_acceptor,
            r_inf_AD=r_inf_AD,
            r_0=r0
        )
        self.assertTupleEqual(
            v,
            (0.4385964912280701, 0.6583029208008411)
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

    def test_fluorescence_anisotropy_decay_calculcate_spectrum(self):
        lifetime_spectrum = np.array([1.0, 4.0])
        anisotropy_spectrum = np.array([1.0, 1.0])
        g_factor = 1.5
        a = calculcate_spectrum(
            lifetime_spectrum=lifetime_spectrum,
            anisotropy_spectrum=anisotropy_spectrum,
            polarization_type='VV',
            g_factor=g_factor,
            l1=0.0,
            l2=0.0
        )
        self.assertEqual(
            np.allclose(
                a,
                np.array([1., 4., 2., 0.8, 0., 4., -0., 0.8])
            ),
            True
        )

        a = calculcate_spectrum(
            lifetime_spectrum=lifetime_spectrum,
            anisotropy_spectrum=anisotropy_spectrum,
            polarization_type='VV',
            g_factor=g_factor,
            l1=0.1,
            l2=0.0
        )
        self.assertEqual(
            np.allclose(
                a,
                np.array([0.9, 4., 1.8, 0.8, 0.15, 4., -0.3, 0.8])
            ),
            True
        )

        a = calculcate_spectrum(
            lifetime_spectrum=lifetime_spectrum,
            anisotropy_spectrum=anisotropy_spectrum,
            polarization_type='VH',
            g_factor=g_factor,
            l1=0.0,
            l2=0.0
        )
        self.assertEqual(
            np.allclose(
                a,
                np.array([0., 4., 0., 0.8, 1.5, 4., -3., 0.8])
            ),
            True
        )

        a = calculcate_spectrum(
            lifetime_spectrum=lifetime_spectrum,
            anisotropy_spectrum=anisotropy_spectrum,
            polarization_type='VH',
            g_factor=g_factor,
            l1=0.0,
            l2=0.1
        )

        self.assertEqual(
            np.allclose(
                a,
                np.array([0.1, 4., 0.2, 0.8, 1.35, 4., -2.7, 0.8])
            ),
            True
        )

        self.assertEqual(
            np.allclose(
                calculcate_spectrum(
                    lifetime_spectrum=lifetime_spectrum,
                    anisotropy_spectrum=anisotropy_spectrum,
                    polarization_type='AAA',
                    g_factor=g_factor,
                    l1=0.0,
                    l2=0.1
                ),
                lifetime_spectrum
            ),
            True
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
        photons = chisurf.fio.fluorescence.photons.Photons(spc_files, reading_routine="bh132")
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

    def test_fluorescence_tcspc_corrections_compute_linearization_table(self):
        x = np.linspace(0, 40, 128)
        dnl_fraction = 0.01
        counts = 10000
        mean = np.sin(x) * dnl_fraction * counts + (1 - dnl_fraction) * counts
        np.random.seed(0)
        data = np.random.poisson(mean).astype(np.float64)
        lin_table = compute_linearization_table(data, 12, "hanning", 10, 90)
        ref_lintable = np.array(
            [1., 1., 1., 1., 1.,
             1., 0.99980049, 0.99922759, 0.99823618, 0.99681489,
             0.99518504, 0.99393009, 0.99341097, 0.99356403, 0.99429588,
             0.99518651, 0.99570281, 0.99588077, 0.99593212, 0.99620644,
             0.99725223, 0.99907418, 1.00121338, 1.00320785, 1.00486314,
             1.00613769, 1.00711954, 1.00770543, 1.00739009, 1.00586985,
             1.00286934, 0.99849134, 0.99377306, 0.98981455, 0.98744986,
             0.98683355, 0.9874091, 0.98868764, 0.99063644, 0.99339915,
             0.99669779, 1.00031741, 1.00394849, 1.00699941, 1.00914148,
             1.01040307, 1.01119998, 1.01183167, 1.01202324, 1.01120659,
             1.00905205, 1.0055979, 1.0012921, 0.99701193, 0.99370229,
             0.99214721, 0.99253283, 0.99448162, 0.99720137, 0.99972508,
             1.001355, 1.00164798, 1.00115969, 1.00083382, 1.00120239,
             1.0025449, 1.00450071, 1.00633953, 1.00702856, 1.00606332,
             1.00373424, 1.00058981, 0.99734557, 0.99442657, 0.9925278,
             0.99219378, 0.99332066, 0.9954457, 0.99805603, 1.00079518,
             1.00285328, 1.00367601, 1.00341471, 1.00277917, 1.00262715,
             1.00308127, 1.00383407, 1.00447143, 1.00466959, 1.00415533,
             1.00289327, 1.00144407, 1.00036365, 0.99984595, 0.99978234,
             0.99991273, 1., 1., 1., 1.,
             1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1.,
             1., 1., 1.]
        )
        self.assertEqual(
            np.allclose(
                lin_table,
                ref_lintable
            ),
            True
        )
