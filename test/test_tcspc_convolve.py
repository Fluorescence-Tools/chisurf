import utils
import os
import unittest
import numpy as np
import scipy.stats


TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)

import chisurf.fluorescence
import chisurf.fluorescence.tcspc.convolve


class Tests(unittest.TestCase):

    def test_convolve_lifetime_spectrum(self):
        # normal convolution (not periodic)
        time_axis = np.linspace(0, 50, 64)
        irf_position = 5.0
        irf_width = 1.0
        irf = scipy.stats.norm.pdf(time_axis, loc=irf_position, scale=irf_width)
        lifetime_spectrum = np.array([0.8, 1.1, 0.2, 4.0])
        decay = np.zeros_like(time_axis)
        chisurf.fluorescence.tcspc.convolve.convolve_lifetime_spectrum(
            decay,
            lifetime_spectrum=lifetime_spectrum,
            irf=irf,
            time_axis=time_axis
        )
        reference = np.array(
            [0.00000000e+00, 2.31003027e-05, 4.93647051e-04, 5.66130319e-03,
             3.59025860e-02, 1.29567103e-01, 2.78241739e-01, 3.79536468e-01,
             3.59991962e-01, 2.67999476e-01, 1.80082384e-01, 1.21758553e-01,
             8.63799410e-02, 6.42284492e-02, 4.94564785e-02, 3.89942937e-02,
             3.12175855e-02, 2.52305284e-02, 2.05105451e-02, 1.67321564e-02,
             1.36785440e-02, 1.11962524e-02, 9.17127452e-03, 7.51587074e-03,
             6.16088666e-03, 5.05097177e-03, 4.14139705e-03, 3.39580416e-03,
             2.78453404e-03, 2.28334115e-03, 1.87238019e-03, 1.53539533e-03,
             1.25906496e-03, 1.03246918e-03, 8.46655362e-04, 6.94283101e-04,
             5.69333503e-04, 4.66871126e-04, 3.82848866e-04, 3.13948029e-04,
             2.57447205e-04, 2.11114770e-04, 1.73120726e-04, 1.41964421e-04,
             1.16415276e-04, 9.54641762e-05, 7.82836177e-05, 6.41950212e-05,
             5.26419303e-05, 4.31680335e-05, 3.53991411e-05, 2.90284058e-05,
             2.38042030e-05, 1.95201929e-05, 1.60071703e-05, 1.31263816e-05,
             1.07640445e-05, 8.82685402e-06, 7.23829709e-06, 5.93563059e-06,
             4.86740321e-06, 3.99142326e-06, 3.27309224e-06, 2.68403827e-06]
        )
        self.assertEqual(
            np.allclose(decay, reference),
            True
        )

    def test_convolve_lifetime_spectrum_periodic(self):
        # normal convolution (not periodic)
        import scipy.stats
        time_axis = np.linspace(0, 16, 64)
        dt = time_axis[1] - time_axis[0]
        irf_position = 5.0
        irf_width = 1.0
        irf = scipy.stats.norm.pdf(time_axis, loc=irf_position, scale=irf_width)
        lifetime_spectrum = np.array([0.8, 1.1, 0.2, 4.0])
        decay = np.zeros_like(time_axis)
        n_decay = decay.shape[0]
        chisurf.fluorescence.tcspc.convolve.convolve_lifetime_spectrum_periodic(
            decay,
            lifetime_spectrum=lifetime_spectrum,
            irf=irf,
            start=0,
            stop=n_decay,
            n_points=n_decay,
            period=16,
            conv_stop=n_decay,
            dt=dt
        )
        reference = np.array(
            [0.01265501, 0.01187065, 0.01113791, 0.01045646, 0.00983215,
             0.00928425, 0.00886104, 0.00866889, 0.0089203, 0.01000387,
             0.01256884, 0.01760087, 0.02644374, 0.04070815, 0.06201623,
             0.09157463, 0.12964723, 0.17508306, 0.22510068, 0.27549207,
             0.32127994, 0.35768572, 0.38112437, 0.38991182, 0.38447305,
             0.36702632, 0.34090036, 0.30973841, 0.27682646, 0.24468516,
             0.21494478, 0.18843437, 0.16538203, 0.14563559, 0.12884727,
             0.11460187, 0.10249176, 0.0921526, 0.08327483, 0.07560255,
             0.06892714, 0.06307955, 0.05792301, 0.05334674, 0.04926082,
             0.04559205, 0.04228057, 0.03927727, 0.03654167, 0.03404019,
             0.03174486, 0.02963222, 0.02768248, 0.02587883, 0.0242069,
             0.02265429, 0.02121027, 0.01986547, 0.01861165, 0.01744149,
             0.01634852, 0.01532691, 0.01437142, 0.01324633]
        )
        self.assertEqual(
            np.allclose(decay, reference),
            True
        )

    def test_convolve_decay(self):
        n_points = 64
        time_axis = np.linspace(0, 16, n_points)
        irf_position = 5.0
        irf_width = 0.5
        dt = time_axis[1] - time_axis[0]
        irf = scipy.stats.norm.pdf(time_axis, loc=irf_position, scale=irf_width)
        decay_u = np.exp(- time_axis / 4.1)
        decay = chisurf.fluorescence.tcspc.convolve.convolve_decay(
            decay_u,
            irf=irf,
            start=0,
            stop=n_points,
            dt=dt
        )
        reference = np.array(
            [0.00000000e+00, 2.77819439e-21, 3.06332831e-19, 2.59556233e-17,
             1.70152833e-15, 8.63429983e-14, 3.39340281e-12, 1.03364648e-10,
             2.44248333e-09, 4.48248284e-08, 6.39847217e-07, 7.11741328e-06,
             6.18433221e-05, 4.21016564e-04, 2.25423039e-03, 9.53868022e-03,
             3.20952829e-02, 8.65558481e-02, 1.89037122e-01, 3.38963884e-01,
             5.08254614e-01, 6.52897353e-01, 7.40765250e-01, 7.68647246e-01,
             7.54802676e-01, 7.20771357e-01, 6.80570553e-01, 6.40352746e-01,
             6.02000410e-01, 5.65856083e-01, 5.31869941e-01, 4.99923824e-01,
             4.69896413e-01, 4.41672562e-01, 4.15143947e-01, 3.90208747e-01,
             3.66771254e-01, 3.44741510e-01, 3.24034961e-01, 3.04572129e-01,
             2.86278313e-01, 2.69083297e-01, 2.52921081e-01, 2.37729633e-01,
             2.23450643e-01, 2.10029307e-01, 1.97414110e-01, 1.85556631e-01,
             1.74411360e-01, 1.63935519e-01, 1.54088898e-01, 1.44833704e-01,
             1.36134415e-01, 1.27957639e-01, 1.20271993e-01, 1.13047978e-01,
             1.06257866e-01, 9.98755948e-02, 9.38766684e-02, 8.82380615e-02,
             8.29381318e-02, 7.79565370e-02, 7.32741566e-02, 6.88730186e-02]
        )
        self.assertEqual(
            np.allclose(decay, reference),
            True
        )


if __name__ == '__main__':
    unittest.main()

