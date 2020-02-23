import unittest
import numpy as np

import chisurf.fluorescence.fret.fret_line
from chisurf.fluorescence.fret.fret_line import FRETLineGenerator
import chisurf.models


class Tests(unittest.TestCase):

    def test_fret_line_1(self):
        # set rda-axis in chisurf.model.tcspc to avoid dependencies on
        # specific settings
        chisurf.models.tcspc.fret.rda_axis = np.logspace(
            start=np.log(1),
            stop=np.log(500)
        )

        n_points = 20
        parameter_range = (1.0, 100.0)
        fl = FRETLineGenerator(
            n_points=n_points,
            parameter_range=parameter_range
        )
        fl.model = chisurf.models.tcspc.fret.GaussianModel
        fl.model.gaussians.append(55.0, 10, 1.0)
        fl.model.find_parameters()
        fl.model.parameter_dict['xDOnly'].value = 0.0

        # The fluorescence/species averaged lifetime of the model is obtained by
        self.assertAlmostEqual(
            fl.fret_fluorescence_averaged_lifetime,
            2.5845124717847283
        )
        self.assertAlmostEqual(
            fl.fret_species_averaged_lifetime,
            2.2266377259027252
        )
        # The model parameters can be changed using their names by the parameter_dict
        fl.model.parameter_dict['R(G,1)'].value = 40.0
        fl.model.parameter_dict['s(G,1)'].value = 8.0
        self.assertAlmostEqual(
            fl.fret_fluorescence_averaged_lifetime,
            1.2711109746252425
        )
        self.assertAlmostEqual(
            fl.fret_species_averaged_lifetime,
            0.8311367744065555
        )

        # Set the name of the parameter changes to calculate the distributions used for the
        # FRET-lines. Here a more common parameter as the donor-acceptor separation distance
        # is used to generate a static FRET-line.
        fl.parameter_name = 'R(G,1)'

        # Set the range in which this parameter is modified to generate the
        # line and calculate the FRET-line
        fl.parameter_range = 0.1, 200
        fl.update()

        self.assertEqual(
            np.allclose(
                np.array([1.00000000e-01, 1.06210526e+01, 2.11421053e+01, 3.16631579e+01,
                          4.21842105e+01, 5.27052632e+01, 6.32263158e+01, 7.37473684e+01,
                          8.42684211e+01, 9.47894737e+01, 1.05310526e+02, 1.15831579e+02,
                          1.26352632e+02, 1.36873684e+02, 1.47394737e+02, 1.57915789e+02,
                          1.68436842e+02, 1.78957895e+02, 1.89478947e+02, 2.00000000e+02]),
                fl.parameter_values
            ),
            True
        )
        self.assertEqual(
            np.allclose(
                np.array([2.52930436e-03, 9.24226541e-03, 6.77909951e-02, 3.40748902e-01,
                          9.95018225e-01, 2.08997394e+00, 2.83589022e+00, 3.55135018e+00,
                          3.71638170e+00, 3.87966541e+00, 3.94678180e+00, 3.94818517e+00,
                          3.97928833e+00, 3.99082746e+00, 3.99086805e+00, 3.99087494e+00,
                          3.99634473e+00, 3.99841265e+00, 3.99841362e+00, 3.99841362e+00]),
                fl.species_averaged_lifetimes
            ),
            True
        )
        self.assertEqual(
            np.allclose(
                np.array([0.04708067, 0.11446077, 0.33399563, 0.80546468, 1.44127727,
                          2.43777787, 2.92090421, 3.58832524, 3.71851104, 3.88254752,
                          3.94684699, 3.94818914, 3.97937967, 3.99082789, 3.99086806,
                          3.99087495, 3.99634757, 3.99841266, 3.99841362, 3.99841362]),
                fl.fluorescence_averaged_lifetimes
            ),
            True
        )
        self.assertEqual(
            '-0.010955*x^0+0.107894*x^1+0.496252*x^2+-0.077779*x^3+0.002407*x^4',
            fl.conversion_function_string
        )
        self.assertEqual(
            '1.0-(-0.010955*x^0+0.107894*x^1+0.496252*x^2+-0.077779*x^3+0.002407*x^4)/(4.000000)',
            fl.transfer_efficency_string
        )
        self.assertEqual(
            '0.8/0.32 / ((4.0)/(-0.010955*x^0+0.107894*x^1+0.496252*x^2+-0.077779*x^3+0.002407*x^4) - 1)',
            fl.fdfa_string
        )

    def test_fret_line_2(self):
        chisurf.models.tcspc.fret.rda_axis = np.logspace(
            start=np.log(1),
            stop=np.log(500)
        )

        n_points = 20
        parameter_range = (1.0, 100.0)
        fl = FRETLineGenerator(
            n_points=n_points,
            parameter_range=parameter_range,
            model=chisurf.models.tcspc.fret.GaussianModel,
            verbose=True
        )
        fl.model.gaussians.append(55.0, 10, 1.0)
        fl.model.find_parameters()

        fl.model.parameter_dict['xDOnly'].value = 0.0
        fl.parameter_name = 'R(G,1)'

        # The fluorescence/species averaged lifetime of the model is obtained by
        self.assertAlmostEqual(
            fl.fret_fluorescence_averaged_lifetime,
            2.5845124717847283
        )
        self.assertAlmostEqual(
            fl.fret_species_averaged_lifetime,
            2.2266377259027252
        )
        fl.model.find_parameters()
        fl.update()

    def test_static_fret_line(self):
        import chisurf.fluorescence.fret.fret_line
        chisurf.models.tcspc.fret.rda_axis = np.logspace(
            start=np.log(1),
            stop=np.log(500)
        )
        fl = chisurf.fluorescence.fret.fret_line.StaticFRETLine()
        fl.update()
        self.assertEqual(
            '0.007988*x^0+-0.068925*x^1+0.437420*x^2+-0.009930*x^3+-0.008274*x^4',
            fl.conversion_function_string
        )

    def test_dynamic_fret_line(self):
        import chisurf.tools.fret_lines.fret_lines
        chisurf.models.tcspc.fret.rda_axis = np.logspace(
            start=np.log(1),
            stop=np.log(500)
        )
        fl = chisurf.fluorescence.fret.fret_line.DynamicFRETLine(
            n_points=10
        )
        fl.update()
        x, f = fl.conversion_function
        self.assertEqual(
            np.allclose(
                np.array([1.74621269, 3.64847205, 3.68138151, 3.69260583, 3.69826697,
                          3.70167948, 3.70396113, 3.70559414, 3.70682068, 3.70777573]),
                x
            ), True
        )
        self.assertEqual(
            np.allclose(
                np.array([1.02892012, 3.41195223, 3.55436922, 3.60570122, 3.63215778,
                          3.6482941, 3.65916325, 3.66698232, 3.67287719, 3.67748034]),
                f
            ), True
        )
