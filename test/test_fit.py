import utils
import os
import unittest

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)

import tempfile
import numpy as np
import copy

import chisurf.experiments
import chisurf.models
import chisurf.fitting


def get_data_values(
        c_value: float = 3.1,
        a_value: float = 1.2,
        n_points: int = 32
):
    x_data = np.linspace(0, 32, n_points)
    y_data = c_value + a_value * x_data ** 2.0
    return x_data, y_data


class FitTests(unittest.TestCase):

    def test_data_group(self):
        a_value = 1.2
        c_value = 3.1
        x_data, y_data = get_data_values(
            a_value=a_value,
            c_value=c_value
        )
        data = chisurf.experiments.data.DataCurve(
            x=x_data,
            y=y_data,
            ey=np.ones_like(y_data)
        )
        data2 = copy.copy(data)
        self.assertEqual(
            np.allclose(
                data2.y,
                data.y
            ),
            True
        )
        self.assertEqual(
            np.allclose(
                data2.x,
                data.x
            ),
            True
        )
        return data, data2

    def test_fit_parse(self):
        test = True
        a_value = 1.2
        c_value = 3.1

        x_data, y_data = get_data_values(
            a_value=a_value,
            c_value=c_value
        )
        data = chisurf.experiments.data.DataCurve(
            x=x_data,
            y=y_data,
            ey=np.ones_like(y_data)
        )
        fit = chisurf.fitting.fit.FitGroup(
            data=chisurf.experiments.data.DataGroup(
                [data]
            ),
            model_class=chisurf.models.parse.ParseModel
        )
        model = fit.model
        model.func = 'c+a*x**2'
        if test:
            self.assertEqual(
                len(model.parameters),
                2
            )
            self.assertSetEqual(
                set(model.parameter_names),
                {'a', 'c'}
            )
            self.assertEqual(
                np.allclose(
                    model.y,
                    np.zeros_like(model.y)
                ),
                True
            )
        fit_range = 0, len(model.y) - 1
        fit.fit_range = fit_range
        y_model = model.parameter_dict['c'].value + model.parameter_dict['a'].value * x_data ** 2.0
        fit.model.update_model()
        if test:
            self.assertEqual(
                np.allclose(
                    y_model,
                    model.y
                ),
                True
            )
            self.assertTupleEqual(
                fit.fit_range,
                fit_range
            )

        # The fit range is bounded to the size of the data
        fit.fit_range = -10, len(model.y) + 30
        if test:
            self.assertTupleEqual(
                fit.fit_range,
                fit_range
            )
            self.assertAlmostEqual(
                fit.chi2,
                248125.85601591066
            )
            self.assertEqual(
                np.allclose(
                    fit.weighted_residuals.y,
                    np.array(
                        [2.1, 2.31311134, 2.95244537, 4.01800208,
                         5.50978148, 7.42778356, 9.77200832, 12.54245578,
                         15.73912591, 19.36201873, 23.41113424, 27.88647242,
                         32.7880333, 38.11581686, 43.8698231, 50.05005203,
                         56.65650364, 63.68917794, 71.14807492, 79.03319459,
                         87.34453694, 96.08210198, 105.2458897, 114.8359001,
                         124.85213319, 135.29458897, 146.16326743,
                         157.45816857,
                         169.1792924, 181.32663892, 193.90020812]
                    )
                ),
                True
            )

        fit.run()
        chi2 = fit.chi2
        chi2r = chi2 / float(model.n_points - model.n_free - 1.0)
        if test:
            self.assertAlmostEqual(
                fit.chi2r,
                chi2r
            )
        fit.run()
        if test:
            self.assertAlmostEqual(
                fit.chi2r,
                0.0
            )
            self.assertAlmostEqual(
                model.parameter_dict['a'].value,
                a_value
            )
            self.assertAlmostEqual(
                model.parameter_dict['c'].value,
                c_value
            )
            # The number of "free" parameters corresponds to the number
            # of fitting parameters
            self.assertEqual(
                fit.n_free,
                len(fit.model.parameters)
            )

        curves = fit.get_curves()

    def test_fit_data_setter(self):
        c_value = 3.1
        a_value = 1.2
        x_data, y_data = get_data_values(
            a_value=a_value,
            c_value=c_value
        )

        data = chisurf.experiments.data.DataCurve(
            x=x_data,
            y=y_data,
            ey=np.ones_like(y_data)
        )
        fit = chisurf.fitting.fit.FitGroup(
            data=chisurf.experiments.data.DataGroup(
                [data]
            ),
            model_class=chisurf.models.parse.ParseModel
        )

        self.assertIs(
            fit.data,
            data
        )

        data_2 = chisurf.experiments.data.DataCurve(
            x=x_data,
            y=y_data,
            ey=np.ones_like(y_data)
        )

        self.assertIsNot(
            data,
            data_2
        )

        fit.data = data_2
        self.assertIs(
            fit.data,
            data_2
        )

    def test_fit_sample(self):
        import chisurf.experiments
        import chisurf.models
        import chisurf.fitting

        c_value = 3.1
        a_value = 1.2
        x_data, y_data = get_data_values(
            a_value=a_value,
            c_value=c_value
        )

        data = chisurf.experiments.data.DataCurve(
            x=x_data,
            y=y_data,
            ey=np.ones_like(y_data)
        )
        fit = chisurf.fitting.fit.FitGroup(
            data=chisurf.experiments.data.DataGroup(
                [data]
            ),
            model_class=chisurf.models.parse.ParseModel
        )
        fit.fit_range = 0, len(fit.model.y)

        model = fit.model
        model.func = 'c+a*x**2'
        fit.model.find_parameters()
        r = chisurf.fitting.sample.sample_emcee(
            fit=fit,
            steps=100,
            nwalkers=5,
            thin=10
        )
        self.assertSetEqual(
            {'chi2r', 'parameter_values', 'parameter_names'},
            set(r.keys())
        )
        self.assertEqual(
            len(r['chi2r']),
            50
        )

        # There is an alternative sampler that directly saves to files
        _, filename = tempfile.mkstemp(
            suffix='.er4'
        )
        n_runs = 5
        sampling_method = 'emcee'
        chisurf.fitting.fit.sample_fit(
            fit=fit,
            filename=filename,
            steps=100,
            thin=1,
            n_runs=n_runs,
            method=sampling_method
        )

        # Every run is a file. Test if all the runs are written out
        for i_run in range(n_runs):
            fn = os.path.splitext(filename)[0] + "_" + str(i_run) + '.er4'
            self.assertTrue(
                os.path.isfile(
                    fn
                )
            )

        fit.run()
        r = chisurf.fitting.sample.walk_mcmc(
            fit=fit,
            steps=50,
            step_size=0.1,
            chi2max=10,
            temp=1,
            thin=1
        )
        self.assertEqual(
            len(r['chi2r']),
            50
        )

        sampling_method = 'mcmc'
        n_runs = 1
        fit.run()
        chisurf.fitting.fit.sample_fit(
            fit=fit,
            filename=filename,
            steps=50,
            thin=1,
            n_runs=n_runs,
            method=sampling_method
        )

        fit.run()

        import chisurf.fitting.support_plane
        r = chisurf.fitting.support_plane.scan_parameter(
            fit=fit,
            parameter_name='c',
            rel_range=0.2,
            n_steps=20
        )
        self.assertEqual(
            len(r['chi2r']),
            20
        )

