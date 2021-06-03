import utils
import os
import unittest

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)

import chisurf.parameter
import chisurf.models
import chisurf.fitting


class Tests(unittest.TestCase):

    def test_get_instances(self):
        p1 = chisurf.parameter.Parameter()
        initial_instances = len(list(p1.get_instances()))
        self.assertEqual(p1 in p1.get_instances(), True)

        p2 = chisurf.parameter.Parameter()
        self.assertEqual(
            len(list(p1.get_instances())), initial_instances + 1
        )
        self.assertEqual(p2 in p1.get_instances(), True)

    def test_create(self):
        p1 = chisurf.parameter.Parameter()
        p1.value = 2.0
        self.assertEqual(p1.value, 2.0)

        p2 = chisurf.parameter.Parameter(value=2.0)
        self.assertEqual(p2.value, 2.0)

    def test_equality(self):
        p1 = chisurf.parameter.Parameter(value=2.0)
        p2 = chisurf.parameter.Parameter(value=2.0)
        self.assertEqual(p1, p2)
        self.assertIsNot(p1, p2)

    def test_arithmetics(self):
        p1 = chisurf.parameter.Parameter(value=2.0)
        p2 = chisurf.parameter.Parameter(value=3.0)

        p3 = p1 + p2
        self.assertEqual(p3.value, 5.0)

        p3 = p1 - p2
        self.assertEqual(p3.value, -1.0)

        p3 = p1 * p2
        self.assertEqual(p3.value, 6.0)

        p3 = p1 / p2
        self.assertEqual(p3.value, 2. / 3.)

        p3 = p1 // p2
        self.assertEqual(p3.value, 2. // 3.)

        p3 = p1 % p2
        self.assertEqual(p3.value, 2. % 3.)

        p3 = p1 ** p2
        self.assertEqual(p3.value, 2. ** 3.)

    def test_linking(self):
        p1 = chisurf.parameter.Parameter(value=2.0)
        p2 = chisurf.parameter.Parameter(value=3.0)
        self.assertEqual(p1.value, 2.0)
        self.assertEqual(p2.value, 3.0)
        self.assertEqual(p2.is_linked, False)

        p2.link = p1
        self.assertEqual(p2.value, 2.0)
        self.assertEqual(p2.is_linked, True)

        # The original value is NOT overwritten with the
        # linked value once the parameters are unlinked
        p2.link = None
        self.assertEqual(p2.value, 3.0)
        self.assertEqual(p2.is_linked, False)

        p2.value = 3
        self.assertEqual(p2.value, 3.0)

    def test_restore_link_from_dict(self):
        p1 = chisurf.parameter.Parameter(value=2.0)
        p2 = chisurf.parameter.Parameter(value=3.0)
        p2.link = p1
        p3 = chisurf.parameter.Parameter()
        p3.from_dict(
            p2.to_dict()
        )
        self.assertEqual(p3.value, 2.0)

    def test_fixing(self):
        p1 = chisurf.parameter.Parameter(value=2.0)
        p1.fixed = True
        self.assertEqual(
            p1.fixed,
            True
        )
        p1.fixed = False
        self.assertEqual(
            p1.fixed,
            False
        )

    def test_bounds(self):
        p1 = chisurf.parameter.Parameter(
            value=2.0,
            bounds_on=True,
            lb=1.,
            ub=2.5
        )
        self.assertEqual(p1.value, 2.0)
        p1.value = 5.0
        self.assertEqual(p1.value, 2.5)

    def test_rep_str(self):
        p1 = chisurf.parameter.Parameter(22)
        self.assertEqual(
            p1.__repr__(),
            "22"
        )

    @unittest.expectedFailure
    def test_dict(self):
        d1 = {
            'value': 2.0,
            'bounds_on': True,
            'lb': 1.,
            'ub': 2.5,
            'unique_identifier': 'b671b0b3-3009-42df-824a-6d690c2b3e54'
        }
        p1 = chisurf.parameter.Parameter(**d1)
        d3 = {
            'name': 'Parameter',
            'verbose': False,
            'unique_identifier': 'b671b0b3-3009-42df-824a-6d690c2b3e54',
            'bounds_on': True,
            'controller': None,
            '_link': None,
            '_port': 2.0,
            'lb': 1.0,
            'ub': 2.5
        }
        self.assertEqual(
            p1.to_dict(),
            d3
        )

    def test_save_load(self):

        import tempfile

        #file = tempfile.NamedTemporaryFile(
        #    suffix='.json'
        #)
        #filename = file.name

        _, filename = tempfile.mkstemp(
            suffix='.json'
        )

        p1 = chisurf.parameter.Parameter(
            value=2.0,
            bounds_on=True,
            lb=1.,
            ub=2.5
        )
        p1.save(
            filename,
            file_type='json'
        )

        p2 = chisurf.parameter.Parameter()
        p2.load(
            filename=filename,
            file_type='json'
        )

    def test_parameter_group(self):
        p1 = chisurf.parameter.Parameter(
            value=22,
            name='p1'
        )
        p2 = chisurf.parameter.Parameter(
            value=11,
            name='p2'
        )
        group_name = 'Parameter Gruppe'
        pg = chisurf.parameter.ParameterGroup(
            name=group_name
        )
        self.assertEqual(
            pg.name,
            group_name
        )
        pg.append(p1)
        pg.append(p2)
        self.assertEqual(
            pg.values,
            [22, 11]
        )
        self.assertListEqual(
            pg.parameter_names,
            ['p1', 'p2']
        )

    def test_fitting_parameter(self):
        p1 = chisurf.fitting.parameter.FittingParameter(value=22)

        value = 11
        link = p1
        lower_bound = 11
        upper_bound = 33
        bounds_on = True
        name = 'Name_P1'
        verbose = True
        unique_identifier = None
        fixed = True
        p2 = chisurf.fitting.parameter.FittingParameter(
            fixed=fixed,
            value=value,
            link=link,
            lb=lower_bound,
            ub=upper_bound,
            bounds_on=bounds_on,
            name=name,
            verbose=verbose,
            unique_identifier=unique_identifier
        )

        self.assertEqual(
            p1.value,
            p2.value
        )

        p3 = chisurf.fitting.parameter.FittingParameter()
        p3.from_dict(
            p2.to_dict()
        )
        self.assertEqual(
            p3.link,
            p1
        )

        self.assertEqual(
            p2.fixed,
            fixed
        )

    def test_fitting_parameter_group(self):
        p1 = chisurf.fitting.parameter.FittingParameter(value=22)
        p2 = chisurf.fitting.parameter.FittingParameter(value=33)
        pg = chisurf.fitting.parameter.FittingParameterGroup(name="jjk")
        pg.append(p1)
        pg.append(p2)
        pg.find_parameters(
            chisurf.fitting.parameter.FittingParameter
        )

    # def test_numpy(self):
    #     import numpy as np
    #     value = 22
    #     p1 = chisurf.fitting.parameter.FittingParameter(value=value)
    #     x = np.linspace(0, 2, 100)
    #     p2 = p1 + x
    #     self.assertEqual(
    #         type(p2),
    #         chisurf.fitting.parameter.FittingParameter
    #     )
    #     self.assertEqual(
    #         np.allclose(
    #             p2.value,
    #             x + value
    #         ),
    #         True
    #     )

    def test_abs(self):
        value = -11
        p1 = chisurf.fitting.parameter.FittingParameter(value=value)
        p2 = abs(p1)
        self.assertEqual(
            abs(p1.value),
            p2.value
        )

    def test_parameter_group(self):

        class A(chisurf.parameter.ParameterGroup):

            def __init__(self):
                self.pv = chisurf.parameter.Parameter(
                    value=11
                )

        # The values of Parameters that are grouped in a ParameterGroup
        # can be written to without explicitly addressing the Parameter
        #  value attribute
        a = A()
        self.assertEqual(
            a.pv.value,
            11
        )
        a.pv = 22
        self.assertEqual(
            a.pv.value,
            22
        )


if __name__ == '__main__':
    unittest.main()
