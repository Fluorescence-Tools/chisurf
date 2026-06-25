import utils
import os
import unittest
import pathlib

TOPDIR = pathlib.Path(__file__).parent.parent

utils.set_search_paths(TOPDIR)

import chisurf.core.parameter
import chisurf.core.models
import chisurf.core.fitting


class Tests(unittest.TestCase):

    def test_get_instances(self):
        p1 = chisurf.core.parameter.Parameter()
        initial_instances = len(list(p1.get_instances()))
        self.assertEqual(p1 in p1.get_instances(), True)

        p2 = chisurf.core.parameter.Parameter()
        self.assertEqual(
            len(list(p1.get_instances())), initial_instances + 1
        )
        self.assertEqual(p2 in p1.get_instances(), True)

    def test_create(self):
        p1 = chisurf.core.parameter.Parameter()
        p1.value = 2.0
        self.assertEqual(p1.value, 2.0)

        p2 = chisurf.core.parameter.Parameter(value=2.0)
        self.assertEqual(p2.value, 2.0)

    def test_equality(self):
        p1 = chisurf.core.parameter.Parameter(value=2.0)
        p2 = chisurf.core.parameter.Parameter(value=2.0)
        self.assertEqual(p1, p2)
        self.assertIsNot(p1, p2)

    def test_arithmetics(self):
        p1 = chisurf.core.parameter.Parameter(value=2.0)
        p2 = chisurf.core.parameter.Parameter(value=3.0)

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
        p1 = chisurf.core.parameter.Parameter(value=2.0)
        p2 = chisurf.core.parameter.Parameter(value=3.0)
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

    def test_cyclic_link_rejected(self):
        # The parameter link graph must remain a DAG. check_recursive_link
        # uses Kahn's algorithm to reject any assignment that would create a
        # cycle, mirroring chinet's node-graph validation.
        Parameter = chisurf.core.parameter.Parameter
        a = Parameter(value=1.0)
        b = Parameter(value=2.0)
        c = Parameter(value=3.0)

        # Self-link is a (degenerate) cycle.
        self.assertTrue(Parameter.check_recursive_link(a, a))

        # Build the chain a <- b <- c (b follows a, c follows b).
        b.link = a
        c.link = b

        # Closing the loop a -> c would create a cycle and must be rejected.
        self.assertTrue(Parameter.check_recursive_link(c, a))
        with self.assertRaises(ValueError):
            a.link = c

        # A non-cyclic cross link is still allowed.
        d = Parameter(value=4.0)
        self.assertFalse(Parameter.check_recursive_link(d, c))
        c.link = d  # should not raise
        self.assertTrue(c.is_linked)

    def test_restore_link_from_dict(self):
        p1 = chisurf.core.parameter.Parameter(value=2.0)
        p2 = chisurf.core.parameter.Parameter(value=3.0)
        p2.link = p1
        p3 = chisurf.core.parameter.Parameter()
        p3.from_dict(
            p2.to_dict()
        )
        self.assertEqual(p3.value, 2.0)

    def test_fixing(self):
        p1 = chisurf.core.parameter.Parameter(value=2.0)
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
        p1 = chisurf.core.parameter.Parameter(
            value=2.0,
            bounds_on=True,
            lb=1.,
            ub=2.5
        )
        self.assertEqual(p1.value, 2.0)
        p1.value = 5.0
        self.assertEqual(p1.value, 2.5)

    def test_rep_str(self):
        p1 = chisurf.core.parameter.Parameter(22)
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
        p1 = chisurf.core.parameter.Parameter(**d1)
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

        p1 = chisurf.core.parameter.Parameter(
            value=2.0,
            bounds_on=True,
            lb=1.,
            ub=2.5
        )
        p1.save(
            filename,
            file_type='json'
        )

        p2 = chisurf.core.parameter.Parameter()
        p2.load(
            filename=filename,
            file_type='json'
        )

    def test_parameter_group(self):
        p1 = chisurf.core.parameter.Parameter(
            value=22,
            name='p1'
        )
        p2 = chisurf.core.parameter.Parameter(
            value=11,
            name='p2'
        )
        group_name = 'Parameter Gruppe'
        pg = chisurf.core.parameter.ParameterGroup(
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
        p1 = chisurf.core.fitting.parameter.FittingParameter(value=22)

        value = 11
        link = p1
        lower_bound = 11
        upper_bound = 33
        bounds_on = True
        name = 'Name_P1'
        verbose = True
        unique_identifier = None
        fixed = True
        p2 = chisurf.core.fitting.parameter.FittingParameter(
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

        p3 = chisurf.core.fitting.parameter.FittingParameter()
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
        p1 = chisurf.core.fitting.parameter.FittingParameter(value=22)
        p2 = chisurf.core.fitting.parameter.FittingParameter(value=33)
        pg = chisurf.core.fitting.parameter.FittingParameterGroup(name="jjk")
        pg.append(p1)
        pg.append(p2)
        pg.find_parameters(
            chisurf.core.fitting.parameter.FittingParameter
        )

    # def test_numpy(self):
    #     import numpy as np
    #     value = 22
    #     p1 = chisurf.core.fitting.parameter.FittingParameter(value=value)
    #     x = np.linspace(0, 2, 100)
    #     p2 = p1 + x
    #     self.assertEqual(
    #         type(p2),
    #         chisurf.core.fitting.parameter.FittingParameter
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
        p1 = chisurf.core.fitting.parameter.FittingParameter(value=value)
        p2 = abs(p1)
        self.assertEqual(
            abs(p1.value),
            p2.value
        )


    def test_parameter_get_set_state_roundtrip(self):
        """Parameter.get_state/set_state should round-trip basic port state."""

        p1 = chisurf.core.parameter.Parameter(
            value=2.0,
            bounds_on=True,
            lb=1.0,
            ub=3.0
        )
        s = p1.get_state()

        p2 = chisurf.core.parameter.Parameter()
        # Ensure defaults differ
        self.assertNotEqual(p2.value, 2.0)

        p2.set_state(s)
        self.assertEqual(p2.bounds_on, True)
        self.assertAlmostEqual(p2.value, 2.0)


    def test_parameter_group_get_set_state_does_not_crash(self):
        """ParameterGroup.get_state/set_state should be callable and benign."""

        p1 = chisurf.core.parameter.Parameter(value=11, name='p1')
        p2 = chisurf.core.parameter.Parameter(value=22, name='p2')
        pg = chisurf.core.parameter.ParameterGroup(parameters=[p1, p2])
        state = pg.get_state()

        # Re-apply to the same group; primarily a smoke test.
        pg.set_state(state)
        self.assertEqual(len(pg.parameters), 2)


    def test_fitting_parameter_group_get_set_state_roundtrip(self):
        """FittingParameterGroup.get_state/set_state should be JSON-safe.

        The test focuses on ensuring that the returned state is
        JSON-serializable and that calling :meth:`set_state` does not raise
        and keeps the parameter structure intact. Detailed value round-trips
        are covered at the individual :class:`Parameter` level.
        """

        import json

        p1 = chisurf.core.fitting.parameter.FittingParameter(value=22, name='p1')
        p2 = chisurf.core.fitting.parameter.FittingParameter(value=33, name='p2')
        pg = chisurf.core.fitting.parameter.FittingParameterGroup(name="grp1")
        pg.append(p1)
        pg.append(p2)
        pg.find_parameters(chisurf.core.fitting.parameter.FittingParameter)

        state = pg.get_state()
        # Must be JSON-serializable without custom encoders
        json.dumps(state)

        # Calling set_state on the same group should not crash and should
        # preserve the parameter structure.
        pg.set_state(state)
        params2 = pg.parameters_all_dict
        self.assertIn('p1', params2)
        self.assertIn('p2', params2)


    def test_parameter_group(self):

        class A(chisurf.core.parameter.ParameterGroup):

            def __init__(self):
                self.pv = chisurf.core.parameter.Parameter(
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
